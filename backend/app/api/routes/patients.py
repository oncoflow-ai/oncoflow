from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.api.deps import (
    get_current_user,
    get_session,
    verify_patient_access,
)
from app.api.schemas.patients import (
    AssignedDoctorResponse,
    AssignmentRequest,
    PatientCreate,
    PatientDetailResponse,
    PatientResponse,
    PatientStudyItemResponse,
    PatientUpdate,
)
from app.core.audit import log_audit_event
from app.infra.db.models import Assignment, Patient, Study, User

router = APIRouter(prefix="/patients", tags=["patients"])
ASSIGNABLE_PATIENT_ROLES = {"doctor", "clinician", "radiologist"}


def _find_patient(patient_id: str, session: Session) -> Patient:
    try:
        patient_uuid = UUID(patient_id)
        patient = session.query(Patient).filter(Patient.public_id == patient_uuid).first()
        if patient:
            return patient
    except ValueError:
        pass

    # Try matching by pseudonym or integer ID
    patient = session.query(Patient).filter(Patient.pseudonym == patient_id).first()
    if patient:
        return patient

    try:
        int_id = int(patient_id)
        patient = session.query(Patient).filter(Patient.id == int_id).first()
        if patient:
            return patient
    except ValueError:
        pass

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Patient {patient_id} not found")


def _find_user(user_id: str, session: Session) -> User:
    try:
        user_uuid = UUID(user_id)
        user = session.query(User).filter(User.public_id == user_uuid).first()
        if user:
            return user
    except ValueError:
        pass

    try:
        int_id = int(user_id)
        user = session.query(User).filter(User.id == int_id).first()
        if user:
            return user
    except ValueError:
        pass

    user = session.query(User).filter(User.email == user_id).first()
    if user:
        return user

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User {user_id} not found")


def _build_patient_response(patient: Patient, session: Session) -> PatientResponse:
    studies = (
        session.query(Study)
        .filter((Study.patient_id == patient.id) | (Study.patient_public_id == patient.public_id))
        .order_by(desc(Study.acquired_at), desc(Study.created_at))
        .all()
    )
    scan_count = len(studies)
    last_scan_date = None
    if studies:
        latest = studies[0]
        last_scan_date = latest.acquired_at or latest.created_at.date()

    # Get primary assigned doctor if any
    first_assignment = (
        session.query(Assignment)
        .filter(Assignment.patient_id == patient.id)
        .first()
    )
    assigned_physician_id = None
    if first_assignment and first_assignment.doctor:
        assigned_physician_id = str(first_assignment.doctor.public_id)

    return PatientResponse(
        id=str(patient.public_id),
        pseudonym=patient.pseudonym,
        name=patient.pseudonym,  # For privacy, pseudonym is the display name
        dob=patient.dob,
        gender=patient.gender,
        diagnosis=patient.diagnosis,
        diagnosis_location=patient.diagnosis_location,
        status=patient.status,
        notes=patient.notes,
        scan_count=scan_count,
        last_scan_date=last_scan_date,
        assigned_physician_id=assigned_physician_id,
        created_at=patient.created_at,
        updated_at=patient.updated_at,
    )


@router.get("", response_model=list[PatientResponse])
def list_patients(
    query: str | None = Query(None, description="Search term for pseudonym or diagnosis"),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> list[PatientResponse]:
    q = session.query(Patient)

    # If user is a clinician/doctor (not admin), scope to assigned patients
    if current_user.role != "admin":
        assigned_patient_ids = (
            session.query(Assignment.patient_id)
            .filter(Assignment.doctor_id == current_user.id)
            .scalar_subquery()
        )
        q = q.filter(Patient.id.in_(assigned_patient_ids))


    if query:
        search_pattern = f"%{query}%"
        q = q.filter(
            (Patient.pseudonym.ilike(search_pattern))
            | (Patient.diagnosis.ilike(search_pattern))
            | (Patient.diagnosis_location.ilike(search_pattern))
        )

    patients = q.order_by(desc(Patient.created_at)).all()
    log_audit_event(
        action="LIST_PATIENTS",
        resource_id="patients_list",
        actor=str(current_user.public_id),
        details={"result_count": len(patients)},
    )
    return [_build_patient_response(p, session) for p in patients]


@router.post("", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
def create_patient(
    payload: PatientCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> PatientResponse:
    doctor_to_assign: User | None = None
    if payload.assigned_physician_id:
        if current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only administrators can assign a physician during patient creation",
            )
        doctor_to_assign = _find_user(payload.assigned_physician_id, session)
        if doctor_to_assign.role not in ASSIGNABLE_PATIENT_ROLES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Patient assignments require a doctor, clinician, or radiologist",
            )

    pseudonym = payload.pseudonym or payload.name
    if not pseudonym:
        pseudonym = f"PAT-{uuid4().hex[:6].upper()}"

    # Check if pseudonym already exists
    existing = session.query(Patient).filter(Patient.pseudonym == pseudonym).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Patient with pseudonym {pseudonym} already exists",
        )

    patient = Patient(
        public_id=uuid4(),
        pseudonym=pseudonym,
        dob=payload.dob,
        gender=payload.gender,
        diagnosis=payload.diagnosis,
        diagnosis_location=payload.diagnosis_location,
        status=payload.status or "active",
        notes=payload.notes,
    )
    session.add(patient)
    session.flush()

    # Assign doctor if specified or if logged in as doctor
    if doctor_to_assign is None and not payload.assigned_physician_id and current_user.role in ASSIGNABLE_PATIENT_ROLES:
        doctor_to_assign = current_user

    if doctor_to_assign:
        assignment = Assignment(
            doctor_id=doctor_to_assign.id,
            patient_id=patient.id,
        )
        session.add(assignment)
        session.flush()

    session.commit()
    session.refresh(patient)

    log_audit_event(
        action="CREATE_PATIENT",
        resource_id=str(patient.public_id),
        actor=str(current_user.public_id),
        details={"pseudonym": patient.pseudonym},
    )

    return _build_patient_response(patient, session)


@router.get("/{patient_id}", response_model=PatientDetailResponse)
def get_patient(
    patient_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> PatientDetailResponse:
    patient = _find_patient(patient_id, session)
    verify_patient_access(patient, current_user, session)

    base_resp = _build_patient_response(patient, session)

    # Get assigned doctors
    assignments = (
        session.query(Assignment)
        .filter(Assignment.patient_id == patient.id)
        .all()
    )
    assigned_doctors = [
        AssignedDoctorResponse(
            id=str(a.doctor.public_id),
            name=a.doctor.name,
            email=a.doctor.email,
            role=a.doctor.role,
            assigned_at=a.assigned_at,
        )
        for a in assignments
        if a.doctor
    ]

    # Get serial studies
    studies = (
        session.query(Study)
        .filter((Study.patient_id == patient.id) | (Study.patient_public_id == patient.public_id))
        .order_by(desc(Study.acquired_at), desc(Study.created_at))
        .all()
    )
    study_items = [
        PatientStudyItemResponse(
            study_id=str(s.public_id),
            study_instance_uid=s.study_instance_uid,
            source_kind=s.source_kind,
            staging_status=s.staging_status,
            acquired_at=s.acquired_at,
            created_at=s.created_at,
        )
        for s in studies
    ]

    log_audit_event(
        action="VIEW_PATIENT",
        resource_id=str(patient.public_id),
        actor=str(current_user.public_id),
        details={"studies_count": len(study_items)},
    )

    return PatientDetailResponse(
        **base_resp.model_dump(),
        assigned_doctors=assigned_doctors,
        studies=study_items,
    )


@router.patch("/{patient_id}", response_model=PatientResponse)
def update_patient(
    patient_id: str,
    payload: PatientUpdate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> PatientResponse:
    patient = _find_patient(patient_id, session)
    verify_patient_access(patient, current_user, session)

    updates = payload.model_dump(exclude_unset=True)
    for field, value in updates.items():
        setattr(patient, field, value)

    session.commit()
    session.refresh(patient)

    log_audit_event(
        action="UPDATE_PATIENT",
        resource_id=str(patient.public_id),
        actor=str(current_user.public_id),
        details={"updated_fields": list(updates.keys())},
    )

    return _build_patient_response(patient, session)


@router.post("/{patient_id}/assignments", response_model=AssignedDoctorResponse, status_code=status.HTTP_201_CREATED)
def assign_doctor_to_patient(
    patient_id: str,
    payload: AssignmentRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> AssignedDoctorResponse:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can manage patient assignments",
        )

    patient = _find_patient(patient_id, session)
    doctor = _find_user(payload.doctor_id, session)
    if doctor.role not in ASSIGNABLE_PATIENT_ROLES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Patient assignments require a doctor, clinician, or radiologist",
        )

    # Check if assignment already exists
    existing = (
        session.query(Assignment)
        .filter(Assignment.doctor_id == doctor.id, Assignment.patient_id == patient.id)
        .first()
    )
    if existing:
        return AssignedDoctorResponse(
            id=str(doctor.public_id),
            name=doctor.name,
            email=doctor.email,
            role=doctor.role,
            assigned_at=existing.assigned_at,
        )

    assignment = Assignment(
        doctor_id=doctor.id,
        patient_id=patient.id,
    )
    session.add(assignment)
    session.commit()
    session.refresh(assignment)

    log_audit_event(
        action="ASSIGN_DOCTOR",
        resource_id=str(patient.public_id),
        actor=str(current_user.public_id),
        details={"doctor_public_id": str(doctor.public_id), "doctor_name": doctor.name},
    )

    return AssignedDoctorResponse(
        id=str(doctor.public_id),
        name=doctor.name,
        email=doctor.email,
        role=doctor.role,
        assigned_at=assignment.assigned_at,
    )


@router.delete("/{patient_id}/assignments/{doctor_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_doctor_assignment(
    patient_id: str,
    doctor_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> None:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can manage patient assignments",
        )

    patient = _find_patient(patient_id, session)
    doctor = _find_user(doctor_id, session)

    assignment = (
        session.query(Assignment)
        .filter(Assignment.doctor_id == doctor.id, Assignment.patient_id == patient.id)
        .first()
    )
    if assignment:
        session.delete(assignment)
        session.commit()
        log_audit_event(
            action="UNASSIGN_DOCTOR",
            resource_id=str(patient.public_id),
            actor=str(current_user.public_id),
            details={"doctor_public_id": str(doctor.public_id)},
        )


@router.get("/{patient_id}/studies", response_model=list[PatientStudyItemResponse])
def get_patient_studies(
    patient_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> list[PatientStudyItemResponse]:
    patient = _find_patient(patient_id, session)
    verify_patient_access(patient, current_user, session)

    studies = (
        session.query(Study)
        .filter((Study.patient_id == patient.id) | (Study.patient_public_id == patient.public_id))
        .order_by(desc(Study.acquired_at), desc(Study.created_at))
        .all()
    )

    return [
        PatientStudyItemResponse(
            study_id=str(s.public_id),
            study_instance_uid=s.study_instance_uid,
            source_kind=s.source_kind,
            staging_status=s.staging_status,
            acquired_at=s.acquired_at,
            created_at=s.created_at,
        )
        for s in studies
    ]
