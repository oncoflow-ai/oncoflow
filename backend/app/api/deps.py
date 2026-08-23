from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt
from sqlalchemy.orm import Session
from uuid import UUID
from typing import Generator

from app.core.config import get_settings
from app.infra.db.session import create_session_factory
from app.infra.db.models import User, Patient, Assignment, Study

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")
optional_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


def get_session() -> Generator[Session, None, None]:
    factory = create_session_factory()
    with factory() as session:
        yield session


def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session)
) -> User:
    settings = get_settings()
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        user_id_str: str | None = payload.get("sub")
        if user_id_str is None:
            raise credentials_exception
    except jwt.InvalidTokenError:
        raise credentials_exception
        
    try:
        user_uuid = UUID(user_id_str)
    except ValueError:
        raise credentials_exception
        
    user = session.query(User).filter(User.public_id == user_uuid).first()
    if user is None:
        raise credentials_exception
        
    return user


def get_optional_current_user(
    token: str | None = Depends(optional_oauth2_scheme),
    session: Session = Depends(get_session)
) -> User | None:
    if not token:
        return None
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        user_id_str = payload.get("sub")
        if not user_id_str:
            return None
        user_uuid = UUID(user_id_str)
        user = session.query(User).filter(User.public_id == user_uuid).first()
        return user
    except Exception:
        return None


def verify_patient_access(
    patient: Patient,
    user: User,
    session: Session,
) -> None:
    """Verifies that the user has permission to access the patient record."""
    if user.role == "admin":
        return  # Admin has global access
    
    # Check if doctor/clinician is assigned
    assignment = session.query(Assignment).filter(
        Assignment.doctor_id == user.id,
        Assignment.patient_id == patient.id,
    ).first()
    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this patient record",
        )


def verify_study_access(
    study: Study,
    user: User,
    session: Session,
) -> None:
    """Verify access to the patient who owns a study."""
    if user.role == "admin":
        return

    patient = None
    if study.patient_id is not None:
        patient = session.query(Patient).filter(Patient.id == study.patient_id).one_or_none()
    if patient is None:
        patient = (
            session.query(Patient)
            .filter(Patient.public_id == study.patient_public_id)
            .one_or_none()
        )
    if patient is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="study not found")
    verify_patient_access(patient, user, session)
