from __future__ import annotations

import io
import zipfile
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.security import create_access_token
from app.infra.db.models import Assignment, AuditLog, Patient, Study, User
from app.infra.db.session import create_session_factory
from app.main import create_app


@pytest.fixture
def app_and_client():
    app = create_app()
    client = TestClient(app)
    return app, client


@pytest.fixture
def auth_tokens(app_and_client):
    app, client = app_and_client
    session_factory = create_session_factory()

    with session_factory() as session:
        # Create Dr A
        dr_a = User(
            email="dr.a@test.local",
            name="Dr. Alpha",
            hashed_password="hash",
            role="doctor",
        )
        # Create Dr B
        dr_b = User(
            email="dr.b@test.local",
            name="Dr. Beta",
            hashed_password="hash",
            role="doctor",
        )
        # Create Admin
        admin = User(
            email="admin@test.local",
            name="Admin User",
            hashed_password="hash",
            role="admin",
        )
        clinician = User(
            email="clinician@test.local",
            name="Clinical User",
            hashed_password="hash",
            role="clinician",
        )
        radiologist = User(
            email="radiologist@test.local",
            name="Radiology User",
            hashed_password="hash",
            role="radiologist",
        )
        researcher = User(
            email="researcher@test.local",
            name="Research User",
            hashed_password="hash",
            role="researcher",
        )
        session.add_all([dr_a, dr_b, admin, clinician, radiologist, researcher])
        session.flush()

        token_dr_a = create_access_token({"sub": str(dr_a.public_id), "role": "doctor"})
        token_dr_b = create_access_token({"sub": str(dr_b.public_id), "role": "doctor"})
        token_admin = create_access_token({"sub": str(admin.public_id), "role": "admin"})

        dr_a_id = str(dr_a.public_id)
        dr_b_id = str(dr_b.public_id)
        admin_id = str(admin.public_id)
        clinician_id = str(clinician.public_id)
        radiologist_id = str(radiologist.public_id)
        researcher_id = str(researcher.public_id)

        session.commit()

    return {
        "dr_a": {"token": token_dr_a, "id": dr_a_id},
        "dr_b": {"token": token_dr_b, "id": dr_b_id},
        "admin": {"token": token_admin, "id": admin_id},
        "clinician": {"id": clinician_id},
        "radiologist": {"id": radiologist_id},
        "researcher": {"id": researcher_id},
    }


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("get", "/api/v1/patients", {}),
        ("post", "/api/v1/patients", {"json": {"name": "PAT-ANON-CREATE"}}),
        ("get", "/api/v1/patients/missing-patient", {}),
        (
            "patch",
            "/api/v1/patients/missing-patient",
            {"json": {"status": "review"}},
        ),
        ("get", "/api/v1/patients/missing-patient/studies", {}),
        (
            "post",
            "/api/v1/patients/missing-patient/assignments",
            {"json": {"doctorId": "missing-doctor"}},
        ),
        (
            "delete",
            "/api/v1/patients/missing-patient/assignments/missing-doctor",
            {},
        ),
    ],
)
def test_patient_routes_require_authentication(
    app_and_client,
    method: str,
    path: str,
    kwargs: dict,
) -> None:
    _, client = app_and_client

    response = client.request(method, path, **kwargs)

    assert response.status_code == 401
    with create_session_factory()() as session:
        assert session.query(Patient).count() == 0
        assert session.query(Assignment).count() == 0


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("get", "/api/v1/patients", {}),
        ("post", "/api/v1/patients", {"json": {"name": "PAT-BAD-TOKEN"}}),
    ],
)
def test_patient_routes_reject_invalid_tokens_without_mutation(
    app_and_client,
    method: str,
    path: str,
    kwargs: dict,
) -> None:
    _, client = app_and_client

    response = client.request(
        method,
        path,
        headers={"Authorization": "Bearer not-a-valid-jwt"},
        **kwargs,
    )

    assert response.status_code == 401
    with create_session_factory()() as session:
        assert session.query(Patient).count() == 0
        assert session.query(Assignment).count() == 0


def test_create_and_list_patients(app_and_client, auth_tokens):
    _, client = app_and_client
    headers_dr_a = {"Authorization": f"Bearer {auth_tokens['dr_a']['token']}"}

    # Create patient as Dr A
    create_resp = client.post(
        "/api/v1/patients",
        json={
            "name": "PAT-TEST-001",
            "diagnosis": "Osteosarcoma",
            "diagnosisLocation": "Left Tibia",
            "status": "active",
        },
        headers=headers_dr_a,
    )
    assert create_resp.status_code == 201
    patient_data = create_resp.json()
    assert patient_data["pseudonym"] == "PAT-TEST-001"
    assert patient_data["diagnosis"] == "Osteosarcoma"
    patient_id = patient_data["id"]

    # Dr A should see this patient in their list
    list_resp = client.get("/api/v1/patients", headers=headers_dr_a)
    assert list_resp.status_code == 200
    my_patients = list_resp.json()
    assert any(p["id"] == patient_id for p in my_patients)

    # Dr B should NOT see this patient in their list (not assigned)
    headers_dr_b = {"Authorization": f"Bearer {auth_tokens['dr_b']['token']}"}
    list_b_resp = client.get("/api/v1/patients", headers=headers_dr_b)
    assert list_b_resp.status_code == 200
    b_patients = list_b_resp.json()
    assert not any(p["id"] == patient_id for p in b_patients)

    # Admin should see all patients
    headers_admin = {"Authorization": f"Bearer {auth_tokens['admin']['token']}"}
    list_admin_resp = client.get("/api/v1/patients", headers=headers_admin)
    assert list_admin_resp.status_code == 200
    admin_patients = list_admin_resp.json()
    assert any(p["id"] == patient_id for p in admin_patients)


def test_create_patient_audit_stores_authenticated_actor(app_and_client, auth_tokens):
    _, client = app_and_client
    response = client.post(
        "/api/v1/patients",
        json={"name": "PAT-AUDIT-ACTOR"},
        headers={"Authorization": f"Bearer {auth_tokens['dr_a']['token']}"},
    )

    assert response.status_code == 201
    with create_session_factory()() as session:
        event = session.query(AuditLog).filter(
            AuditLog.action == "CREATE_PATIENT",
            AuditLog.resource_id == response.json()["id"],
        ).one()
        assert event.actor_id == auth_tokens["dr_a"]["id"]


def test_abac_patient_detail_access(app_and_client, auth_tokens):
    _, client = app_and_client
    headers_dr_a = {"Authorization": f"Bearer {auth_tokens['dr_a']['token']}"}
    headers_dr_b = {"Authorization": f"Bearer {auth_tokens['dr_b']['token']}"}
    headers_admin = {"Authorization": f"Bearer {auth_tokens['admin']['token']}"}

    create_resp = client.post(
        "/api/v1/patients",
        json={"name": "PAT-RESTRICTED-01", "diagnosis": "Chondrosarcoma"},
        headers=headers_dr_a,
    )
    assert create_resp.status_code == 201
    patient_id = create_resp.json()["id"]

    # Dr A (assigned) -> 200
    resp_a = client.get(f"/api/v1/patients/{patient_id}", headers=headers_dr_a)
    assert resp_a.status_code == 200
    assert resp_a.json()["pseudonym"] == "PAT-RESTRICTED-01"

    # Dr B (unassigned) -> 403 Forbidden
    resp_b = client.get(f"/api/v1/patients/{patient_id}", headers=headers_dr_b)
    assert resp_b.status_code == 403
    assert "access" in resp_b.json()["detail"].lower()

    # Admin -> 200 OK
    resp_admin = client.get(f"/api/v1/patients/{patient_id}", headers=headers_admin)
    assert resp_admin.status_code == 200


def test_doctor_assignment_management(app_and_client, auth_tokens):
    _, client = app_and_client
    headers_admin = {"Authorization": f"Bearer {auth_tokens['admin']['token']}"}
    headers_dr_b = {"Authorization": f"Bearer {auth_tokens['dr_b']['token']}"}

    create_resp = client.post(
        "/api/v1/patients",
        json={"name": "PAT-ASSIGN-01"},
        headers=headers_admin,
    )
    patient_id = create_resp.json()["id"]

    # Initially Dr B cannot access
    resp_b_before = client.get(f"/api/v1/patients/{patient_id}", headers=headers_dr_b)
    assert resp_b_before.status_code == 403

    # Admin assigns Dr B
    assign_resp = client.post(
        f"/api/v1/patients/{patient_id}/assignments",
        json={"doctorId": auth_tokens["dr_b"]["id"]},
        headers=headers_admin,
    )
    assert assign_resp.status_code == 201
    assert assign_resp.json()["id"] == auth_tokens["dr_b"]["id"]

    # Now Dr B CAN access
    resp_b_after = client.get(f"/api/v1/patients/{patient_id}", headers=headers_dr_b)
    assert resp_b_after.status_code == 200

    # Remove assignment
    del_resp = client.delete(
        f"/api/v1/patients/{patient_id}/assignments/{auth_tokens['dr_b']['id']}",
        headers=headers_admin,
    )
    assert del_resp.status_code == 204

    # Dr B cannot access again
    resp_b_final = client.get(f"/api/v1/patients/{patient_id}", headers=headers_dr_b)
    assert resp_b_final.status_code == 403


def test_non_admin_cannot_manage_assignments(app_and_client, auth_tokens):
    _, client = app_and_client
    headers_dr_a = {"Authorization": f"Bearer {auth_tokens['dr_a']['token']}"}

    create_resp = client.post(
        "/api/v1/patients",
        json={"name": "PAT-NON-ADMIN-ASSIGN"},
        headers=headers_dr_a,
    )
    assert create_resp.status_code == 201
    patient_id = create_resp.json()["id"]

    assign_resp = client.post(
        f"/api/v1/patients/{patient_id}/assignments",
        json={"doctorId": auth_tokens["dr_b"]["id"]},
        headers=headers_dr_a,
    )
    remove_resp = client.delete(
        f"/api/v1/patients/{patient_id}/assignments/{auth_tokens['dr_a']['id']}",
        headers=headers_dr_a,
    )

    assert assign_resp.status_code == 403
    assert remove_resp.status_code == 403
    with create_session_factory()() as session:
        patient = session.query(Patient).filter(Patient.public_id == UUID(patient_id)).one()
        assignments = session.query(Assignment).filter(Assignment.patient_id == patient.id).all()
        assert [assignment.doctor_id for assignment in assignments] == [
            session.query(User)
            .filter(User.public_id == UUID(auth_tokens["dr_a"]["id"]))
            .one()
            .id
        ]


def test_non_admin_cannot_assign_during_patient_creation(app_and_client, auth_tokens):
    _, client = app_and_client
    response = client.post(
        "/api/v1/patients",
        json={
            "name": "PAT-CREATE-ASSIGN-BYPASS",
            "assignedPhysicianId": auth_tokens["researcher"]["id"],
        },
        headers={"Authorization": f"Bearer {auth_tokens['dr_a']['token']}"},
    )

    assert response.status_code == 403
    with create_session_factory()() as session:
        assert session.query(Patient).filter(
            Patient.pseudonym == "PAT-CREATE-ASSIGN-BYPASS"
        ).count() == 0


def test_admin_creation_rejects_non_clinical_assignment(app_and_client, auth_tokens):
    _, client = app_and_client
    response = client.post(
        "/api/v1/patients",
        json={
            "name": "PAT-CREATE-INVALID-ROLE",
            "assignedPhysicianId": auth_tokens["researcher"]["id"],
        },
        headers={"Authorization": f"Bearer {auth_tokens['admin']['token']}"},
    )

    assert response.status_code == 422


def test_admin_creation_propagates_missing_assignment_target(app_and_client, auth_tokens):
    _, client = app_and_client
    response = client.post(
        "/api/v1/patients",
        json={"name": "PAT-CREATE-MISSING-TARGET", "assignedPhysicianId": str(uuid4())},
        headers={"Authorization": f"Bearer {auth_tokens['admin']['token']}"},
    )

    assert response.status_code == 404


@pytest.mark.parametrize("target_key", ["clinician", "radiologist"])
def test_admin_can_assign_other_clinical_roles(app_and_client, auth_tokens, target_key: str):
    _, client = app_and_client
    headers_admin = {"Authorization": f"Bearer {auth_tokens['admin']['token']}"}
    patient_id = client.post(
        "/api/v1/patients",
        json={"name": f"PAT-ASSIGN-{target_key.upper()}"},
        headers=headers_admin,
    ).json()["id"]

    assign_resp = client.post(
        f"/api/v1/patients/{patient_id}/assignments",
        json={"doctorId": auth_tokens[target_key]["id"]},
        headers=headers_admin,
    )
    remove_resp = client.delete(
        f"/api/v1/patients/{patient_id}/assignments/{auth_tokens[target_key]['id']}",
        headers=headers_admin,
    )

    assert assign_resp.status_code == 201
    assert assign_resp.json()["role"] == target_key
    assert remove_resp.status_code == 204


@pytest.mark.parametrize("target_key", ["admin", "researcher"])
def test_admin_cannot_assign_non_clinical_roles(app_and_client, auth_tokens, target_key: str):
    _, client = app_and_client
    headers_admin = {"Authorization": f"Bearer {auth_tokens['admin']['token']}"}
    patient_id = client.post(
        "/api/v1/patients",
        json={"name": f"PAT-INVALID-TARGET-{target_key.upper()}"},
        headers=headers_admin,
    ).json()["id"]

    response = client.post(
        f"/api/v1/patients/{patient_id}/assignments",
        json={"doctorId": auth_tokens[target_key]["id"]},
        headers=headers_admin,
    )

    assert response.status_code == 422
    with create_session_factory()() as session:
        assert session.query(Assignment).count() == 0


def test_update_patient(app_and_client, auth_tokens):
    _, client = app_and_client
    headers_admin = {"Authorization": f"Bearer {auth_tokens['admin']['token']}"}

    create_resp = client.post(
        "/api/v1/patients",
        json={"name": "PAT-UPDATE-01", "status": "active"},
        headers=headers_admin,
    )
    patient_id = create_resp.json()["id"]

    patch_resp = client.patch(
        f"/api/v1/patients/{patient_id}",
        json={"status": "review", "diagnosis": "Updated Diagnosis"},
        headers=headers_admin,
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["status"] == "review"
    assert patch_resp.json()["diagnosis"] == "Updated Diagnosis"


def test_update_patient_clears_nullable_fields_and_preserves_omitted_values(
    app_and_client,
    auth_tokens,
):
    _, client = app_and_client
    headers_admin = {"Authorization": f"Bearer {auth_tokens['admin']['token']}"}
    created = client.post(
        "/api/v1/patients",
        json={
            "name": "PAT-CLEAR-NULLABLE",
            "diagnosis": "Original diagnosis",
            "notes": "keep this note",
            "status": "review",
        },
        headers=headers_admin,
    ).json()

    response = client.patch(
        f"/api/v1/patients/{created['id']}",
        json={"diagnosis": None},
        headers=headers_admin,
    )

    assert response.status_code == 200
    assert response.json()["diagnosis"] is None
    assert response.json()["notes"] == "keep this note"
    assert response.json()["status"] == "review"


@pytest.mark.parametrize("field_name", ["pseudonym", "status"])
def test_update_patient_rejects_null_non_nullable_fields(
    app_and_client,
    auth_tokens,
    field_name: str,
):
    _, client = app_and_client
    headers_admin = {"Authorization": f"Bearer {auth_tokens['admin']['token']}"}
    created = client.post(
        "/api/v1/patients",
        json={"name": f"PAT-NONNULL-{field_name}"},
        headers=headers_admin,
    ).json()

    response = client.patch(
        f"/api/v1/patients/{created['id']}",
        json={field_name: None},
        headers=headers_admin,
    )

    assert response.status_code == 422
