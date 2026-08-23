from __future__ import annotations

from pathlib import Path
import json
from uuid import UUID, uuid4

import pytest

from app.infra.db.models import Assignment, Artifact, Patient, StoredLesionResult, Study, StudyResult, User
from app.infra.db.session import create_session_factory
from app.modules.results.service import ResultNotFoundError, get_case_result_payload
from app.api.deps import get_current_user
from app.modules.artifacts.storage import resolve_artifact_location


@pytest.fixture(autouse=True)
def configure_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONCOFLOW_DATABASE_URL", f"sqlite+pysqlite:///{tmp_path / 'results-api.sqlite3'}")
    monkeypatch.setenv("ONCOFLOW_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("ONCOFLOW_STORAGE_STAGING_DIR", "raw")


@pytest.fixture(autouse=True)
def bypass_auth(client) -> None:
    client.app.dependency_overrides[get_current_user] = lambda: User(
        id="test-user",
        email="test@oncoflow.local",
        name="Test User",
        role="admin"
    )
    yield
    client.app.dependency_overrides.clear()


def _seed_results():
    session_factory = create_session_factory()
    with session_factory() as session:
        patient = Patient(pseudonym=f"PAT-RESULT-{uuid4().hex[:8]}", status="active")
        session.add(patient)
        session.flush()
        study = Study(
            public_id=uuid4(),
            patient_id=patient.id,
            patient_public_id=patient.public_id,
            study_instance_uid="1.2.3.4.5.6",
            source_kind="dicom-study",
            source_metadata={},
            staging_status="processed",
        )
        session.add(study)
        session.flush()
        result = StudyResult(
            study_id=study.id,
            result_kind="single-scan",
            needs_review=False,
            summary_metadata={"case_qc_reasons": []},
        )
        session.add(result)
        session.flush()
        session.add(
            StoredLesionResult(
                study_result_id=result.id,
                study_id=study.id,
                lesion_id="study-001:lesion-001",
                measurement_payload={"volume_mm3": 12.0, "longest_diameter_mm": 4.0},
                bounding_box={"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1, "z_min": 0, "z_max": 1},
                artifact_refs={
                    "mask": {
                        "artifact_kind": "segmentation-mask",
                        "storage_root": "derived",
                        "relative_path": f"studies/{study.public_id}/lesions/lesion-001-mask.nii.gz",
                    },
                    "review": [],
                },
                result_metadata={"slot_provenance": []},
            )
        )
        session.add(
            Artifact(
                study_id=study.id,
                artifact_kind="study-result-bundle",
                storage_root="derived",
                relative_path=f"studies/{study.public_id}/results/study-result.json",
                source_metadata={"study_result_id": result.id},
            )
        )
        session.commit()
        return str(study.public_id)


def test_result_service_returns_machine_readable_case_payload() -> None:
    study_id = _seed_results()
    payload = get_case_result_payload(study_id=study_id)
    assert payload.study_id == study_id
    assert payload.lesions[0].lesion_id == "study-001:lesion-001"
    assert payload.lesions[0].measurements.volume_mm3 == 12.0
    assert not payload.lesions[0].mask_artifact.relative_path.startswith("/")


def test_result_service_raises_not_found_for_missing_results() -> None:
    with pytest.raises(ResultNotFoundError):
        get_case_result_payload(study_id=str(uuid4()))


def test_results_endpoint_returns_payload(client) -> None:
    study_id = _seed_results()
    response = client.get(f"/api/v1/results/{study_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["studyId"] == study_id
    assert body["lesions"][0]["lesionId"] == "study-001:lesion-001"
    assert not body["lesions"][0]["maskArtifact"]["relativePath"].startswith("/")


def test_results_endpoint_returns_not_found(client) -> None:
    response = client.get(f"/api/v1/results/{uuid4()}")
    assert response.status_code == 404


def test_results_endpoint_rejects_invalid_study_ids(client) -> None:
    response = client.get("/api/v1/results/not-a-uuid")
    assert response.status_code == 400


def test_result_service_returns_empty_lesion_case_payload() -> None:
    session_factory = create_session_factory()
    with session_factory() as session:
        study = Study(
            public_id=uuid4(),
            study_instance_uid="1.2.840.empty",
            source_kind="dicom-study",
            source_metadata={},
            staging_status="processed",
        )
        session.add(study)
        session.flush()
        result = StudyResult(
            study_id=study.id,
            result_kind="single-scan",
            needs_review=True,
            summary_metadata={"case_qc_reasons": ["selected canonical series do not share geometry"], "lesion_count": 0},
        )
        session.add(result)
        session.flush()
        session.add(
            Artifact(
                study_id=study.id,
                artifact_kind="study-result-bundle",
                storage_root="derived",
                relative_path=f"studies/{study.public_id}/results/study-result.json",
                source_metadata={"study_result_id": result.id, "lesion_ids": []},
            )
        )
        session.commit()
        study_id = str(study.public_id)

    payload = get_case_result_payload(study_id=study_id)
    assert payload.study_id == study_id
    assert payload.lesions == ()
    assert payload.needs_review is True


def test_results_and_study_listing_are_scoped_to_assigned_patients(client) -> None:
    assert client.get("/api/v1/health").status_code == 200
    restricted_study_id = _seed_results()
    session_factory = create_session_factory()
    with session_factory() as session:
        clinician = User(email="results-clinician@test.local", name="Clinician", hashed_password="hash", role="clinician")
        assigned_patient = Patient(pseudonym="PAT-RESULT-ASSIGNED", status="active")
        restricted_patient = Patient(pseudonym="PAT-RESULT-RESTRICTED", status="active")
        session.add_all([clinician, assigned_patient, restricted_patient])
        session.flush()
        session.add(Assignment(doctor_id=clinician.id, patient_id=assigned_patient.id))
        assigned_study = Study(patient_id=assigned_patient.id, patient_public_id=assigned_patient.public_id, study_instance_uid="result-assigned", source_kind="nifti-upload", source_metadata={}, staging_status="staged")
        session.add(assigned_study)
        restricted_study = session.query(Study).filter(Study.public_id == UUID(restricted_study_id)).one()
        restricted_study.patient_id = restricted_patient.id
        restricted_study.patient_public_id = restricted_patient.public_id
        session.commit()
        assigned_study_id = str(assigned_study.public_id)

    client.app.dependency_overrides[get_current_user] = lambda: clinician
    forbidden = client.get(f"/api/v1/results/{restricted_study_id}")
    listing = client.get("/api/v1/results/studies")

    assert forbidden.status_code == 403
    assert listing.status_code == 200
    assert [item["studyId"] for item in listing.json()] == [assigned_study_id]


def test_stored_comparison_rejects_user_unassigned_from_patient(client) -> None:
    assert client.get("/api/v1/health").status_code == 200
    session_factory = create_session_factory()
    with session_factory() as session:
        clinician = User(email="stored-comparison-outsider@test.local", name="Outsider", hashed_password="hash", role="clinician")
        patient = Patient(pseudonym="PAT-STORED-COMPARISON", status="active")
        session.add_all([clinician, patient])
        session.flush()
        baseline = Study(patient_id=patient.id, patient_public_id=patient.public_id, study_instance_uid="stored-comp-a", source_kind="nifti-upload", source_metadata={}, staging_status="staged")
        followup = Study(patient_id=patient.id, patient_public_id=patient.public_id, study_instance_uid="stored-comp-b", source_kind="nifti-upload", source_metadata={}, staging_status="staged")
        session.add_all([baseline, followup])
        session.commit()
        baseline_id = str(baseline.public_id)
        followup_id = str(followup.public_id)

    comparison_id = str(uuid4())
    location = resolve_artifact_location("derived", f"comparisons/{comparison_id}/comparison.json")
    location.absolute_path.parent.mkdir(parents=True, exist_ok=True)
    location.absolute_path.write_text(json.dumps({
        "baseline_study_id": baseline_id,
        "followup_study_id": followup_id,
        "metrics": {},
    }))
    client.app.dependency_overrides[get_current_user] = lambda: clinician

    response = client.get(f"/api/v1/results/comparisons/{comparison_id}")

    assert response.status_code == 403
