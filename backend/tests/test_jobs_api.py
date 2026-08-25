from __future__ import annotations

import asyncio
import io
import threading
import zipfile
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.infra.db.models import Artifact, Assignment, AuditLog, Job, Patient, Study, User
from app.infra.db.session import create_session_factory, get_engine
from app.modules.jobs.service import JobService
from app.modules.jobs.state_machine import transition_job
from app.modules.jobs.worker_tasks import (
    WorkerDispatchEnvelope,
    execute_demo_mri_segmentation_job,
)
from app.api.deps import get_current_user


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return payload.getvalue()


@pytest.fixture(autouse=True)
def configure_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONCOFLOW_DATABASE_URL", f"sqlite+pysqlite:///{tmp_path / 'jobs.sqlite3'}")
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


@pytest.mark.parametrize(
    ("endpoint", "files"),
    [
        (
            "/api/v1/jobs/mri-ingestion",
            {"study_archive": ("exam.zip", _zip_bytes({"exam/file.dcm": b"dicom"}), "application/zip")},
        ),
        (
            "/api/v1/jobs/nifti-segmentation",
            {"scan_file": ("scan.nii.gz", b"nifti-bytes", "application/gzip")},
        ),
        (
            "/api/v1/jobs/demo-mri-segmentation",
            {"scan_file": ("demo.nii.gz", b"demo-bytes", "application/gzip")},
        ),
    ],
)
def test_ingestion_routes_require_authentication(client, endpoint: str, files: dict) -> None:
    client.app.dependency_overrides.clear()

    response = client.post(endpoint, files=files)

    assert response.status_code == 401


@pytest.mark.parametrize(
    ("endpoint", "files"),
    [
        (
            "/api/v1/jobs/mri-ingestion",
            {"study_archive": ("exam.zip", _zip_bytes({"exam/file.dcm": b"dicom"}), "application/zip")},
        ),
        (
            "/api/v1/jobs/nifti-segmentation",
            {"scan_file": ("scan.nii.gz", b"nifti-bytes", "application/gzip")},
        ),
        (
            "/api/v1/jobs/demo-mri-segmentation",
            {"scan_file": ("demo.nii.gz", b"demo-bytes", "application/gzip")},
        ),
    ],
)
def test_ingestion_rejects_inaccessible_existing_patient_without_db_side_effects(
    client,
    endpoint: str,
    files: dict,
) -> None:
    assert client.get("/api/v1/health").status_code == 200
    session_factory = create_session_factory()
    with session_factory() as session:
        clinician = User(
            email="unassigned-clinician@test.local",
            name="Unassigned Clinician",
            hashed_password="hash",
            role="clinician",
        )
        patient = Patient(pseudonym="PAT-INGESTION-RESTRICTED", status="active")
        session.add_all([clinician, patient])
        session.commit()
        patient_id = str(patient.public_id)
        baseline_counts = {
            "patients": session.query(Patient).count(),
            "assignments": session.query(Assignment).count(),
            "studies": session.query(Study).count(),
            "jobs": session.query(Job).count(),
        }
    storage_root = Path(get_settings().storage_root)
    baseline_storage_entries = {
        path.relative_to(storage_root) for path in storage_root.rglob("*")
    }

    client.app.dependency_overrides[get_current_user] = lambda: clinician
    response = client.post(endpoint, files=files, data={"patient_id": patient_id})

    assert response.status_code == 403
    with session_factory() as session:
        assert session.query(Patient).count() == baseline_counts["patients"]
        assert session.query(Assignment).count() == baseline_counts["assignments"]
        assert session.query(Study).count() == baseline_counts["studies"]
        assert session.query(Job).count() == baseline_counts["jobs"]
    assert {
        path.relative_to(storage_root) for path in storage_root.rglob("*")
    } == baseline_storage_entries


def test_ingestion_allows_assigned_clinician_for_existing_patient(client) -> None:
    assert client.get("/api/v1/health").status_code == 200
    session_factory = create_session_factory()
    with session_factory() as session:
        clinician = User(
            email="assigned-clinician@test.local",
            name="Assigned Clinician",
            hashed_password="hash",
            role="clinician",
        )
        patient = Patient(pseudonym="PAT-INGESTION-ASSIGNED", status="active")
        session.add_all([clinician, patient])
        session.flush()
        session.add(Assignment(doctor_id=clinician.id, patient_id=patient.id))
        session.commit()
        patient_id = str(patient.public_id)

    client.app.dependency_overrides[get_current_user] = lambda: clinician
    response = client.post(
        "/api/v1/jobs/demo-mri-segmentation",
        files={"scan_file": ("demo.nii.gz", b"demo-bytes", "application/gzip")},
        data={"patient_id": patient_id},
    )

    assert response.status_code == 201
    with session_factory() as session:
        assert session.query(Assignment).filter(Assignment.patient_id == patient.id).count() == 1


@pytest.mark.parametrize(
    ("endpoint", "files"),
    [
        (
            "/api/v1/jobs/mri-ingestion",
            {"study_archive": ("exam.zip", _zip_bytes({"exam/file.dcm": b"dicom"}), "application/zip")},
        ),
        (
            "/api/v1/jobs/nifti-segmentation",
            {"scan_file": ("scan.nii.gz", b"nifti-bytes", "application/gzip")},
        ),
        (
            "/api/v1/jobs/demo-mri-segmentation",
            {"scan_file": ("demo.nii.gz", b"demo-bytes", "application/gzip")},
        ),
    ],
)
def test_ingestion_rejects_unknown_explicit_patient_without_creating_one(
    client,
    endpoint: str,
    files: dict,
) -> None:
    assert client.get("/api/v1/health").status_code == 200
    session_factory = create_session_factory()
    with session_factory() as session:
        baseline_patient_count = session.query(Patient).count()
    storage_root = Path(get_settings().storage_root)
    baseline_storage_entries = {
        path.relative_to(storage_root) for path in storage_root.rglob("*")
    }

    response = client.post(
        endpoint,
        files=files,
        data={"patient_id": "PAT-DOES-NOT-EXIST"},
    )

    assert response.status_code == 404
    with session_factory() as session:
        assert session.query(Patient).count() == baseline_patient_count
    assert {
        path.relative_to(storage_root) for path in storage_root.rglob("*")
    } == baseline_storage_entries


def test_post_mri_ingestion_stages_archive_and_returns_queued_job(client) -> None:
    archive_bytes = _zip_bytes({"exam/DICOMDIR": b"directory", "exam/series/file1.dcm": b"dicom"})

    response = client.post(
        "/api/v1/jobs/mri-ingestion",
        files={"study_archive": ("exam.zip", archive_bytes, "application/zip")},
        data={"source_label": "sample-exam"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "queued"
    assert body["stage"] == "staged"
    UUID(body["jobId"])
    UUID(body["studyId"])

    session_factory = create_session_factory()
    with session_factory() as session:
        job = session.query(Job).one()
        artifacts = session.query(Artifact).order_by(Artifact.id.asc()).all()

        assert job.status == "queued"
        assert job.stage == "staged"
        assert [artifact.artifact_kind for artifact in artifacts] == [
            "raw-study-archive",
            "extracted-study-root",
        ]
        assert not any("storage" in str(value) for value in body.values())


@pytest.mark.parametrize(
    ("files", "expected_status"),
    [
        ({}, 422),
        ({"study_archive": ("empty.zip", b"", "application/zip")}, 400),
        ({"study_archive": ("bad.txt", b"plain-text", "text/plain")}, 415),
    ],
)
def test_post_mri_ingestion_rejects_invalid_payloads(client, files, expected_status: int) -> None:
    response = client.post("/api/v1/jobs/mri-ingestion", files=files)
    assert response.status_code == expected_status


def test_invalid_zip_cleanup_leaves_no_raw_study_directory(client) -> None:
    storage_root = Path(get_settings().storage_root)
    baseline_storage_entries = {
        path.relative_to(storage_root) for path in storage_root.rglob("*")
    }

    response = client.post(
        "/api/v1/jobs/mri-ingestion",
        files={"study_archive": ("invalid.zip", b"not-a-zip", "application/zip")},
    )

    assert response.status_code == 400
    assert {
        path.relative_to(storage_root) for path in storage_root.rglob("*")
    } == baseline_storage_entries


def test_nifti_database_failure_cleans_up_raw_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_root = Path(get_settings().storage_root)
    baseline_storage_entries = {
        path.relative_to(storage_root) for path in storage_root.rglob("*")
    }
    original_commit = Session.commit

    def fail_study_commit(session: Session) -> None:
        if any(isinstance(value, Study) for value in session.identity_map.values()):
            raise RuntimeError("database unavailable")
        original_commit(session)

    monkeypatch.setattr(Session, "commit", fail_study_commit)

    with pytest.raises(RuntimeError, match="database unavailable"):
        asyncio.run(JobService().submit_nifti_study(
            scan_filename="scan.nii.gz",
            scan_bytes=b"nifti-bytes",
            mask_filename=None,
            mask_bytes=None,
            source_label="failure cleanup",
            acquired_at=None,
        ))

    assert {
        path.relative_to(storage_root) for path in storage_root.rglob("*")
    } == baseline_storage_entries
    with create_session_factory()() as session:
        assert session.query(Study).count() == 0


@pytest.mark.parametrize("submission_kind", ["dicom", "nifti", "demo"])
def test_study_audit_failure_rolls_back_mutation_and_cleans_raw_files(
    submission_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_root = Path(get_settings().storage_root)
    baseline_storage_entries = {
        path.relative_to(storage_root) for path in storage_root.rglob("*")
    }
    with create_session_factory()() as session:
        admin = User(
            email="study-audit-admin@test.local",
            name="Study Audit Admin",
            hashed_password="hash",
            role="admin",
        )
        session.add(admin)
        session.commit()
        session.refresh(admin)
        session.expunge(admin)

    original_flush = Session.flush

    def fail_when_auditing(session: Session, *args, **kwargs) -> None:
        if any(isinstance(value, AuditLog) for value in session.new):
            raise RuntimeError("audit database unavailable")
        original_flush(session, *args, **kwargs)

    monkeypatch.setattr(Session, "flush", fail_when_auditing)

    service = JobService()
    if submission_kind == "dicom":
        submission = service.submit_mri_study(
            filename="study.zip",
            content_type="application/zip",
            archive_bytes=_zip_bytes({"exam/file.dcm": b"dicom"}),
            source_label="audit failure cleanup",
            current_user=admin,
        )
    elif submission_kind == "nifti":
        submission = service.submit_nifti_study(
            scan_filename="scan.nii.gz",
            scan_bytes=b"nifti-bytes",
            mask_filename=None,
            mask_bytes=None,
            source_label="audit failure cleanup",
            acquired_at=None,
            current_user=admin,
        )
    else:
        submission = service.submit_demo_mri_segmentation(
            scan_filename="demo.nii.gz",
            scan_bytes=b"demo-bytes",
            content_type="application/gzip",
            source_label="audit failure cleanup",
            acquired_at=None,
            current_user=admin,
        )

    with pytest.raises(RuntimeError, match="audit database unavailable"):
        asyncio.run(submission)

    assert {
        path.relative_to(storage_root) for path in storage_root.rglob("*")
    } == baseline_storage_entries
    with create_session_factory()() as session:
        assert session.query(Study).count() == 0
        assert session.query(AuditLog).filter(AuditLog.action == "CREATE_STUDY").count() == 0


def test_submission_dispatches_identifiers_not_raw_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    archive_bytes = _zip_bytes({"exam/file1.dcm": b"dicom"})
    captured: dict[str, str] = {}

    def fake_dispatch(*, job_id: str, study_id: str, extracted_relative_path: str) -> WorkerDispatchEnvelope:
        captured["job_id"] = job_id
        captured["study_id"] = study_id
        captured["extracted_relative_path"] = extracted_relative_path
        return WorkerDispatchEnvelope(
            job_id=job_id,
            study_id=study_id,
            extracted_relative_path=extracted_relative_path,
        )

    service = JobService()
    monkeypatch.setattr(service, "_dispatch_worker", fake_dispatch)

    result = asyncio.run(service.submit_mri_study(
        filename="exam.zip",
        content_type="application/zip",
        archive_bytes=archive_bytes,
        source_label="dispatch-check",
    ))

    assert captured["job_id"] == str(result.job_public_id)
    assert captured["study_id"] == str(result.study_public_id)
    assert "dicom" not in captured["extracted_relative_path"]


def test_threaded_execution_mode_starts_background_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONCOFLOW_JOB_EXECUTION_MODE", "threaded")
    get_settings.cache_clear()
    service = JobService()
    captured: dict[str, str] = {}

    def fake_execute_ingestion_job(*, job_id: str) -> None:
        captured["job_id"] = job_id

    class FakeThread:
        def __init__(self, *, target, kwargs, daemon, name):
            self._target = target
            self._kwargs = kwargs
            self.daemon = daemon
            self.name = name

        def start(self) -> None:
            self._target(**self._kwargs)

    monkeypatch.setattr("app.modules.jobs.service.execute_ingestion_job", fake_execute_ingestion_job)
    monkeypatch.setattr("app.modules.jobs.service.threading.Thread", FakeThread)

    dispatch = service._dispatch_worker(
        job_id="job-123",
        study_id="study-456",
        extracted_relative_path="studies/demo/extracted",
    )

    assert dispatch.job_id == "job-123"
    assert captured["job_id"] == "job-123"


def test_get_job_status_returns_failure_payload(client) -> None:
    archive_bytes = _zip_bytes({"exam/file1.dcm": b"dicom"})
    submit_response = client.post(
        "/api/v1/jobs/mri-ingestion",
        files={"study_archive": ("exam.zip", archive_bytes, "application/zip")},
    )
    job_id = UUID(submit_response.json()["jobId"])

    JobService().mark_job_failed(job_id, code="conversion-error", message="NIfTI conversion failed")

    response = client.get(f"/api/v1/jobs/{job_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert body["stage"] == "persisting"
    assert body["error"]["code"] == "conversion-error"


def test_get_job_status_rejects_unassigned_clinician(client) -> None:
    archive_bytes = _zip_bytes({"exam/file1.dcm": b"dicom"})
    submitted = client.post(
        "/api/v1/jobs/mri-ingestion",
        files={"study_archive": ("exam.zip", archive_bytes, "application/zip")},
    ).json()
    with create_session_factory()() as session:
        outsider = User(
            email="job-outsider@test.local",
            name="Job Outsider",
            hashed_password="hash",
            role="clinician",
        )
        session.add(outsider)
        session.commit()

    client.app.dependency_overrides[get_current_user] = lambda: outsider
    response = client.get(f"/api/v1/jobs/{submitted['jobId']}")

    assert response.status_code == 403


def test_comparison_rejects_studies_from_different_patients(client) -> None:
    with create_session_factory()() as session:
        patients = [Patient(pseudonym=f"PAT-COMP-{index}") for index in (1, 2)]
        session.add_all(patients)
        session.flush()
        studies = []
        for index, patient in enumerate(patients, start=1):
            study = Study(
                public_id=uuid4(),
                patient_id=patient.id,
                patient_public_id=patient.public_id,
                study_instance_uid=f"comparison-cross-patient-{index}",
                source_kind="nifti",
                staging_status="staged",
            )
            session.add(study)
            session.flush()
            session.add(Artifact(
                study_id=study.id,
                artifact_kind="nifti-source",
                storage_root="raw",
                relative_path=f"cross-patient/{index}.nii.gz",
            ))
            studies.append(str(study.public_id))
        session.commit()

    response = client.post(
        "/api/v1/jobs/longitudinal-comparison",
        json={"baselineStudyId": studies[0], "followupStudyId": studies[1]},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "studies must belong to the same patient"


def test_invalid_state_transition_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid job transition"):
        transition_job("completed", "running", stage="profiling")


def test_post_demo_mri_segmentation_returns_queued_job(client) -> None:
    response = client.post(
        "/api/v1/jobs/demo-mri-segmentation",
        files={"scan_file": ("class-demo.nii.gz", b"mri-bytes", "application/gzip")},
        data={"source_label": "Class demo MRI", "acquired_at": "2024-01-15"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "queued"
    assert body["stage"] == "staged"
    UUID(body["jobId"])
    UUID(body["studyId"])

    session_factory = create_session_factory()
    with session_factory() as session:
        job = session.query(Job).one()
        artifacts = session.query(Artifact).all()

        assert job.job_type == "demo-mri-segmentation"
        assert job.status == "queued"
        assert artifacts == []
        assert job.study.source_metadata["upload_discarded"] is True


@pytest.mark.parametrize(
    ("files", "data", "expected_status"),
    [
        ({}, {}, 422),
        ({"scan_file": ("empty.nii.gz", b"", "application/gzip")}, {}, 400),
        (
            {"scan_file": ("class-demo.nii.gz", b"mri-bytes", "application/gzip")},
            {"acquired_at": "not-a-date"},
            400,
        ),
    ],
)
def test_post_demo_mri_segmentation_rejects_invalid_payloads(
    client,
    files,
    data,
    expected_status: int,
) -> None:
    response = client.post(
        "/api/v1/jobs/demo-mri-segmentation",
        files=files,
        data=data,
    )

    assert response.status_code == expected_status


def test_demo_mri_worker_exposes_running_state_before_delay(
    client,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class StopAfterRunningStateCheck(Exception):
        pass

    monkeypatch.setenv("ONCOFLOW_DATABASE_URL", f"sqlite+pysqlite:///{tmp_path / 'polling.sqlite3'}")
    get_settings.cache_clear()
    get_engine.cache_clear()

    submitted = client.post(
        "/api/v1/jobs/demo-mri-segmentation",
        files={"scan_file": ("class-demo.nii.gz", b"mri-bytes", "application/gzip")},
        data={"source_label": "Class demo MRI", "acquired_at": "2024-01-15"},
    ).json()
    job_id = UUID(submitted["jobId"])
    delay_entered = threading.Event()
    release_delay = threading.Event()
    worker_errors: list[Exception] = []

    def pause_delay(_delay_seconds: float) -> None:
        delay_entered.set()
        assert release_delay.wait(timeout=3)
        raise StopAfterRunningStateCheck

    monkeypatch.setattr(
        "app.modules.jobs.worker_tasks.time.sleep",
        pause_delay,
    )

    def run_worker() -> None:
        try:
            execute_demo_mri_segmentation_job(job_id=str(job_id))
        except StopAfterRunningStateCheck as exc:
            worker_errors.append(exc)

    worker_thread = threading.Thread(target=run_worker)
    worker_thread.start()
    assert delay_entered.wait(timeout=3)

    try:
        with create_session_factory()() as polling_session:
            polled_job = polling_session.query(Job).filter(Job.public_id == job_id).one()
            assert polled_job.status == "running"
            assert polled_job.stage in {"data-fetching", "demo-inference"}
            assert polled_job.progress >= 15
    finally:
        release_delay.set()
        worker_thread.join(timeout=3)

    assert not worker_thread.is_alive()
    assert len(worker_errors) == 1


def test_demo_mri_segmentation_completes_and_exposes_report(
    client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONCOFLOW_JOB_EXECUTION_MODE", "threaded")
    monkeypatch.setenv("ONCOFLOW_DEMO_JOB_DELAY_SECONDS", "0")
    get_settings.cache_clear()

    def dispatch_inline(_self, *, job_id: str, study_id: str) -> None:
        execute_demo_mri_segmentation_job(job_id=job_id)

    monkeypatch.setattr(
        "app.modules.jobs.service.JobService._dispatch_demo_mri_worker",
        dispatch_inline,
    )

    submit_response = client.post(
        "/api/v1/jobs/demo-mri-segmentation",
        files={"scan_file": ("class-demo.nii.gz", b"mri-bytes", "application/gzip")},
        data={"source_label": "Class demo MRI", "acquired_at": "2024-01-15"},
    )

    assert submit_response.status_code == 201
    submitted = submit_response.json()

    status_response = client.get(f"/api/v1/jobs/{submitted['jobId']}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "completed"
    assert status_response.json()["stage"] == "completed"
    assert status_response.json()["progress"] == 100
    assert status_response.json()["stageMessage"] is not None

    result_response = client.get(f"/api/v1/results/{submitted['studyId']}")
    assert result_response.status_code == 200
    body = result_response.json()
    assert body["studyId"] == submitted["studyId"]
    assert body["needsReview"] is False
    assert body["lesions"][0]["lesionId"] == "lesion-001"
    assert body["lesions"][0]["measurements"]["volumeMm3"] > 0
    assert body["lesions"][0]["metadata"]["runner"]["model_id"] == (
        "oncoflow-demo-ensemble"
    )
    assert body["lesions"][0]["metadata"]["runner"]["execution_backend"] == (
        "simulated"
    )
    assert body["metadata"]["source"] == "ground-truth-demo-mask"
    assert body["metadata"]["report"]["title"] == (
        "AI brain MRI segmentation report"
    )
    assert "previous scan" in body["metadata"]["report"]["comparison"]
    assert body["metadata"]["report"]["quantitative"]["volume_change_pct"] > 0
    assert len(body["metadata"]["report"]["recommendations"]) == 3
    assert "disclaimer" not in body["metadata"]["report"]
