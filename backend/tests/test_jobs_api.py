from __future__ import annotations

import io
import zipfile
from pathlib import Path
from uuid import UUID

import pytest

from app.infra.db.models import Artifact, Job
from app.infra.db.session import create_session_factory
from app.modules.jobs.service import JobService
from app.modules.jobs.state_machine import transition_job
from app.modules.jobs.worker_tasks import WorkerDispatchEnvelope


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

    result = pytest.run(async_fn=service.submit_mri_study)(
        filename="exam.zip",
        content_type="application/zip",
        archive_bytes=archive_bytes,
        source_label="dispatch-check",
    )

    assert captured["job_id"] == str(result.job_public_id)
    assert captured["study_id"] == str(result.study_public_id)
    assert "dicom" not in captured["extracted_relative_path"]


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


def test_invalid_state_transition_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid job transition"):
        transition_job("completed", "running", stage="profiling")
