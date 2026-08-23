from __future__ import annotations

from uuid import uuid4

from app.modules.jobs.contracts import ProcessingJobContract, StagedStudyReference
from app.main import PhiSafeLogFilter
from app.infra.db.models import Patient, User
from app.infra.db.session import create_session_factory
from fastapi.testclient import TestClient


def test_app_exposes_health_and_readiness_routes(client) -> None:
    health_response = client.get("/api/v1/health")
    readiness_response = client.get("/api/v1/ready")

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}

    assert readiness_response.status_code == 200
    assert readiness_response.json()["status"] == "ready"
    assert readiness_response.json()["queue"] == "deferred"


def test_phi_safe_log_filter_redacts_paths_and_dicom_tags() -> None:
    log_filter = PhiSafeLogFilter()

    class Record:
        def __init__(self) -> None:
            self.msg = "received /tmp/study/file.dcm from tag (0010,0010)"
            self.args = ()

        def getMessage(self) -> str:
            return self.msg

    record = Record()

    assert log_filter.filter(record) is True
    assert "[redacted-path]" in record.msg
    assert "[redacted-dicom-tag]" in record.msg


def test_job_contracts_are_importable() -> None:
    staged_study = StagedStudyReference(
        study_id=uuid4(),
        staging_uri="s3://oncoflow-local/staging/study-1",
        storage_bucket="oncoflow-local",
        storage_key="staging/study-1",
    )
    contract = ProcessingJobContract(
        job_id=uuid4(),
        study=staged_study,
        status="queued",
        stage="profile",
    )

    running_contract = contract.transition(status="running", stage="validate")

    assert running_contract.job_id == contract.job_id
    assert running_contract.study.storage_key == "staging/study-1"
    assert running_contract.status == "running"
    assert running_contract.stage == "validate"


def test_demo_bootstrap_is_disabled_without_development_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("ONCOFLOW_ENVIRONMENT", "production")
    monkeypatch.delenv("ONCOFLOW_BOOTSTRAP_DEMO_DATA", raising=False)
    from app.core.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    with TestClient(create_app()):
        pass

    with create_session_factory()() as session:
        assert session.query(User).count() == 0
        assert session.query(Patient).count() == 0
