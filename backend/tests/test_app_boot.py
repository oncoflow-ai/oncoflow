from __future__ import annotations

from uuid import uuid4

from app.core.config import get_settings
from app.infra.queue.celery_app import celery_app
from app.modules.jobs.contracts import ProcessingJobContract, StagedStudyReference
from app.core.config import Settings
from app.main import PhiSafeLogFilter


def test_app_exposes_health_and_readiness_routes(client) -> None:
    health_response = client.get("/api/v1/health")
    readiness_response = client.get("/api/v1/ready")

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}

    assert readiness_response.status_code == 200
    assert readiness_response.json()["status"] == "ready"
    assert readiness_response.json()["queue"] == "configured"


def test_settings_build_broker_dsn_from_environment() -> None:
    settings = Settings(
        redis_host="queue",
        redis_port=6380,
        redis_db=2,
        redis_password="secret",
    )

    assert settings.broker_dsn == "redis://:secret@queue:6380/2"
    assert settings.result_backend_dsn == "redis://:secret@queue:6380/2"


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


def test_queue_boundary_and_job_contracts_are_importable() -> None:
    settings = get_settings()

    assert celery_app.conf["broker_url"] == settings.broker_dsn
    assert celery_app.conf["result_backend"] == settings.result_backend_dsn

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
