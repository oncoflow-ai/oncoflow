from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from app.modules.artifacts.storage import ensure_storage_layout, resolve_artifact_location


def test_job_lifecycle_and_relationships_persist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "persistence.sqlite3"
    monkeypatch.setenv("ONCOFLOW_DATABASE_URL", f"sqlite+pysqlite:///{db_path}")

    from app.infra.db.models import Artifact, Job, JobEvent, Series, Study
    from app.infra.db.session import create_session_factory

    session_factory = create_session_factory()

    with session_factory() as session:
        study = Study(
            study_instance_uid="1.2.840.113619.2.55.3.604688654.100.1",
            source_kind="dicom-study",
            source_metadata={"source": "unit-test"},
            staging_status="staged",
        )
        session.add(study)
        session.flush()

        series = Series(
            study_id=study.id,
            series_instance_uid="1.2.840.113619.2.55.3.604688654.100.2",
            modality="MR",
            series_description="t2_tse_stir_tra_RT",
            protocol_name="t2_tse_stir",
            classification="processable",
            scanner_vendor="Siemens",
            source_metadata={"images": 24},
        )
        session.add(series)
        session.flush()

        artifact = Artifact(
            study_id=study.id,
            series_id=series.id,
            artifact_kind="nifti-volume",
            storage_root="derived",
            relative_path="studies/study-1/series-1/volume.nii.gz",
            source_metadata={"format": "nifti"},
        )
        session.add(artifact)

        job = Job(
            study_id=study.id,
            public_id=uuid4(),
            job_type="ingest-study",
            status="queued",
            stage="profile",
        )
        session.add(job)
        session.flush()

        job_event = JobEvent(
            job_id=job.id,
            status="queued",
            stage="profile",
            event_type="transition",
            payload={"detail": "job queued"},
        )
        session.add(job_event)
        session.commit()

    with session_factory() as session:
        persisted_job = session.query(Job).one()
        persisted_job.status = "failed"
        persisted_job.stage = "persist"
        persisted_job.failure_payload = {
            "code": "conversion-error",
            "message": "NIfTI conversion failed",
        }
        session.add(
            JobEvent(
                job_id=persisted_job.id,
                status="failed",
                stage="persist",
                event_type="failure",
                payload=persisted_job.failure_payload,
            )
        )
        session.commit()

    with session_factory() as session:
        persisted_job = session.query(Job).one()
        persisted_study = session.query(Study).one()
        persisted_series = session.query(Series).one()
        persisted_artifact = session.query(Artifact).one()
        events = session.query(JobEvent).order_by(JobEvent.created_at.asc()).all()

        assert persisted_job.status == "failed"
        assert persisted_job.stage == "persist"
        assert persisted_job.failure_payload == {
            "code": "conversion-error",
            "message": "NIfTI conversion failed",
        }
        assert persisted_study.series[0].id == persisted_series.id
        assert persisted_study.artifacts[0].id == persisted_artifact.id
        assert persisted_job.study_id == persisted_study.id
        assert [event.status for event in events] == ["queued", "failed"]
        assert events[0].job_id == persisted_job.id


def test_alembic_migration_creates_phase_one_tables(tmp_path: Path) -> None:
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect

    db_path = tmp_path / "migration.sqlite3"
    config = Config("alembic.ini")
    config.set_main_option("script_location", "alembic")
    config.set_main_option("sqlalchemy.url", f"sqlite+pysqlite:///{db_path}")

    command.upgrade(config, "head")

    tables = set(inspect(create_engine(f"sqlite+pysqlite:///{db_path}")).get_table_names())
    assert {
        "artifacts",
        "job_events",
        "jobs",
        "series",
        "studies",
    }.issubset(tables)


def test_storage_helpers_normalize_managed_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONCOFLOW_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("ONCOFLOW_STORAGE_STAGING_DIR", "raw")

    roots = ensure_storage_layout()
    artifact = resolve_artifact_location("derived", "studies/study-1/series-2/mask.nii.gz")

    assert roots["raw"].exists()
    assert roots["derived"].exists()
    assert artifact.relative_path == "studies/study-1/series-2/mask.nii.gz"
    assert str(artifact.absolute_path).startswith(str(roots["derived"]))


@pytest.mark.parametrize("bad_path", ["/tmp/escape.txt", "../escape.txt", "studies/../escape.txt"])
def test_storage_helpers_reject_path_escape_attempts(bad_path: str) -> None:
    with pytest.raises(ValueError):
        resolve_artifact_location("raw", bad_path)
