from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def configure_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONCOFLOW_DATABASE_URL", f"sqlite+pysqlite:///{tmp_path / 'results.sqlite3'}")


def test_result_schema_persists_study_and_lesion_rows() -> None:
    from app.infra.db.models import StoredLesionResult, Study, StudyResult
    from app.infra.db.session import create_session_factory

    session_factory = create_session_factory()
    with session_factory() as session:
        study = Study(
            study_instance_uid="1.2.3.4",
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
            summary_metadata={"case_qc_reasons": ["review"]},
        )
        session.add(result)
        session.flush()

        lesion = StoredLesionResult(
            study_result_id=result.id,
            study_id=study.id,
            lesion_id="study-001:lesion-001",
            measurement_payload={"volume_mm3": 12.0, "longest_diameter_mm": 4.0},
            bounding_box={"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1, "z_min": 0, "z_max": 1},
            artifact_refs={"mask": {"artifact_kind": "segmentation-mask", "storage_root": "derived", "relative_path": "studies/study/results/mask.nii.gz"}},
            result_metadata={"slot_provenance": []},
        )
        session.add(lesion)
        session.commit()

    with session_factory() as session:
        result = session.query(StudyResult).one()
        lesion = session.query(StoredLesionResult).one()
        assert result.needs_review is True
        assert lesion.lesion_id == "study-001:lesion-001"
        assert lesion.artifact_refs["mask"]["relative_path"] == "studies/study/results/mask.nii.gz"


def test_phase3_migration_creates_result_tables(tmp_path: Path) -> None:
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect

    db_path = tmp_path / "migration.sqlite3"
    config = Config("alembic.ini")
    config.set_main_option("script_location", "alembic")
    config.set_main_option("sqlalchemy.url", f"sqlite+pysqlite:///{db_path}")
    command.upgrade(config, "head")

    tables = set(inspect(create_engine(f"sqlite+pysqlite:///{db_path}")).get_table_names())
    assert {"study_results", "lesion_results"}.issubset(tables)
