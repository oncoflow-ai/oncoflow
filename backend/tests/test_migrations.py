from __future__ import annotations

from pathlib import Path
import sys
from datetime import datetime, timezone
from uuid import UUID, uuid4

import sqlalchemy as sa


_BACKEND_DIR = Path(__file__).resolve().parents[1]
_ORIGINAL_SYS_PATH = sys.path[:]
sys.path[:] = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() != _BACKEND_DIR
]
try:
    from alembic import command
    from alembic.config import Config
    from alembic.script import ScriptDirectory
finally:
    sys.path[:] = _ORIGINAL_SYS_PATH


def _alembic_config(database_path: Path) -> Config:
    config = Config(str(_BACKEND_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(_BACKEND_DIR / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    return config


def _assert_patient_schema(database_path: Path) -> None:
    engine = sa.create_engine(f"sqlite:///{database_path}")
    inspector = sa.inspect(engine)

    assert {"patients", "assignments", "studies"}.issubset(inspector.get_table_names())
    assert "patient_id" in {column["name"] for column in inspector.get_columns("studies")}
    assert "ix_studies_patient_id" in {
        index["name"] for index in inspector.get_indexes("studies")
    }
    assert any(
        foreign_key["constrained_columns"] == ["patient_id"]
        and foreign_key["referred_table"] == "patients"
        and foreign_key["referred_columns"] == ["id"]
        for foreign_key in inspector.get_foreign_keys("studies")
    )

    engine.dispose()


def test_patient_migration_round_trip_on_fresh_sqlite(tmp_path: Path) -> None:
    database_path = tmp_path / "migration.sqlite3"
    config = _alembic_config(database_path)

    command.upgrade(config, "head")
    _assert_patient_schema(database_path)
    command.downgrade(config, "base")
    downgraded_engine = sa.create_engine(f"sqlite:///{database_path}")
    downgraded_inspector = sa.inspect(downgraded_engine)
    downgraded_tables = set(downgraded_inspector.get_table_names())
    assert "patients" not in downgraded_tables
    assert "assignments" not in downgraded_tables
    assert "studies" not in downgraded_tables or "patient_id" not in {
        column["name"] for column in downgraded_inspector.get_columns("studies")
    }
    downgraded_engine.dispose()

    command.upgrade(config, "head")
    _assert_patient_schema(database_path)


def test_all_migration_revision_ids_fit_alembic_version_column() -> None:
    config = _alembic_config(Path("unused.sqlite3"))
    script = ScriptDirectory.from_config(config)

    assert all(len(revision.revision) <= 32 for revision in script.walk_revisions())


def test_patient_migration_backfills_populated_studies(tmp_path: Path) -> None:
    database_path = tmp_path / "populated-migration.sqlite3"
    config = _alembic_config(database_path)
    command.upgrade(config, "f61c96c5c275")

    engine = sa.create_engine(f"sqlite:///{database_path}")
    metadata = sa.MetaData()
    studies = sa.Table("studies", metadata, autoload_with=engine)
    shared_patient_id = uuid4()
    other_patient_id = uuid4()
    now = datetime.now(timezone.utc)
    with engine.begin() as connection:
        connection.execute(
            studies.insert(),
            [
                {
                    "public_id": uuid4().hex,
                    "patient_public_id": shared_patient_id.hex,
                    "study_instance_uid": "legacy-study-1",
                    "source_kind": "nifti",
                    "source_metadata": {},
                    "staging_status": "staged",
                    "acquired_at": None,
                    "created_at": now,
                    "updated_at": now,
                },
                {
                    "public_id": uuid4().hex,
                    "patient_public_id": shared_patient_id.hex,
                    "study_instance_uid": "legacy-study-2",
                    "source_kind": "nifti",
                    "source_metadata": {},
                    "staging_status": "staged",
                    "acquired_at": None,
                    "created_at": now,
                    "updated_at": now,
                },
                {
                    "public_id": uuid4().hex,
                    "patient_public_id": other_patient_id.hex,
                    "study_instance_uid": "legacy-study-3",
                    "source_kind": "dicom",
                    "source_metadata": {},
                    "staging_status": "staged",
                    "acquired_at": None,
                    "created_at": now,
                    "updated_at": now,
                },
            ],
        )
    engine.dispose()

    command.upgrade(config, "a1b2c3d4e5f6")

    upgraded_engine = sa.create_engine(f"sqlite:///{database_path}")
    upgraded_metadata = sa.MetaData()
    upgraded_studies = sa.Table(
        "studies", upgraded_metadata, autoload_with=upgraded_engine
    )
    patients = sa.Table("patients", upgraded_metadata, autoload_with=upgraded_engine)
    with upgraded_engine.connect() as connection:
        patient_rows = connection.execute(sa.select(patients)).mappings().all()
        study_rows = connection.execute(
            sa.select(upgraded_studies).order_by(upgraded_studies.c.study_instance_uid)
        ).mappings().all()

    assert len(patient_rows) == 2
    patient_by_public_id = {
        str(UUID(str(row["public_id"]))): row for row in patient_rows
    }
    assert set(patient_by_public_id) == {str(shared_patient_id), str(other_patient_id)}
    assert all(row["pseudonym"].startswith("MIG-") for row in patient_rows)
    assert all(row["patient_id"] is not None for row in study_rows)
    assert study_rows[0]["patient_id"] == study_rows[1]["patient_id"]
    assert study_rows[2]["patient_id"] != study_rows[0]["patient_id"]
    upgraded_engine.dispose()
