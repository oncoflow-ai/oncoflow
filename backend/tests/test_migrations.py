from __future__ import annotations

from pathlib import Path
import sys
from datetime import datetime, timezone
from uuid import uuid4

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


def test_patient_migration_backfills_legacy_studies(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy-migration.sqlite3"
    config = _alembic_config(database_path)
    command.upgrade(config, "f61c96c5c275")

    engine = sa.create_engine(f"sqlite:///{database_path}")
    metadata = sa.MetaData()
    studies = sa.Table("studies", metadata, autoload_with=engine)
    legacy_patient_id = uuid4().hex
    now = datetime.now(timezone.utc)
    with engine.begin() as connection:
        connection.execute(
            studies.insert(),
            [
                {
                    "public_id": uuid4().hex,
                    "patient_public_id": legacy_patient_id,
                    "study_instance_uid": f"legacy-{index}",
                    "source_kind": "nifti-upload",
                    "source_metadata": {},
                    "staging_status": "staged",
                    "created_at": now,
                    "updated_at": now,
                }
                for index in range(2)
            ],
        )
    engine.dispose()

    command.upgrade(config, "head")
    engine = sa.create_engine(f"sqlite:///{database_path}")
    metadata = sa.MetaData()
    patients = sa.Table("patients", metadata, autoload_with=engine)
    studies = sa.Table("studies", metadata, autoload_with=engine)
    with engine.connect() as connection:
        patient_rows = connection.execute(
            sa.select(patients).where(patients.c.public_id == legacy_patient_id)
        ).mappings().all()
        study_rows = connection.execute(
            sa.select(studies).where(studies.c.patient_public_id == legacy_patient_id)
        ).mappings().all()

    assert len(patient_rows) == 1
    assert {row["patient_id"] for row in study_rows} == {patient_rows[0]["id"]}
    engine.dispose()
