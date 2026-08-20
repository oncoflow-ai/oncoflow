from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect

from app.core.audit import log_audit_event
from app.core.security import create_access_token, get_password_hash
from app.infra.db.base import Base
from app.infra.db.models import (
    Artifact,
    AuditLog,
    Comparison,
    Patient,
    Report,
    Study,
    User,
)
from app.infra.db.session import create_session_factory
from app.main import create_app


def test_comparison_and_report_model_persistence() -> None:
    session_factory = create_session_factory()
    with session_factory() as session:
        # Create Patient
        patient = Patient(
            public_id=uuid4(),
            pseudonym="PAT-TEST-001",
            diagnosis="Osteosarcoma",
            status="active",
        )
        session.add(patient)
        session.flush()

        # Create Studies
        study_a = Study(
            public_id=uuid4(),
            patient_id=patient.id,
            patient_public_id=patient.public_id,
            study_instance_uid="1.2.3.4.5.1",
            source_kind="nifti",
            staging_status="staged",
            acquired_at=date(2026, 1, 1),
        )
        study_b = Study(
            public_id=uuid4(),
            patient_id=patient.id,
            patient_public_id=patient.public_id,
            study_instance_uid="1.2.3.4.5.2",
            source_kind="nifti",
            staging_status="staged",
            acquired_at=date(2026, 4, 1),
        )
        session.add_all([study_a, study_b])
        session.flush()

        # Create Comparison
        comparison = Comparison(
            public_id=uuid4(),
            study_a_id=study_a.id,
            study_b_id=study_b.id,
            volume_a=12.5,
            volume_b=15.0,
            delta_cm3=2.5,
            pct_change=20.0,
            dice_overlap=0.88,
            hd95_mm=3.2,
            growth_rate_cm3_per_day=0.028,
            interpretation_flag="Progression",
            recist_ratio=1.15,
            vol_delta_ci_half_cm3=0.35,
            registration_ncc=0.92,
            comparison_metadata={"model_versions": {"nnunet": "v2"}},
        )
        session.add(comparison)
        session.flush()

        # Create PDF Artifact
        pdf_artifact = Artifact(
            study_id=study_b.id,
            artifact_kind="report-pdf",
            storage_root="derived",
            relative_path="reports/test_report.pdf",
            source_metadata={"pages": 2},
        )
        session.add(pdf_artifact)
        session.flush()

        # Create Report
        report = Report(
            public_id=uuid4(),
            patient_id=patient.id,
            comparison_id=comparison.id,
            pdf_artifact_id=pdf_artifact.id,
            signature="SHA256-RSA-SIG-TEST-12345",
            generated_at=datetime.now(timezone.utc),
        )
        session.add(report)
        session.commit()

        # Verification
        saved_comp = session.query(Comparison).filter(Comparison.id == comparison.id).one()
        assert saved_comp.volume_a == 12.5
        assert saved_comp.volume_b == 15.0
        assert saved_comp.delta_cm3 == 2.5
        assert saved_comp.pct_change == 20.0
        assert saved_comp.dice_overlap == 0.88
        assert saved_comp.hd95_mm == 3.2
        assert saved_comp.growth_rate_cm3_per_day == 0.028
        assert saved_comp.interpretation_flag == "Progression"
        assert saved_comp.recist_ratio == 1.15
        assert saved_comp.registration_ncc == 0.92
        assert saved_comp.study_a.id == study_a.id
        assert saved_comp.study_b.id == study_b.id

        saved_report = session.query(Report).filter(Report.id == report.id).one()
        assert saved_report.patient_id == patient.id
        assert saved_report.comparison_id == comparison.id
        assert saved_report.pdf_artifact_id == pdf_artifact.id
        assert saved_report.signature == "SHA256-RSA-SIG-TEST-12345"
        assert saved_report.patient.pseudonym == "PAT-TEST-001"
        assert saved_report.comparison.delta_cm3 == 2.5
        assert saved_report.pdf_artifact.relative_path == "reports/test_report.pdf"


def test_audit_log_model_and_helper_persistence() -> None:
    session_factory = create_session_factory()
    with session_factory() as session:
        log_audit_event(
            action="TEST_PERSIST_ACTION",
            resource_id="res-abc-999",
            actor="dr.house@oncoflow.local",
            details={"ip": "127.0.0.1", "action_code": 42},
            db=session,
        )
        session.commit()

        log_entry = (
            session.query(AuditLog)
            .filter(AuditLog.resource_id == "res-abc-999")
            .order_by(AuditLog.id.desc())
            .first()
        )
        assert log_entry is not None
        assert log_entry.action == "TEST_PERSIST_ACTION"
        assert log_entry.actor_id == "dr.house@oncoflow.local"
        assert log_entry.details == {"ip": "127.0.0.1", "action_code": 42}
        assert log_entry.timestamp is not None


def test_audit_logs_api_endpoint() -> None:
    session_factory = create_session_factory()
    with session_factory() as session:
        admin_user = User(
            public_id=uuid4(),
            email="admin_audit_test@oncoflow.local",
            name="Admin User",
            hashed_password=get_password_hash("password"),
            role="admin",
        )
        doctor_user = User(
            public_id=uuid4(),
            email="doctor_audit_test@oncoflow.local",
            name="Doctor User",
            hashed_password=get_password_hash("password"),
            role="doctor",
        )
        session.add_all([admin_user, doctor_user])
        session.commit()

        admin_token = create_access_token({"sub": str(admin_user.public_id)})
        doctor_token = create_access_token({"sub": str(doctor_user.public_id)})

        # Seed audit logs
        log1 = AuditLog(
            actor_id=str(admin_user.public_id),
            action="CREATE_PATIENT",
            resource_id="PAT-100",
            details={"note": "first patient"},
        )
        log2 = AuditLog(
            actor_id=str(doctor_user.public_id),
            action="VIEW_PATIENT",
            resource_id="PAT-100",
            details={"note": "viewed"},
        )
        session.add_all([log1, log2])
        session.commit()

    app = create_app()
    client = TestClient(app)

    # 1. Non-admin should be forbidden (403)
    resp = client.get(
        "/api/v1/audit-logs",
        headers={"Authorization": f"Bearer {doctor_token}"},
    )
    assert resp.status_code == 403

    # 2. Admin should get 200 and list of logs
    resp = client.get(
        "/api/v1/audit-logs",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 2

    # 3. Filter by action
    resp = client.get(
        "/api/v1/audit-logs?action=CREATE_PATIENT",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 200
    filtered_data = resp.json()
    assert all(item["action"] == "CREATE_PATIENT" for item in filtered_data)


def test_alembic_migration_0004(tmp_path: Path) -> None:
    from alembic import command
    from alembic.config import Config

    db_path = tmp_path / "migration_0004.sqlite3"
    backend_dir = Path(__file__).resolve().parent.parent
    alembic_ini_path = backend_dir / "alembic.ini"
    alembic_dir_path = backend_dir / "alembic"

    config = Config(str(alembic_ini_path))
    config.set_main_option("script_location", str(alembic_dir_path))
    config.set_main_option("sqlalchemy.url", f"sqlite+pysqlite:///{db_path}")

    # Upgrade to head
    command.upgrade(config, "head")

    engine = create_engine(f"sqlite+pysqlite:///{db_path}")
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())

    assert "comparisons" in table_names
    assert "reports" in table_names
    assert "audit_logs" in table_names

    comp_cols = {col["name"] for col in inspector.get_columns("comparisons")}
    assert {
        "id",
        "public_id",
        "study_a_id",
        "study_b_id",
        "volume_a",
        "volume_b",
        "delta_cm3",
        "pct_change",
        "dice_overlap",
        "hd95_mm",
        "growth_rate_cm3_per_day",
        "interpretation_flag",
        "recist_ratio",
        "vol_delta_ci_half_cm3",
        "registration_ncc",
        "metadata",
        "created_at",
    }.issubset(comp_cols)

    rep_cols = {col["name"] for col in inspector.get_columns("reports")}
    assert {
        "id",
        "public_id",
        "patient_id",
        "comparison_id",
        "pdf_artifact_id",
        "signature",
        "generated_at",
        "created_at",
    }.issubset(rep_cols)

    audit_cols = {col["name"] for col in inspector.get_columns("audit_logs")}
    assert {
        "id",
        "actor_id",
        "action",
        "resource_id",
        "details",
        "timestamp",
    }.issubset(audit_cols)
