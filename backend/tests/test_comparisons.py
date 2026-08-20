from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from app.infra.db.models import Artifact, Comparison, Patient, Study, User
from app.infra.db.session import create_session_factory
from app.modules.results.comparisons import ComparisonError, run_longitudinal_comparison


class _FakeResult:
    def summary(self) -> dict:
        return {"metrics": {}, "interpretation": {}, "registration": {}}


def test_comparison_persistence_failure_returns_server_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_factory = create_session_factory()
    with session_factory() as session:
        patient = Patient(pseudonym="PAT-PERSIST-FAIL")
        admin = User(
            email="comparison-admin@test.local",
            name="Comparison Admin",
            hashed_password="hash",
            role="admin",
        )
        session.add_all([patient, admin])
        session.flush()
        studies = []
        for index in (1, 2):
            study = Study(
                public_id=uuid4(),
                patient_id=patient.id,
                patient_public_id=patient.public_id,
                study_instance_uid=f"persist-failure-{index}",
                source_kind="nifti",
                staging_status="staged",
            )
            session.add(study)
            session.flush()
            session.add(Artifact(
                study_id=study.id,
                artifact_kind="nifti-source",
                storage_root="raw",
                relative_path=f"persist-failure/{index}.nii.gz",
            ))
            studies.append(str(study.public_id))
        session.commit()

    monkeypatch.setattr("app.modules.results.comparisons._build_inference_config", lambda: object())
    monkeypatch.setattr(
        "ml.inference.pipeline.longitudinal.compare_studies",
        lambda **_kwargs: _FakeResult(),
    )
    original_commit = Session.commit

    def fail_comparison_commit(session: Session) -> None:
        if any(isinstance(value, Comparison) for value in session.new):
            raise RuntimeError("database unavailable")
        original_commit(session)

    monkeypatch.setattr(Session, "commit", fail_comparison_commit)

    with pytest.raises(ComparisonError) as caught:
        run_longitudinal_comparison(
            baseline_study_id=studies[0],
            followup_study_id=studies[1],
            current_user=admin,
        )

    assert caught.value.status_code == 500
    assert caught.value.message == "failed to persist comparison"
    comparison_root = tmp_path / "storage" / "derived" / "comparisons"
    assert not comparison_root.exists() or list(comparison_root.iterdir()) == []
