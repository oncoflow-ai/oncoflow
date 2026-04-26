from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from app.infra.db.models import Artifact, Series, Study
from app.infra.db.session import create_session_factory
from app.modules.segmentation.input_bundle import build_canonical_series_bundle
from app.modules.segmentation.runner import get_runner, run_segmentation
from app.modules.segmentation.runtime import ModelPackageMissingError


@pytest.fixture(autouse=True)
def configure_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONCOFLOW_DATABASE_URL", f"sqlite+pysqlite:///{tmp_path / 'segmentation-fusion.sqlite3'}")
    monkeypatch.setenv("ONCOFLOW_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("ONCOFLOW_STORAGE_STAGING_DIR", "raw")


def _add_series(session, *, study_id: int, description: str, protocol: str, uid: str, spacing: tuple[float, float, float] = (1.0, 1.0, 2.0), shape: tuple[int, int, int] = (64, 64, 12)) -> None:
    series = Series(
        study_id=study_id,
        series_instance_uid=uid,
        modality="MR",
        series_description=description,
        protocol_name=protocol,
        classification="processable",
        scanner_vendor="Siemens",
        source_metadata={},
    )
    session.add(series)
    session.flush()
    session.add(
        Artifact(
            study_id=study_id,
            series_id=series.id,
            artifact_kind="nifti-volume",
            storage_root="derived",
            relative_path=f"studies/demo/series/{series.id}/volume.nii.gz",
            source_metadata={"geometry": {"spacing_mm": list(spacing), "shape": list(shape)}},
        )
    )


def test_canonical_bundle_selects_one_series_per_slot() -> None:
    session_factory = create_session_factory()
    with session_factory() as session:
        study = Study(
            public_id=uuid4(),
            study_instance_uid=f"study-{uuid4()}",
            source_kind="dicom-study",
            source_metadata={},
            staging_status="processed",
        )
        session.add(study)
        session.flush()

        _add_series(session, study_id=study.id, description="tra_t1_tse", protocol="tra_t1_tse", uid="series-a")
        _add_series(session, study_id=study.id, description="tra_t1_tse_low_priority", protocol="tra_t1_tse", uid="series-b")
        _add_series(session, study_id=study.id, description="sag_t1_tse_fs+c", protocol="post contrast fs", uid="series-c")
        _add_series(session, study_id=study.id, description="t2_tse_stir_tra_RT", protocol="t2_tse_stir", uid="series-d")
        session.commit()

        bundle = build_canonical_series_bundle(session=session, study_public_id=study.public_id)

    assert [assignment.slot_name for assignment in bundle.slot_assignments] == ["t1_pre", "t1_post_or_fs", "t2_or_stir"]
    assert [assignment.series_instance_uid for assignment in bundle.slot_assignments] == ["series-a", "series-c", "series-d"]
    assert bundle.missing_slots == ()
    assert bundle.degradation_reasons == ()


def test_canonical_bundle_surfaces_geometry_mismatch_as_degradation() -> None:
    session_factory = create_session_factory()
    with session_factory() as session:
        study = Study(
            public_id=uuid4(),
            study_instance_uid=f"study-{uuid4()}",
            source_kind="dicom-study",
            source_metadata={},
            staging_status="processed",
        )
        session.add(study)
        session.flush()

        _add_series(session, study_id=study.id, description="tra_t1_tse", protocol="tra_t1_tse", uid="series-a", spacing=(1.0, 1.0, 2.0))
        _add_series(session, study_id=study.id, description="t2_tse_stir_tra_RT", protocol="t2_tse_stir", uid="series-b", spacing=(0.5, 0.5, 3.0))
        session.commit()

        bundle = build_canonical_series_bundle(session=session, study_public_id=study.public_id)

    assert bundle.missing_slots == ("t1_post_or_fs",)
    assert any("missing canonical slots" in reason for reason in bundle.degradation_reasons)
    assert any("share geometry" in reason for reason in bundle.degradation_reasons)


def test_runner_defaults_to_phase_two_baseline_and_rejects_unknown_models() -> None:
    session_factory = create_session_factory()
    with session_factory() as session:
        study = Study(
            public_id=uuid4(),
            study_instance_uid=f"study-{uuid4()}",
            source_kind="dicom-study",
            source_metadata={},
            staging_status="processed",
        )
        session.add(study)
        session.flush()
        _add_series(session, study_id=study.id, description="tra_t1_tse", protocol="tra_t1_tse", uid="series-a")
        session.commit()

        bundle = build_canonical_series_bundle(session=session, study_public_id=study.public_id)

    with pytest.raises(ModelPackageMissingError, match="ONCOFLOW_NNUNET_MODEL_DIR"):
        run_segmentation(bundle=bundle)

    result = run_segmentation(bundle=bundle, warnings=("stub-test-path",))
    assert result.runner.model_id == "nnunet-v2-resenc"
    assert result.runner.execution_backend == "stub"

    with pytest.raises(ValueError, match="Unknown benchmark model id"):
        get_runner(model_id="unknown-model")
