from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.config import Settings
from app.modules.segmentation.contracts import (
    CanonicalSeriesBundle,
    CanonicalSeriesSlotAssignment,
    ManagedArtifactRef,
)
from app.modules.segmentation.oncoflow_runner import OncoFlowInferenceRunner
from app.modules.segmentation.runner import run_segmentation


class _PanelAgreement:
    level = "high"
    mean_agreement = 0.92
    models_used = ("nnunet",)

    def as_dict(self) -> dict[str, object]:
        return {
            "mean_agreement": self.mean_agreement,
            "agreement_level": self.level,
            "models_used": list(self.models_used),
        }


def _write_nifti(path: Path, data: np.ndarray) -> None:
    nib = pytest.importorskip("nibabel")
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), str(path))


def _bundle(study_id: str = "study-001") -> CanonicalSeriesBundle:
    return CanonicalSeriesBundle(
        study_id=study_id,
        slot_assignments=(
            CanonicalSeriesSlotAssignment(
                slot_name="t1_post_or_fs",
                series_instance_uid="series-1",
                source_artifact=ManagedArtifactRef(
                    storage_root="raw",
                    relative_path="studies/study-001/t1c.nii.gz",
                ),
            ),
        ),
        missing_slots=("t1_pre", "t2_or_stir"),
        degradation_reasons=("missing canonical slots: t1_pre, t2_or_stir",),
    )


def test_oncoflow_runner_maps_segmentation_components_to_backend_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONCOFLOW_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("ONCOFLOW_STORAGE_STAGING_DIR", "raw")
    source_path = tmp_path / "storage" / "raw" / "studies" / "study-001" / "t1c.nii.gz"
    preprocessed_path = tmp_path / "preprocessed.nii.gz"
    _write_nifti(source_path, np.zeros((5, 5, 5), dtype=np.float32))
    _write_nifti(preprocessed_path, np.zeros((5, 5, 5), dtype=np.float32))

    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[1:3, 1:3, 2] = 1
    mask[4, 4, 4] = 1

    def fake_segment_study(nifti_path, cfg, *, output_dir, use_cache):
        assert Path(nifti_path) == source_path
        assert cfg.enabled_models == ("nnunet",)
        return SimpleNamespace(
            ensemble_mask=mask,
            preprocessed_path=str(preprocessed_path),
            output_dir=str(output_dir),
            ensemble_volume_cm3=0.005,
            panel_agreement=_PanelAgreement(),
            adapter_meta={"nnunet": {"mode": "test"}},
            preprocessed_spacing=(1.0, 1.0, 1.0),
        )

    monkeypatch.setattr(
        "app.modules.segmentation.oncoflow_runner.segment_study",
        fake_segment_study,
    )
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        storage_root=str(tmp_path / "storage"),
        storage_staging_dir="raw",
        inference_enabled_models="nnunet",
    )

    result = OncoFlowInferenceRunner(
        model_id="nnunet-v2-resenc",
        settings=settings,
    ).run(bundle=_bundle())

    assert result.runner.execution_backend == "oncoflow-inference:local"
    assert len(result.lesions) == 2
    assert result.lesions[0].confidence_score == pytest.approx(0.92)
    assert result.lesions[0].source_mask_path
    assert Path(result.lesions[0].source_mask_path).exists()
    assert result.lesions[0].metadata["panel_agreement"]["agreement_level"] == "high"


def test_backend_segmentation_dispatch_uses_oncoflow_runner_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONCOFLOW_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("ONCOFLOW_STORAGE_STAGING_DIR", "raw")
    source_path = tmp_path / "storage" / "raw" / "studies" / "study-001" / "t1c.nii.gz"
    preprocessed_path = tmp_path / "preprocessed.nii.gz"
    _write_nifti(source_path, np.zeros((3, 3, 3), dtype=np.float32))
    _write_nifti(preprocessed_path, np.zeros((3, 3, 3), dtype=np.float32))
    monkeypatch.setenv("OFLOW_ENABLED_MODELS", "")

    def fake_segment_study(nifti_path, cfg, *, output_dir, use_cache):
        return SimpleNamespace(
            ensemble_mask=np.zeros((3, 3, 3), dtype=np.uint8),
            preprocessed_path=str(preprocessed_path),
            output_dir=str(output_dir),
            ensemble_volume_cm3=0.0,
            panel_agreement=SimpleNamespace(
                level="low",
                mean_agreement=0.0,
                models_used=(),
                as_dict=lambda: {"agreement_level": "low"},
            ),
            adapter_meta={},
            preprocessed_spacing=(1.0, 1.0, 1.0),
        )

    monkeypatch.setattr(
        "app.modules.segmentation.oncoflow_runner.segment_study",
        fake_segment_study,
    )

    result = run_segmentation(bundle=_bundle())

    assert result.runner.execution_backend == "oncoflow-inference:local"
    assert result.lesions == ()
    assert "inference produced an empty ensemble mask" in result.warnings


def test_backend_oncoflow_runner_accepts_p01_example_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("nibabel")
    pytest.importorskip("SimpleITK")
    repo_root = Path(__file__).resolve().parents[2]
    p01_t1c = repo_root / "data" / "P01" / "BraTS" / "baseline" / "t1c.nii.gz"
    if not p01_t1c.exists():
        pytest.skip("P01 example data not present")

    monkeypatch.setenv("ONCOFLOW_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("ONCOFLOW_STORAGE_STAGING_DIR", "raw")
    monkeypatch.setenv("OFLOW_ENABLED_MODELS", "")
    target = tmp_path / "storage" / "raw" / "studies" / "p01" / "t1c.nii.gz"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p01_t1c, target)
    bundle = CanonicalSeriesBundle(
        study_id="p01",
        slot_assignments=(
            CanonicalSeriesSlotAssignment(
                slot_name="t1_post_or_fs",
                series_instance_uid="p01-t1c",
                source_artifact=ManagedArtifactRef(
                    storage_root="raw",
                    relative_path="studies/p01/t1c.nii.gz",
                ),
            ),
        ),
        missing_slots=("t1_pre", "t2_or_stir"),
        degradation_reasons=("P01 single-modality service smoke test",),
    )

    result = run_segmentation(bundle=bundle)

    assert result.runner.execution_backend == "oncoflow-inference:local"
    assert result.lesions == ()
    assert "inference produced an empty ensemble mask" in result.warnings
