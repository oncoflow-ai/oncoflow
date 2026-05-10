from __future__ import annotations

from pathlib import Path

import pytest

from app.modules.segmentation.contracts import CanonicalSeriesBundle, CanonicalSeriesSlotAssignment, ManagedArtifactRef
from app.modules.segmentation.nnunet_runner import (
    ModelInputPreparationError,
    NnUnetAutomaticRunner,
    PredictorOutput,
    PreparedNnUnetInput,
    prepare_nnunet_input,
)
from app.modules.segmentation.runtime import RuntimeReadiness


def _runtime(tmp_path: Path) -> RuntimeReadiness:
    model_root = tmp_path / "model"
    model_root.mkdir(parents=True, exist_ok=True)
    return RuntimeReadiness(
        model_id="nnunet-v1-brats",
        model_directory=str(model_root),
        checkpoint_relative_path="model_final_checkpoint.model",
        package_manifest_relative_path="dataset.json",
        weights_digest="abc123def456abc123def456abc123def456abc123def456abc123def456abcd",
        device="cpu",
        execution_backend="nnunetv1",
    )


def _touch_channel_files(tmp_path: Path) -> tuple[ManagedArtifactRef, ManagedArtifactRef, ManagedArtifactRef]:
    files = []
    for idx in range(3):
        relative = f"studies/study-001/series/{idx + 1}/volume.nii.gz"
        absolute = tmp_path / "storage" / "derived" / relative
        absolute.parent.mkdir(parents=True, exist_ok=True)
        absolute.write_bytes(b"nifti")
        files.append(ManagedArtifactRef(storage_root="derived", relative_path=relative))
    return files[0], files[1], files[2]


def _bundle(tmp_path: Path) -> CanonicalSeriesBundle:
    t1_pre, t1_post, t2 = _touch_channel_files(tmp_path)
    return CanonicalSeriesBundle(
        study_id="study-001",
        slot_assignments=(
            CanonicalSeriesSlotAssignment(slot_name="t1_pre", series_instance_uid="series-1", source_artifact=t1_pre),
            CanonicalSeriesSlotAssignment(slot_name="t1_post_or_fs", series_instance_uid="series-2", source_artifact=t1_post),
            CanonicalSeriesSlotAssignment(slot_name="t2_or_stir", series_instance_uid="series-3", source_artifact=t2),
        ),
    )


def test_prepare_nnunet_input_maps_canonical_slots_into_fixed_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONCOFLOW_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("ONCOFLOW_STORAGE_STAGING_DIR", "raw")
    prepared = prepare_nnunet_input(bundle=_bundle(tmp_path), runtime=_runtime(tmp_path))

    assert [channel.slot_name for channel in prepared.channels] == ["t1_pre", "t1_post_or_fs", "t2_or_stir"]
    assert [channel.channel_index for channel in prepared.channels] == [0, 1, 2]
    assert all(Path(channel.source_absolute_path).exists() for channel in prepared.channels)


def test_prepare_nnunet_input_rejects_missing_slots_or_geometry_degradation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONCOFLOW_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("ONCOFLOW_STORAGE_STAGING_DIR", "raw")
    bundle = _bundle(tmp_path)

    degraded = CanonicalSeriesBundle(
        study_id=bundle.study_id,
        slot_assignments=bundle.slot_assignments[:2],
        missing_slots=("t2_or_stir",),
        degradation_reasons=("missing canonical slots: t2_or_stir",),
    )
    with pytest.raises(ModelInputPreparationError, match="missing"):
        prepare_nnunet_input(bundle=degraded, runtime=_runtime(tmp_path))

    geometry = CanonicalSeriesBundle(
        study_id=bundle.study_id,
        slot_assignments=bundle.slot_assignments,
        degradation_reasons=("selected canonical series do not share geometry",),
    )
    with pytest.raises(ModelInputPreparationError, match="non-degraded"):
        prepare_nnunet_input(bundle=geometry, runtime=_runtime(tmp_path))


def test_nnunet_runner_normalizes_predictor_output_into_lesion_predictions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONCOFLOW_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("ONCOFLOW_STORAGE_STAGING_DIR", "raw")

    class FakePredictor:
        def predict(self, *, prepared_input: PreparedNnUnetInput) -> PredictorOutput:
            assert prepared_input.case_id == "study_001"
            return PredictorOutput(
                binary_mask=[
                    [
                        [1, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    [
                        [0, 0, 0],
                        [0, 0, 1],
                        [0, 0, 1],
                    ],
                ],
                warnings=("uncertain logits",),
                component_confidences=(0.91, 0.41),
            )

    runner = NnUnetAutomaticRunner(
        runtime=_runtime(tmp_path),
        predictor_factory=lambda runtime: FakePredictor(),
    )
    result = runner.run(bundle=_bundle(tmp_path))

    assert result.runner.execution_backend == "nnunetv1"
    assert [prediction.bounding_box.z_min for prediction in result.lesions] == [0, 1]
    assert result.lesions[0].confidence_score == 0.91
    assert result.lesions[1].confidence_score == 0.41
    assert result.lesions[1].occupied_voxels_ijk[-1] == (2, 2, 1)


def test_nnunet_runner_treats_empty_masks_as_valid_empty_prediction_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONCOFLOW_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("ONCOFLOW_STORAGE_STAGING_DIR", "raw")

    class EmptyPredictor:
        def predict(self, *, prepared_input: PreparedNnUnetInput) -> PredictorOutput:
            return PredictorOutput(binary_mask=[[[0]]])

    runner = NnUnetAutomaticRunner(
        runtime=_runtime(tmp_path),
        predictor_factory=lambda runtime: EmptyPredictor(),
    )
    result = runner.run(bundle=_bundle(tmp_path))

    assert result.lesions == ()
