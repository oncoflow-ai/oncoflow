from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.config import Settings, get_settings
from app.modules.artifacts.storage import resolve_artifact_location
from app.modules.segmentation.contracts import (
    BoundingBox3D,
    CanonicalSeriesBundle,
    CanonicalSeriesSlotAssignment,
    ManagedArtifactRef,
    RunnerProvenance,
)
from app.modules.segmentation.runner import (
    NormalizedLesionPrediction,
    RunnerExecutionResult,
    SegmentationRunner,
)
from ml.inference import __version__ as inference_version
from ml.inference.config import InferenceConfig
from ml.inference.io import load_nifti, save_nifti
from ml.inference.pipeline.segment import StudySegmentation, segment_study


_FALLBACK_SLOT_ORDER = ("t1_post_or_fs", "t1_pre", "t2_or_stir")


class OncoFlowInferenceRunner(SegmentationRunner):
    """Backend runner that delegates automatic segmentation to `ml.inference`."""

    def __init__(self, *, model_id: str, settings: Settings | None = None) -> None:
        self._model_id = model_id
        self._settings = settings or get_settings()

    def run(self, *, bundle: CanonicalSeriesBundle) -> RunnerExecutionResult:
        assignment, input_warnings = _select_input_assignment(bundle, self._settings)
        source_location = resolve_artifact_location(
            assignment.source_artifact.storage_root,
            assignment.source_artifact.relative_path,
        )
        run_relative_dir = f"studies/{bundle.study_id}/inference/oncoflow"
        run_location = resolve_artifact_location("derived", run_relative_dir)
        cfg = _build_inference_config(self._settings, run_location.absolute_path)

        segmentation = segment_study(
            source_location.absolute_path,
            cfg,
            output_dir=run_location.absolute_path,
            use_cache=True,
        )
        warnings = tuple(input_warnings + _segmentation_warnings(segmentation))
        predictions = _predictions_from_segmentation(
            bundle=bundle,
            segmentation=segmentation,
            warnings=warnings,
        )

        return RunnerExecutionResult(
            bundle=bundle,
            runner=RunnerProvenance(
                model_id=self._model_id,
                runner_version=inference_version,
                execution_backend=f"oncoflow-inference:{cfg.backend}",
                warnings=warnings,
            ),
            lesions=predictions,
            warnings=warnings,
        )


def _build_inference_config(settings: Settings, run_dir: Path) -> InferenceConfig:
    enabled_models = tuple(
        model.strip()
        for model in settings.inference_enabled_models.split(",")
        if model.strip()
    )
    weights_dir = (
        Path(settings.inference_weights_dir).expanduser()
        if settings.inference_weights_dir
        else InferenceConfig().weights_dir
    )
    cache_dir = (
        Path(settings.inference_cache_dir).expanduser()
        if settings.inference_cache_dir
        else run_dir / "cache"
    )
    return InferenceConfig(
        backend=settings.inference_backend,
        device=settings.inference_device,
        enabled_models=enabled_models,
        weights_dir=weights_dir,
        cache_dir=cache_dir,
        n4_bias_correction=settings.inference_n4_bias_correction,
        skull_strip=settings.inference_skull_strip,
        isotropic_spacing_mm=settings.inference_isotropic_spacing_mm,
    )


def _select_input_assignment(
    bundle: CanonicalSeriesBundle,
    settings: Settings,
) -> tuple[CanonicalSeriesSlotAssignment, list[str]]:
    by_slot = {assignment.slot_name: assignment for assignment in bundle.slot_assignments}
    if settings.inference_input_slot in by_slot:
        return by_slot[settings.inference_input_slot], []

    for slot_name in _FALLBACK_SLOT_ORDER:
        if slot_name in by_slot:
            return by_slot[slot_name], [
                f"configured inference input slot {settings.inference_input_slot} missing; used {slot_name}",
            ]

    raise ValueError("No canonical NIfTI input is available for inference")


def _segmentation_warnings(segmentation: StudySegmentation) -> list[str]:
    warnings: list[str] = []
    adapter_meta = segmentation.adapter_meta
    if adapter_meta and all(meta.get("stub") for meta in adapter_meta.values()):
        warnings.append("all inference adapters were unavailable; ensemble mask is empty")
    for model_name, meta in adapter_meta.items():
        if meta.get("stub"):
            error = meta.get("error", "adapter unavailable")
            warnings.append(f"{model_name}: {error}")
    if segmentation.ensemble_mask.sum() == 0:
        warnings.append("inference produced an empty ensemble mask")
    if segmentation.panel_agreement.level == "low":
        warnings.append("low model agreement")
    return warnings


def _predictions_from_segmentation(
    *,
    bundle: CanonicalSeriesBundle,
    segmentation: StudySegmentation,
    warnings: tuple[str, ...],
) -> tuple[NormalizedLesionPrediction, ...]:
    components = _connected_components(segmentation.ensemble_mask)
    if not components:
        return ()

    preprocessed = load_nifti(segmentation.preprocessed_path)
    base_confidence = _confidence_from_agreement(segmentation)
    predictions: list[NormalizedLesionPrediction] = []
    for index, component in enumerate(components):
        component_mask = np.zeros(segmentation.ensemble_mask.shape, dtype=np.uint8)
        if component.size:
            component_mask[tuple(component.T)] = 1
        relative_path = f"studies/{bundle.study_id}/lesions/component-{index + 1:03d}.nii.gz"
        mask_location = resolve_artifact_location("derived", relative_path)
        save_nifti(preprocessed.copy_with(component_mask), mask_location.absolute_path)
        predictions.append(
            NormalizedLesionPrediction(
                mask_artifact=ManagedArtifactRef(
                    storage_root="derived",
                    relative_path=relative_path,
                ),
                bounding_box=_bbox_from_component(component),
                confidence_score=base_confidence,
                occupied_voxels_ijk=_occupied_voxels(component),
                warning_reasons=warnings,
                source_mask_path=str(mask_location.absolute_path),
                metadata={
                    "ensemble_volume_cm3": segmentation.ensemble_volume_cm3,
                    "panel_agreement": segmentation.panel_agreement.as_dict(),
                    "adapter_meta": segmentation.adapter_meta,
                    "preprocessed_spacing": list(segmentation.preprocessed_spacing),
                    "inference_output_dir": segmentation.output_dir,
                },
            )
        )
    return tuple(predictions)


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []
    try:
        from scipy import ndimage

        labels, count = ndimage.label(binary)
        components = [np.argwhere(labels == label) for label in range(1, count + 1)]
    except Exception:
        components = [np.argwhere(binary > 0)]
    components.sort(key=lambda component: _bbox_sort_key(_bbox_from_component(component)))
    return components


def _bbox_from_component(component: np.ndarray) -> BoundingBox3D:
    mins = component.min(axis=0)
    maxs = component.max(axis=0)
    return BoundingBox3D(
        x_min=int(mins[0]),
        x_max=int(maxs[0]),
        y_min=int(mins[1]),
        y_max=int(maxs[1]),
        z_min=int(mins[2]),
        z_max=int(maxs[2]),
    )


def _bbox_sort_key(bbox: BoundingBox3D) -> tuple[int, int, int, int, int, int]:
    return (bbox.z_min, bbox.y_min, bbox.x_min, bbox.z_max, bbox.y_max, bbox.x_max)


def _occupied_voxels(component: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (int(coord[0]), int(coord[1]), int(coord[2]))
        for coord in sorted(component.tolist(), key=lambda item: (item[2], item[1], item[0]))
    )


def _confidence_from_agreement(segmentation: StudySegmentation) -> float:
    if segmentation.panel_agreement.models_used:
        return max(0.0, min(1.0, float(segmentation.panel_agreement.mean_agreement)))
    return 0.0
