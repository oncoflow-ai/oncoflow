from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.modules.benchmark.model_registry import get_model_spec
from app.modules.segmentation.contracts import BoundingBox3D, CanonicalSeriesBundle, ManagedArtifactRef, RunnerProvenance
from app.modules.segmentation.runtime import resolve_runtime_readiness


@dataclass(frozen=True)
class NormalizedLesionPrediction:
    mask_artifact: ManagedArtifactRef
    bounding_box: BoundingBox3D
    confidence_score: float
    occupied_voxels_ijk: tuple[tuple[int, int, int], ...] = ()
    warning_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("confidence_score must be between 0.0 and 1.0")
        if not self.occupied_voxels_ijk:
            raise ValueError("occupied_voxels_ijk must include at least one voxel")


@dataclass(frozen=True)
class RunnerExecutionResult:
    bundle: CanonicalSeriesBundle
    runner: RunnerProvenance
    lesions: tuple[NormalizedLesionPrediction, ...]
    warnings: tuple[str, ...] = ()


class SegmentationRunner(Protocol):
    def run(self, *, bundle: CanonicalSeriesBundle) -> RunnerExecutionResult:
        ...


class StubAutomaticRunner:
    def __init__(self, *, model_id: str, predictions: tuple[NormalizedLesionPrediction, ...] = (), warnings: tuple[str, ...] = ()) -> None:
        self._model_id = model_id
        self._predictions = predictions
        self._warnings = warnings

    def run(self, *, bundle: CanonicalSeriesBundle) -> RunnerExecutionResult:
        get_model_spec(self._model_id)
        return RunnerExecutionResult(
            bundle=bundle,
            runner=RunnerProvenance(
                model_id=self._model_id,
                runner_version="2026.04",
                execution_backend="stub",
                warnings=self._warnings,
            ),
            lesions=self._predictions,
            warnings=self._warnings,
        )


def get_runner(*, model_id: str = "nnunet-v2-resenc", predictions: tuple[NormalizedLesionPrediction, ...] = (), warnings: tuple[str, ...] = ()) -> SegmentationRunner:
    get_model_spec(model_id)
    if not predictions and not warnings and model_id == "nnunet-v2-resenc":
        from app.modules.segmentation.nnunet_runner import NnUnetAutomaticRunner

        return NnUnetAutomaticRunner(runtime=resolve_runtime_readiness(model_id=model_id))
    return StubAutomaticRunner(model_id=model_id, predictions=predictions, warnings=warnings)


def run_segmentation(*, bundle: CanonicalSeriesBundle, model_id: str = "nnunet-v2-resenc", predictions: tuple[NormalizedLesionPrediction, ...] = (), warnings: tuple[str, ...] = ()) -> RunnerExecutionResult:
    runner = get_runner(model_id=model_id, predictions=predictions, warnings=warnings)
    return runner.run(bundle=bundle)
