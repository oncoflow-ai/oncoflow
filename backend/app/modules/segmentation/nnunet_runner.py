from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory
from typing import Callable, Protocol

from app.modules.artifacts.storage import resolve_artifact_location
from app.modules.segmentation.contracts import BoundingBox3D, CanonicalSeriesBundle, ManagedArtifactRef, RunnerProvenance
from app.modules.segmentation.runner import NormalizedLesionPrediction, RunnerExecutionResult, SegmentationRunner
from app.modules.segmentation.runtime import RuntimeReadiness, SegmentationRuntimeError

REQUIRED_SLOT_ORDER = ("t1_pre", "t1_post_or_fs", "t2_or_stir")


class ModelInputPreparationError(SegmentationRuntimeError):
    pass


@dataclass(frozen=True)
class PreparedNnUnetChannel:
    channel_index: int
    slot_name: str
    series_instance_uid: str
    source_relative_path: str
    source_absolute_path: str


@dataclass(frozen=True)
class PreparedNnUnetInput:
    study_id: str
    runtime: RuntimeReadiness
    channels: tuple[PreparedNnUnetChannel, ...]

    @property
    def case_id(self) -> str:
        return self.study_id.replace("-", "_")


@dataclass(frozen=True)
class PredictorOutput:
    binary_mask: object
    warnings: tuple[str, ...] = ()
    component_confidences: tuple[float, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


class Predictor(Protocol):
    def predict(self, *, prepared_input: PreparedNnUnetInput) -> PredictorOutput:
        ...


def prepare_nnunet_input(*, bundle: CanonicalSeriesBundle, runtime: RuntimeReadiness) -> PreparedNnUnetInput:
    if bundle.missing_slots:
        raise ModelInputPreparationError(
            "Real nnU-Net inference requires all canonical slots; missing: "
            + ", ".join(bundle.missing_slots)
        )
    if bundle.degradation_reasons:
        raise ModelInputPreparationError(
            "Real nnU-Net inference requires a non-degraded canonical bundle: "
            + "; ".join(bundle.degradation_reasons)
        )

    assignments_by_slot = {assignment.slot_name: assignment for assignment in bundle.slot_assignments}
    channels: list[PreparedNnUnetChannel] = []
    for channel_index, slot_name in enumerate(REQUIRED_SLOT_ORDER):
        assignment = assignments_by_slot.get(slot_name)
        if assignment is None:
            raise ModelInputPreparationError(f"Canonical slot {slot_name} is required for real nnU-Net inference")

        location = resolve_artifact_location(
            assignment.source_artifact.storage_root,
            assignment.source_artifact.relative_path,
        )
        if not location.absolute_path.exists():
            raise ModelInputPreparationError(
                f"Canonical slot {slot_name} source artifact is missing: {location.absolute_path}"
            )

        channels.append(
            PreparedNnUnetChannel(
                channel_index=channel_index,
                slot_name=slot_name,
                series_instance_uid=assignment.series_instance_uid,
                source_relative_path=assignment.source_artifact.relative_path,
                source_absolute_path=str(location.absolute_path),
            )
        )

    return PreparedNnUnetInput(
        study_id=bundle.study_id,
        runtime=runtime,
        channels=tuple(channels),
    )


def _mask_to_nested_lists(mask: object) -> list[list[list[int]]]:
    if hasattr(mask, "tolist"):
        mask = mask.tolist()
    if not isinstance(mask, list):
        raise ValueError("binary mask output must be convertible to nested lists")
    return mask


def _occupied_voxels(mask: object) -> set[tuple[int, int, int]]:
    nested = _mask_to_nested_lists(mask)
    occupied: set[tuple[int, int, int]] = set()
    for z, plane in enumerate(nested):
        for y, row in enumerate(plane):
            for x, value in enumerate(row):
                if value:
                    occupied.add((x, y, z))
    return occupied


def _connected_components(mask: object) -> list[set[tuple[int, int, int]]]:
    remaining = _occupied_voxels(mask)
    components: list[set[tuple[int, int, int]]] = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        frontier: deque[tuple[int, int, int]] = deque([seed])
        while frontier:
            x, y, z = frontier.popleft()
            for neighbor in (
                (x - 1, y, z),
                (x + 1, y, z),
                (x, y - 1, z),
                (x, y + 1, z),
                (x, y, z - 1),
                (x, y, z + 1),
            ):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    frontier.append(neighbor)
        components.append(component)
    return components


def _bbox(component: set[tuple[int, int, int]]) -> BoundingBox3D:
    xs = [voxel[0] for voxel in component]
    ys = [voxel[1] for voxel in component]
    zs = [voxel[2] for voxel in component]
    return BoundingBox3D(
        x_min=min(xs),
        x_max=max(xs),
        y_min=min(ys),
        y_max=max(ys),
        z_min=min(zs),
        z_max=max(zs),
    )


def _bbox_sort_key(component: set[tuple[int, int, int]]) -> tuple[int, int, int, int, int, int]:
    bbox = _bbox(component)
    return (bbox.z_min, bbox.y_min, bbox.x_min, bbox.z_max, bbox.y_max, bbox.x_max)


def normalize_predictor_output(
    *,
    prepared_input: PreparedNnUnetInput,
    output: PredictorOutput,
) -> tuple[NormalizedLesionPrediction, ...]:
    components = sorted(_connected_components(output.binary_mask), key=_bbox_sort_key)
    predictions: list[NormalizedLesionPrediction] = []

    for index, component in enumerate(components):
        confidence = output.component_confidences[index] if index < len(output.component_confidences) else 1.0
        predictions.append(
            NormalizedLesionPrediction(
                mask_artifact=ManagedArtifactRef(
                    storage_root="derived",
                    relative_path=f"studies/{prepared_input.study_id}/lesions/component-{index + 1:03d}.nii.gz",
                ),
                bounding_box=_bbox(component),
                confidence_score=float(confidence),
                occupied_voxels_ijk=tuple(sorted(component, key=lambda voxel: (voxel[2], voxel[1], voxel[0]))),
                warning_reasons=tuple(output.warnings),
            )
        )
    return tuple(predictions)


class DefaultNnUnetPredictor:
    def __init__(self, runtime: RuntimeReadiness) -> None:
        self._runtime = runtime

    def predict(self, *, prepared_input: PreparedNnUnetInput) -> PredictorOutput:
        try:
            import nibabel as nib
            import torch
            from nnunetv1.inference.predict_from_raw_data import nnUNetPredictor
        except ModuleNotFoundError as exc:  # pragma: no cover - covered by runtime gate tests
            raise SegmentationRuntimeError(
                "nnU-Net runtime dependencies are missing from the active Python environment"
            ) from exc

        with TemporaryDirectory(prefix="oncoflow-nnunet-") as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            channel_paths: list[str] = []
            for channel in prepared_input.channels:
                target = input_dir / f"{prepared_input.case_id}_{channel.channel_index:04d}.nii.gz"
                copy2(channel.source_absolute_path, target)
                channel_paths.append(str(target))

            output_path = output_dir / f"{prepared_input.case_id}.nii.gz"
            device = torch.device(prepared_input.runtime.device)
            predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=False,
                perform_everything_on_device=prepared_input.runtime.device != "cpu",
                device=device,
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=False,
            )
            predictor.initialize_from_trained_model_folder(
                prepared_input.runtime.model_directory,
                use_folds=("all",),
                checkpoint_name=Path(prepared_input.runtime.checkpoint_relative_path).name,
            )
            predictor.predict_from_files(
                [channel_paths],
                [str(output_path)],
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
            )

            predicted = nib.load(str(output_path)).get_fdata()
            binary_mask = (predicted > 0).astype(int).tolist()
            return PredictorOutput(binary_mask=binary_mask)


class NnUnetAutomaticRunner(SegmentationRunner):
    def __init__(
        self,
        *,
        runtime: RuntimeReadiness,
        predictor_factory: Callable[[RuntimeReadiness], Predictor] | None = None,
    ) -> None:
        self._runtime = runtime
        self._predictor_factory = predictor_factory or DefaultNnUnetPredictor

    def run(self, *, bundle: CanonicalSeriesBundle) -> RunnerExecutionResult:
        prepared_input = prepare_nnunet_input(bundle=bundle, runtime=self._runtime)
        predictor = self._predictor_factory(self._runtime)
        output = predictor.predict(prepared_input=prepared_input)
        predictions = normalize_predictor_output(prepared_input=prepared_input, output=output)
        return RunnerExecutionResult(
            bundle=bundle,
            runner=RunnerProvenance(
                model_id=self._runtime.model_id,
                runner_version=self._runtime.runner_version,
                execution_backend=self._runtime.execution_backend,
                warnings=output.warnings,
            ),
            lesions=predictions,
            warnings=output.warnings,
        )
