from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.modules.benchmark.model_registry import get_model_spec

ArtifactRootKind = Literal["raw", "derived"]
CanonicalSlotName = Literal["t1_pre", "t1_post_or_fs", "t2_or_stir"]
ConfidenceBucket = Literal["high", "medium", "low"]
ReviewArtifactKind = Literal["overlay", "thumbnail", "index"]


def _require_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} is required")
    return cleaned


def build_lesion_id(*, study_id: str, lesion_index: int) -> str:
    if lesion_index < 0:
        raise ValueError("lesion_index must be non-negative")
    return f"{_require_text(study_id, field_name='study_id')}:lesion-{lesion_index + 1:03d}"


@dataclass(frozen=True)
class ManagedArtifactRef:
    storage_root: ArtifactRootKind
    relative_path: str

    def __post_init__(self) -> None:
        _require_text(self.relative_path, field_name="relative_path")

        if self.relative_path.startswith("/"):
            raise ValueError("relative_path must stay inside a managed storage root")
        if ".." in self.relative_path.split("/"):
            raise ValueError("relative_path cannot escape the managed storage root")


@dataclass(frozen=True)
class ReviewArtifactRef:
    artifact_kind: ReviewArtifactKind
    artifact_ref: ManagedArtifactRef
    provenance_label: str

    def __post_init__(self) -> None:
        _require_text(self.provenance_label, field_name="provenance_label")


@dataclass(frozen=True)
class BoundingBox3D:
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    z_min: int
    z_max: int

    def __post_init__(self) -> None:
        if self.x_min < 0 or self.y_min < 0 or self.z_min < 0:
            raise ValueError("bounding box coordinates must be non-negative")
        if self.x_min > self.x_max or self.y_min > self.y_max or self.z_min > self.z_max:
            raise ValueError("bounding box minima must not exceed maxima")


@dataclass(frozen=True)
class LesionQc:
    confidence_bucket: ConfidenceBucket
    flagged_for_review: bool
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.flagged_for_review and not self.reasons:
            raise ValueError("flagged_for_review lesions must include at least one QC reason")


@dataclass(frozen=True)
class RunnerProvenance:
    model_id: str
    runner_version: str
    execution_backend: str = "python"
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.model_id, field_name="model_id")
        _require_text(self.runner_version, field_name="runner_version")
        _require_text(self.execution_backend, field_name="execution_backend")
        get_model_spec(self.model_id)


@dataclass(frozen=True)
class CanonicalSeriesSlotAssignment:
    slot_name: CanonicalSlotName
    series_instance_uid: str
    source_artifact: ManagedArtifactRef

    def __post_init__(self) -> None:
        _require_text(self.series_instance_uid, field_name="series_instance_uid")


@dataclass(frozen=True)
class CanonicalSeriesBundle:
    study_id: str
    slot_assignments: tuple[CanonicalSeriesSlotAssignment, ...]
    missing_slots: tuple[CanonicalSlotName, ...] = ()
    degradation_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.study_id, field_name="study_id")
        if not self.slot_assignments:
            raise ValueError("slot_assignments must include at least one canonical series")

        slot_names = [assignment.slot_name for assignment in self.slot_assignments]
        if len(slot_names) != len(set(slot_names)):
            raise ValueError("slot_assignments must not contain duplicate canonical slots")

        if self.missing_slots and not self.degradation_reasons:
            raise ValueError("missing canonical slots must surface degradation_reasons")


@dataclass(frozen=True)
class LesionResult:
    lesion_id: str
    mask_artifact: ManagedArtifactRef
    bounding_box: BoundingBox3D
    qc: LesionQc
    review_artifacts: tuple[ReviewArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.lesion_id, field_name="lesion_id")


@dataclass(frozen=True)
class CaseSegmentationResult:
    study_id: str
    input_bundle: CanonicalSeriesBundle
    runner: RunnerProvenance
    lesions: tuple[LesionResult, ...]
    needs_review: bool
    case_qc_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.study_id, field_name="study_id")
        if self.needs_review and not self.case_qc_reasons and not any(
            lesion.qc.flagged_for_review for lesion in self.lesions
        ):
            raise ValueError("needs_review cases must include case_qc_reasons or review-flagged lesions")
