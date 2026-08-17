from __future__ import annotations

from dataclasses import dataclass


def _require_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} is required")
    return cleaned


@dataclass(frozen=True)
class StoredArtifactRef:
    artifact_kind: str
    storage_root: str
    relative_path: str

    def __post_init__(self) -> None:
        _require_text(self.artifact_kind, field_name="artifact_kind")
        _require_text(self.storage_root, field_name="storage_root")
        _require_text(self.relative_path, field_name="relative_path")
        if self.relative_path.startswith("/") or ".." in self.relative_path.split("/"):
            raise ValueError("relative_path must be retrieval-safe")


@dataclass(frozen=True)
class StoredLesionMeasurement:
    volume_mm3: float
    longest_diameter_mm: float

    def __post_init__(self) -> None:
        if self.volume_mm3 < 0 or self.longest_diameter_mm < 0:
            raise ValueError("measurement values must be non-negative")


@dataclass(frozen=True)
class StoredLesionResult:
    lesion_id: str
    bounding_box: dict[str, int]
    measurements: StoredLesionMeasurement
    mask_artifact: StoredArtifactRef
    review_artifacts: tuple[StoredArtifactRef, ...] = ()
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        _require_text(self.lesion_id, field_name="lesion_id")
        if not self.bounding_box:
            raise ValueError("bounding_box is required")


@dataclass(frozen=True)
class StoredCaseResult:
    study_id: str
    result_artifact: StoredArtifactRef
    lesions: tuple[StoredLesionResult, ...]
    needs_review: bool
    case_qc_reasons: tuple[str, ...] = ()
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        _require_text(self.study_id, field_name="study_id")
        if self.needs_review and not self.case_qc_reasons and not self.lesions:
            raise ValueError("needs_review cases must include reasons or lesions")
