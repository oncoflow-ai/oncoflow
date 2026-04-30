from __future__ import annotations

from dataclasses import dataclass, field

from app.modules.segmentation.contracts import ManagedArtifactRef, ReviewArtifactRef

PHI_FORBIDDEN_KEYS = {
    "patient_name",
    "patient_id",
    "patient_birth_date",
    "accession_number",
}


@dataclass(frozen=True)
class ReviewArtifactDescriptor:
    artifact: ReviewArtifactRef
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        forbidden = sorted(key for key in self.metadata if key.lower() in PHI_FORBIDDEN_KEYS)
        if forbidden:
            raise ValueError("review artifact metadata contains PHI-bearing keys: " + ", ".join(forbidden))


def make_review_artifact_ref(
    *,
    artifact_kind: str,
    relative_path: str,
    provenance_label: str,
    metadata: dict[str, str] | None = None,
) -> ReviewArtifactDescriptor:
    descriptor = ReviewArtifactDescriptor(
        artifact=ReviewArtifactRef(
            artifact_kind=artifact_kind,  # type: ignore[arg-type]
            artifact_ref=ManagedArtifactRef(storage_root="derived", relative_path=relative_path),
            provenance_label=provenance_label,
        ),
        metadata=metadata or {},
    )
    return descriptor


def determine_case_review(*, bundle_degradation_reasons: tuple[str, ...], lesion_flagged_for_review: tuple[bool, ...], runner_warnings: tuple[str, ...] = ()) -> tuple[bool, tuple[str, ...]]:
    reasons = list(bundle_degradation_reasons)
    if any(lesion_flagged_for_review):
        reasons.append("one or more lesions were flagged for review")
    reasons.extend(runner_warnings)
    return bool(reasons), tuple(reasons)
