from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from app.infra.db.models import Artifact, Series, Study
from app.modules.segmentation.contracts import (
    CanonicalSeriesBundle,
    CanonicalSeriesSlotAssignment,
    ManagedArtifactRef,
)

ALL_CANONICAL_SLOTS = ("t1_pre", "t1_post_or_fs", "t2_or_stir")


@dataclass(frozen=True)
class SeriesArtifactCandidate:
    series: Series
    artifact: Artifact
    slot_name: str
    priority: int


def _normalized_text(series: Series) -> str:
    description = series.series_description or ""
    protocol = series.protocol_name or ""
    return f"{description} {protocol}".lower()


def _infer_slot(series: Series) -> tuple[str | None, int]:
    text = _normalized_text(series)

    if "t2" in text or "stir" in text:
        score = 100
        if "stir" in text:
            score += 25
        return "t2_or_stir", score

    if "t1" not in text:
        return None, -1

    if any(term in text for term in ("+c", "post", "contrast", "fs")):
        score = 90
        if "+c" in text or "post" in text or "contrast" in text:
            score += 20
        if "fs" in text:
            score += 5
        return "t1_post_or_fs", score

    return "t1_pre", 80


def _geometry_signature(artifact: Artifact) -> tuple[tuple[float | None, ...], tuple[int | None, ...]]:
    geometry = artifact.source_metadata.get("geometry", {}) if artifact.source_metadata else {}
    spacing = tuple(geometry.get("spacing_mm", ()))
    shape = tuple(geometry.get("shape", ()))
    return spacing, shape


def _geometry_matches(candidates: list[SeriesArtifactCandidate]) -> bool:
    signatures = {_geometry_signature(candidate.artifact) for candidate in candidates}
    return len(signatures) <= 1


def build_canonical_series_bundle(*, session, study_public_id: UUID) -> CanonicalSeriesBundle:
    study = session.query(Study).filter(Study.public_id == study_public_id).one()
    rows = (
        session.query(Series, Artifact)
        .join(Artifact, Artifact.series_id == Series.id)
        .filter(
            Series.study_id == study.id,
            Series.classification == "processable",
            Artifact.artifact_kind == "nifti-volume",
        )
        .all()
    )

    candidates: list[SeriesArtifactCandidate] = []
    for series, artifact in rows:
        slot_name, priority = _infer_slot(series)
        if slot_name is None:
            continue
        candidates.append(
            SeriesArtifactCandidate(
                series=series,
                artifact=artifact,
                slot_name=slot_name,
                priority=priority,
            )
        )

    selected: list[SeriesArtifactCandidate] = []
    for slot_name in ALL_CANONICAL_SLOTS:
        slot_candidates = [candidate for candidate in candidates if candidate.slot_name == slot_name]
        if not slot_candidates:
            continue
        slot_candidates.sort(
            key=lambda candidate: (
                -candidate.priority,
                candidate.series.series_description or "",
                candidate.series.series_instance_uid,
            )
        )
        selected.append(slot_candidates[0])

    missing_slots = tuple(slot for slot in ALL_CANONICAL_SLOTS if slot not in {candidate.slot_name for candidate in selected})
    degradation_reasons: list[str] = []
    if missing_slots:
        degradation_reasons.append(
            "missing canonical slots: " + ", ".join(sorted(missing_slots))
        )
    if selected and not _geometry_matches(selected):
        degradation_reasons.append("selected canonical series do not share geometry")

    assignments = tuple(
        CanonicalSeriesSlotAssignment(
            slot_name=candidate.slot_name,  # type: ignore[arg-type]
            series_instance_uid=candidate.series.series_instance_uid,
            source_artifact=ManagedArtifactRef(
                storage_root=candidate.artifact.storage_root,  # type: ignore[arg-type]
                relative_path=candidate.artifact.relative_path,
            ),
        )
        for candidate in sorted(selected, key=lambda candidate: (ALL_CANONICAL_SLOTS.index(candidate.slot_name), candidate.series.series_instance_uid))
    )

    return CanonicalSeriesBundle(
        study_id=str(study.public_id),
        slot_assignments=assignments,
        missing_slots=missing_slots,  # type: ignore[arg-type]
        degradation_reasons=tuple(degradation_reasons),
    )
