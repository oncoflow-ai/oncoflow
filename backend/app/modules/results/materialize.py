from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from app.infra.db.models import Artifact, StoredLesionResult as StoredLesionResultModel, Study, StudyResult
from app.modules.artifacts.catalog import record_study_artifact
from app.modules.results.measurements import compute_longest_diameter_mm, compute_volume_mm3


@dataclass(frozen=True)
class MaterializationResult:
    study_result_id: int
    lesion_count: int


def _mask_from_voxels(occupied_voxels_ijk: list[list[int]] | tuple[tuple[int, int, int], ...]) -> list[list[list[int]]]:
    voxels = [tuple(int(axis) for axis in voxel) for voxel in occupied_voxels_ijk]
    if not voxels:
        raise ValueError("segmentation-mask artifacts must include occupied_voxels_ijk")

    x_max = max(voxel[0] for voxel in voxels)
    y_max = max(voxel[1] for voxel in voxels)
    z_max = max(voxel[2] for voxel in voxels)
    mask = [
        [[0 for _x in range(x_max + 1)] for _y in range(y_max + 1)]
        for _z in range(z_max + 1)
    ]
    for x, y, z in voxels:
        mask[z][y][x] = 1
    return mask


def materialize_study_results(*, session, study_public_id: UUID) -> MaterializationResult:
    study = session.query(Study).filter(Study.public_id == study_public_id).one()
    case_artifact = (
        session.query(Artifact)
        .filter(Artifact.study_id == study.id, Artifact.artifact_kind == "segmentation-case-result")
        .order_by(Artifact.id.desc())
        .first()
    )
    if case_artifact is None:
        raise ValueError("segmentation case result artifact not found")
    segmentation_run_id = case_artifact.source_metadata.get("segmentation_run_id")
    mask_artifacts = (
        session.query(Artifact)
        .filter(Artifact.study_id == study.id, Artifact.artifact_kind == "segmentation-mask")
        .order_by(Artifact.id.asc())
        .all()
    )
    review_artifacts = (
        session.query(Artifact)
        .filter(Artifact.study_id == study.id, Artifact.artifact_kind.like("review-%"))
        .all()
    )
    if segmentation_run_id is not None:
        mask_artifacts = [
            artifact
            for artifact in mask_artifacts
            if artifact.source_metadata.get("segmentation_run_id") == segmentation_run_id
        ]
        review_artifacts = [
            artifact
            for artifact in review_artifacts
            if artifact.source_metadata.get("segmentation_run_id") == segmentation_run_id
        ]

    study_result = StudyResult(
        study_id=study.id,
        result_kind="single-scan",
        needs_review=bool(case_artifact.source_metadata.get("needs_review", False)),
        summary_metadata={
            "segmentation_run_id": segmentation_run_id,
            "case_qc_reasons": list(case_artifact.source_metadata.get("case_qc_reasons", [])),
            "lesion_count": int(case_artifact.source_metadata.get("lesion_count", len(mask_artifacts))),
        },
    )
    session.add(study_result)
    session.flush()

    for mask_artifact in mask_artifacts:
        bbox = dict(mask_artifact.source_metadata.get("bounding_box", {}))
        geometry = mask_artifact.source_metadata.get("geometry", {}) if mask_artifact.source_metadata else {}
        spacing = tuple(geometry.get("spacing_mm", (1.0, 1.0, 1.0))) or (1.0, 1.0, 1.0)
        mask = _mask_from_voxels(mask_artifact.source_metadata.get("occupied_voxels_ijk", []))
        measurement_payload = {
            "volume_mm3": compute_volume_mm3(mask, spacing),
            "longest_diameter_mm": compute_longest_diameter_mm(mask, spacing),
        }
        linked_reviews = [
            {
                "artifact_kind": artifact.artifact_kind,
                "storage_root": artifact.storage_root,
                "relative_path": artifact.relative_path,
            }
            for artifact in review_artifacts
            if artifact.source_metadata.get("lesion_id") == mask_artifact.source_metadata.get("lesion_id")
        ]

        lesion_row = StoredLesionResultModel(
            study_result_id=study_result.id,
            study_id=study.id,
            lesion_id=mask_artifact.source_metadata["lesion_id"],
            measurement_payload=measurement_payload,
            bounding_box=bbox,
            artifact_refs={
                "mask": {
                    "artifact_kind": mask_artifact.artifact_kind,
                    "storage_root": mask_artifact.storage_root,
                    "relative_path": mask_artifact.relative_path,
                },
                "review": linked_reviews,
            },
            result_metadata={
                "slot_provenance": mask_artifact.source_metadata.get("slot_provenance", []),
                "runner": mask_artifact.source_metadata.get("runner", {}),
            },
        )
        session.add(lesion_row)

    bundle_relative_path = f"studies/{study.public_id}/results/study-result.json"
    record_study_artifact(
        session,
        study_id=study.id,
        artifact_kind="study-result-bundle",
        relative_path=bundle_relative_path,
        metadata={
            "segmentation_run_id": segmentation_run_id,
            "study_result_id": study_result.id,
            "lesion_ids": [artifact.source_metadata["lesion_id"] for artifact in mask_artifacts],
        },
    )
    session.flush()
    return MaterializationResult(study_result_id=study_result.id, lesion_count=len(mask_artifacts))
