from __future__ import annotations

import json
from shutil import copy2
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from uuid import UUID, uuid4

from app.infra.db.models import Artifact, Study
from app.modules.artifacts.catalog import record_study_artifact
from app.modules.artifacts.storage import resolve_artifact_location
from app.modules.segmentation.contracts import CaseSegmentationResult
from app.modules.segmentation.input_bundle import build_canonical_series_bundle
from app.modules.segmentation.packaging import package_predictions
from app.modules.segmentation.review import determine_case_review, make_review_artifact_ref
from app.modules.segmentation.runner import run_segmentation


@dataclass(frozen=True)
class SegmentationPipelineResult:
    case_result: CaseSegmentationResult
    persisted_artifact_count: int


def _write_placeholder(relative_path: str, *, payload: bytes | str = b"") -> None:
    location = resolve_artifact_location("derived", relative_path)
    location.absolute_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        location.absolute_path.write_text(payload)
    else:
        location.absolute_path.write_bytes(payload)


def _persist_mask_artifact(relative_path: str, *, source_mask_path: str | None) -> None:
    location = resolve_artifact_location("derived", relative_path)
    location.absolute_path.parent.mkdir(parents=True, exist_ok=True)
    if source_mask_path and Path(source_mask_path).exists():
        source = Path(source_mask_path).resolve()
        if source != location.absolute_path.resolve():
            copy2(source, location.absolute_path)
        return
    _write_placeholder(relative_path, payload=b"mask")


def run_study_segmentation(
    *,
    session,
    study_public_id: UUID,
    model_id: str = "nnunet-v2-resenc",
    stage_callback: Callable[[str], None] | None = None,
) -> SegmentationPipelineResult:
    stage_callback = stage_callback or (lambda _: None)
    study = session.query(Study).filter(Study.public_id == study_public_id).one()
    segmentation_run_id = str(uuid4())

    stage_callback("prepare-inputs")
    bundle = build_canonical_series_bundle(session=session, study_public_id=study_public_id)

    stage_callback("infer")
    runner_result = run_segmentation(bundle=bundle, model_id=model_id)
    effective_bundle = runner_result.bundle

    review_artifacts_by_index: dict[int, tuple] = {}
    for index, prediction in enumerate(sorted(runner_result.lesions, key=lambda item: item.bounding_box.z_min)):
        if prediction.confidence_score < 0.5 or prediction.warning_reasons:
            descriptor = make_review_artifact_ref(
                artifact_kind="overlay",
                relative_path=f"studies/{study.public_id}/review/lesion-{index + 1:03d}-overlay.png",
                provenance_label="segmentation-qc-overlay",
                metadata={"generator": "phase-02-review"},
            )
            _write_placeholder(descriptor.artifact.artifact_ref.relative_path, payload=b"overlay")
            review_artifacts_by_index[index] = (descriptor.artifact,)

    stage_callback("postprocess")
    lesions = package_predictions(
        study_id=str(study.public_id),
        predictions=runner_result.lesions,
        review_artifacts_by_index=review_artifacts_by_index,
    )
    needs_review, case_qc_reasons = determine_case_review(
        bundle_degradation_reasons=effective_bundle.degradation_reasons,
        lesion_flagged_for_review=tuple(lesion.qc.flagged_for_review for lesion in lesions),
        runner_warnings=runner_result.warnings,
    )
    case_result = CaseSegmentationResult(
        study_id=str(study.public_id),
        input_bundle=effective_bundle,
        runner=runner_result.runner,
        lesions=lesions,
        needs_review=needs_review,
        case_qc_reasons=case_qc_reasons,
    )

    stage_callback("package-results")
    persisted_count = 0
    source_artifacts = (
        session.query(Artifact)
        .filter(
            Artifact.study_id == study.id,
            Artifact.relative_path.in_(
                [assignment.source_artifact.relative_path for assignment in effective_bundle.slot_assignments]
            ),
        )
        .all()
    )
    geometry_by_relative_path = {
        artifact.relative_path: dict(artifact.source_metadata.get("geometry", {}))
        for artifact in source_artifacts
    }
    slot_provenance = [
        {
            "slot_name": assignment.slot_name,
            "series_instance_uid": assignment.series_instance_uid,
            "relative_path": assignment.source_artifact.relative_path,
            "geometry": geometry_by_relative_path.get(assignment.source_artifact.relative_path, {}),
        }
        for assignment in effective_bundle.slot_assignments
    ]

    predictions_by_lesion_id = {
        lesion.lesion_id: prediction
        for lesion, prediction in zip(lesions, sorted(runner_result.lesions, key=lambda item: item.bounding_box.z_min), strict=True)
    }

    primary_geometry = next(
        (
            provenance["geometry"]
            for provenance in slot_provenance
            if provenance.get("geometry")
        ),
        {},
    )

    for lesion in lesions:
        relative_path = lesion.mask_artifact.relative_path
        prediction = predictions_by_lesion_id[lesion.lesion_id]
        _persist_mask_artifact(
            relative_path,
            source_mask_path=prediction.source_mask_path,
        )
        record_study_artifact(
            session,
            study_id=study.id,
            artifact_kind="segmentation-mask",
            relative_path=relative_path,
            metadata={
                "segmentation_run_id": segmentation_run_id,
                "lesion_id": lesion.lesion_id,
                "bounding_box": lesion.bounding_box.__dict__,
                "slot_provenance": slot_provenance,
                "runner": {"model_id": case_result.runner.model_id, "runner_version": case_result.runner.runner_version},
                "needs_review": lesion.qc.flagged_for_review,
                "geometry": primary_geometry,
                "occupied_voxels_ijk": [list(voxel) for voxel in prediction.occupied_voxels_ijk],
                "inference": prediction.metadata or {},
            },
        )
        persisted_count += 1

        for review_artifact in lesion.review_artifacts:
            record_study_artifact(
                session,
                study_id=study.id,
                artifact_kind=f"review-{review_artifact.artifact_kind}",
                relative_path=review_artifact.artifact_ref.relative_path,
                metadata={
                    "segmentation_run_id": segmentation_run_id,
                    "lesion_id": lesion.lesion_id,
                    "provenance_label": review_artifact.provenance_label,
                },
            )
            persisted_count += 1

    case_relative_path = f"studies/{study.public_id}/segmentation/result.json"
    _write_placeholder(
        case_relative_path,
        payload=json.dumps(
            {
                "study_id": case_result.study_id,
                "needs_review": case_result.needs_review,
                "case_qc_reasons": list(case_result.case_qc_reasons),
                "lesion_ids": [lesion.lesion_id for lesion in case_result.lesions],
                "runner": {
                    "model_id": case_result.runner.model_id,
                    "runner_version": case_result.runner.runner_version,
                    "execution_backend": case_result.runner.execution_backend,
                    "warnings": list(case_result.runner.warnings),
                },
            }
        ),
    )
    record_study_artifact(
        session,
        study_id=study.id,
        artifact_kind="segmentation-case-result",
        relative_path=case_relative_path,
        metadata={
            "segmentation_run_id": segmentation_run_id,
            "needs_review": case_result.needs_review,
            "case_qc_reasons": list(case_result.case_qc_reasons),
            "lesion_count": len(case_result.lesions),
            "runner": {
                "model_id": case_result.runner.model_id,
                "runner_version": case_result.runner.runner_version,
                "execution_backend": case_result.runner.execution_backend,
                "warnings": list(case_result.runner.warnings),
            },
        },
    )
    persisted_count += 1

    session.flush()
    return SegmentationPipelineResult(case_result=case_result, persisted_artifact_count=persisted_count)
