from __future__ import annotations

import pytest

from app.modules.segmentation.contracts import (
    BoundingBox3D,
    CanonicalSeriesBundle,
    CanonicalSeriesSlotAssignment,
    CaseSegmentationResult,
    LesionQc,
    LesionResult,
    ManagedArtifactRef,
    ReviewArtifactRef,
    RunnerProvenance,
    build_lesion_id,
)


def _bundle() -> CanonicalSeriesBundle:
    return CanonicalSeriesBundle(
        study_id="study-001",
        slot_assignments=(
            CanonicalSeriesSlotAssignment(
                slot_name="t1_pre",
                series_instance_uid="series-001",
                source_artifact=ManagedArtifactRef(
                    storage_root="derived",
                    relative_path="studies/study-001/series/1/volume.nii.gz",
                ),
            ),
            CanonicalSeriesSlotAssignment(
                slot_name="t2_or_stir",
                series_instance_uid="series-002",
                source_artifact=ManagedArtifactRef(
                    storage_root="derived",
                    relative_path="studies/study-001/series/2/volume.nii.gz",
                ),
            ),
        ),
        missing_slots=("t1_post_or_fs",),
        degradation_reasons=("missing t1 post contrast slot",),
    )


def _runner(model_id: str = "nnunet-v2-resenc") -> RunnerProvenance:
    return RunnerProvenance(
        model_id=model_id,
        runner_version="2026.04",
        execution_backend="stub",
    )


def _lesion(*, lesion_id: str = "study-001:lesion-001", review: bool = False) -> LesionResult:
    review_artifacts = ()
    reasons = ()
    if review:
        reasons = ("low confidence logits",)
        review_artifacts = (
            ReviewArtifactRef(
                artifact_kind="overlay",
                artifact_ref=ManagedArtifactRef(
                    storage_root="derived",
                    relative_path="studies/study-001/review/lesion-001-overlay.png",
                ),
                provenance_label="qc-overlay",
            ),
        )

    return LesionResult(
        lesion_id=lesion_id,
        mask_artifact=ManagedArtifactRef(
            storage_root="derived",
            relative_path="studies/study-001/lesions/lesion-001-mask.nii.gz",
        ),
        bounding_box=BoundingBox3D(x_min=1, x_max=4, y_min=2, y_max=6, z_min=0, z_max=3),
        qc=LesionQc(
            confidence_bucket="low" if review else "high",
            flagged_for_review=review,
            reasons=reasons,
        ),
        review_artifacts=review_artifacts,
    )


def test_case_segmentation_result_requires_required_review_fields() -> None:
    result = CaseSegmentationResult(
        study_id="study-001",
        input_bundle=_bundle(),
        runner=_runner(),
        lesions=(_lesion(review=True),),
        needs_review=True,
    )

    assert result.needs_review is True
    assert result.runner.model_id == "nnunet-v2-resenc"
    assert result.input_bundle.slot_assignments[0].slot_name == "t1_pre"

    with pytest.raises(ValueError, match="needs_review cases must include"):
        CaseSegmentationResult(
            study_id="study-001",
            input_bundle=_bundle(),
            runner=_runner(),
            lesions=(),
            needs_review=True,
        )


def test_lesion_result_rejects_missing_bbox_mask_or_qc() -> None:
    with pytest.raises(ValueError, match="relative_path is required"):
        ManagedArtifactRef(storage_root="derived", relative_path="")

    with pytest.raises(ValueError, match="bounding box minima"):
        BoundingBox3D(x_min=4, x_max=1, y_min=0, y_max=2, z_min=0, z_max=1)

    lesion = LesionResult(
        lesion_id="study-001:lesion-001",
        mask_artifact=ManagedArtifactRef(
            storage_root="derived",
            relative_path="studies/study-001/lesions/lesion-001-mask.nii.gz",
        ),
        bounding_box=BoundingBox3D(x_min=1, x_max=4, y_min=2, y_max=6, z_min=0, z_max=3),
        qc=LesionQc(
            confidence_bucket="low",
            flagged_for_review=True,
            reasons=("uncertain boundary",),
        ),
    )

    assert lesion.qc.flagged_for_review is True


def test_runner_provenance_rejects_unknown_model_ids() -> None:
    with pytest.raises(ValueError, match="Unknown benchmark model id"):
        _runner(model_id="unknown-model")


def test_stable_lesion_ids_are_deterministic_for_same_ordered_inputs() -> None:
    first = (
        build_lesion_id(study_id="study-001", lesion_index=0),
        build_lesion_id(study_id="study-001", lesion_index=1),
    )
    second = (
        build_lesion_id(study_id="study-001", lesion_index=0),
        build_lesion_id(study_id="study-001", lesion_index=1),
    )

    assert first == second
    assert first[0] == "study-001:lesion-001"
    assert first[1] == "study-001:lesion-002"


def test_flagged_lesion_can_exist_before_review_artifacts_are_generated() -> None:
    lesion = LesionResult(
        lesion_id="study-001:lesion-003",
        mask_artifact=ManagedArtifactRef(
            storage_root="derived",
            relative_path="studies/study-001/lesions/lesion-003-mask.nii.gz",
        ),
        bounding_box=BoundingBox3D(x_min=0, x_max=2, y_min=0, y_max=2, z_min=0, z_max=2),
        qc=LesionQc(
            confidence_bucket="low",
            flagged_for_review=True,
            reasons=("low confidence score",),
        ),
        review_artifacts=(),
    )

    assert lesion.review_artifacts == ()
