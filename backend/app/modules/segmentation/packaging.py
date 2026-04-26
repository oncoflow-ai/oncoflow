from __future__ import annotations

from app.modules.segmentation.contracts import LesionQc, LesionResult, ReviewArtifactRef, build_lesion_id
from app.modules.segmentation.runner import NormalizedLesionPrediction


def _bbox_sort_key(prediction: NormalizedLesionPrediction) -> tuple[int, int, int, int, int, int]:
    bbox = prediction.bounding_box
    return (bbox.z_min, bbox.y_min, bbox.x_min, bbox.z_max, bbox.y_max, bbox.x_max)


def _confidence_bucket(confidence_score: float) -> str:
    if confidence_score >= 0.8:
        return "high"
    if confidence_score >= 0.5:
        return "medium"
    return "low"


def package_predictions(
    *,
    study_id: str,
    predictions: tuple[NormalizedLesionPrediction, ...],
    review_artifacts_by_index: dict[int, tuple[ReviewArtifactRef, ...]] | None = None,
) -> tuple[LesionResult, ...]:
    review_artifacts_by_index = review_artifacts_by_index or {}
    packaged: list[LesionResult] = []
    ordered = sorted(predictions, key=_bbox_sort_key)

    for index, prediction in enumerate(ordered):
        reasons = tuple(prediction.warning_reasons)
        flagged = prediction.confidence_score < 0.5 or bool(reasons)
        if prediction.confidence_score < 0.5 and not reasons:
            reasons = ("low confidence score",)
        review_artifacts = review_artifacts_by_index.get(index, ())

        lesion = LesionResult(
            lesion_id=build_lesion_id(study_id=study_id, lesion_index=index),
            mask_artifact=prediction.mask_artifact,
            bounding_box=prediction.bounding_box,
            qc=LesionQc(
                confidence_bucket=_confidence_bucket(prediction.confidence_score),  # type: ignore[arg-type]
                flagged_for_review=flagged,
                reasons=reasons,
            ),
            review_artifacts=review_artifacts,
        )
        packaged.append(lesion)

    return tuple(packaged)
