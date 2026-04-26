from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas.results import (
    ArtifactRefResponse,
    LesionMeasurementResponse,
    StoredCaseResultResponse,
    StoredLesionResultResponse,
)
from app.modules.results.service import InvalidResultRequestError, ResultNotFoundError, get_case_result_payload

router = APIRouter(prefix="/results", tags=["results"])


@router.get("/{study_id}", response_model=StoredCaseResultResponse)
def get_case_results(study_id: str) -> StoredCaseResultResponse:
    try:
        result = get_case_result_payload(study_id=study_id)
    except InvalidResultRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ResultNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return StoredCaseResultResponse(
        study_id=result.study_id,
        result_artifact=ArtifactRefResponse(**result.result_artifact.__dict__),
        lesions=[
            StoredLesionResultResponse(
                lesion_id=lesion.lesion_id,
                bounding_box=lesion.bounding_box,
                measurements=LesionMeasurementResponse(**lesion.measurements.__dict__),
                mask_artifact=ArtifactRefResponse(**lesion.mask_artifact.__dict__),
                review_artifacts=[ArtifactRefResponse(**artifact.__dict__) for artifact in lesion.review_artifacts],
                metadata=lesion.metadata,
            )
            for lesion in result.lesions
        ],
        needs_review=result.needs_review,
        case_qc_reasons=list(result.case_qc_reasons),
    )
