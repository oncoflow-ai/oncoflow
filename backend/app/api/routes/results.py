from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_session, verify_study_access
from app.api.schemas.results import (
    ArtifactRefResponse,
    ComparisonMetricsResponse,
    ComparisonResponse,
    LesionMeasurementResponse,
    StoredCaseResultResponse,
    StoredLesionResultResponse,
    StudyListItemResponse,
)
from app.modules.artifacts.storage import resolve_artifact_location
from app.infra.db.models import Comparison, Study, User
from app.modules.results.service import (
    InvalidResultRequestError,
    ResultNotFoundError,
    get_case_result_payload,
)
from app.modules.results.studies_listing import list_studies

router = APIRouter(prefix="/results", tags=["results"])


@router.get("/studies", response_model=list[StudyListItemResponse])
def list_studies_endpoint(
    current_user: User = Depends(get_current_user),
) -> list[StudyListItemResponse]:
    items = list_studies(current_user=current_user)
    return [
        StudyListItemResponse(
            study_id=item.study_id,
            source_kind=item.source_kind,
            source_label=item.source_label,
            acquired_at=item.acquired_at,
            created_at=item.created_at,
            job_status=item.job_status,
            has_results=item.has_results,
        )
        for item in items
    ]


@router.get("/comparisons/{comparison_id}", response_model=ComparisonResponse)
def get_comparison(
    comparison_id: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> ComparisonResponse:
    try:
        parsed_comparison_id = UUID(comparison_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="comparison not found") from exc
    comparison = (
        session.query(Comparison)
        .filter(Comparison.public_id == parsed_comparison_id)
        .one_or_none()
    )
    if comparison is None:
        raise HTTPException(status_code=404, detail="comparison not found")
    baseline = session.query(Study).filter(Study.id == comparison.study_a_id).one_or_none()
    followup = session.query(Study).filter(Study.id == comparison.study_b_id).one_or_none()
    if baseline is None or followup is None:
        raise HTTPException(status_code=404, detail="comparison not found")
    verify_study_access(baseline, current_user, session)
    verify_study_access(followup, current_user, session)
    if baseline.patient_id is None or baseline.patient_id != followup.patient_id:
        raise HTTPException(status_code=422, detail="studies must belong to the same patient")

    location = resolve_artifact_location(
        "derived", f"comparisons/{comparison_id}/comparison.json"
    )
    if not Path(location.absolute_path).exists():
        raise HTTPException(status_code=404, detail="comparison not found")

    raw = json.loads(Path(location.absolute_path).read_text())
    metrics = raw.get("metrics", {})
    registration = raw.get("registration", {})
    notes_field = raw.get("notes", "")
    notes_list = (
        [n for n in str(notes_field).split("\n") if n.strip()] if notes_field else []
    )
    interpretation = raw.get("interpretation", {})
    metrics_payload = ComparisonMetricsResponse(
        volume_a_cm3=float(metrics.get("volume_a_cm3", 0.0) or 0.0),
        volume_b_cm3=float(metrics.get("volume_b_cm3", 0.0) or 0.0),
        delta_cm3=float(metrics.get("delta_cm3", 0.0) or 0.0),
        pct_change=float(metrics.get("pct_change", 0.0) or 0.0),
        dice_overlap=metrics.get("dice_overlap"),
        hd95_mm=metrics.get("hd95_mm"),
        recist_a_mm=metrics.get("recist_a_mm"),
        recist_b_mm=metrics.get("recist_b_mm"),
        recist_ratio=metrics.get("recist_ratio"),
        growth_rate_cm3_per_day=metrics.get("growth_rate_cm3_per_day"),
        registration_ncc=metrics.get("registration_ncc")
        or registration.get("ncc_after"),
        vol_delta_ci_half_cm3=metrics.get("vol_delta_ci_half_cm3"),
        method=registration.get("method"),
        backend=registration.get("backend"),
        did_resegment=metrics.get("did_resegment"),
    )
    return ComparisonResponse(
        comparison_id=comparison_id,
        baseline_study_id=str(baseline.public_id),
        followup_study_id=str(followup.public_id),
        baseline_acquired_at=baseline.acquired_at,
        followup_acquired_at=followup.acquired_at,
        metrics=metrics_payload,
        interpretation=(
            interpretation.get("label")
            if isinstance(interpretation, dict)
            else None
        ),
        notes=notes_list,
        output_relative_path=f"comparisons/{comparison_id}",
    )


@router.get("/{study_id}", response_model=StoredCaseResultResponse)
def get_case_results(
    study_id: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> StoredCaseResultResponse:
    try:
        parsed_study_id = UUID(study_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="study_id must be a valid UUID") from exc
    study = session.query(Study).filter(Study.public_id == parsed_study_id).one_or_none()
    if study is None:
        raise HTTPException(status_code=404, detail="study not found")
    verify_study_access(study, current_user, session)

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
        metadata=result.metadata,
    )
