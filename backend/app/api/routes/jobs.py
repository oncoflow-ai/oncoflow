from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.deps import get_current_user
from app.api.schemas.jobs import (
    JobStatusResponse,
    JobSubmissionResponse,
    LongitudinalComparisonRequest,
)
from app.api.schemas.results import ComparisonResponse
from app.infra.db.models import User
from app.modules.jobs.service import JobService, SubmissionValidationError
from app.modules.results.comparisons import (
    ComparisonError,
    run_longitudinal_comparison,
)

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post(
    "/mri-ingestion",
    response_model=JobSubmissionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def submit_mri_ingestion_job(
    study_archive: UploadFile = File(...),
    source_label: str | None = Form(default=None),
    patient_id: str | None = Form(default=None),
    current_user: User = Depends(get_current_user),
) -> JobSubmissionResponse:
    try:
        submission = await JobService().submit_mri_study(
            filename=study_archive.filename or "study.zip",
            content_type=study_archive.content_type,
            archive_bytes=await study_archive.read(),
            source_label=source_label,
            patient_id=patient_id,
            current_user=current_user,
        )
    except SubmissionValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    return JobSubmissionResponse(
        job_id=str(submission.job_public_id),
        study_id=str(submission.study_public_id),
        status=submission.status,
        stage=submission.stage,
        submitted_at=submission.submitted_at,
    )


@router.post(
    "/nifti-segmentation",
    response_model=JobSubmissionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def submit_nifti_segmentation_job(
    scan_file: UploadFile = File(...),
    mask_file: UploadFile | None = File(default=None),
    source_label: str | None = Form(default=None),
    acquired_at: str | None = Form(default=None),
    patient_id: str | None = Form(default=None),
    current_user: User = Depends(get_current_user),
) -> JobSubmissionResponse:
    parsed_date: date | None = None
    if acquired_at:
        try:
            parsed_date = date.fromisoformat(acquired_at)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail="acquired_at must be an ISO date (YYYY-MM-DD)",
            ) from exc

    mask_bytes: bytes | None = None
    mask_filename: str | None = None
    if mask_file is not None and mask_file.filename:
        mask_bytes = await mask_file.read()
        mask_filename = mask_file.filename

    try:
        submission = await JobService().submit_nifti_study(
            scan_filename=scan_file.filename or "scan.nii.gz",
            scan_bytes=await scan_file.read(),
            mask_filename=mask_filename,
            mask_bytes=mask_bytes,
            source_label=source_label,
            acquired_at=parsed_date,
            patient_id=patient_id,
            current_user=current_user,
        )
    except SubmissionValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    return JobSubmissionResponse(
        job_id=str(submission.job_public_id),
        study_id=str(submission.study_public_id),
        status=submission.status,
        stage=submission.stage,
        submitted_at=submission.submitted_at,
    )


@router.post(
    "/demo-mri-segmentation",
    response_model=JobSubmissionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def submit_demo_mri_segmentation_job(
    scan_file: UploadFile = File(...),
    source_label: str | None = Form(default=None),
    acquired_at: str | None = Form(default=None),
    patient_id: str | None = Form(default=None),
    current_user: User = Depends(get_current_user),
) -> JobSubmissionResponse:
    parsed_date: date | None = None
    if acquired_at:
        try:
            parsed_date = date.fromisoformat(acquired_at)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail="acquired_at must be an ISO date (YYYY-MM-DD)",
            ) from exc

    try:
        submission = await JobService().submit_demo_mri_segmentation(
            scan_filename=scan_file.filename or "demo-mri-upload.bin",
            scan_bytes=await scan_file.read(),
            content_type=scan_file.content_type,
            source_label=source_label,
            acquired_at=parsed_date,
            patient_id=patient_id,
            current_user=current_user,
        )
    except SubmissionValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    return JobSubmissionResponse(
        job_id=str(submission.job_public_id),
        study_id=str(submission.study_public_id),
        status=submission.status,
        stage=submission.stage,
        submitted_at=submission.submitted_at,
    )



@router.post(
    "/longitudinal-comparison",
    response_model=ComparisonResponse,
    status_code=status.HTTP_200_OK,
)
def submit_longitudinal_comparison(payload: LongitudinalComparisonRequest) -> ComparisonResponse:
    try:
        result = run_longitudinal_comparison(
            baseline_study_id=payload.baseline_study_id,
            followup_study_id=payload.followup_study_id,
        )
    except ComparisonError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    return ComparisonResponse.model_validate(result)


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    try:
        job_status = JobService().get_job_status(job_id)
    except SubmissionValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    return JobStatusResponse(
        job_id=str(job_status.job_public_id),
        study_id=str(job_status.study_public_id),
        status=job_status.status,
        stage=job_status.stage,
        submitted_at=job_status.submitted_at,
        error=job_status.error,
    )
