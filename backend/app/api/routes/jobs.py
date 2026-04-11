from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.api.schemas.jobs import JobStatusResponse, JobSubmissionResponse
from app.modules.jobs.service import JobService, SubmissionValidationError

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post(
    "/mri-ingestion",
    response_model=JobSubmissionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def submit_mri_ingestion_job(
    study_archive: UploadFile = File(...),
    source_label: str | None = Form(default=None),
) -> JobSubmissionResponse:
    try:
        submission = await JobService().submit_mri_study(
            filename=study_archive.filename or "study.zip",
            content_type=study_archive.content_type,
            archive_bytes=await study_archive.read(),
            source_label=source_label,
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
