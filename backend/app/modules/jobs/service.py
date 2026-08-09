from __future__ import annotations

import io
import logging
import threading
import zipfile
from dataclasses import dataclass
from datetime import timezone, date, datetime
from pathlib import Path
from uuid import UUID, uuid4

from app.api.schemas.jobs import JobErrorPayload
from app.core.audit import log_audit_event
from app.core.config import get_settings
from app.modules.artifacts.storage import resolve_artifact_location
from app.modules.jobs.state_machine import transition_job
from app.modules.jobs.worker_tasks import (
    WorkerDispatchEnvelope,
    dispatch_ingestion_job,
    execute_ingestion_job,
    execute_nifti_segmentation_job,
    forget_worker_thread,
    register_worker_thread,
)
from app.infra.db.models import Artifact, Job, JobEvent, Study
from app.infra.db.session import create_session_factory

logger = logging.getLogger(__name__)

SUPPORTED_ARCHIVE_TYPES = {
    "application/zip",
    "application/x-zip-compressed",
    "application/octet-stream",
}

NIFTI_FILE_SUFFIXES = (".nii.gz", ".nii")


def _is_nifti_filename(filename: str) -> bool:
    lowered = filename.lower()
    return any(lowered.endswith(suffix) for suffix in NIFTI_FILE_SUFFIXES)


def _nifti_suffix(filename: str) -> str:
    lowered = filename.lower()
    for suffix in NIFTI_FILE_SUFFIXES:
        if lowered.endswith(suffix):
            return suffix
    return ".nii.gz"


class SubmissionValidationError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass(frozen=True)
class JobSubmissionResult:
    job_public_id: UUID
    study_public_id: UUID
    status: str
    stage: str
    submitted_at: datetime


@dataclass(frozen=True)
class JobStatusResult(JobSubmissionResult):
    error: JobErrorPayload | None = None


class JobService:
    def __init__(self) -> None:
        self._session_factory = create_session_factory()

    async def submit_mri_study(
        self,
        *,
        filename: str,
        content_type: str | None,
        archive_bytes: bytes,
        source_label: str | None,
    ) -> JobSubmissionResult:
        if not archive_bytes:
            raise SubmissionValidationError(400, "study_archive must not be empty")
        if content_type not in SUPPORTED_ARCHIVE_TYPES:
            raise SubmissionValidationError(415, "study_archive must be a zip upload")

        archive_id = uuid4()
        archive_name = f"{archive_id}.zip"
        archive_location = resolve_artifact_location("raw", f"studies/{archive_id}/{archive_name}")
        archive_location.absolute_path.parent.mkdir(parents=True, exist_ok=True)
        archive_location.absolute_path.write_bytes(archive_bytes)

        extracted_relative_path = f"studies/{archive_id}/extracted"
        extracted_location = resolve_artifact_location("raw", extracted_relative_path)
        extracted_location.absolute_path.mkdir(parents=True, exist_ok=True)

        series_files = self._extract_archive(archive_bytes, extracted_location.absolute_path)

        study_public_id = uuid4()
        submitted_at = datetime.now(timezone.utc)

        with self._session_factory() as session:
            study = Study(
                public_id=study_public_id,
                study_instance_uid=f"staged-{study_public_id}",
                source_kind="dicom-study",
                source_metadata={
                    "source_label": source_label,
                    "uploaded_filename": filename,
                    "archive_relative_path": archive_location.relative_path,
                    "extracted_relative_path": extracted_location.relative_path,
                    "series_file_count": series_files,
                },
                staging_status="staged",
            )
            session.add(study)
            session.flush()

            archive_artifact = Artifact(
                study_id=study.id,
                artifact_kind="raw-study-archive",
                storage_root="raw",
                relative_path=archive_location.relative_path,
                source_metadata={"filename": filename},
            )
            extracted_artifact = Artifact(
                study_id=study.id,
                artifact_kind="extracted-study-root",
                storage_root="raw",
                relative_path=extracted_location.relative_path,
                source_metadata={"file_count": series_files},
            )
            session.add_all([archive_artifact, extracted_artifact])

            job = Job(
                study_id=study.id,
                job_type="ingest-study",
                status="queued",
                stage="staged",
                created_at=submitted_at,
                updated_at=submitted_at,
            )
            session.add(job)
            session.flush()

            session.add(
                JobEvent(
                    job_id=job.id,
                    status="queued",
                    stage="staged",
                    event_type="transition",
                    payload={"reason": "job submitted"},
                    created_at=submitted_at,
                )
            )
            session.commit()

            log_audit_event(
                action="CREATE_STUDY",
                resource_id=str(study.public_id),
                details={"job_id": str(job.public_id), "study_type": "dicom"},
            )

            dispatch = self._dispatch_worker(
                job_id=str(job.public_id),
                study_id=str(study.public_id),
                extracted_relative_path=extracted_location.relative_path,
            )

        if not isinstance(dispatch, WorkerDispatchEnvelope):
            raise SubmissionValidationError(500, "worker dispatch failed")

        return JobSubmissionResult(
            job_public_id=job.public_id,
            study_public_id=study.public_id,
            status=job.status,
            stage=job.stage,
            submitted_at=submitted_at,
        )

    async def submit_nifti_study(
        self,
        *,
        scan_filename: str,
        scan_bytes: bytes,
        mask_filename: str | None,
        mask_bytes: bytes | None,
        source_label: str | None,
        acquired_at: date | None,
    ) -> JobSubmissionResult:
        if not scan_bytes:
            raise SubmissionValidationError(400, "scan_file must not be empty")
        if not _is_nifti_filename(scan_filename):
            raise SubmissionValidationError(
                415, "scan_file must be a .nii or .nii.gz NIfTI volume"
            )
        if mask_bytes is not None:
            if not mask_bytes:
                raise SubmissionValidationError(400, "mask_file must not be empty")
            if not _is_nifti_filename(mask_filename or ""):
                raise SubmissionValidationError(
                    415, "mask_file must be a .nii or .nii.gz NIfTI volume"
                )

        archive_id = uuid4()
        scan_relative_path = (
            f"studies/{archive_id}/scan{_nifti_suffix(scan_filename)}"
        )
        scan_location = resolve_artifact_location("raw", scan_relative_path)
        scan_location.absolute_path.parent.mkdir(parents=True, exist_ok=True)
        scan_location.absolute_path.write_bytes(scan_bytes)

        mask_relative_path: str | None = None
        if mask_bytes is not None:
            mask_relative_path = (
                f"studies/{archive_id}/mask{_nifti_suffix(mask_filename or '')}"
            )
            mask_location = resolve_artifact_location("raw", mask_relative_path)
            mask_location.absolute_path.parent.mkdir(parents=True, exist_ok=True)
            mask_location.absolute_path.write_bytes(mask_bytes)

        study_public_id = uuid4()
        submitted_at = datetime.now(timezone.utc)

        with self._session_factory() as session:
            study = Study(
                public_id=study_public_id,
                study_instance_uid=f"nifti-{study_public_id}",
                source_kind="nifti-upload",
                source_metadata={
                    "source_label": source_label,
                    "uploaded_filename": scan_filename,
                    "scan_relative_path": scan_location.relative_path,
                    "mask_relative_path": mask_relative_path,
                    "acquired_at": acquired_at.isoformat() if acquired_at else None,
                },
                staging_status="staged",
                acquired_at=acquired_at,
            )
            session.add(study)
            session.flush()

            scan_artifact = Artifact(
                study_id=study.id,
                artifact_kind="nifti-source",
                storage_root="raw",
                relative_path=scan_location.relative_path,
                source_metadata={"filename": scan_filename},
            )
            session.add(scan_artifact)

            if mask_relative_path is not None:
                mask_artifact = Artifact(
                    study_id=study.id,
                    artifact_kind="tumor-mask-input",
                    storage_root="raw",
                    relative_path=mask_relative_path,
                    source_metadata={"filename": mask_filename},
                )
                session.add(mask_artifact)

            job = Job(
                study_id=study.id,
                job_type="ingest-nifti",
                status="queued",
                stage="staged",
                created_at=submitted_at,
                updated_at=submitted_at,
            )
            session.add(job)
            session.flush()

            session.add(
                JobEvent(
                    job_id=job.id,
                    status="queued",
                    stage="staged",
                    event_type="transition",
                    payload={"reason": "nifti job submitted"},
                    created_at=submitted_at,
                )
            )
            session.commit()

            log_audit_event(
                action="CREATE_STUDY",
                resource_id=str(study.public_id),
                details={"job_id": str(job.public_id), "study_type": "nifti"},
            )

            self._dispatch_nifti_worker(
                job_id=str(job.public_id),
                study_id=str(study.public_id),
            )

        return JobSubmissionResult(
            job_public_id=job.public_id,
            study_public_id=study.public_id,
            status=job.status,
            stage=job.stage,
            submitted_at=submitted_at,
        )

    def get_job_status(self, job_public_id: str) -> JobStatusResult:
        try:
            parsed_job_id = UUID(job_public_id)
        except ValueError as exc:
            raise SubmissionValidationError(404, "job not found") from exc

        with self._session_factory() as session:
            job = session.query(Job).filter(Job.public_id == parsed_job_id).one_or_none()
            if job is None:
                raise SubmissionValidationError(404, "job not found")

            study = session.query(Study).filter(Study.id == job.study_id).one()
            error = None
            if job.failure_payload:
                error = JobErrorPayload(
                    code=job.failure_payload.get("code", "unknown"),
                    message=job.failure_payload.get("message", "Job failed"),
                    details=job.failure_payload,
                )

            return JobStatusResult(
                job_public_id=job.public_id,
                study_public_id=study.public_id,
                status=job.status,
                stage=job.stage,
                submitted_at=job.created_at,
                error=error,
            )

    def mark_job_failed(self, job_public_id: UUID, *, code: str, message: str) -> None:
        with self._session_factory() as session:
            job = session.query(Job).filter(Job.public_id == job_public_id).one()
            next_state = transition_job(
                job.status,
                "failed",
                stage="persisting",
                error=JobErrorPayload(code=code, message=message, details={"jobId": str(job.public_id)}),
            )
            job.status = next_state.status
            job.stage = next_state.stage
            job.failure_payload = next_state.error.model_dump(by_alias=True) if next_state.error else None
            session.add(
                JobEvent(
                    job_id=job.id,
                    status=job.status,
                    stage=job.stage,
                    event_type="failure",
                    payload=job.failure_payload or {},
                )
            )
            session.commit()

            log_audit_event(
                action="JOB_FAILED",
                resource_id=str(job.public_id),
                details={"code": code, "message": message},
            )

    def _dispatch_worker(self, *, job_id: str, study_id: str, extracted_relative_path: str) -> WorkerDispatchEnvelope:
        settings = get_settings()
        if settings.job_execution_mode == "threaded":
            logger.info(
                "Dispatching ingestion job on background thread",
                extra={"job_id": job_id, "study_id": study_id, "mode": settings.job_execution_mode},
            )
            thread = threading.Thread(
                target=execute_ingestion_job,
                kwargs={"job_id": job_id},
                daemon=False,
                name=f"oncoflow-job-{job_id[:8]}",
            )
            register_worker_thread(job_id, thread)
            thread.start()
            if not getattr(thread, "is_alive", lambda: False)():
                forget_worker_thread(job_id)
            return WorkerDispatchEnvelope(
                job_id=job_id,
                study_id=study_id,
                extracted_relative_path=extracted_relative_path,
            )

        logger.info(
            "Queued ingestion job for external worker dispatch",
            extra={"job_id": job_id, "study_id": study_id, "mode": settings.job_execution_mode},
        )
        return dispatch_ingestion_job(
            job_id=job_id,
            study_id=study_id,
            extracted_relative_path=extracted_relative_path,
        )

    def _dispatch_nifti_worker(self, *, job_id: str, study_id: str) -> None:
        settings = get_settings()
        if settings.job_execution_mode == "threaded":
            logger.info(
                "Dispatching NIfTI job on background thread",
                extra={"job_id": job_id, "study_id": study_id, "mode": settings.job_execution_mode},
            )
            thread = threading.Thread(
                target=execute_nifti_segmentation_job,
                kwargs={"job_id": job_id},
                daemon=False,
                name=f"oncoflow-nifti-{job_id[:8]}",
            )
            register_worker_thread(job_id, thread)
            thread.start()
            if not getattr(thread, "is_alive", lambda: False)():
                forget_worker_thread(job_id)
            return

        logger.info(
            "Queued NIfTI job for external worker dispatch (no-op in deferred mode)",
            extra={"job_id": job_id, "study_id": study_id, "mode": settings.job_execution_mode},
        )

    def _extract_archive(self, archive_bytes: bytes, destination: Path) -> int:
        try:
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
                members = [member for member in archive.infolist() if not member.is_dir()]
                if not members:
                    raise SubmissionValidationError(400, "study_archive must contain at least one file")
                for member in members:
                    member_path = Path(member.filename)
                    if member_path.is_absolute() or ".." in member_path.parts:
                        raise SubmissionValidationError(400, "study_archive contains an invalid file path")
                    target_path = destination / member_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, target_path.open("wb") as target:
                        target.write(source.read())
                return len(members)
        except zipfile.BadZipFile as exc:
            raise SubmissionValidationError(400, "study_archive must be a valid zip file") from exc
