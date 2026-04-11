from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

JobStatus = Literal["queued", "running", "failed", "completed"]
WorkerStage = Literal["profile", "validate", "convert", "persist"]
StagedStudyKind = Literal["dicom-study", "series-bundle"]


@dataclass(frozen=True)
class StagedStudyReference:
    study_id: UUID
    staging_uri: str
    storage_bucket: str
    storage_key: str
    source_kind: StagedStudyKind = "dicom-study"


@dataclass(frozen=True)
class ProcessingJobContract:
    job_id: UUID
    study: StagedStudyReference
    status: JobStatus
    stage: WorkerStage
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_code: str | None = None
    error_message: str | None = None

    def transition(self, *, status: JobStatus, stage: WorkerStage) -> "ProcessingJobContract":
        return ProcessingJobContract(
            job_id=self.job_id,
            study=self.study,
            status=status,
            stage=stage,
            submitted_at=self.submitted_at,
            error_code=self.error_code,
            error_message=self.error_message,
        )
