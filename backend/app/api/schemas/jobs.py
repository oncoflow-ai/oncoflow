from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


def to_camel(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class JobErrorPayload(CamelModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class JobSubmissionResponse(CamelModel):
    job_id: str
    study_id: str
    status: str
    stage: str
    progress: int = 0
    stage_message: str | None = None
    submitted_at: datetime


class JobStatusResponse(JobSubmissionResponse):
    error: JobErrorPayload | None = None


class LongitudinalComparisonRequest(CamelModel):
    baseline_study_id: str
    followup_study_id: str
