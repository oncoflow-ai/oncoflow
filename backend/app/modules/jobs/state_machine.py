from __future__ import annotations

from dataclasses import dataclass

from app.api.schemas.jobs import JobErrorPayload

ALLOWED_TRANSITIONS = {
    ("queued", "running"),
    ("queued", "failed"),
    ("running", "failed"),
    ("running", "completed"),
}


@dataclass(frozen=True)
class JobState:
    status: str
    stage: str
    error: JobErrorPayload | None = None


def transition_job(current_status: str, next_status: str, *, stage: str, error: JobErrorPayload | None = None) -> JobState:
    if current_status != next_status and (current_status, next_status) not in ALLOWED_TRANSITIONS:
        raise ValueError(f"Invalid job transition: {current_status} -> {next_status}")
    return JobState(status=next_status, stage=stage, error=error)
