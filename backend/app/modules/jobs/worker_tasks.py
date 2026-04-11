from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerDispatchEnvelope:
    job_id: str
    study_id: str
    extracted_relative_path: str


def dispatch_ingestion_job(*, job_id: str, study_id: str, extracted_relative_path: str) -> WorkerDispatchEnvelope:
    return WorkerDispatchEnvelope(
        job_id=job_id,
        study_id=study_id,
        extracted_relative_path=extracted_relative_path,
    )
