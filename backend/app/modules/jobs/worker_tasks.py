from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from app.api.schemas.jobs import JobErrorPayload
from app.infra.db.models import Artifact, Job, JobEvent, Study
from app.infra.db.session import create_session_factory
from app.modules.ingestion.pipeline import process_staged_study
from app.modules.jobs.state_machine import transition_job


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


def execute_ingestion_job(*, job_id: str) -> WorkerDispatchEnvelope:
    session_factory = create_session_factory()
    with session_factory() as session:
        job = session.query(Job).filter(Job.public_id == UUID(job_id)).one()
        study = session.query(Study).filter(Study.id == job.study_id).one()
        extracted_artifact = (
            session.query(Artifact)
            .filter(Artifact.study_id == study.id, Artifact.artifact_kind == "extracted-study-root")
            .one()
        )

        running_state = transition_job(job.status, "running", stage="profiling")
        job.status = running_state.status
        job.stage = running_state.stage
        session.add(
            JobEvent(
                job_id=job.id,
                status=job.status,
                stage=job.stage,
                event_type="transition",
                payload={"detail": "worker started"},
            )
        )
        session.flush()

        try:
            process_staged_study(
                session=session,
                study_public_id=study.public_id,
                extracted_relative_path=extracted_artifact.relative_path,
            )
            completed_state = transition_job(job.status, "completed", stage="completed")
            job.status = completed_state.status
            job.stage = completed_state.stage
            job.failure_payload = None
            session.add(
                JobEvent(
                    job_id=job.id,
                    status=job.status,
                    stage=job.stage,
                    event_type="transition",
                    payload={"detail": "ingestion completed"},
                )
            )
            session.commit()
        except Exception as exc:
            failed_state = transition_job(
                job.status,
                "failed",
                stage="converting",
                error=JobErrorPayload(
                    code="ingestion-failed",
                    message=str(exc),
                    details={"jobId": job_id, "studyId": str(study.public_id)},
                ),
            )
            job.status = failed_state.status
            job.stage = failed_state.stage
            job.failure_payload = failed_state.error.model_dump(by_alias=True) if failed_state.error else None
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
            raise

        return WorkerDispatchEnvelope(
            job_id=str(job.public_id),
            study_id=str(study.public_id),
            extracted_relative_path=extracted_artifact.relative_path,
        )
