from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from uuid import UUID

from app.api.schemas.jobs import JobErrorPayload
from app.infra.db.models import Artifact, Job, JobEvent, Study
from app.infra.db.session import create_session_factory
from app.core.config import get_settings
from app.modules.ingestion.pipeline import process_staged_study_with_stages
from app.modules.jobs.state_machine import transition_job
from app.modules.results.materialize import materialize_study_results
from app.modules.segmentation.pipeline import run_study_segmentation

logger = logging.getLogger(__name__)
_ACTIVE_WORKER_THREADS: dict[str, threading.Thread] = {}
_ACTIVE_WORKER_THREADS_LOCK = threading.Lock()


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


def register_worker_thread(job_id: str, thread: threading.Thread) -> None:
    with _ACTIVE_WORKER_THREADS_LOCK:
        _ACTIVE_WORKER_THREADS[job_id] = thread


def forget_worker_thread(job_id: str) -> None:
    with _ACTIVE_WORKER_THREADS_LOCK:
        _ACTIVE_WORKER_THREADS.pop(job_id, None)


def shutdown_background_workers(timeout_seconds: float = 30.0) -> None:
    with _ACTIVE_WORKER_THREADS_LOCK:
        active_threads = list(_ACTIVE_WORKER_THREADS.items())

    deadline = time.monotonic() + timeout_seconds
    for _job_id, thread in active_threads:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        thread.join(timeout=remaining)

    with _ACTIVE_WORKER_THREADS_LOCK:
        finished_job_ids = [job_id for job_id, thread in _ACTIVE_WORKER_THREADS.items() if not thread.is_alive()]
        for job_id in finished_job_ids:
            _ACTIVE_WORKER_THREADS.pop(job_id, None)
        if _ACTIVE_WORKER_THREADS:
            logger.warning(
                "Background worker shutdown timed out",
                extra={"active_worker_count": len(_ACTIVE_WORKER_THREADS)},
            )


def execute_ingestion_job(*, job_id: str) -> WorkerDispatchEnvelope:
    session_factory = create_session_factory()
    settings = get_settings()

    def log_stage(level: str, message: str, **extra) -> None:
        payload = {"job_id": job_id, **extra}
        if level == "debug" and settings.verbose_worker_logs:
            logger.info(message, extra=payload)
        elif level == "info":
            logger.info(message, extra=payload)
        elif level == "error":
            logger.error(message, extra=payload)

    with session_factory() as session:
        job = session.query(Job).filter(Job.public_id == UUID(job_id)).one()
        study = session.query(Study).filter(Study.id == job.study_id).one()
        extracted_artifact = (
            session.query(Artifact)
            .filter(Artifact.study_id == study.id, Artifact.artifact_kind == "extracted-study-root")
            .one()
        )

        current_stage = "profiling"
        log_stage("info", "Starting ingestion worker", study_id=str(study.public_id), stage=current_stage)
        running_state = transition_job(job.status, "running", stage=current_stage)
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

        def _update_stage(stage: str, detail: str) -> None:
            nonlocal current_stage
            current_stage = stage
            log_stage("debug", "Worker stage transition", study_id=str(study.public_id), stage=stage, detail=detail)
            stage_state = transition_job(job.status, job.status, stage=stage)
            job.status = stage_state.status
            job.stage = stage_state.stage
            session.add(
                JobEvent(
                    job_id=job.id,
                    status=job.status,
                    stage=job.stage,
                    event_type="transition",
                    payload={"detail": detail},
                )
            )
            session.flush()

        try:
            process_staged_study_with_stages(
                session=session,
                study_public_id=study.public_id,
                extracted_relative_path=extracted_artifact.relative_path,
                stage_callback=lambda stage: _update_stage(stage, f"ingestion {stage}"),
            )
            run_study_segmentation(
                session=session,
                study_public_id=study.public_id,
                stage_callback=lambda stage: _update_stage(stage, f"segmentation {stage}"),
            )
            _update_stage("materialize-results", "results materialize-results")
            materialize_study_results(
                session=session,
                study_public_id=study.public_id,
            )
            completed_state = transition_job(job.status, "completed", stage="completed")
            job.status = completed_state.status
            job.stage = completed_state.stage
            job.failure_payload = None
            log_stage("info", "Ingestion worker completed", study_id=str(study.public_id), stage="completed")
            session.add(
                JobEvent(
                    job_id=job.id,
                    status=job.status,
                    stage=job.stage,
                    event_type="transition",
                    payload={"detail": "analysis completed"},
                )
            )
            session.commit()
        except Exception as exc:
            log_stage("error", "Ingestion worker failed", study_id=str(study.public_id), stage=current_stage, error=str(exc))
            session.rollback()

            failure_session = session_factory()
            try:
                failed_job = failure_session.query(Job).filter(Job.public_id == UUID(job_id)).one()
                failed_study = failure_session.query(Study).filter(Study.id == failed_job.study_id).one()
                failed_state = transition_job(
                    failed_job.status,
                    "failed",
                    stage=current_stage,
                    error=JobErrorPayload(
                        code="ingestion-failed",
                        message=str(exc),
                        details={"jobId": job_id, "studyId": str(failed_study.public_id)},
                    ),
                )
                failed_job.status = failed_state.status
                failed_job.stage = failed_state.stage
                failed_job.failure_payload = failed_state.error.model_dump(by_alias=True) if failed_state.error else None
                failure_session.add(
                    JobEvent(
                        job_id=failed_job.id,
                        status=failed_job.status,
                        stage=failed_job.stage,
                        event_type="failure",
                        payload=failed_job.failure_payload or {},
                    )
                )
                failure_session.commit()
            finally:
                failure_session.close()

            raise
        finally:
            forget_worker_thread(job_id)

        return WorkerDispatchEnvelope(
            job_id=str(job.public_id),
            study_id=str(study.public_id),
            extracted_relative_path=extracted_artifact.relative_path,
        )
