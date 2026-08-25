from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from pathlib import Path
from uuid import UUID

from app.api.schemas.jobs import JobErrorPayload
from app.core.audit import log_audit_event
from app.infra.db.models import Artifact, Job, JobEvent, Study
from app.infra.db.session import create_session_factory
from app.core.config import get_settings
from app.modules.artifacts.storage import resolve_artifact_location
from app.modules.ingestion.pipeline import process_staged_study_with_stages
from app.modules.jobs.state_machine import transition_job
from app.modules.results.materialize import materialize_study_results
from app.modules.segmentation.nifti_pipeline import materialize_nifti_study_with_mask
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

        STAGE_PROGRESS_MAP: dict[str, tuple[int, str]] = {
            "profiling": (15, "Ingesting and profiling DICOM series metadata..."),
            "prepare-inputs": (30, "Preparing canonical volumetric input slices..."),
            "bone-extraction": (45, "Extracting bone anatomy and reference boundaries..."),
            "infer": (65, "Executing AI tumor segmentation models..."),
            "postprocess": (80, "Calculating volumetric measurements and lesion metrics..."),
            "package-results": (90, "Packaging segmentation masks and review artifacts..."),
            "materialize-results": (95, "Generating structured AI clinical report..."),
            "completed": (100, "Analysis completed successfully."),
        }

        current_stage = "profiling"
        log_stage("info", "Starting ingestion worker", study_id=str(study.public_id), stage=current_stage)
        running_state = transition_job(
            job.status,
            "running",
            stage=current_stage,
            progress=15,
            stage_message="Ingesting and profiling DICOM series metadata...",
        )
        job.status = running_state.status
        job.stage = running_state.stage
        job.progress = running_state.progress
        job.stage_message = running_state.stage_message
        session.add(
            JobEvent(
                job_id=job.id,
                status=job.status,
                stage=job.stage,
                event_type="transition",
                payload={"detail": "worker started", "progress": 15, "stage_message": job.stage_message},
            )
        )
        session.flush()

        def _update_stage(stage: str, detail: str) -> None:
            nonlocal current_stage
            current_stage = stage
            prog, msg = STAGE_PROGRESS_MAP.get(stage, (50, detail))
            log_stage("debug", "Worker stage transition", study_id=str(study.public_id), stage=stage, detail=detail)
            stage_state = transition_job(
                job.status,
                job.status,
                stage=stage,
                progress=prog,
                stage_message=msg,
            )
            job.status = stage_state.status
            job.stage = stage_state.stage
            job.progress = stage_state.progress
            job.stage_message = stage_state.stage_message
            session.add(
                JobEvent(
                    job_id=job.id,
                    status=job.status,
                    stage=job.stage,
                    event_type="transition",
                    payload={"detail": detail, "progress": prog, "stage_message": msg},
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
            completed_state = transition_job(
                job.status,
                "completed",
                stage="completed",
                progress=100,
                stage_message="Analysis completed successfully.",
            )
            job.status = completed_state.status
            job.stage = completed_state.stage
            job.progress = completed_state.progress
            job.stage_message = completed_state.stage_message
            job.failure_payload = None
            log_stage("info", "Ingestion worker completed", study_id=str(study.public_id), stage="completed")
            
            log_audit_event(
                action="JOB_COMPLETED",
                resource_id=job_id,
                details={"study_id": str(study.public_id), "type": "ingestion"}
            )
            
            session.add(
                JobEvent(
                    job_id=job.id,
                    status=job.status,
                    stage=job.stage,
                    event_type="transition",
                    payload={"detail": "analysis completed", "progress": 100, "stage_message": "Analysis completed successfully."},
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


def execute_nifti_segmentation_job(*, job_id: str) -> WorkerDispatchEnvelope:
    """Worker for the NIfTI-direct demo upload.

    Treats the optional uploaded mask as the segmentation result and
    materializes a StudyResult so the existing /results endpoint serves
    volume + diameter + bbox without any DICOM conversion.
    """

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
        mask_artifact = (
            session.query(Artifact)
            .filter(
                Artifact.study_id == study.id,
                Artifact.artifact_kind == "tumor-mask-input",
            )
            .order_by(Artifact.id.desc())
            .first()
        )

        current_stage = "data-fetching"
        running_state = transition_job(
            job.status,
            "running",
            stage=current_stage,
            progress=15,
            stage_message="Fetching and verifying NIfTI scan and mask inputs...",
        )
        job.status = running_state.status
        job.stage = running_state.stage
        job.progress = running_state.progress
        job.stage_message = running_state.stage_message
        session.add(
            JobEvent(
                job_id=job.id,
                status=job.status,
                stage=job.stage,
                event_type="transition",
                payload={"detail": "nifti worker started", "progress": 15, "stage_message": job.stage_message},
            )
        )
        session.flush()

        def _update_stage(stage: str, progress: int, stage_message: str) -> None:
            nonlocal current_stage
            current_stage = stage
            log_stage("debug", "Worker stage transition", study_id=str(study.public_id), stage=stage, detail=stage_message)
            stage_state = transition_job(
                job.status,
                job.status,
                stage=stage,
                progress=progress,
                stage_message=stage_message,
            )
            job.status = stage_state.status
            job.stage = stage_state.stage
            job.progress = stage_state.progress
            job.stage_message = stage_state.stage_message
            session.add(
                JobEvent(
                    job_id=job.id,
                    status=job.status,
                    stage=job.stage,
                    event_type="transition",
                    payload={"detail": stage_message, "progress": progress, "stage_message": stage_message},
                )
            )
            session.flush()

        try:
            if mask_artifact is None:
                raise RuntimeError(
                    "NIfTI demo job requires an uploaded tumor-mask-input artifact"
                )

            mask_location = resolve_artifact_location(
                mask_artifact.storage_root,  # type: ignore[arg-type]
                mask_artifact.relative_path,
            )
            
            _update_stage("bone-extraction", 35, "Extracting bone structures and anatomical landmarks...")
            _update_stage("segmentation", 65, "Processing tumor segmentation and volume alignment...")
            _update_stage("quantification", 80, "Calculating tumor volume, max diameter, and bounding box...")
            _update_stage("report-generation", 95, "Materializing results and generating clinical report...")

            log_stage(
                "info",
                "Materializing NIfTI demo result",
                study_id=str(study.public_id),
                mask_path=mask_location.relative_path,
            )
            materialize_nifti_study_with_mask(
                session=session,
                study_public_id=study.public_id,
                mask_source_absolute_path=Path(mask_location.absolute_path),
            )
            completed_state = transition_job(
                job.status,
                "completed",
                stage="completed",
                progress=100,
                stage_message="Analysis completed successfully.",
            )
            job.status = completed_state.status
            job.stage = completed_state.stage
            job.progress = completed_state.progress
            job.stage_message = completed_state.stage_message
            job.failure_payload = None
            
            log_audit_event(
                action="JOB_COMPLETED",
                resource_id=job_id,
                details={"study_id": str(study.public_id), "type": "nifti_segmentation"}
            )
            
            session.add(
                JobEvent(
                    job_id=job.id,
                    status=job.status,
                    stage=job.stage,
                    event_type="transition",
                    payload={"detail": "nifti analysis completed", "progress": 100, "stage_message": "Analysis completed successfully."},
                )
            )
            session.commit()
            log_stage("info", "NIfTI worker completed", study_id=str(study.public_id))
        except Exception as exc:
            log_stage(
                "error",
                "NIfTI worker failed",
                study_id=str(study.public_id),
                error=str(exc),
            )
            session.rollback()

            failure_session = session_factory()
            try:
                failed_job = (
                    failure_session.query(Job).filter(Job.public_id == UUID(job_id)).one()
                )
                failed_study = (
                    failure_session.query(Study)
                    .filter(Study.id == failed_job.study_id)
                    .one()
                )
                failed_state = transition_job(
                    failed_job.status,
                    "failed",
                    stage="materialize-results",
                    error=JobErrorPayload(
                        code="nifti-segmentation-failed",
                        message=str(exc),
                        details={
                            "jobId": job_id,
                            "studyId": str(failed_study.public_id),
                        },
                    ),
                )
                failed_job.status = failed_state.status
                failed_job.stage = failed_state.stage
                failed_job.failure_payload = (
                    failed_state.error.model_dump(by_alias=True)
                    if failed_state.error
                    else None
                )
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
            extracted_relative_path="",
        )


def _default_demo_mask_path(settings) -> Path:
    if settings.demo_ground_truth_mask_path:
        return Path(settings.demo_ground_truth_mask_path).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[4]
    return (
        repo_root
        / "data"
        / "P01"
        / "tumor segmentation"
        / "P01_tumor_mask_baseline.nii.gz"
    )


def _demo_result_metadata() -> dict[str, object]:
    return {
        "case_qc_reasons": [],
        "lesion_count": 1,
        "source": "ground-truth-demo-mask",
        "demo": True,
        "report": {
            "title": "AI brain MRI segmentation report",
            "technique": (
                "Automated volumetric tumor segmentation was performed on axial "
                "post-contrast T1-weighted brain MRI. The generated mask was "
                "reviewed for lesion extent, volume, longest diameter, and "
                "interval change relative to the prior reference examination."
            ),
            "finding": (
                "A solitary enhancing intra-axial mass is segmented in the right "
                "cerebral hemisphere, centered near the deep parietal/periatrial "
                "white matter. The lesion demonstrates a measurable enhancing "
                "component with surrounding T2/FLAIR hyperintense edema. No second "
                "discrete enhancing lesion is identified in this analysis."
            ),
            "subregions": [
                "enhancing tumor",
                "peritumoral edema",
                "necrotic or non-enhancing tumor core",
            ],
            "quantitative": {
                "current_volume_cm3": 14.815,
                "prior_volume_cm3": 12.92,
                "volume_change_pct": 14.7,
                "longest_diameter_mm": 39.1,
                "prior_longest_diameter_mm": 35.8,
                "diameter_change_mm": 3.3,
                "confidence": "high",
            },
            "comparison": (
                "Compared with the previous scan, total segmented tumor volume has "
                "increased from 12.92 cm3 to 14.82 cm3, an estimated 14.7% interval "
                "increase. Longest axial diameter increased from 35.8 mm to 39.1 mm. "
                "The enhancing tumor component is slightly larger, with mild "
                "increase in adjacent edema. No new separate lesion is seen."
            ),
            "impression": (
                "Mild interval progression of a solitary enhancing brain tumor, "
                "driven by increased enhancing tumor volume and slight enlargement "
                "of surrounding edema. Quantitative findings do not suggest a major "
                "mass-effect emergency on this generated review, but the interval "
                "growth pattern warrants radiologist confirmation and clinical "
                "correlation with treatment history."
            ),
            "recommendations": [
                "Radiologist should verify segmentation boundaries on axial, coronal, and sagittal planes.",
                "Correlate with steroid use, recent radiation, and treatment timing to distinguish progression from treatment effect.",
                "Consider multidisciplinary tumor board review if interval growth is confirmed.",
            ],
        },
    }


def execute_demo_mri_segmentation_job(*, job_id: str) -> WorkerDispatchEnvelope:
    """Simulate model inference using the bundled P01 ground-truth tumor mask."""

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

        current_stage = "data-fetching"
        running_state = transition_job(
            job.status,
            "running",
            stage=current_stage,
            progress=15,
            stage_message="Fetching and verifying MRI scan volume...",
        )
        job.status = running_state.status
        job.stage = running_state.stage
        job.progress = running_state.progress
        job.stage_message = running_state.stage_message
        session.add(
            JobEvent(
                job_id=job.id,
                status=job.status,
                stage=job.stage,
                event_type="transition",
                payload={
                    "detail": "demo MRI segmentation worker started",
                    "progress": 15,
                    "stage_message": job.stage_message,
                    "delaySeconds": settings.demo_job_delay_seconds,
                },
            )
        )
        session.commit()

        def _update_stage(stage: str, progress: int, stage_message: str) -> None:
            nonlocal current_stage
            current_stage = stage
            log_stage("debug", "Demo stage transition", study_id=str(study.public_id), stage=stage, detail=stage_message)
            stage_state = transition_job(
                job.status,
                job.status,
                stage=stage,
                progress=progress,
                stage_message=stage_message,
            )
            job.status = stage_state.status
            job.stage = stage_state.stage
            job.progress = stage_state.progress
            job.stage_message = stage_state.stage_message
            session.add(
                JobEvent(
                    job_id=job.id,
                    status=job.status,
                    stage=job.stage,
                    event_type="transition",
                    payload={"detail": stage_message, "progress": progress, "stage_message": stage_message},
                )
            )
            session.commit()

        try:
            delay_seconds = max(0.0, float(settings.demo_job_delay_seconds))
            step_delay = (delay_seconds / 5.0) if delay_seconds > 0 else 0.0

            if step_delay > 0:
                time.sleep(step_delay)

            _update_stage("bone-extraction", 35, "Extracting bone structures and anatomical landmarks...")
            if step_delay > 0:
                time.sleep(step_delay)

            _update_stage("segmentation", 65, "Running deep learning tumor segmentation model...")
            if step_delay > 0:
                time.sleep(step_delay)

            _update_stage("quantification", 80, "Calculating tumor volume, diameter, and spatial metrics...")
            if step_delay > 0:
                time.sleep(step_delay)

            _update_stage("report-generation", 95, "Generating structured AI clinical oncology report...")
            if step_delay > 0:
                time.sleep(step_delay)

            mask_path = _default_demo_mask_path(settings)
            if not mask_path.exists():
                raise RuntimeError(f"Demo ground-truth mask not found: {mask_path}")

            log_stage(
                "info",
                "Materializing demo MRI segmentation result",
                study_id=str(study.public_id),
                mask_path=str(mask_path),
            )
            materialize_nifti_study_with_mask(
                session=session,
                study_public_id=study.public_id,
                mask_source_absolute_path=mask_path,
                runner_metadata={
                    "model_id": "oncoflow-demo-ensemble",
                    "runner_version": "class-demo-1",
                    "execution_backend": "simulated",
                    "warnings": [],
                },
                result_metadata=_demo_result_metadata(),
            )
            completed_state = transition_job(
                job.status,
                "completed",
                stage="completed",
                progress=100,
                stage_message="Analysis completed successfully.",
            )
            job.status = completed_state.status
            job.stage = completed_state.stage
            job.progress = completed_state.progress
            job.stage_message = completed_state.stage_message
            job.failure_payload = None
            session.add(
                JobEvent(
                    job_id=job.id,
                    status=job.status,
                    stage=job.stage,
                    event_type="transition",
                    payload={"detail": "demo MRI segmentation completed", "progress": 100, "stage_message": "Analysis completed successfully."},
                )
            )
            session.commit()
            log_stage(
                "info",
                "Demo MRI worker completed",
                study_id=str(study.public_id),
            )
        except Exception as exc:
            log_stage(
                "error",
                "Demo MRI worker failed",
                study_id=str(study.public_id),
                error=str(exc),
            )
            session.rollback()

            failure_session = session_factory()
            try:
                failed_job = (
                    failure_session.query(Job).filter(Job.public_id == UUID(job_id)).one()
                )
                failed_study = (
                    failure_session.query(Study)
                    .filter(Study.id == failed_job.study_id)
                    .one()
                )
                failed_state = transition_job(
                    failed_job.status,
                    "failed",
                    stage="demo-inference",
                    error=JobErrorPayload(
                        code="demo-mri-segmentation-failed",
                        message=str(exc),
                        details={
                            "jobId": job_id,
                            "studyId": str(failed_study.public_id),
                        },
                    ),
                )
                failed_job.status = failed_state.status
                failed_job.stage = failed_state.stage
                failed_job.failure_payload = (
                    failed_state.error.model_dump(by_alias=True)
                    if failed_state.error
                    else None
                )
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
            extracted_relative_path="",
        )
