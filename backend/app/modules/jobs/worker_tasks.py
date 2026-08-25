from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from pathlib import Path
from uuid import UUID

from app.api.schemas.jobs import JobErrorPayload
from app.core.audit import log_audit_event
from app.infra.db.models import Artifact, Job, JobEvent, Patient, Study
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


DEMO_STAGE_CONFIGS: list[dict[str, Any]] = [
    {
        "stage_index": 0,
        "stage_key": "baseline",
        "stage_label": "Baseline Scan",
        "mask_file": "P01_tumor_mask_baseline.nii.gz",
        "report": {
            "title": "AI brain MRI segmentation report (Baseline Study)",
            "technique": (
                "Automated volumetric tumor segmentation was performed on initial "
                "baseline axial post-contrast T1-weighted brain MRI establishing "
                "pre-treatment tumor burden."
            ),
            "finding": (
                "A solitary enhancing intra-axial mass is segmented in the right "
                "cerebral hemisphere, centered near the deep fronto-parietal white matter. "
                "The lesion demonstrates an enhancing component (14.82 cm3 volume, longest "
                "diameter 39.1 mm) with surrounding T2/FLAIR hyperintense vasogenic edema. "
                "No second discrete enhancing lesion is identified."
            ),
            "subregions": [
                "enhancing tumor core",
                "peritumoral vasogenic edema",
                "non-enhancing core component",
            ],
            "quantitative": {
                "current_volume_cm3": 14.815,
                "prior_volume_cm3": None,
                "volume_change_pct": None,
                "longest_diameter_mm": 39.1,
                "prior_longest_diameter_mm": None,
                "diameter_change_mm": None,
                "confidence": "high",
            },
            "comparison": (
                "Baseline examination. No prior reference MRI study available for "
                "comparison. Establishes baseline quantitative volumetric metrics "
                "for longitudinal response tracking."
            ),
            "impression": (
                "Solitary right cerebral enhancing lesion establishing initial "
                "baseline tumor burden (14.82 cm3 volume). Findings serve as the "
                "quantitative benchmark for subsequent longitudinal interval response assessment."
            ),
            "recommendations": [
                "Radiologist should verify baseline segmentation boundaries on orthogonal planes.",
                "Initiate treatment protocol and schedule serial follow-up MRI to assess interval therapeutic response.",
                "Correlate baseline volumetric burden with neurological baseline examination.",
            ],
        },
    },
    {
        "stage_index": 1,
        "stage_key": "fu1",
        "stage_label": "Follow-Up Study #1 (Marked Tumor Shrinkage)",
        "mask_file": "P01_tumor_mask_fu1.nii.gz",
        "report": {
            "title": "AI brain MRI segmentation report (Follow-Up #1)",
            "technique": (
                "Automated volumetric tumor segmentation was performed on axial "
                "post-contrast T1-weighted brain MRI with longitudinal interval "
                "comparison against baseline scan."
            ),
            "finding": (
                "Marked interval regression of the previously segmented right parietal "
                "enhancing intra-axial mass. Substantial reduction in both enhancing "
                "solid core and surrounding peritumoral edema."
            ),
            "subregions": [
                "residual enhancing tumor",
                "reduced peritumoral edema",
                "treated core",
            ],
            "quantitative": {
                "current_volume_cm3": 3.101,
                "prior_volume_cm3": 14.815,
                "volume_change_pct": -79.1,
                "longest_diameter_mm": 21.2,
                "prior_longest_diameter_mm": 39.1,
                "diameter_change_mm": -17.9,
                "confidence": "high",
            },
            "comparison": (
                "Compared with baseline (14.82 cm3), total tumor volume decreased "
                "significantly to 3.10 cm3 (-79.1% volume reduction). Longest axial "
                "diameter decreased from 39.1 mm to 21.2 mm (-17.9 mm decrease). "
                "Significant reduction in surrounding edema."
            ),
            "impression": (
                "Marked interval therapeutic response and tumor shrinkage following "
                "treatment initiation (-79.1% volume reduction). Findings consistent "
                "with major radiological response."
            ),
            "recommendations": [
                "Confirm segmentation boundaries against baseline landmarks.",
                "Continue current treatment protocol given positive therapeutic response.",
                "Schedule next surveillance MRI study in 8-12 weeks.",
            ],
        },
    },
    {
        "stage_index": 2,
        "stage_key": "fu2",
        "stage_label": "Follow-Up Study #2 (Stable Post-Treatment Bed)",
        "mask_file": "P01_tumor_mask_fu2.nii.gz",
        "report": {
            "title": "AI brain MRI segmentation report (Follow-Up #2)",
            "technique": (
                "Automated volumetric tumor segmentation and interval tracking "
                "relative to prior serial MRI studies."
            ),
            "finding": (
                "Stable appearance of the treated right parietal cavity with "
                "thin peripheral rim enhancement. Residual edema remains minimal."
            ),
            "subregions": [
                "thin peripheral rim",
                "minimal edema",
                "post-treatment cavity",
            ],
            "quantitative": {
                "current_volume_cm3": 3.911,
                "prior_volume_cm3": 3.101,
                "volume_change_pct": -73.6,
                "longest_diameter_mm": 23.5,
                "prior_longest_diameter_mm": 21.2,
                "diameter_change_mm": 2.3,
                "confidence": "high",
            },
            "comparison": (
                "Compared with baseline, tumor burden remains substantially reduced "
                "(-73.6% volume reduction vs baseline). Minor interval margin remodeling "
                "noted relative to Follow-Up #1 (3.10 -> 3.91 cm3), consistent with stable "
                "post-treatment cavity dynamics."
            ),
            "impression": (
                "Stable post-treatment appearances without evidence of true nodular "
                "recurrence. Substantial overall response preserved relative to baseline."
            ),
            "recommendations": [
                "Review subtraction series to distinguish post-radiation enhancement from progression.",
                "Maintain planned imaging surveillance schedule.",
            ],
        },
    },
    {
        "stage_index": 3,
        "stage_key": "fu3",
        "stage_label": "Follow-Up Study #3 (Continued Regression)",
        "mask_file": "P01_tumor_mask_fu3.nii.gz",
        "report": {
            "title": "AI brain MRI segmentation report (Follow-Up #3)",
            "technique": (
                "Automated volumetric tumor segmentation with multi-study longitudinal comparison."
            ),
            "finding": (
                "Continued interval reduction in residual enhancing tumor tissue in the right "
                "parietal region. Surrounding parenchymal edema is nearly completely resolved."
            ),
            "subregions": [
                "focal residual enhancement",
                "resolved edema",
            ],
            "quantitative": {
                "current_volume_cm3": 2.285,
                "prior_volume_cm3": 3.911,
                "volume_change_pct": -84.6,
                "longest_diameter_mm": 19.1,
                "prior_longest_diameter_mm": 23.5,
                "diameter_change_mm": -4.4,
                "confidence": "high",
            },
            "comparison": (
                "Compared with prior study (3.91 cm3), total volume decreased to 2.29 cm3 "
                "(-41.6% interval decrease; -84.6% reduction from baseline). Longest diameter "
                "decreased to 19.1 mm."
            ),
            "impression": (
                "Excellent ongoing therapeutic response with progressive interval shrinkage "
                "of residual enhancing tissue."
            ),
            "recommendations": [
                "Continue maintenance therapy as clinically indicated.",
                "Routine interval follow-up MRI.",
            ],
        },
    },
    {
        "stage_index": 4,
        "stage_key": "fu4",
        "stage_label": "Follow-Up Study #4 (Minimal Residual Focus)",
        "mask_file": "P01_tumor_mask_fu4.nii.gz",
        "report": {
            "title": "AI brain MRI segmentation report (Follow-Up #4)",
            "technique": (
                "Automated volumetric tumor segmentation assessing residual post-treatment tissue."
            ),
            "finding": (
                "Only a minute focal area of faint residual enhancement is detected in the right "
                "parietal cavity. No surrounding edema, midline shift, or mass effect."
            ),
            "subregions": [
                "minimal focal enhancement",
            ],
            "quantitative": {
                "current_volume_cm3": 1.264,
                "prior_volume_cm3": 2.285,
                "volume_change_pct": -91.5,
                "longest_diameter_mm": 14.8,
                "prior_longest_diameter_mm": 19.1,
                "diameter_change_mm": -4.3,
                "confidence": "high",
            },
            "comparison": (
                "Compared with baseline (14.82 cm3), overall tumor volume has decreased "
                "by -91.5% to 1.26 cm3. Longest diameter reduced from 39.1 mm to 14.8 mm. "
                "Sustained durable regression."
            ),
            "impression": (
                "Minimal residual enhancing disease with sustained durable response to therapy "
                "(-91.5% from baseline). No evidence of aggressive relapse."
            ),
            "recommendations": [
                "Reassuring imaging appearance. Continue surveillance intervals.",
            ],
        },
    },
]


def _demo_mask_path_for_stage(stage_index: int, settings) -> Path:
    cfg = DEMO_STAGE_CONFIGS[stage_index % len(DEMO_STAGE_CONFIGS)]
    if stage_index == 0 and settings.demo_ground_truth_mask_path:
        return Path(settings.demo_ground_truth_mask_path).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[4]
    return (
        repo_root
        / "data"
        / "P01"
        / "tumor segmentation"
        / cfg["mask_file"]
    )


def _default_demo_mask_path(settings) -> Path:
    return _demo_mask_path_for_stage(0, settings)


def _demo_result_metadata(stage_index: int = 0) -> dict[str, object]:
    cfg = DEMO_STAGE_CONFIGS[stage_index % len(DEMO_STAGE_CONFIGS)]
    return {
        "case_qc_reasons": [],
        "lesion_count": 1,
        "source": "ground-truth-demo-mask",
        "demo": True,
        "demo_stage": cfg["stage_key"],
        "stage_index": cfg["stage_index"],
        "stage_label": cfg["stage_label"],
        "report": cfg["report"],
    }


def execute_demo_mri_segmentation_job(*, job_id: str) -> WorkerDispatchEnvelope:
    """Simulate model inference using bundled multi-stage ground-truth tumor masks."""

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

        # Count prior demo uploads:
        # If the patient has a fixed designated pseudonym (like P-9001), count prior uploads for that patient.
        # If the upload is anonymous/auto-generated patient without explicit patient_id, count global prior uploads.
        patient = session.query(Patient).filter(Patient.id == study.patient_id).first() if study.patient_id else None
        is_auto_generated_patient = patient is not None and patient.pseudonym.startswith("PAT-")

        if study.patient_id is not None and not is_auto_generated_patient:
            patient_prior_count = session.query(Study).filter(
                Study.source_kind == "demo-mri-upload",
                Study.patient_id == study.patient_id,
                Study.id < study.id,
            ).count()
            demo_stage_index = patient_prior_count % len(DEMO_STAGE_CONFIGS)
        else:
            global_prior_count = session.query(Study).filter(
                Study.source_kind == "demo-mri-upload",
                Study.id < study.id,
            ).count()
            demo_stage_index = global_prior_count % len(DEMO_STAGE_CONFIGS)

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
                    "stageIndex": demo_stage_index,
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

            mask_path = _demo_mask_path_for_stage(demo_stage_index, settings)
            if not mask_path.exists():
                raise RuntimeError(f"Demo ground-truth mask not found: {mask_path}")

            log_stage(
                "info",
                "Materializing demo MRI segmentation result",
                study_id=str(study.public_id),
                mask_path=str(mask_path),
                stage_index=demo_stage_index,
            )
            materialize_nifti_study_with_mask(
                session=session,
                study_public_id=study.public_id,
                mask_source_absolute_path=mask_path,
                runner_metadata={
                    "model_id": "oncoflow-demo-ensemble",
                    "runner_version": f"class-demo-{demo_stage_index + 1}",
                    "execution_backend": "simulated",
                    "warnings": [],
                },
                result_metadata=_demo_result_metadata(demo_stage_index),
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
