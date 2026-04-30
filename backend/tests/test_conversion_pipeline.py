from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian, generate_uid

from app.infra.db.models import Artifact, Job, Study, StudyResult
from app.infra.db.session import create_session_factory
from app.modules.artifacts.catalog import record_study_artifact
from app.modules.jobs.service import JobService
from app.modules.jobs.worker_tasks import execute_ingestion_job
from app.modules.segmentation.pipeline import SegmentationPipelineResult
from app.modules.segmentation.contracts import (
    CanonicalSeriesBundle,
    CanonicalSeriesSlotAssignment,
    CaseSegmentationResult,
    ManagedArtifactRef,
    RunnerProvenance,
)
from app.modules.segmentation.runner import RunnerExecutionResult


_DEFAULT_STUDY_INSTANCE_UID = generate_uid()


def _write_dicom(
    path: Path,
    *,
    series_instance_uid: str,
    study_instance_uid: str | None = None,
    modality: str = "MR",
    series_description: str = "t2_tse_stir_tra_RT",
    protocol_name: str = "t2_tse_stir",
) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.PatientName = "Test^Patient"
    dataset.PatientID = "123"
    dataset.StudyInstanceUID = study_instance_uid or _DEFAULT_STUDY_INSTANCE_UID
    dataset.SeriesInstanceUID = series_instance_uid
    dataset.Modality = modality
    dataset.SeriesDescription = series_description
    dataset.ProtocolName = protocol_name
    dataset.ImageType = ["ORIGINAL", "PRIMARY"]
    dataset.StudyDescription = "MRI FOOT"
    dataset.Manufacturer = "Siemens"
    dataset.ManufacturerModelName = "MAGNETOM Cima.X Fit"
    dataset.MagneticFieldStrength = 3.0
    dataset.Rows = 128
    dataset.Columns = 128
    dataset.PixelSpacing = [0.28, 0.28]
    dataset.SliceThickness = 2.0
    dataset.SpacingBetweenSlices = 2.0
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_as(str(path), write_like_original=False)


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    import io
    import zipfile

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return payload.getvalue()


def _stub_segmentation_result(*, session, study_public_id, model_id="nnunet-v2-resenc", stage_callback=None):
    if stage_callback:
        for stage in ("prepare-inputs", "infer", "postprocess", "package-results"):
            stage_callback(stage)

    from app.infra.db.models import Study

    study = session.query(Study).filter(Study.public_id == study_public_id).one()
    case_relative_path = f"studies/{study_public_id}/segmentation/result.json"
    record_study_artifact(
        session,
        study_id=study.id,
        artifact_kind="segmentation-case-result",
        relative_path=case_relative_path,
        metadata={
            "needs_review": False,
            "case_qc_reasons": [],
            "lesion_count": 0,
        },
    )
    session.flush()
    return SegmentationPipelineResult(
        case_result=CaseSegmentationResult(
            study_id=str(study_public_id),
            input_bundle=CanonicalSeriesBundle(
                study_id=str(study_public_id),
                slot_assignments=(
                    CanonicalSeriesSlotAssignment(
                        slot_name="t1_pre",
                        series_instance_uid="series-1",
                        source_artifact=ManagedArtifactRef(
                            storage_root="derived",
                            relative_path=f"studies/{study_public_id}/series/1/volume.nii.gz",
                        ),
                    ),
                ),
            ),
            runner=RunnerProvenance(model_id=model_id, runner_version="stub", execution_backend="stub"),
            lesions=(),
            needs_review=False,
        ),
        persisted_artifact_count=1,
    )


@pytest.fixture(autouse=True)
def configure_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONCOFLOW_DATABASE_URL", f"sqlite+pysqlite:///{tmp_path / 'conversion.sqlite3'}")
    monkeypatch.setenv("ONCOFLOW_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("ONCOFLOW_STORAGE_STAGING_DIR", "raw")


def test_worker_execution_creates_nifti_and_sidecar_artifacts(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    _write_dicom(source_dir / "exam" / "series1" / "slice1.dcm", series_instance_uid=generate_uid())
    archive_bytes = _zip_bytes({"exam/series1/slice1.dcm": (source_dir / "exam" / "series1" / "slice1.dcm").read_bytes()})

    submission = asyncio.run(
        JobService().submit_mri_study(
            filename="exam.zip",
            content_type="application/zip",
            archive_bytes=archive_bytes,
            source_label="pipeline-test",
        )
    )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("app.modules.jobs.worker_tasks.run_study_segmentation", _stub_segmentation_result)
    try:
        envelope = execute_ingestion_job(job_id=str(submission.job_public_id))
    finally:
        monkeypatch.undo()
    assert envelope.job_id == str(submission.job_public_id)

    session_factory = create_session_factory()
    with session_factory() as session:
        job = session.query(Job).filter(Job.public_id == UUID(str(submission.job_public_id))).one()
        study = session.query(Study).filter(Study.public_id == UUID(str(submission.study_public_id))).one()
        artifacts = session.query(Artifact).filter(Artifact.artifact_kind.in_(["nifti-volume", "nifti-sidecar", "conversion-log"])).all()
        result_bundle = session.query(Artifact).filter(Artifact.artifact_kind == "study-result-bundle").one()
        study_result = session.query(StudyResult).filter(StudyResult.study_id == study.id).one()
        assert job.status == "completed"
        assert job.stage == "completed"
        assert study.study_instance_uid != f"staged-{submission.study_public_id}"
        assert study.staging_status == "processed"
        assert len(artifacts) == 3
        assert "geometry" in artifacts[0].source_metadata
        assert result_bundle.relative_path.endswith("study-result.json")
        assert study_result.summary_metadata["lesion_count"] == 0


def test_worker_failure_sets_actionable_error_payload(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    _write_dicom(
        source_dir / "exam" / "series1" / "slice1.dcm",
        series_instance_uid=generate_uid(),
        series_description="localizer_tra",
        protocol_name="localizer",
    )
    archive_bytes = _zip_bytes({"exam/series1/slice1.dcm": (source_dir / "exam" / "series1" / "slice1.dcm").read_bytes()})

    submission = asyncio.run(
        JobService().submit_mri_study(
            filename="exam.zip",
            content_type="application/zip",
            archive_bytes=archive_bytes,
            source_label="pipeline-failure",
        )
    )

    with pytest.raises(ValueError):
        execute_ingestion_job(job_id=str(submission.job_public_id))

    session_factory = create_session_factory()
    with session_factory() as session:
        job = session.query(Job).filter(Job.public_id == UUID(str(submission.job_public_id))).one()
        assert job.status == "failed"
        assert job.stage == "validating"
        assert job.failure_payload["code"] == "ingestion-failed"


def test_worker_failure_in_segmentation_keeps_actionable_failure_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_dir = tmp_path / "source"
    _write_dicom(source_dir / "exam" / "series1" / "slice1.dcm", series_instance_uid=generate_uid(), series_description="tra_t1_tse", protocol_name="tra_t1_tse")
    _write_dicom(source_dir / "exam" / "series2" / "slice1.dcm", series_instance_uid=generate_uid(), series_description="sag_t1_tse_fs+c", protocol_name="post contrast fs")
    _write_dicom(source_dir / "exam" / "series3" / "slice1.dcm", series_instance_uid=generate_uid(), series_description="t2_tse_stir_tra_RT", protocol_name="t2_tse_stir")
    archive_bytes = _zip_bytes({
        "exam/series1/slice1.dcm": (source_dir / "exam" / "series1" / "slice1.dcm").read_bytes(),
        "exam/series2/slice1.dcm": (source_dir / "exam" / "series2" / "slice1.dcm").read_bytes(),
        "exam/series3/slice1.dcm": (source_dir / "exam" / "series3" / "slice1.dcm").read_bytes(),
    })

    submission = asyncio.run(
        JobService().submit_mri_study(
            filename="exam.zip",
            content_type="application/zip",
            archive_bytes=archive_bytes,
            source_label="segmentation-failure",
        )
    )

    def boom(*, session, study_public_id, model_id="nnunet-v2-resenc", stage_callback=None):
        if stage_callback:
            stage_callback("infer")
        raise RuntimeError("runner crashed")

    monkeypatch.setattr("app.modules.jobs.worker_tasks.run_study_segmentation", boom)

    with pytest.raises(RuntimeError, match="runner crashed"):
        execute_ingestion_job(job_id=str(submission.job_public_id))

    session_factory = create_session_factory()
    with session_factory() as session:
        job = session.query(Job).filter(Job.public_id == UUID(str(submission.job_public_id))).one()
        assert job.status == "failed"
        assert job.stage == "infer"
        assert job.failure_payload["message"] == "runner crashed"


def test_worker_failure_in_result_materialization_keeps_actionable_failure_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_dir = tmp_path / "source"
    _write_dicom(source_dir / "exam" / "series1" / "slice1.dcm", series_instance_uid=generate_uid(), series_description="tra_t1_tse", protocol_name="tra_t1_tse")
    _write_dicom(source_dir / "exam" / "series2" / "slice1.dcm", series_instance_uid=generate_uid(), series_description="sag_t1_tse_fs+c", protocol_name="post contrast fs")
    _write_dicom(source_dir / "exam" / "series3" / "slice1.dcm", series_instance_uid=generate_uid(), series_description="t2_tse_stir_tra_RT", protocol_name="t2_tse_stir")
    archive_bytes = _zip_bytes({
        "exam/series1/slice1.dcm": (source_dir / "exam" / "series1" / "slice1.dcm").read_bytes(),
        "exam/series2/slice1.dcm": (source_dir / "exam" / "series2" / "slice1.dcm").read_bytes(),
        "exam/series3/slice1.dcm": (source_dir / "exam" / "series3" / "slice1.dcm").read_bytes(),
    })

    submission = asyncio.run(
        JobService().submit_mri_study(
            filename="exam.zip",
            content_type="application/zip",
            archive_bytes=archive_bytes,
            source_label="materialization-failure",
        )
    )

    monkeypatch.setattr("app.modules.jobs.worker_tasks.run_study_segmentation", _stub_segmentation_result)

    def materialize_boom(*, session, study_public_id):
        raise RuntimeError("materialization crashed")

    monkeypatch.setattr("app.modules.jobs.worker_tasks.materialize_study_results", materialize_boom)

    with pytest.raises(RuntimeError, match="materialization crashed"):
        execute_ingestion_job(job_id=str(submission.job_public_id))

    session_factory = create_session_factory()
    with session_factory() as session:
        job = session.query(Job).filter(Job.public_id == UUID(str(submission.job_public_id))).one()
        assert job.status == "failed"
        assert job.stage == "materialize-results"
        assert job.failure_payload["message"] == "materialization crashed"


def test_duplicate_study_instance_uid_fails_cleanly_without_pending_rollback(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    shared_study_uid = generate_uid()

    def write_with_shared_uid(path: Path, *, series_instance_uid: str) -> None:
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = generate_uid()
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

        dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.PatientName = "Test^Patient"
        dataset.PatientID = "123"
        dataset.StudyInstanceUID = shared_study_uid
        dataset.SeriesInstanceUID = series_instance_uid
        dataset.Modality = "MR"
        dataset.SeriesDescription = "t2_tse_stir_tra_RT"
        dataset.ProtocolName = "t2_tse_stir"
        dataset.ImageType = ["ORIGINAL", "PRIMARY"]
        dataset.StudyDescription = "MRI FOOT"
        dataset.Manufacturer = "Siemens"
        dataset.ManufacturerModelName = "MAGNETOM Cima.X Fit"
        dataset.MagneticFieldStrength = 3.0
        dataset.PixelSpacing = [0.28, 0.28]
        dataset.SliceThickness = 2.0
        dataset.SpacingBetweenSlices = 2.0
        dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_as(str(path), write_like_original=False)

    write_with_shared_uid(source_dir / "exam" / "series1" / "slice1.dcm", series_instance_uid=generate_uid())
    first_archive = _zip_bytes({
        "exam/series1/slice1.dcm": (source_dir / "exam" / "series1" / "slice1.dcm").read_bytes(),
    })

    first_submission = asyncio.run(
        JobService().submit_mri_study(
            filename="exam-first.zip",
            content_type="application/zip",
            archive_bytes=first_archive,
            source_label="duplicate-test-1",
        )
    )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("app.modules.jobs.worker_tasks.run_study_segmentation", _stub_segmentation_result)
    try:
        execute_ingestion_job(job_id=str(first_submission.job_public_id))
    finally:
        monkeypatch.undo()

    write_with_shared_uid(source_dir / "exam2" / "series2" / "slice1.dcm", series_instance_uid=generate_uid())
    second_archive = _zip_bytes({
        "exam2/series2/slice1.dcm": (source_dir / "exam2" / "series2" / "slice1.dcm").read_bytes(),
    })

    second_submission = asyncio.run(
        JobService().submit_mri_study(
            filename="exam-second.zip",
            content_type="application/zip",
            archive_bytes=second_archive,
            source_label="duplicate-test-2",
        )
    )

    with pytest.raises(ValueError, match="StudyInstanceUID already exists"):
        execute_ingestion_job(job_id=str(second_submission.job_public_id))

    session_factory = create_session_factory()
    with session_factory() as session:
        job = session.query(Job).filter(Job.public_id == UUID(str(second_submission.job_public_id))).one()
        assert job.status == "failed"
        assert job.stage == "validating"
        assert job.failure_payload["message"] == "StudyInstanceUID already exists for another uploaded study"


def test_mixed_study_instance_uid_archive_is_rejected(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    shared_study_uid = generate_uid()
    other_study_uid = generate_uid()

    _write_dicom(
        source_dir / "exam" / "series1" / "slice1.dcm",
        series_instance_uid=generate_uid(),
        study_instance_uid=shared_study_uid,
        series_description="tra_t1_tse",
        protocol_name="tra_t1_tse",
    )
    _write_dicom(
        source_dir / "exam" / "series2" / "slice1.dcm",
        series_instance_uid=generate_uid(),
        study_instance_uid=other_study_uid,
        series_description="t2_tse_stir_tra_RT",
        protocol_name="t2_tse_stir",
    )
    archive_bytes = _zip_bytes({
        "exam/series1/slice1.dcm": (source_dir / "exam" / "series1" / "slice1.dcm").read_bytes(),
        "exam/series2/slice1.dcm": (source_dir / "exam" / "series2" / "slice1.dcm").read_bytes(),
    })

    submission = asyncio.run(
        JobService().submit_mri_study(
            filename="mixed-study.zip",
            content_type="application/zip",
            archive_bytes=archive_bytes,
            source_label="mixed-study-test",
        )
    )

    with pytest.raises(ValueError, match="multiple StudyInstanceUID"):
        execute_ingestion_job(job_id=str(submission.job_public_id))

    session_factory = create_session_factory()
    with session_factory() as session:
        job = session.query(Job).filter(Job.public_id == UUID(str(submission.job_public_id))).one()
        assert job.status == "failed"
        assert job.stage == "profiling"
        assert job.failure_payload["message"] == "Uploaded series contain multiple StudyInstanceUID values"
