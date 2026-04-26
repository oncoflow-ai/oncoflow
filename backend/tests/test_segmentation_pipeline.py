from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian, generate_uid

from app.infra.db.models import Artifact, Job, StudyResult
from app.infra.db.session import create_session_factory
from app.modules.jobs.service import JobService
from app.modules.jobs.worker_tasks import execute_ingestion_job
from app.modules.segmentation.contracts import BoundingBox3D, ManagedArtifactRef
from app.modules.segmentation.runner import NormalizedLesionPrediction, RunnerExecutionResult


_DEFAULT_STUDY_INSTANCE_UID = generate_uid()


def _write_dicom(
    path: Path,
    *,
    series_instance_uid: str,
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
    dataset.StudyInstanceUID = _DEFAULT_STUDY_INSTANCE_UID
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


@pytest.fixture(autouse=True)
def configure_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONCOFLOW_DATABASE_URL", f"sqlite+pysqlite:///{tmp_path / 'segmentation-pipeline.sqlite3'}")
    monkeypatch.setenv("ONCOFLOW_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("ONCOFLOW_STORAGE_STAGING_DIR", "raw")


def _submit_supported_study(tmp_path: Path) -> tuple[str, str]:
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
            source_label="pipeline-test",
        )
    )
    return str(submission.job_public_id), str(submission.study_public_id)


def test_segmentation_pipeline_persists_mask_and_review_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id, study_id = _submit_supported_study(tmp_path)

    def fake_run_segmentation(*, bundle, model_id="nnunet-v2-resenc", predictions=(), warnings=()):
        return RunnerExecutionResult(
            bundle=bundle,
            runner=type("RunnerInfo", (), {
                "model_id": model_id,
                "runner_version": "2026.04",
                "execution_backend": "stub",
                "warnings": (),
            })(),
            lesions=(
                NormalizedLesionPrediction(
                    mask_artifact=ManagedArtifactRef(storage_root="derived", relative_path=f"studies/{study_id}/lesions/component-001.nii.gz"),
                    bounding_box=BoundingBox3D(x_min=1, x_max=4, y_min=2, y_max=6, z_min=1, z_max=3),
                    confidence_score=0.21,
                    occupied_voxels_ijk=((0, 0, 0), (1, 1, 1), (1, 2, 1)),
                    warning_reasons=("uncertain boundary",),
                ),
            ),
            warnings=(),
        )

    monkeypatch.setattr("app.modules.segmentation.pipeline.run_segmentation", fake_run_segmentation)

    execute_ingestion_job(job_id=job_id)

    session_factory = create_session_factory()
    with session_factory() as session:
        artifacts = session.query(Artifact).filter(Artifact.artifact_kind.in_(["segmentation-mask", "review-overlay", "segmentation-case-result", "study-result-bundle"])).all()
        job = session.query(Job).filter(Job.public_id == UUID(job_id)).one()
        study_result = session.query(StudyResult).one()

        assert job.status == "completed"
        assert job.stage == "completed"
        assert {artifact.artifact_kind for artifact in artifacts} == {"segmentation-mask", "review-overlay", "segmentation-case-result", "study-result-bundle"}
        mask_artifact = next(artifact for artifact in artifacts if artifact.artifact_kind == "segmentation-mask")
        assert mask_artifact.source_metadata["runner"]["model_id"] == "nnunet-v2-resenc"
        assert "slot_provenance" in mask_artifact.source_metadata
        assert mask_artifact.source_metadata["occupied_voxels_ijk"] == [[0, 0, 0], [1, 1, 1], [1, 2, 1]]
        assert not mask_artifact.relative_path.startswith("/")
        assert study_result.summary_metadata["lesion_count"] == 1


def test_segmentation_pipeline_persists_review_required_case_on_bundle_degradation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id, _study_id = _submit_supported_study(tmp_path)

    def fake_run_segmentation(*, bundle, model_id="nnunet-v2-resenc", predictions=(), warnings=()):
        degraded_bundle = type(bundle)(
            study_id=bundle.study_id,
            slot_assignments=bundle.slot_assignments[:2],
            missing_slots=("t1_post_or_fs",),
            degradation_reasons=("missing canonical slots: t1_post_or_fs",),
        )
        return RunnerExecutionResult(
            bundle=degraded_bundle,
            runner=type("RunnerInfo", (), {
                "model_id": model_id,
                "runner_version": "2026.04",
                "execution_backend": "stub",
                "warnings": ("fallback review path",),
            })(),
            lesions=(),
            warnings=("fallback review path",),
        )

    monkeypatch.setattr("app.modules.segmentation.pipeline.run_segmentation", fake_run_segmentation)

    execute_ingestion_job(job_id=job_id)

    session_factory = create_session_factory()
    with session_factory() as session:
        case_artifact = session.query(Artifact).filter(Artifact.artifact_kind == "segmentation-case-result").one()
        result_bundle = session.query(Artifact).filter(Artifact.artifact_kind == "study-result-bundle").one()
        assert case_artifact.source_metadata["needs_review"] is True
        assert "missing canonical slots: t1_post_or_fs" in case_artifact.source_metadata["case_qc_reasons"]
        assert result_bundle.relative_path.endswith("study-result.json")
