from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian, generate_uid

from app.infra.db.models import Artifact, Job, Study
from app.infra.db.session import create_session_factory
from app.modules.jobs.service import JobService
from app.modules.jobs.worker_tasks import execute_ingestion_job


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
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = series_instance_uid
    dataset.Modality = modality
    dataset.SeriesDescription = series_description
    dataset.ProtocolName = protocol_name
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

    envelope = execute_ingestion_job(job_id=str(submission.job_public_id))
    assert envelope.job_id == str(submission.job_public_id)

    session_factory = create_session_factory()
    with session_factory() as session:
        job = session.query(Job).filter(Job.public_id == UUID(str(submission.job_public_id))).one()
        artifacts = session.query(Artifact).filter(Artifact.artifact_kind.in_(["nifti-volume", "nifti-sidecar", "conversion-log"])).all()
        assert job.status == "completed"
        assert job.stage == "completed"
        assert len(artifacts) == 3
        assert "geometry" in artifacts[0].source_metadata


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
        assert job.stage == "converting"
        assert job.failure_payload["code"] == "ingestion-failed"
