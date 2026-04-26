from __future__ import annotations

from pathlib import Path

import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian, generate_uid

from app.modules.ingestion.profiling import profile_staged_study
from app.modules.ingestion.validation import validate_study_profile


def _write_dicom(
    path: Path,
    *,
    modality: str,
    series_description: str,
    protocol_name: str,
    image_type: list[str],
    study_description: str = "MRI FOOT",
    pixel_spacing: tuple[float, float] | None = (0.28, 0.28),
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
    dataset.SeriesInstanceUID = generate_uid()
    dataset.Modality = modality
    dataset.SeriesDescription = series_description
    dataset.ProtocolName = protocol_name
    dataset.ImageType = image_type
    dataset.StudyDescription = study_description
    dataset.Manufacturer = "Siemens"
    dataset.ManufacturerModelName = "MAGNETOM Cima.X Fit"
    dataset.MagneticFieldStrength = 3.0
    if pixel_spacing is not None:
        dataset.Rows = 128
        dataset.Columns = 128
        dataset.PixelSpacing = list(pixel_spacing)
        dataset.SliceThickness = 2.0
        dataset.SpacingBetweenSlices = 2.0
        dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_as(str(path), write_like_original=False)


def test_profile_classifies_supported_series_families(tmp_path: Path) -> None:
    _write_dicom(tmp_path / "exam" / "t1.dcm", modality="MR", series_description="tra_t1_tse_fs+c", protocol_name="tra_t1_tse_fs+c", image_type=["ORIGINAL", "PRIMARY"])
    _write_dicom(tmp_path / "exam" / "stir.dcm", modality="MR", series_description="t2_tse_stir_tra_RT", protocol_name="t2_tse_stir", image_type=["ORIGINAL", "PRIMARY"])

    profile = profile_staged_study(tmp_path / "exam")
    processable = [series for series in profile.series if series.classification == "processable"]

    assert len(processable) == 2
    assert all(series.reason_code == "supported-mr-series" for series in processable)


def test_profile_marks_localizers_and_derived_objects_as_metadata_only(tmp_path: Path) -> None:
    _write_dicom(tmp_path / "exam" / "localizer.dcm", modality="MR", series_description="localizer_tra", protocol_name="localizer", image_type=["ORIGINAL", "PRIMARY"])
    _write_dicom(tmp_path / "exam" / "document.dcm", modality="OT", series_description="Scanned Document", protocol_name="document", image_type=["DERIVED", "SECONDARY"], pixel_spacing=None)

    profile = profile_staged_study(tmp_path / "exam")
    classes = {series.record.series_description: series.classification for series in profile.series}

    assert classes["localizer_tra"] == "metadata-only"
    assert classes["Scanned Document"] == "metadata-only"


def test_validation_returns_explicit_message_when_no_processable_series(tmp_path: Path) -> None:
    _write_dicom(tmp_path / "exam" / "localizer.dcm", modality="MR", series_description="localizer_tra", protocol_name="localizer", image_type=["ORIGINAL", "PRIMARY"])

    profile = profile_staged_study(tmp_path / "exam")
    messages = validate_study_profile(profile)

    assert messages[0].code == "no-processable-series"
    assert "supported T1/T2 STIR" in messages[0].message
