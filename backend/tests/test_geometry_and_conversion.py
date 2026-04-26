from __future__ import annotations

from pathlib import Path

from app.infra.imaging.dcm2niix_wrapper import convert_dicom_series
from app.infra.imaging.dicom_inventory import DicomSeriesRecord
from app.infra.imaging.geometry import extract_geometry_metadata


def test_geometry_metadata_uses_recorded_matrix_dimensions(tmp_path: Path) -> None:
    record = DicomSeriesRecord(
        study_instance_uid="1.2.3",
        series_instance_uid="4.5.6",
        modality="MR",
        series_description="t1",
        protocol_name="t1",
        image_type=("ORIGINAL", "PRIMARY"),
        manufacturer="Siemens",
        manufacturer_model_name="Cima.X",
        magnetic_field_strength=3.0,
        pixel_spacing=(0.5, 0.5),
        slice_thickness=2.0,
        spacing_between_slices=2.0,
        orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        rows=128,
        columns=256,
        files=(tmp_path / "slice1.dcm", tmp_path / "slice2.dcm"),
        extra_metadata={},
    )

    geometry = extract_geometry_metadata(record)

    assert geometry["shape"] == [128, 256, 2]


def test_convert_dicom_series_uses_common_series_root_for_conversion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    files = (
        tmp_path / "exam" / "series-a" / "part-1" / "slice1.dcm",
        tmp_path / "exam" / "series-a" / "part-2" / "slice2.dcm",
    )
    record = DicomSeriesRecord(
        study_instance_uid="1.2.3",
        series_instance_uid="4.5.6",
        modality="MR",
        series_description="t1",
        protocol_name="t1",
        image_type=("ORIGINAL", "PRIMARY"),
        manufacturer="Siemens",
        manufacturer_model_name="Cima.X",
        magnetic_field_strength=3.0,
        pixel_spacing=(0.5, 0.5),
        slice_thickness=2.0,
        spacing_between_slices=2.0,
        orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        rows=128,
        columns=256,
        files=files,
        extra_metadata={},
    )

    calls: list[list[str]] = []

    monkeypatch.setattr("app.infra.imaging.dcm2niix_wrapper.shutil.which", lambda _: "/usr/bin/dcm2niix")

    class FakeCompletedProcess:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, capture_output, text, check):
        calls.append(list(cmd))
        return FakeCompletedProcess()

    monkeypatch.setattr("app.infra.imaging.dcm2niix_wrapper.subprocess.run", fake_run)

    result = convert_dicom_series(record, tmp_path / "out", filename_stem="volume")

    assert calls[0][-1] == str(tmp_path / "exam" / "series-a")
    assert result["converter"] == "dcm2niix"
