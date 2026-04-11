from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pydicom


@dataclass(frozen=True)
class DicomSeriesRecord:
    series_instance_uid: str
    modality: str
    series_description: str
    protocol_name: str
    image_type: tuple[str, ...]
    manufacturer: str
    manufacturer_model_name: str
    magnetic_field_strength: float | None
    pixel_spacing: tuple[float, float] | None
    slice_thickness: float | None
    spacing_between_slices: float | None
    orientation: tuple[float, ...] | None
    files: tuple[Path, ...]
    extra_metadata: dict[str, Any]


def _read_dataset(path: Path):
    return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)


def _as_float_tuple(value: Any, *, expected_len: int | None = None) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raw = [value]
    else:
        raw = list(value)
    converted = tuple(float(item) for item in raw)
    if expected_len is not None and len(converted) < expected_len:
        return None
    return converted


def scan_staged_study(study_root: str | Path, *, max_files: int = 5000) -> list[DicomSeriesRecord]:
    root = Path(study_root)
    if not root.exists():
        raise ValueError(f"Staged study root not found: {root}")

    files = [path for path in root.rglob("*") if path.is_file()]
    if len(files) > max_files:
        raise ValueError("Study inventory exceeds the maximum supported file count")

    grouped: dict[str, dict[str, Any]] = {}
    for path in files:
        try:
            dataset = _read_dataset(path)
        except Exception:
            continue

        series_uid = str(getattr(dataset, "SeriesInstanceUID", "")).strip()
        if not series_uid:
            continue

        bucket = grouped.setdefault(
            series_uid,
            {"dataset": dataset, "files": []},
        )
        bucket["files"].append(path)

    records: list[DicomSeriesRecord] = []
    for bucket in grouped.values():
        dataset = bucket["dataset"]
        image_type = tuple(str(item).upper() for item in getattr(dataset, "ImageType", []))
        pixel_spacing = _as_float_tuple(getattr(dataset, "PixelSpacing", None), expected_len=2)
        spacing = (pixel_spacing[0], pixel_spacing[1]) if pixel_spacing else None
        orientation_values = _as_float_tuple(getattr(dataset, "ImageOrientationPatient", None))
        records.append(
            DicomSeriesRecord(
                series_instance_uid=str(getattr(dataset, "SeriesInstanceUID", "")).strip(),
                modality=str(getattr(dataset, "Modality", "")).strip().upper(),
                series_description=str(getattr(dataset, "SeriesDescription", "")).strip(),
                protocol_name=str(getattr(dataset, "ProtocolName", "")).strip(),
                image_type=image_type,
                manufacturer=str(getattr(dataset, "Manufacturer", "")).strip(),
                manufacturer_model_name=str(getattr(dataset, "ManufacturerModelName", "")).strip(),
                magnetic_field_strength=float(getattr(dataset, "MagneticFieldStrength", 0.0) or 0.0) or None,
                pixel_spacing=spacing,
                slice_thickness=float(getattr(dataset, "SliceThickness", 0.0) or 0.0) or None,
                spacing_between_slices=float(getattr(dataset, "SpacingBetweenSlices", 0.0) or 0.0) or None,
                orientation=orientation_values,
                files=tuple(sorted(bucket["files"])),
                extra_metadata={
                    "study_description": str(getattr(dataset, "StudyDescription", "")).strip(),
                    "sequence_name": str(getattr(dataset, "SequenceName", "")).strip(),
                },
            )
        )
    return sorted(records, key=lambda record: record.series_description or record.series_instance_uid)
