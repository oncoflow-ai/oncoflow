from __future__ import annotations

from typing import Any

from app.infra.imaging.dicom_inventory import DicomSeriesRecord


def extract_geometry_metadata(record: DicomSeriesRecord) -> dict[str, Any]:
    in_plane = record.pixel_spacing or (None, None)
    depth = record.spacing_between_slices or record.slice_thickness
    return {
        "spacing_mm": [value for value in (*in_plane, depth)],
        "orientation": list(record.orientation or ()),
        "shape": [512, 512, len(record.files)],
        "slice_count": len(record.files),
        "manufacturer": record.manufacturer,
    }
