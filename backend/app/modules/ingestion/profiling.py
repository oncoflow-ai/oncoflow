from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.infra.imaging.dicom_inventory import DicomSeriesRecord, scan_staged_study

SeriesClassification = Literal["processable", "metadata-only", "rejected"]


@dataclass(frozen=True)
class ProfiledSeries:
    record: DicomSeriesRecord
    classification: SeriesClassification
    reason_code: str
    reason_message: str


@dataclass(frozen=True)
class StudyProfile:
    study_root: Path
    series: tuple[ProfiledSeries, ...]


def _classify_series(record: DicomSeriesRecord) -> tuple[SeriesClassification, str, str]:
    description = f"{record.series_description} {record.protocol_name}".lower()
    image_type = " ".join(record.image_type).lower()

    if record.modality != "MR":
        return "metadata-only", "non-mr-modality", "Series is not an MR acquisition"
    if "localizer" in description or "scout" in description:
        return "metadata-only", "localizer", "Localizer series is retained only for traceability"
    if "scanned document" in description or "secondary" in image_type:
        return "metadata-only", "derived-document", "Derived or document-only object excluded from processing"
    if "derived" in image_type and "primary" not in image_type:
        return "metadata-only", "derived-series", "Derived MR series excluded from primary processing"

    supported_terms = ("t1", "stir", "t2")
    if not any(term in description for term in supported_terms):
        return "rejected", "unsupported-series-family", "Series does not match the supported Phase 1 MRI families"
    if record.pixel_spacing is None:
        return "rejected", "missing-geometry", "Series is missing pixel spacing metadata"

    return "processable", "supported-mr-series", "Series matches the supported Phase 1 MRI families"


def profile_staged_study(study_root: str | Path) -> StudyProfile:
    root = Path(study_root)
    series = tuple(
        ProfiledSeries(
            record=record,
            classification=classification,
            reason_code=reason_code,
            reason_message=reason_message,
        )
        for record in scan_staged_study(root)
        for classification, reason_code, reason_message in [_classify_series(record)]
    )
    return StudyProfile(study_root=root, series=series)


def summarize_profile(profile: StudyProfile) -> dict[str, Any]:
    counts = {"processable": 0, "metadata-only": 0, "rejected": 0}
    for series in profile.series:
        counts[series.classification] += 1
    return {"study_root": str(profile.study_root), "counts": counts}
