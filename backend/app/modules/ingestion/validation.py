from __future__ import annotations

from dataclasses import dataclass

from app.modules.ingestion.profiling import StudyProfile


@dataclass(frozen=True)
class ValidationMessage:
    code: str
    message: str


def validate_study_profile(profile: StudyProfile) -> list[ValidationMessage]:
    if not profile.series:
        return [ValidationMessage(code="empty-study", message="No readable DICOM series were found in the staged study")]

    processable = [series for series in profile.series if series.classification == "processable"]
    if not processable:
        return [
            ValidationMessage(
                code="no-processable-series",
                message="Study does not contain a supported T1/T2 STIR MR series for Phase 1 ingestion",
            )
        ]

    return []
