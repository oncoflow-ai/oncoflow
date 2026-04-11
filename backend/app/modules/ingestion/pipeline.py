from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from app.infra.db.models import Series, Study
from app.modules.artifacts.catalog import record_derived_artifact
from app.modules.artifacts.storage import resolve_artifact_location
from app.modules.ingestion.profiling import StudyProfile, profile_staged_study
from app.modules.ingestion.validation import validate_study_profile
from app.infra.imaging.dcm2niix_wrapper import convert_dicom_series


@dataclass(frozen=True)
class PipelineResult:
    study_public_id: UUID
    processable_series: int
    derived_artifact_count: int


def process_staged_study(*, session, study_public_id: UUID, extracted_relative_path: str) -> tuple[PipelineResult, StudyProfile]:
    study = session.query(Study).filter(Study.public_id == study_public_id).one()
    extracted_root = resolve_artifact_location("raw", extracted_relative_path).absolute_path
    profile = profile_staged_study(extracted_root)
    validation_messages = validate_study_profile(profile)
    if validation_messages:
        raise ValueError(validation_messages[0].message)

    derived_count = 0
    processable_series = 0
    for profiled in profile.series:
        record = profiled.record
        series_row = Series(
            study_id=study.id,
            series_instance_uid=record.series_instance_uid,
            modality=record.modality,
            series_description=record.series_description,
            protocol_name=record.protocol_name,
            classification=profiled.classification,
            scanner_vendor=record.manufacturer,
            source_metadata={
                "reason_code": profiled.reason_code,
                "reason_message": profiled.reason_message,
                "image_type": list(record.image_type),
            },
        )
        session.add(series_row)
        session.flush()

        if profiled.classification != "processable":
            continue

        processable_series += 1
        relative_dir = f"studies/{study.public_id}/series/{series_row.id}"
        output_dir = resolve_artifact_location("derived", relative_dir).absolute_path
        result = convert_dicom_series(record, output_dir, filename_stem="volume")
        record_derived_artifact(
            session,
            study_id=study.id,
            series_id=series_row.id,
            artifact_kind="nifti-volume",
            relative_path=f"{relative_dir}/volume.nii.gz",
            metadata={"geometry": result["geometry"], "converter": result["converter"]},
        )
        record_derived_artifact(
            session,
            study_id=study.id,
            series_id=series_row.id,
            artifact_kind="nifti-sidecar",
            relative_path=f"{relative_dir}/volume.json",
            metadata={"geometry": result["geometry"]},
        )
        record_derived_artifact(
            session,
            study_id=study.id,
            series_id=series_row.id,
            artifact_kind="conversion-log",
            relative_path=f"{relative_dir}/volume.log",
            metadata={"converter": result["converter"]},
        )
        derived_count += 3

    session.flush()
    return PipelineResult(
        study_public_id=study_public_id,
        processable_series=processable_series,
        derived_artifact_count=derived_count,
    ), profile
