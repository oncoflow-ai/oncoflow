from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable
from uuid import UUID

from app.infra.db.models import Series, Study
from app.modules.artifacts.catalog import record_derived_artifact
from app.modules.artifacts.storage import resolve_artifact_location
from app.modules.ingestion.profiling import StudyProfile, profile_staged_study
from app.modules.ingestion.validation import validate_study_profile
from app.infra.imaging.dcm2niix_wrapper import convert_dicom_series
from app.infra.imaging.dicom_anonymizer import anonymize_dicom_series


@dataclass(frozen=True)
class PipelineResult:
    study_public_id: UUID
    processable_series: int
    derived_artifact_count: int


class DuplicateStudyInstanceUidError(ValueError):
    pass


def process_staged_study(*, session, study_public_id: UUID, extracted_relative_path: str) -> tuple[PipelineResult, StudyProfile]:
    return process_staged_study_with_stages(
        session=session,
        study_public_id=study_public_id,
        extracted_relative_path=extracted_relative_path,
    )


def process_staged_study_with_stages(
    *,
    session,
    study_public_id: UUID,
    extracted_relative_path: str,
    stage_callback: Callable[[str], None] | None = None,
) -> tuple[PipelineResult, StudyProfile]:
    study = session.query(Study).filter(Study.public_id == study_public_id).one()
    extracted_root = resolve_artifact_location("raw", extracted_relative_path).absolute_path

    stage_callback = stage_callback or (lambda _: None)
    stage_callback("profiling")
    profile = profile_staged_study(extracted_root)
    study_instance_uids = {
        series.record.study_instance_uid
        for series in profile.series
        if series.record.study_instance_uid
    }
    if len(study_instance_uids) > 1:
        raise DuplicateStudyInstanceUidError("Uploaded series contain multiple StudyInstanceUID values")
    first_processable = next(
        (series.record for series in profile.series if series.classification == "processable"),
        None,
    )

    stage_callback("validating")
    validation_messages = validate_study_profile(profile)
    if validation_messages:
        raise ValueError(validation_messages[0].message)

    if first_processable and first_processable.study_instance_uid:
        existing_study = (
            session.query(Study)
            .filter(
                Study.study_instance_uid == first_processable.study_instance_uid,
                Study.id != study.id,
            )
            .one_or_none()
        )
        if existing_study is not None:
            raise DuplicateStudyInstanceUidError(
                "StudyInstanceUID already exists for another uploaded study"
            )
        study.study_instance_uid = first_processable.study_instance_uid
        study.staging_status = "profiled"

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

        stage_callback("anonymizing")
        relative_dir = f"studies/{study.public_id}/series/{series_row.id}"
        anonymized_dir = resolve_artifact_location("anonymized", relative_dir).absolute_path
        anonymized_files = anonymize_dicom_series(
            record=record,
            output_dir=Path(anonymized_dir),
            patient_uuid=study.patient_public_id,
        )
        record = replace(record, files=anonymized_files)

        stage_callback("converting")
        processable_series += 1
        output_dir = resolve_artifact_location("derived", relative_dir).absolute_path
        result = convert_dicom_series(record, output_dir, filename_stem="volume")

        stage_callback("persisting")
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

    study.staging_status = "processed"
    session.flush()
    return PipelineResult(
        study_public_id=study_public_id,
        processable_series=processable_series,
        derived_artifact_count=derived_count,
    ), profile
