from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, model_validator


def to_camel(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class PatientCreate(CamelModel):
    pseudonym: str | None = None
    name: str | None = None
    dob: date | None = None
    gender: str | None = None
    diagnosis: str | None = None
    diagnosis_location: str | None = None
    status: str = "active"
    notes: str | None = None
    assigned_physician_id: str | None = None


class PatientUpdate(CamelModel):
    pseudonym: str | None = None
    dob: date | None = None
    gender: str | None = None
    diagnosis: str | None = None
    diagnosis_location: str | None = None
    status: str | None = None
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_null_required_fields(cls, value: Any) -> Any:
        if isinstance(value, dict):
            aliases = {"pseudonym", "status"}
            null_fields = sorted(field for field in aliases if field in value and value[field] is None)
            if null_fields:
                raise ValueError(f"{', '.join(null_fields)} cannot be null")
        return value


class AssignedDoctorResponse(CamelModel):
    id: str
    name: str
    email: str
    role: str
    assigned_at: datetime


class PatientStudyItemResponse(CamelModel):
    study_id: str
    study_instance_uid: str
    source_kind: str
    staging_status: str
    acquired_at: date | None = None
    created_at: datetime


class PatientResponse(CamelModel):
    id: str
    pseudonym: str
    name: str | None = None
    dob: date | None = None
    gender: str | None = None
    diagnosis: str | None = None
    diagnosis_location: str | None = None
    status: str
    notes: str | None = None
    scan_count: int = 0
    last_scan_date: date | None = None
    assigned_physician_id: str | None = None
    created_at: datetime
    updated_at: datetime


class PatientDetailResponse(PatientResponse):
    assigned_doctors: list[AssignedDoctorResponse] = []
    studies: list[PatientStudyItemResponse] = []


class AssignmentRequest(CamelModel):
    doctor_id: str
