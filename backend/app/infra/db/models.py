from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Float, JSON, Date, DateTime, ForeignKey, String, Text
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid


from app.infra.db.base import Base, utc_now


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    public_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), unique=True, default=uuid4, index=True)
    pseudonym: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    dob: Mapped[date | None] = mapped_column(Date, nullable=True)
    gender: Mapped[str | None] = mapped_column(String(32), nullable=True)
    diagnosis: Mapped[str | None] = mapped_column(String(255), nullable=True)
    diagnosis_location: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    studies: Mapped[list["Study"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )
    doctor_assignments: Mapped[list["Assignment"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )
    reports: Mapped[list["Report"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )



class Assignment(Base):
    __tablename__ = "assignments"
    __table_args__ = (UniqueConstraint("doctor_id", "patient_id"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    doctor_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    assigned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    doctor: Mapped["User"] = relationship(back_populates="patient_assignments")
    patient: Mapped["Patient"] = relationship(back_populates="doctor_assignments")


class Study(Base):
    __tablename__ = "studies"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    public_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), unique=True, default=uuid4, index=True)
    patient_id: Mapped[int | None] = mapped_column(ForeignKey("patients.id"), nullable=True, index=True)
    patient_public_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), default=uuid4, index=True)
    study_instance_uid: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    source_kind: Mapped[str] = mapped_column(String(64))
    source_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    staging_status: Mapped[str] = mapped_column(String(64))
    acquired_at: Mapped[date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    patient: Mapped[Patient | None] = relationship(back_populates="studies")
    series: Mapped[list["Series"]] = relationship(
        back_populates="study",
        cascade="all, delete-orphan",
    )
    artifacts: Mapped[list["Artifact"]] = relationship(
        back_populates="study",
        cascade="all, delete-orphan",
    )
    jobs: Mapped[list["Job"]] = relationship(
        back_populates="study",
        cascade="all, delete-orphan",
    )
    result_sets: Mapped[list["StudyResult"]] = relationship(
        back_populates="study",
        cascade="all, delete-orphan",
    )


class Series(Base):
    __tablename__ = "series"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    study_id: Mapped[int] = mapped_column(ForeignKey("studies.id"), index=True)
    series_instance_uid: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    modality: Mapped[str] = mapped_column(String(16))
    series_description: Mapped[str | None] = mapped_column(String(255), nullable=True)
    protocol_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    classification: Mapped[str] = mapped_column(String(64))
    scanner_vendor: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    study: Mapped[Study] = relationship(back_populates="series")
    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="series")


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    study_id: Mapped[int] = mapped_column(ForeignKey("studies.id"), index=True)
    series_id: Mapped[int | None] = mapped_column(ForeignKey("series.id"), nullable=True, index=True)
    artifact_kind: Mapped[str] = mapped_column(String(64))
    storage_root: Mapped[str] = mapped_column(String(64))
    relative_path: Mapped[str] = mapped_column(Text)
    source_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    study: Mapped[Study] = relationship(back_populates="artifacts")
    series: Mapped[Series | None] = relationship(back_populates="artifacts")


class StudyResult(Base):
    __tablename__ = "study_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    study_id: Mapped[int] = mapped_column(ForeignKey("studies.id"), index=True)
    result_kind: Mapped[str] = mapped_column(String(64), default="single-scan")
    needs_review: Mapped[bool] = mapped_column(default=False)
    summary_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    study: Mapped[Study] = relationship(back_populates="result_sets")
    lesions: Mapped[list["StoredLesionResult"]] = relationship(
        back_populates="study_result",
        cascade="all, delete-orphan",
    )


class StoredLesionResult(Base):
    __tablename__ = "lesion_results"
    __table_args__ = (UniqueConstraint("study_result_id", "lesion_id"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    study_result_id: Mapped[int] = mapped_column(ForeignKey("study_results.id"), index=True)
    study_id: Mapped[int] = mapped_column(ForeignKey("studies.id"), index=True)
    lesion_id: Mapped[str] = mapped_column(String(255))
    measurement_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    bounding_box: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    artifact_refs: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    study_result: Mapped[StudyResult] = relationship(back_populates="lesions")
    study: Mapped[Study] = relationship()


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    study_id: Mapped[int] = mapped_column(ForeignKey("studies.id"), index=True)
    public_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), unique=True, default=uuid4, index=True)
    job_type: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), index=True)
    stage: Mapped[str] = mapped_column(String(32))
    failure_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    study: Mapped[Study] = relationship(back_populates="jobs")
    events: Mapped[list["JobEvent"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
    )


class JobEvent(Base):
    __tablename__ = "job_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"), index=True)
    status: Mapped[str] = mapped_column(String(32))
    stage: Mapped[str] = mapped_column(String(32))
    event_type: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)

    job: Mapped[Job] = relationship(back_populates="events")


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    public_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), unique=True, default=uuid4, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    patient_assignments: Mapped[list["Assignment"]] = relationship(
        back_populates="doctor",
        cascade="all, delete-orphan",
    )


class Comparison(Base):
    __tablename__ = "comparisons"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    public_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), unique=True, default=uuid4, index=True)
    study_a_id: Mapped[int] = mapped_column(ForeignKey("studies.id"), index=True)
    study_b_id: Mapped[int] = mapped_column(ForeignKey("studies.id"), index=True)
    volume_a: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume_b: Mapped[float | None] = mapped_column(Float, nullable=True)
    delta_cm3: Mapped[float | None] = mapped_column(Float, nullable=True)
    pct_change: Mapped[float | None] = mapped_column(Float, nullable=True)
    dice_overlap: Mapped[float | None] = mapped_column(Float, nullable=True)
    hd95_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    growth_rate_cm3_per_day: Mapped[float | None] = mapped_column(Float, nullable=True)
    interpretation_flag: Mapped[str | None] = mapped_column(String(255), nullable=True)
    recist_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    vol_delta_ci_half_cm3: Mapped[float | None] = mapped_column(Float, nullable=True)
    registration_ncc: Mapped[float | None] = mapped_column(Float, nullable=True)
    comparison_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    study_a: Mapped[Study] = relationship(foreign_keys=[study_a_id])
    study_b: Mapped[Study] = relationship(foreign_keys=[study_b_id])
    reports: Mapped[list["Report"]] = relationship(
        back_populates="comparison",
        cascade="all, delete-orphan",
    )


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    public_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), unique=True, default=uuid4, index=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    comparison_id: Mapped[int | None] = mapped_column(ForeignKey("comparisons.id"), nullable=True, index=True)
    pdf_artifact_id: Mapped[int | None] = mapped_column(ForeignKey("artifacts.id"), nullable=True, index=True)
    signature: Mapped[str | None] = mapped_column(Text, nullable=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    patient: Mapped[Patient] = relationship(back_populates="reports")
    comparison: Mapped[Comparison | None] = relationship(back_populates="reports")
    pdf_artifact: Mapped[Artifact | None] = relationship()


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    actor_id: Mapped[str] = mapped_column(String(255), index=True)
    action: Mapped[str] = mapped_column(String(128), index=True)
    resource_id: Mapped[str] = mapped_column(String(255), index=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)


