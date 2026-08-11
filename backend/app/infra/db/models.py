from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, Date, DateTime, ForeignKey, String, Text
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from app.infra.db.base import Base, utc_now


class Study(Base):
    __tablename__ = "studies"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    public_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), unique=True, default=uuid4, index=True)
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
