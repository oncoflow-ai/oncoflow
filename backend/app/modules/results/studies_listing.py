"""Read-only study listing for UI dropdowns."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from app.infra.db.models import Artifact, Job, Study
from app.infra.db.session import create_session_factory


@dataclass(frozen=True)
class StudyListItem:
    study_id: str
    source_kind: str
    source_label: str | None
    acquired_at: date | None
    created_at: datetime
    job_status: str
    has_results: bool


def list_studies() -> list[StudyListItem]:
    session_factory = create_session_factory()
    with session_factory() as session:
        studies = session.query(Study).order_by(Study.created_at.desc()).all()
        items: list[StudyListItem] = []
        for study in studies:
            latest_job = (
                session.query(Job)
                .filter(Job.study_id == study.id)
                .order_by(Job.id.desc())
                .first()
            )
            bundle = (
                session.query(Artifact)
                .filter(
                    Artifact.study_id == study.id,
                    Artifact.artifact_kind == "study-result-bundle",
                )
                .first()
            )
            metadata = study.source_metadata or {}
            label = metadata.get("source_label") if isinstance(metadata, dict) else None
            items.append(
                StudyListItem(
                    study_id=str(study.public_id),
                    source_kind=study.source_kind,
                    source_label=label,
                    acquired_at=study.acquired_at,
                    created_at=study.created_at,
                    job_status=latest_job.status if latest_job is not None else "unknown",
                    has_results=bundle is not None,
                )
            )
        return items
