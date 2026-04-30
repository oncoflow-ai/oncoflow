from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class StagedStudyArtifacts:
    study_public_id: UUID
    archive_artifact_id: int
    extracted_artifact_id: int
    archive_relative_path: str
    extracted_relative_path: str
