from __future__ import annotations

from typing import Any

from app.infra.db.models import Artifact


def record_derived_artifact(
    session,
    *,
    study_id: int,
    series_id: int,
    artifact_kind: str,
    relative_path: str,
    metadata: dict[str, Any],
) -> Artifact:
    artifact = Artifact(
        study_id=study_id,
        series_id=series_id,
        artifact_kind=artifact_kind,
        storage_root="derived",
        relative_path=relative_path,
        source_metadata=metadata,
    )
    session.add(artifact)
    session.flush()
    return artifact
