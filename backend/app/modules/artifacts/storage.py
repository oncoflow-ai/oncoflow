from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from app.core.config import get_settings

ArtifactRoot = Literal["raw", "derived"]


@dataclass(frozen=True)
class ArtifactLocation:
    root_kind: ArtifactRoot
    relative_path: str
    absolute_path: Path


def get_storage_roots() -> dict[ArtifactRoot, Path]:
    settings = get_settings()
    base = Path(settings.storage_root).expanduser().resolve()
    return {
        "raw": base / settings.storage_staging_dir,
        "derived": base / "derived",
    }


def ensure_storage_layout() -> dict[ArtifactRoot, Path]:
    roots = get_storage_roots()
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)
    return roots


def normalize_relative_path(relative_path: str) -> str:
    candidate = PurePosixPath(relative_path)
    if candidate.is_absolute():
        raise ValueError("Artifact paths must be relative to a managed storage root")

    normalized = candidate.as_posix()
    if normalized in {"", "."}:
        raise ValueError("Artifact path cannot be empty")

    if any(part in {"..", ""} for part in candidate.parts):
        raise ValueError("Artifact path cannot escape the managed storage root")

    return normalized


def resolve_artifact_location(root_kind: ArtifactRoot, relative_path: str) -> ArtifactLocation:
    roots = ensure_storage_layout()
    normalized = normalize_relative_path(relative_path)
    absolute = (roots[root_kind] / normalized).resolve()
    root = roots[root_kind].resolve()

    if root not in absolute.parents and absolute != root:
        raise ValueError("Artifact path resolved outside the managed storage root")

    return ArtifactLocation(
        root_kind=root_kind,
        relative_path=normalized,
        absolute_path=absolute,
    )
