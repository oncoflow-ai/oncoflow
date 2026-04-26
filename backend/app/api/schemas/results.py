from __future__ import annotations

from app.api.schemas.jobs import CamelModel


class ArtifactRefResponse(CamelModel):
    artifact_kind: str
    storage_root: str
    relative_path: str


class LesionMeasurementResponse(CamelModel):
    volume_mm3: float
    longest_diameter_mm: float


class StoredLesionResultResponse(CamelModel):
    lesion_id: str
    bounding_box: dict[str, int]
    measurements: LesionMeasurementResponse
    mask_artifact: ArtifactRefResponse
    review_artifacts: list[ArtifactRefResponse]
    metadata: dict[str, object] | None = None


class StoredCaseResultResponse(CamelModel):
    study_id: str
    result_artifact: ArtifactRefResponse
    lesions: list[StoredLesionResultResponse]
    needs_review: bool
    case_qc_reasons: list[str]
