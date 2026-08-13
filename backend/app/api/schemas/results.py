from __future__ import annotations

from datetime import date, datetime

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
    metadata: dict[str, object] | None = None


class StudyListItemResponse(CamelModel):
    study_id: str
    source_kind: str
    source_label: str | None = None
    acquired_at: date | None = None
    created_at: datetime
    job_status: str
    has_results: bool


class ComparisonMetricsResponse(CamelModel):
    volume_a_cm3: float
    volume_b_cm3: float
    delta_cm3: float
    pct_change: float
    dice_overlap: float | None = None
    hd95_mm: float | None = None
    recist_a_mm: float | None = None
    recist_b_mm: float | None = None
    recist_ratio: float | None = None
    growth_rate_cm3_per_day: float | None = None
    registration_ncc: float | None = None
    vol_delta_ci_half_cm3: float | None = None
    method: str | None = None
    backend: str | None = None
    did_resegment: bool | None = None


class ComparisonResponse(CamelModel):
    comparison_id: str
    baseline_study_id: str
    followup_study_id: str
    baseline_acquired_at: date | None = None
    followup_acquired_at: date | None = None
    metrics: ComparisonMetricsResponse
    interpretation: str | None = None
    notes: list[str] = []
    output_relative_path: str
