"""Synchronous longitudinal comparison orchestrator.

Resolves two studies' uploaded NIfTI scan + tumor mask paths from the
Artifact table and calls `ml.inference.compare_studies` with the masks
provided so segmentation is bypassed (no model weights required for the
demo). Writes the resulting `comparison.json` to disk under
`derived/comparisons/{comparison_id}/` and returns a normalised payload
ready for the API layer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from app.core.config import get_settings
from app.infra.db.models import Artifact, Comparison, Study
from app.infra.db.session import create_session_factory
from app.modules.artifacts.storage import resolve_artifact_location

logger = logging.getLogger(__name__)


class ComparisonError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass(frozen=True)
class _StudyAssets:
    internal_id: int
    public_id: str
    scan_absolute_path: Path
    mask_absolute_path: Path | None
    acquired_at: date | None


def _resolve_study_assets(session, study_public_id: str) -> _StudyAssets:
    try:
        parsed = UUID(study_public_id)
    except ValueError as exc:
        raise ComparisonError(400, "studyId must be a valid UUID") from exc

    study = session.query(Study).filter(Study.public_id == parsed).one_or_none()
    if study is None:
        raise ComparisonError(404, f"study not found: {study_public_id}")

    scan_artifact = (
        session.query(Artifact)
        .filter(Artifact.study_id == study.id, Artifact.artifact_kind == "nifti-source")
        .order_by(Artifact.id.desc())
        .first()
    )
    if scan_artifact is None:
        raise ComparisonError(
            422,
            f"study {study_public_id} has no nifti-source artifact (only NIfTI uploads can be compared)",
        )

    mask_artifact = (
        session.query(Artifact)
        .filter(
            Artifact.study_id == study.id,
            Artifact.artifact_kind == "tumor-mask-input",
        )
        .order_by(Artifact.id.desc())
        .first()
    )

    scan_location = resolve_artifact_location(
        scan_artifact.storage_root,  # type: ignore[arg-type]
        scan_artifact.relative_path,
    )
    mask_location = (
        resolve_artifact_location(
            mask_artifact.storage_root,  # type: ignore[arg-type]
            mask_artifact.relative_path,
        )
        if mask_artifact is not None
        else None
    )

    return _StudyAssets(
        internal_id=study.id,
        public_id=str(study.public_id),
        scan_absolute_path=Path(scan_location.absolute_path),
        mask_absolute_path=(
            Path(mask_location.absolute_path) if mask_location is not None else None
        ),
        acquired_at=study.acquired_at,
    )



def _build_inference_config():
    """Build an InferenceConfig pointed at base-models only (nnunet)."""

    from ml.inference.config import load_config  # local import: heavy module

    cfg = load_config(None)
    settings = get_settings()
    cache_root = (
        Path(settings.inference_cache_dir)
        if settings.inference_cache_dir
        else Path(settings.storage_root) / "cache"
    ).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    overrides: dict[str, Any] = {
        "enabled_models": ("nnunet",),
        "cache_dir": cache_root,
    }
    return cfg.with_(**overrides)


def run_longitudinal_comparison(
    *,
    baseline_study_id: str,
    followup_study_id: str,
) -> dict[str, Any]:
    if baseline_study_id == followup_study_id:
        raise ComparisonError(400, "baseline and follow-up study IDs must differ")

    session_factory = create_session_factory()
    with session_factory() as session:
        baseline = _resolve_study_assets(session, baseline_study_id)
        followup = _resolve_study_assets(session, followup_study_id)

    settings = get_settings()
    comparison_id = str(uuid4())
    output_relative_path = f"comparisons/{comparison_id}"
    output_location = resolve_artifact_location("derived", output_relative_path)
    output_location.absolute_path.mkdir(parents=True, exist_ok=True)

    try:
        cfg = _build_inference_config()
        from ml.inference.pipeline.longitudinal import compare_studies
    except Exception as exc:  # pragma: no cover - missing optional deps
        raise ComparisonError(
            500,
            f"ml.inference is not available on this backend: {exc}",
        ) from exc

    logger.info(
        "Running longitudinal comparison",
        extra={
            "comparison_id": comparison_id,
            "baseline_study_id": baseline.public_id,
            "followup_study_id": followup.public_id,
        },
    )

    try:
        result = compare_studies(
            baseline=baseline.scan_absolute_path,
            followup=followup.scan_absolute_path,
            cfg=cfg,
            date_a=baseline.acquired_at,
            date_b=followup.acquired_at,
            output_dir=Path(output_location.absolute_path),
            baseline_mask=baseline.mask_absolute_path,
            followup_mask=followup.mask_absolute_path,
            use_cache=True,
        )
    except Exception as exc:
        logger.exception("compare_studies failed")
        raise ComparisonError(500, f"comparison failed: {exc}") from exc

    summary = result.summary()

    json_path = Path(output_location.absolute_path) / "comparison.json"
    json_path.write_text(json.dumps(summary, default=str, indent=2))

    metrics = summary.get("metrics", {})
    interpretation = summary.get("interpretation", {})
    registration = summary.get("registration", {})
    notes_field = summary.get("notes", "")
    notes_list = (
        [n for n in str(notes_field).split("\n") if n.strip()] if notes_field else []
    )

    metrics_payload: dict[str, Any] = {
        "volume_a_cm3": float(metrics.get("volume_a_cm3", 0.0) or 0.0),
        "volume_b_cm3": float(metrics.get("volume_b_cm3", 0.0) or 0.0),
        "delta_cm3": float(metrics.get("delta_cm3", 0.0) or 0.0),
        "pct_change": float(metrics.get("pct_change", 0.0) or 0.0),
        "dice_overlap": _safe_float(metrics.get("dice_overlap")),
        "hd95_mm": _safe_float(metrics.get("hd95_mm")),
        "recist_a_mm": _safe_float(metrics.get("recist_a_mm")),
        "recist_b_mm": _safe_float(metrics.get("recist_b_mm")),
        "recist_ratio": _safe_float(metrics.get("recist_ratio")),
        "growth_rate_cm3_per_day": _safe_float(metrics.get("growth_rate_cm3_per_day")),
        "registration_ncc": _safe_float(
            metrics.get("registration_ncc")
            or registration.get("ncc_after")
        ),
        "vol_delta_ci_half_cm3": _safe_float(metrics.get("vol_delta_ci_half_cm3")),
        "method": registration.get("method"),
        "backend": registration.get("backend"),
        "did_resegment": bool(metrics.get("did_resegment", False)),
    }

    try:
        with session_factory() as session:
            db_comparison = Comparison(
                public_id=UUID(comparison_id),
                study_a_id=baseline.internal_id,
                study_b_id=followup.internal_id,
                volume_a=metrics_payload["volume_a_cm3"],
                volume_b=metrics_payload["volume_b_cm3"],
                delta_cm3=metrics_payload["delta_cm3"],
                pct_change=metrics_payload["pct_change"],
                dice_overlap=metrics_payload["dice_overlap"],
                hd95_mm=metrics_payload["hd95_mm"],
                growth_rate_cm3_per_day=metrics_payload["growth_rate_cm3_per_day"],
                interpretation_flag=(
                    interpretation.get("label")
                    if isinstance(interpretation, dict)
                    else None
                ),
                recist_ratio=metrics_payload["recist_ratio"],
                vol_delta_ci_half_cm3=metrics_payload["vol_delta_ci_half_cm3"],
                registration_ncc=metrics_payload["registration_ncc"],
                comparison_metadata=summary,
            )
            session.add(db_comparison)
            session.commit()
    except Exception as exc:
        logger.warning("Failed to persist comparison to database: %s", exc)

    return {
        "comparison_id": comparison_id,
        "baseline_study_id": baseline.public_id,
        "followup_study_id": followup.public_id,
        "baseline_acquired_at": baseline.acquired_at,
        "followup_acquired_at": followup.acquired_at,
        "metrics": metrics_payload,
        "interpretation": (
            interpretation.get("label")
            if isinstance(interpretation, dict)
            else None
        ),
        "notes": notes_list,
        "output_relative_path": output_relative_path,
    }



def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    if result in (float("inf"), float("-inf")):
        return None
    return result
