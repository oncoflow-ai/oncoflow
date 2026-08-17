"""Lightweight segmentation pipeline for NIfTI uploads with a pre-computed mask.

Bypasses the DICOM-oriented canonical bundle / ensemble runner used by the
DICOM ingestion path. The provided tumor mask is treated as the segmentation
output: we copy it under the canonical derived path, compute volume +
bounding box / longest diameter directly from the voxel array, and
materialize a StudyResult so that GET /api/v1/results/{study_id} works
unchanged.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from uuid import UUID, uuid4

from app.infra.db.models import Artifact, Study, StoredLesionResult, StudyResult
from app.modules.artifacts.catalog import record_study_artifact
from app.modules.artifacts.storage import resolve_artifact_location

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NiftiMaskMeasurements:
    volume_mm3: float
    longest_diameter_mm: float
    voxel_count: int
    spacing_mm: tuple[float, float, float]
    bounding_box: dict[str, int]


def _ensure_nibabel():
    try:
        import nibabel as nib  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - dep guarded by install profile
        raise RuntimeError(
            "nibabel and numpy are required for the NIfTI demo pipeline. "
            "Install backend with the [ml] extra."
        ) from exc
    return nib, np


def measure_mask(mask_absolute_path: Path) -> NiftiMaskMeasurements:
    """Compute volume, longest-axis diameter, bbox and spacing from a NIfTI mask."""

    nib, np = _ensure_nibabel()
    img = nib.load(str(mask_absolute_path))
    data = np.asarray(img.dataobj)
    occupied = data > 0
    spacing_raw = img.header.get_zooms()[:3]
    spacing = (
        float(spacing_raw[0]) or 1.0,
        float(spacing_raw[1]) or 1.0,
        float(spacing_raw[2]) or 1.0,
    )

    voxel_count = int(occupied.sum())
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    volume_mm3 = float(voxel_count * voxel_volume_mm3)

    if voxel_count == 0:
        bounding_box = {"x_min": 0, "y_min": 0, "z_min": 0, "x_max": 0, "y_max": 0, "z_max": 0}
        return NiftiMaskMeasurements(
            volume_mm3=0.0,
            longest_diameter_mm=0.0,
            voxel_count=0,
            spacing_mm=spacing,
            bounding_box=bounding_box,
        )

    coords = np.argwhere(occupied)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    bounding_box = {
        "x_min": int(mins[0]),
        "y_min": int(mins[1]),
        "z_min": int(mins[2]),
        "x_max": int(maxs[0]),
        "y_max": int(maxs[1]),
        "z_max": int(maxs[2]),
    }

    # 3D bounding-box diagonal in millimetres serves as a fast longest-axis
    # proxy. For a true RECIST measurement we would walk the convex hull of
    # the largest connected component, but the diagonal is a sane upper-bound
    # surrogate for the demo and runs in O(N) instead of O(N^2).
    dx = (bounding_box["x_max"] - bounding_box["x_min"]) * spacing[0]
    dy = (bounding_box["y_max"] - bounding_box["y_min"]) * spacing[1]
    dz = (bounding_box["z_max"] - bounding_box["z_min"]) * spacing[2]
    longest_diameter_mm = math.sqrt(dx * dx + dy * dy + dz * dz)

    return NiftiMaskMeasurements(
        volume_mm3=volume_mm3,
        longest_diameter_mm=float(longest_diameter_mm),
        voxel_count=voxel_count,
        spacing_mm=spacing,
        bounding_box=bounding_box,
    )


def materialize_nifti_study_with_mask(
    *,
    session,
    study_public_id: UUID,
    mask_source_absolute_path: Path,
    runner_metadata: dict | None = None,
    result_metadata: dict | None = None,
) -> int:
    """Persist segmentation artifacts + StudyResult for a NIfTI demo upload."""

    study = session.query(Study).filter(Study.public_id == study_public_id).one()
    measurements = measure_mask(mask_source_absolute_path)
    segmentation_run_id = str(uuid4())
    lesion_id = "lesion-001"

    mask_relative_path = (
        f"studies/{study.public_id}/lesions/{lesion_id}.nii.gz"
    )
    mask_location = resolve_artifact_location("derived", mask_relative_path)
    mask_location.absolute_path.parent.mkdir(parents=True, exist_ok=True)
    if mask_source_absolute_path.resolve() != mask_location.absolute_path.resolve():
        copy2(mask_source_absolute_path, mask_location.absolute_path)

    geometry = {
        "spacing_mm": list(measurements.spacing_mm),
        "voxel_count": measurements.voxel_count,
    }
    runner_meta = runner_metadata or {
        "model_id": "ground-truth-mask",
        "runner_version": "demo-1",
        "execution_backend": "passthrough",
        "warnings": [],
    }

    record_study_artifact(
        session,
        study_id=study.id,
        artifact_kind="segmentation-mask",
        relative_path=mask_relative_path,
        metadata={
            "segmentation_run_id": segmentation_run_id,
            "lesion_id": lesion_id,
            "bounding_box": measurements.bounding_box,
            "geometry": geometry,
            "runner": runner_meta,
            "needs_review": False,
            "slot_provenance": [],
            "inference": {"source": "user-uploaded-mask"},
        },
    )

    case_relative_path = f"studies/{study.public_id}/segmentation/result.json"
    case_location = resolve_artifact_location("derived", case_relative_path)
    case_location.absolute_path.parent.mkdir(parents=True, exist_ok=True)
    case_location.absolute_path.write_text(
        json.dumps(
            {
                "study_id": str(study.public_id),
                "needs_review": False,
                "case_qc_reasons": [],
                "lesion_ids": [lesion_id],
                "runner": runner_meta,
            }
        )
    )
    record_study_artifact(
        session,
        study_id=study.id,
        artifact_kind="segmentation-case-result",
        relative_path=case_relative_path,
        metadata={
            "segmentation_run_id": segmentation_run_id,
            "needs_review": False,
            "case_qc_reasons": [],
            "lesion_count": 1,
            "runner": runner_meta,
        },
    )

    summary_metadata = {
        "segmentation_run_id": segmentation_run_id,
        "case_qc_reasons": [],
        "lesion_count": 1,
        "source": "nifti-demo",
    }
    if result_metadata:
        summary_metadata.update(result_metadata)

    study_result = StudyResult(
        study_id=study.id,
        result_kind="single-scan",
        needs_review=False,
        summary_metadata=summary_metadata,
    )
    session.add(study_result)
    session.flush()

    lesion_row = StoredLesionResult(
        study_result_id=study_result.id,
        study_id=study.id,
        lesion_id=lesion_id,
        measurement_payload={
            "volume_mm3": measurements.volume_mm3,
            "longest_diameter_mm": measurements.longest_diameter_mm,
        },
        bounding_box=measurements.bounding_box,
        artifact_refs={
            "mask": {
                "artifact_kind": "segmentation-mask",
                "storage_root": "derived",
                "relative_path": mask_relative_path,
            },
            "review": [],
        },
        result_metadata={
            "runner": runner_meta,
            "spacing_mm": list(measurements.spacing_mm),
            "voxel_count": measurements.voxel_count,
        },
    )
    session.add(lesion_row)

    bundle_relative_path = f"studies/{study.public_id}/results/study-result.json"
    bundle_location = resolve_artifact_location("derived", bundle_relative_path)
    bundle_location.absolute_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_location.absolute_path.write_text(
        json.dumps(
            {
                "study_id": str(study.public_id),
                "study_result_id": study_result.id,
                "lesion_count": 1,
                "lesions": [
                    {
                        "lesion_id": lesion_id,
                        "volume_mm3": measurements.volume_mm3,
                        "longest_diameter_mm": measurements.longest_diameter_mm,
                        "bounding_box": measurements.bounding_box,
                    }
                ],
            }
        )
    )
    record_study_artifact(
        session,
        study_id=study.id,
        artifact_kind="study-result-bundle",
        relative_path=bundle_relative_path,
        metadata={
            "segmentation_run_id": segmentation_run_id,
            "study_result_id": study_result.id,
            "lesion_ids": [lesion_id],
            "source": "nifti-demo",
        },
    )

    session.flush()
    logger.info(
        "Materialized NIfTI demo result",
        extra={
            "study_id": str(study.public_id),
            "study_result_id": study_result.id,
            "volume_mm3": measurements.volume_mm3,
        },
    )
    return study_result.id
