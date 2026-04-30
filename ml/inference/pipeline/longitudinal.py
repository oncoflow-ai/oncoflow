"""
pipeline/longitudinal.py – End-to-end longitudinal comparison pipeline.

Wires Stages 1–5 of IMPLEMENTATION_PLAN.md Step 4.7:

    1. Preprocess (via segment_study)
    2. Register follow-up -> baseline (ANTsPy or SimpleITK)
    3. Warp follow-up mask OR re-segment registered volume (NCC-gated)
    4. Compute volume / Dice / HD95 / RECIST / growth-rate
    5. Uncertainty CI + interpretation flag

Inputs are two NIfTI paths (baseline, follow-up). Each is either an intensity
volume (the pipeline segments it) or already a segmented mask (then the
pipeline consumes the mask as the "ensemble mask"). Switch via the
`use_provided_masks` kwarg.

All numeric outputs live in `ComparisonResult` (frozen-friendly dataclass);
every field maps 1:1 to a column planned in the `comparisons` /
`registration_results` DB tables so the backend wrapper can persist directly.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from ml.inference.config import InferenceConfig
from ml.inference.io import (
    Volume,
    load_nifti,
    load_nifti_mask,
    save_nifti,
    cache_dir_for,
)
from ml.inference.longitudinal.interpretation import (
    InterpretationFlag,
    interpret,
)
from ml.inference.longitudinal.metrics import (
    LongitudinalMetrics,
    compute_dice,
    compute_growth_rate,
    compute_hausdorff_95,
    compute_recist_diameter_mm,
    compute_volume_cm3,
)
from ml.inference.longitudinal.uncertainty import jackknife_volume_ci
from ml.inference.pipeline.segment import StudySegmentation, segment_study
from ml.inference.preprocessing import preprocess_from_path
from ml.inference.registration.register import (
    RegistrationResult,
    register_followup_to_baseline,
    warp_mask,
)
from ml.inference.registration.resegment import (
    get_followup_mask,
    should_resegment,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """End-to-end longitudinal comparison between two studies."""

    baseline_path: str
    followup_path: str
    output_dir: str

    metrics: LongitudinalMetrics
    interpretation: InterpretationFlag

    # Stage-level artefacts (paths relative to output_dir unless absolute)
    baseline_ensemble_mask_path: str
    followup_warped_mask_path: str
    registered_volume_path: str

    registration_method: str
    registration_backend: str
    registration_ncc_before: float
    registration_ncc_after: float

    baseline_segmentation: Optional[StudySegmentation] = None
    followup_segmentation: Optional[StudySegmentation] = None

    elapsed_s: float = 0.0
    config_snapshot: dict = field(default_factory=dict)
    notes: str = ""

    def summary(self) -> dict:
        out = {
            "baseline_path": self.baseline_path,
            "followup_path": self.followup_path,
            "output_dir": self.output_dir,
            "metrics": self.metrics.to_dict(),
            "interpretation": self.interpretation.as_dict(),
            "registration": {
                "method": self.registration_method,
                "backend": self.registration_backend,
                "ncc_before": round(self.registration_ncc_before, 4),
                "ncc_after": round(self.registration_ncc_after, 4),
            },
            "paths": {
                "baseline_ensemble_mask": self.baseline_ensemble_mask_path,
                "followup_warped_mask": self.followup_warped_mask_path,
                "registered_volume": self.registered_volume_path,
            },
            "elapsed_s": round(self.elapsed_s, 3),
            "notes": self.notes,
            "config_snapshot": self.config_snapshot,
        }
        return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compare_studies(
    baseline: str | Path,
    followup: str | Path,
    cfg: Optional[InferenceConfig] = None,
    *,
    date_a: Optional[date] = None,
    date_b: Optional[date] = None,
    output_dir: Optional[Path] = None,
    baseline_mask: Optional[str | Path] = None,
    followup_mask: Optional[str | Path] = None,
    use_cache: bool = True,
) -> ComparisonResult:
    """
    Full Stage 1–5 longitudinal pipeline.

    Inputs
    ------
    baseline, followup: NIfTI intensity volumes.
    baseline_mask, followup_mask: if supplied, use these instead of running
        the segmentation pipeline. Useful for tests and for comparing
        ground-truth masks directly.
    date_a, date_b: acquisition dates (for the growth rate).

    Outputs
    -------
    ComparisonResult with numeric metrics and all intermediate NIfTI paths.
    """
    cfg = cfg or InferenceConfig()
    baseline = Path(baseline).expanduser().resolve()
    followup = Path(followup).expanduser().resolve()
    if not baseline.exists():
        raise FileNotFoundError(baseline)
    if not followup.exists():
        raise FileNotFoundError(followup)

    t0 = time.perf_counter()

    run_dir = output_dir or cache_dir_for(
        cfg.cache_dir,
        baseline,
        "longitudinal",
        str(followup.name),
        cfg.backend,
        cfg.ensemble_strategy,
    )
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Stage 1: Segment (or load) + preprocess each timepoint
    # ---------------------------------------------------------------
    baseline_seg: Optional[StudySegmentation] = None
    followup_seg: Optional[StudySegmentation] = None

    if baseline_mask is not None:
        base_preproc, _ = preprocess_from_path(baseline, cfg)
        base_mask_vol = _load_and_align_mask(baseline_mask, base_preproc)
        baseline_ensemble = base_mask_vol.data.astype(np.uint8)
        per_model_vols_a: Dict[str, float] = {}
    else:
        baseline_seg = segment_study(
            baseline,
            cfg,
            output_dir=run_dir / "baseline",
            use_cache=use_cache,
        )
        base_preproc, _ = preprocess_from_path(baseline, cfg)
        baseline_ensemble = baseline_seg.ensemble_mask
        per_model_vols_a = baseline_seg.per_model_volumes_cm3

    if followup_mask is not None:
        followup_preproc, _ = preprocess_from_path(followup, cfg)
        fu_mask_vol = _load_and_align_mask(followup_mask, followup_preproc)
        followup_ensemble = fu_mask_vol.data.astype(np.uint8)
        per_model_vols_b: Dict[str, float] = {}
    else:
        followup_seg = segment_study(
            followup,
            cfg,
            output_dir=run_dir / "followup",
            use_cache=use_cache,
        )
        followup_preproc, _ = preprocess_from_path(followup, cfg)
        followup_ensemble = followup_seg.ensemble_mask
        per_model_vols_b = followup_seg.per_model_volumes_cm3

    # ---------------------------------------------------------------
    # Stage 2: Register follow-up -> baseline space
    # ---------------------------------------------------------------
    registration: RegistrationResult = register_followup_to_baseline(
        base_preproc,
        followup_preproc,
        method=cfg.registration_type,
    )

    # ---------------------------------------------------------------
    # Stage 3: Warp mask OR re-segment the registered volume
    # ---------------------------------------------------------------
    followup_ensemble_vol = followup_preproc.copy_with(
        followup_ensemble.astype(np.uint8)
    )

    did_resegment = False
    resegment_cb = None
    if followup_mask is None and baseline_seg is not None:
        # Re-segmentation callback: run the same pipeline on the WARPED volume.
        def _resegment(registered_vol: Volume, cfg_inner: InferenceConfig) -> Volume:
            nonlocal did_resegment
            did_resegment = True
            tmp_path = run_dir / "followup_registered_for_reseg.nii.gz"
            save_nifti(registered_vol, tmp_path)
            reseg = segment_study(
                tmp_path,
                cfg_inner,
                output_dir=run_dir / "followup_reseg",
                use_cache=False,
            )
            return registered_vol.copy_with(reseg.ensemble_mask.astype(np.uint8))

        resegment_cb = _resegment

    followup_mask_in_baseline = get_followup_mask(
        followup_ensemble_vol,
        registration,
        base_preproc,
        cfg,
        resegment_fn=resegment_cb,
    )

    # Make sure shapes agree (pad/crop if ITK produced an off-by-one difference).
    fu_mask_arr = _conform_shape(
        followup_mask_in_baseline.data.astype(np.uint8),
        baseline_ensemble.shape,
    )
    base_mask_arr = baseline_ensemble.astype(np.uint8)

    # ---------------------------------------------------------------
    # Stage 4: Change metrics
    # ---------------------------------------------------------------
    spacing = base_preproc.spacing
    vol_a = compute_volume_cm3(base_mask_arr, spacing)
    vol_b = compute_volume_cm3(fu_mask_arr, spacing)
    delta = vol_b - vol_a
    pct = (delta / vol_a * 100.0) if vol_a > 1e-6 else 0.0
    dice_overlap = compute_dice(base_mask_arr, fu_mask_arr)
    try:
        hd95 = compute_hausdorff_95(base_mask_arr, fu_mask_arr, spacing)
    except Exception as exc:
        logger.warning("HD95 failed: %s", exc)
        hd95 = float("inf")
    recist_a = compute_recist_diameter_mm(base_mask_arr, spacing)
    recist_b = compute_recist_diameter_mm(fu_mask_arr, spacing)
    growth = compute_growth_rate(vol_a, vol_b, date_a, date_b)

    # ---------------------------------------------------------------
    # Stage 5: Uncertainty + interpretation
    # ---------------------------------------------------------------
    _delta_mean, ci_half = jackknife_volume_ci(per_model_vols_a, per_model_vols_b)

    metrics = LongitudinalMetrics(
        volume_a_cm3=vol_a,
        volume_b_cm3=vol_b,
        delta_cm3=delta,
        pct_change=pct,
        dice_overlap=dice_overlap,
        hd95_mm=hd95,
        recist_a_mm=recist_a,
        recist_b_mm=recist_b,
        recist_ratio=(recist_b / (recist_a + 1e-6)) if recist_a > 0 else 0.0,
        growth_rate_cm3_per_day=growth,
        registration_ncc=registration.ncc_after,
        vol_delta_ci_half_cm3=ci_half,
        registration_method=registration.method,
        registration_backend=registration.backend,
        did_resegment=did_resegment,
        per_model_volumes_a_cm3=dict(per_model_vols_a),
        per_model_volumes_b_cm3=dict(per_model_vols_b),
    )

    flag = interpret(
        delta_cm3=delta,
        pct_change=pct,
        registration_ncc=registration.ncc_after,
        ci_half_cm3=ci_half,
        ncc_fail_threshold=cfg.ncc_fail_threshold,
    )

    # ---------------------------------------------------------------
    # Persist artefacts
    # ---------------------------------------------------------------
    base_mask_path = run_dir / "baseline_ensemble_mask.nii.gz"
    fu_warped_path = run_dir / "followup_warped_mask.nii.gz"
    registered_vol_path = run_dir / "followup_registered_volume.nii.gz"
    save_nifti(base_preproc.copy_with(base_mask_arr.astype(np.uint8)), base_mask_path)
    save_nifti(base_preproc.copy_with(fu_mask_arr.astype(np.uint8)), fu_warped_path)
    save_nifti(registration.warped_image, registered_vol_path)

    cfg_snapshot = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in asdict(cfg).items()
    }

    result = ComparisonResult(
        baseline_path=str(baseline),
        followup_path=str(followup),
        output_dir=str(run_dir),
        metrics=metrics,
        interpretation=flag,
        baseline_ensemble_mask_path=str(base_mask_path),
        followup_warped_mask_path=str(fu_warped_path),
        registered_volume_path=str(registered_vol_path),
        registration_method=registration.method,
        registration_backend=registration.backend,
        registration_ncc_before=registration.ncc_before,
        registration_ncc_after=registration.ncc_after,
        baseline_segmentation=baseline_seg,
        followup_segmentation=followup_seg,
        elapsed_s=time.perf_counter() - t0,
        config_snapshot=cfg_snapshot,
    )

    with open(run_dir / "comparison.json", "w") as f:
        json.dump(result.summary(), f, indent=2, default=str)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_and_align_mask(mask_path: str | Path, reference: Volume) -> Volume:
    """Load a mask and resample it to the reference voxel grid if needed."""
    m = load_nifti_mask(mask_path)
    if m.shape == reference.shape:
        return m

    # Resample to reference grid via SimpleITK nearest-neighbour.
    import SimpleITK as sitk

    src_img = sitk.GetImageFromArray(np.ascontiguousarray(m.data.transpose(2, 1, 0)))
    src_img.SetSpacing(tuple(float(s) for s in m.spacing))
    src_img.SetOrigin(tuple(float(x) for x in m.affine[:3, 3]))

    ref_img = sitk.GetImageFromArray(
        np.ascontiguousarray(reference.data.transpose(2, 1, 0))
    )
    ref_img.SetSpacing(tuple(float(s) for s in reference.spacing))
    ref_img.SetOrigin(tuple(float(x) for x in reference.affine[:3, 3]))

    resampled = sitk.Resample(
        src_img,
        ref_img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
    )
    arr = sitk.GetArrayFromImage(resampled).transpose(2, 1, 0).astype(np.uint8)
    return Volume(
        data=arr,
        affine=reference.affine.copy(),
        spacing=reference.spacing,
        source_path=str(mask_path),
    )


def _conform_shape(arr: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    if arr.shape == target:
        return arr
    out = np.zeros(target, dtype=arr.dtype)
    slicers_src = []
    slicers_dst = []
    for src_sz, dst_sz in zip(arr.shape, target):
        if src_sz >= dst_sz:
            start = (src_sz - dst_sz) // 2
            slicers_src.append(slice(start, start + dst_sz))
            slicers_dst.append(slice(0, dst_sz))
        else:
            start = (dst_sz - src_sz) // 2
            slicers_src.append(slice(0, src_sz))
            slicers_dst.append(slice(start, start + src_sz))
    out[tuple(slicers_dst)] = arr[tuple(slicers_src)]
    return out
