"""
benchmark/p01.py – P01 longitudinal benchmark harness.

Runs the full inference pipeline on the bundled P01 dataset
(baseline + fu1..fu4) and reports:

  * Per-model Dice / IoU / HD95 / volume per timepoint
  * Ensemble Dice vs ground truth per timepoint
  * Longitudinal ComparisonResult for every baseline↔follow-up pair
  * Wall-clock totals per stage
  * A volume-vs-time chart (PNG) comparing ensemble volumes to ground-truth volumes

Usage
-----
    from ml.inference.benchmark.p01 import run_p01_benchmark
    result = run_p01_benchmark(Path("data/P01"), Path("runs/"))
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ml.inference.config import InferenceConfig
from ml.inference.io import load_nifti_mask
from ml.inference.longitudinal.metrics import (
    compute_dice,
    compute_hausdorff_95,
    compute_iou,
    compute_volume_cm3,
)
from ml.inference.pipeline.longitudinal import compare_studies
from ml.inference.pipeline.segment import segment_study

logger = logging.getLogger(__name__)

TIMEPOINTS = ["baseline", "fu1", "fu2", "fu3", "fu4"]


def run_p01_benchmark(
    data_root: Path | str,
    output_root: Path | str,
    cfg: Optional[InferenceConfig] = None,
    *,
    modality: str = "t1c",
    use_gt_masks: bool = False,
) -> dict:
    """
    Run segmentation + longitudinal comparison over P01.

    Args:
        data_root:  Path to the `P01` folder (must contain `BraTS/{baseline,fu1..}/`
                    and `tumor segmentation/`).
        output_root: Where to write per-run CSVs, JSON, and plots.
        cfg:        Optional InferenceConfig override.
        modality:   Which BraTS modality to feed adapters (t1c default).
        use_gt_masks: If True, skip segmentation entirely and run the longitudinal
                    algorithm directly on the ground-truth masks. Useful as a
                    sanity check for registration + metric code in isolation.

    Returns:
        Dict with `segmentation`, `longitudinal`, and `elapsed_s` entries.
    """
    cfg = cfg or InferenceConfig()
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    brats_root = data_root / "BraTS"
    gt_root = data_root / "tumor segmentation"
    if not brats_root.exists():
        raise FileNotFoundError(brats_root)
    if not gt_root.exists():
        raise FileNotFoundError(gt_root)

    t0 = time.perf_counter()

    # -------- Collect per-timepoint paths --------
    inputs: Dict[str, Path] = {}
    gts: Dict[str, Path] = {}
    for tp in TIMEPOINTS:
        vol_path = brats_root / tp / f"{modality}.nii.gz"
        gt_path = gt_root / f"P01_tumor_mask_{tp}.nii.gz"
        if vol_path.exists():
            inputs[tp] = vol_path
        else:
            logger.warning("P01 benchmark: missing input %s", vol_path)
        if gt_path.exists():
            gts[tp] = gt_path
        else:
            logger.warning("P01 benchmark: missing GT mask %s", gt_path)

    seg_rows = _run_segmentation_pass(
        inputs, gts, cfg, output_root, use_gt_masks=use_gt_masks
    )
    long_rows = _run_longitudinal_pass(
        inputs, gts, cfg, output_root, use_gt_masks=use_gt_masks
    )

    # -------- Write CSVs --------
    _write_csv(output_root / "segmentation_leaderboard.csv", seg_rows)
    _write_csv(output_root / "longitudinal_results.csv", long_rows)

    # -------- Volume-vs-time plot (best effort) --------
    try:
        _plot_volume_curve(seg_rows, output_root / "volume_curve.png")
    except Exception as exc:  # pragma: no cover
        logger.warning("volume curve plot failed: %s", exc)

    out = {
        "segmentation": seg_rows,
        "longitudinal": long_rows,
        "output_root": str(output_root),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    with open(output_root / "summary.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


# ---------------------------------------------------------------------------
# Segmentation pass
# ---------------------------------------------------------------------------


def _run_segmentation_pass(
    inputs: Dict[str, Path],
    gts: Dict[str, Path],
    cfg: InferenceConfig,
    output_root: Path,
    *,
    use_gt_masks: bool,
) -> List[dict]:
    rows: List[dict] = []
    for tp in TIMEPOINTS:
        if tp not in inputs:
            continue
        vol_path = inputs[tp]
        gt = load_nifti_mask(gts[tp]) if tp in gts else None

        if use_gt_masks:
            # Skip segmentation entirely – record GT volumes only.
            if gt is None:
                continue
            gt_vol = compute_volume_cm3(gt.data, gt.spacing)
            rows.append({
                "timepoint": tp,
                "model": "ground_truth",
                "dice": 1.0,
                "iou": 1.0,
                "hd95_mm": 0.0,
                "volume_cm3": gt_vol,
                "runtime_s": 0.0,
                "agreement_level": "",
                "note": "GT mask used directly",
            })
            continue

        seg = segment_study(
            vol_path,
            cfg,
            output_dir=output_root / "segmentation" / tp,
            use_cache=True,
        )

        # Align GT to the preprocessed grid if needed.
        gt_aligned = _align_mask_to_shape(gt, seg.ensemble_mask.shape) if gt is not None else None

        # Per-model rows
        for name, mask in seg.per_model_masks.items():
            row = {
                "timepoint": tp,
                "model": name,
                "dice": _safe_dice(mask, gt_aligned),
                "iou": _safe_iou(mask, gt_aligned),
                "hd95_mm": _safe_hd95(mask, gt_aligned, seg.preprocessed_spacing),
                "volume_cm3": seg.per_model_volumes_cm3.get(name, 0.0),
                "runtime_s": seg.per_model_runtime_s.get(name, 0.0),
                "agreement_level": seg.panel_agreement.level,
                "note": seg.adapter_meta.get(name, {}).get("error", ""),
            }
            rows.append(row)

        # Ensemble row
        rows.append({
            "timepoint": tp,
            "model": "ensemble",
            "dice": _safe_dice(seg.ensemble_mask, gt_aligned),
            "iou": _safe_iou(seg.ensemble_mask, gt_aligned),
            "hd95_mm": _safe_hd95(seg.ensemble_mask, gt_aligned, seg.preprocessed_spacing),
            "volume_cm3": seg.ensemble_volume_cm3,
            "runtime_s": seg.elapsed_s,
            "agreement_level": seg.panel_agreement.level,
            "note": f"strategy={cfg.ensemble_strategy}",
        })

        # GT row for reference
        if gt is not None:
            rows.append({
                "timepoint": tp,
                "model": "ground_truth",
                "dice": 1.0,
                "iou": 1.0,
                "hd95_mm": 0.0,
                "volume_cm3": compute_volume_cm3(gt.data, gt.spacing),
                "runtime_s": 0.0,
                "agreement_level": "",
                "note": "",
            })
    return rows


# ---------------------------------------------------------------------------
# Longitudinal pass
# ---------------------------------------------------------------------------


def _run_longitudinal_pass(
    inputs: Dict[str, Path],
    gts: Dict[str, Path],
    cfg: InferenceConfig,
    output_root: Path,
    *,
    use_gt_masks: bool,
) -> List[dict]:
    if "baseline" not in inputs:
        logger.warning("No baseline available – skipping longitudinal pass")
        return []

    rows: List[dict] = []
    base = inputs["baseline"]
    for tp in ["fu1", "fu2", "fu3", "fu4"]:
        if tp not in inputs:
            continue

        kwargs = {}
        if use_gt_masks:
            if "baseline" in gts and tp in gts:
                kwargs["baseline_mask"] = gts["baseline"]
                kwargs["followup_mask"] = gts[tp]
            else:
                logger.warning("GT masks missing for pair baseline↔%s – skipping", tp)
                continue

        result = compare_studies(
            base,
            inputs[tp],
            cfg,
            output_dir=output_root / "longitudinal" / f"baseline_vs_{tp}",
            **kwargs,
        )
        m = result.metrics
        rows.append({
            "pair": f"baseline_vs_{tp}",
            "volume_a_cm3": round(m.volume_a_cm3, 4),
            "volume_b_cm3": round(m.volume_b_cm3, 4),
            "delta_cm3": round(m.delta_cm3, 4),
            "pct_change": round(m.pct_change, 2),
            "dice_overlap": round(m.dice_overlap, 4),
            "hd95_mm": round(m.hd95_mm, 3) if np.isfinite(m.hd95_mm) else float("inf"),
            "recist_a_mm": round(m.recist_a_mm, 2),
            "recist_b_mm": round(m.recist_b_mm, 2),
            "recist_ratio": round(m.recist_ratio, 4),
            "growth_rate_cm3_per_day": round(m.growth_rate_cm3_per_day, 4),
            "registration_ncc_before": round(result.registration_ncc_before, 4),
            "registration_ncc_after": round(result.registration_ncc_after, 4),
            "registration_method": result.registration_method,
            "registration_backend": result.registration_backend,
            "did_resegment": m.did_resegment,
            "ci_half_cm3": round(m.vol_delta_ci_half_cm3, 4),
            "interpretation_level": result.interpretation.level,
            "interpretation_label": result.interpretation.label,
            "elapsed_s": round(result.elapsed_s, 2),
        })
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _align_mask_to_shape(
    gt_vol, target_shape: tuple
):
    """Center-pad/crop a mask Volume to match the preprocessed grid shape."""
    if gt_vol is None:
        return None
    if gt_vol.data.shape == target_shape:
        return gt_vol.data.astype(np.uint8)
    src = gt_vol.data.astype(np.uint8)
    out = np.zeros(target_shape, dtype=np.uint8)
    slicers_src = []
    slicers_dst = []
    for s_sz, d_sz in zip(src.shape, target_shape):
        if s_sz >= d_sz:
            start = (s_sz - d_sz) // 2
            slicers_src.append(slice(start, start + d_sz))
            slicers_dst.append(slice(0, d_sz))
        else:
            start = (d_sz - s_sz) // 2
            slicers_src.append(slice(0, s_sz))
            slicers_dst.append(slice(start, start + s_sz))
    out[tuple(slicers_dst)] = src[tuple(slicers_src)]
    return out


def _safe_dice(pred, gt):
    if gt is None:
        return None
    try:
        return round(float(compute_dice(pred, gt)), 4)
    except Exception:
        return None


def _safe_iou(pred, gt):
    if gt is None:
        return None
    try:
        return round(float(compute_iou(pred, gt)), 4)
    except Exception:
        return None


def _safe_hd95(pred, gt, spacing):
    if gt is None:
        return None
    try:
        val = float(compute_hausdorff_95(pred, gt, spacing))
        if not np.isfinite(val):
            return None
        return round(val, 3)
    except Exception:
        return None


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot_volume_curve(rows: List[dict], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return

    order = {tp: i for i, tp in enumerate(TIMEPOINTS)}
    by_model: Dict[str, List[tuple]] = {}
    for r in rows:
        tp = r.get("timepoint")
        if tp not in order or "volume_cm3" not in r:
            continue
        by_model.setdefault(r["model"], []).append((order[tp], r["volume_cm3"]))

    if not by_model:
        return

    plt.figure(figsize=(8, 4.5))
    for model_name, points in by_model.items():
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        style = "--" if model_name == "ground_truth" else "-"
        plt.plot(xs, ys, style, marker="o", label=model_name, linewidth=2)
    plt.xticks(list(order.values()), list(order.keys()))
    plt.ylabel("Volume (cm³)")
    plt.xlabel("Timepoint")
    plt.title("P01 tumour volume across timepoints")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
