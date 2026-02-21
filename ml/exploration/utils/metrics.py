"""
metrics.py – Segmentation evaluation metrics for OncoFlow exploration.

Provides:
  - Dice coefficient (binary and multi-label)
  - IoU / Jaccard index
  - Hausdorff distance (95th percentile)
  - Volumetric metrics (cm³, % change)
  - Agreement score across N models (for ensemble evaluation)
  - A summary DataFrame builder for the benchmark report
"""

import time
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import binary_erosion, label as scipy_label


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def dice_coefficient(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Compute binary Dice coefficient.

    Args:
        pred: Predicted binary mask (0/1).
        gt:   Ground-truth binary mask (0/1).
        smooth: Laplace smoothing to avoid 0/0.

    Returns:
        Dice score in [0, 1].
    """
    pred = (pred > 0.5).astype(bool)
    gt = (gt > 0.5).astype(bool)
    intersection = (pred & gt).sum()
    return float(2.0 * intersection + smooth) / float(pred.sum() + gt.sum() + smooth)


def iou_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Compute Intersection-over-Union (Jaccard index).

    Returns:
        IoU score in [0, 1].
    """
    pred = (pred > 0.5).astype(bool)
    gt = (gt > 0.5).astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(intersection + smooth) / float(union + smooth)


def hausdorff_distance_95(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """
    Compute 95th-percentile Hausdorff distance between two binary masks.

    Uses surface voxel extraction (boundary erosion method).
    Falls back to np.inf if either mask is empty.

    Args:
        pred:    Predicted binary mask.
        gt:      Ground-truth binary mask.
        spacing: Voxel spacing in mm (x, y, z).

    Returns:
        HD95 in mm.
    """
    pred = (pred > 0.5).astype(bool)
    gt = (gt > 0.5).astype(bool)

    if not pred.any() or not gt.any():
        return float("inf")

    # Surface extraction via boundary erosion
    def get_surface(mask):
        eroded = binary_erosion(mask)
        return mask & ~eroded

    pred_surf = get_surface(pred)
    gt_surf = get_surface(gt)

    pred_pts = np.argwhere(pred_surf) * np.array(spacing)
    gt_pts = np.argwhere(gt_surf) * np.array(spacing)

    # Compute point-to-surface distances (subsample for speed if large)
    max_pts = 2000
    if len(pred_pts) > max_pts:
        idx = np.random.choice(len(pred_pts), max_pts, replace=False)
        pred_pts = pred_pts[idx]
    if len(gt_pts) > max_pts:
        idx = np.random.choice(len(gt_pts), max_pts, replace=False)
        gt_pts = gt_pts[idx]

    # Pairwise distances (pred→gt and gt→pred)
    from scipy.spatial import cKDTree
    tree_gt = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)

    d_pred_to_gt, _ = tree_gt.query(pred_pts)
    d_gt_to_pred, _ = tree_pred.query(gt_pts)

    all_dists = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_dists, 95))


# ---------------------------------------------------------------------------
# Volumetric metrics
# ---------------------------------------------------------------------------

def compute_volume_cm3(mask: np.ndarray, spacing_mm: Tuple[float, float, float]) -> float:
    """
    Compute tumor volume in cm³ from a binary mask and voxel spacing.

    Args:
        mask:       Binary mask array.
        spacing_mm: Voxel spacing (x, y, z) in mm.

    Returns:
        Volume in cm³.
    """
    voxel_vol_mm3 = float(np.prod(spacing_mm))
    n_voxels = int((mask > 0.5).sum())
    return (n_voxels * voxel_vol_mm3) / 1000.0


def compute_volume_from_nifti(nifti_path: str) -> float:
    """Load a NIfTI mask and compute volume in cm³."""
    img = nib.load(nifti_path)
    spacing = img.header.get_zooms()[:3]
    mask = img.get_fdata()
    return compute_volume_cm3(mask, spacing)


def volume_delta(vol_a: float, vol_b: float) -> Dict[str, float]:
    """
    Compute longitudinal volume change between two timepoints.

    Returns:
        {'delta_cm3': ..., 'pct_change': ...}
    """
    delta = vol_b - vol_a
    pct = (delta / vol_a * 100.0) if vol_a > 0 else 0.0
    return {"delta_cm3": round(delta, 4), "pct_change": round(pct, 2)}


# ---------------------------------------------------------------------------
# Ensemble agreement
# ---------------------------------------------------------------------------

def pairwise_dice_matrix(masks: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Compute pairwise Dice matrix between N model masks.

    Args:
        masks: Dict mapping model_name → binary numpy array.

    Returns:
        DataFrame of shape (N, N) with Dice scores.
    """
    names = list(masks.keys())
    mat = np.zeros((len(names), len(names)))
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            mat[i, j] = dice_coefficient(masks[n1], masks[n2])
    return pd.DataFrame(mat, index=names, columns=names)


def agreement_score(masks: Dict[str, np.ndarray], ensemble: np.ndarray) -> Dict[str, float]:
    """
    Compute each model's Dice vs the ensemble mask, and mean agreement score.

    Args:
        masks:    Dict model_name → binary mask.
        ensemble: Ensemble binary mask.

    Returns:
        Dict with per-model dice_vs_ensemble and mean_agreement.
    """
    result = {}
    dices = []
    for name, mask in masks.items():
        d = dice_coefficient(mask, ensemble)
        result[f"{name}_dice_vs_ensemble"] = round(d, 4)
        dices.append(d)
    result["mean_agreement"] = round(float(np.mean(dices)), 4)
    return result


# ---------------------------------------------------------------------------
# Benchmark result tracking
# ---------------------------------------------------------------------------

class BenchmarkTracker:
    """
    Collects per-run metrics and builds a comparison DataFrame.

    Usage:
        tracker = BenchmarkTracker()
        tracker.add(model="nnunet", timepoint="baseline",
                    pred=mask, gt=gt_mask, spacing=(1,1,1),
                    inference_s=3.2, vram_gb=4.1)
        df = tracker.to_dataframe()
    """

    def __init__(self):
        self._rows: List[Dict] = []

    def add(
        self,
        model: str,
        timepoint: str,
        pred: np.ndarray,
        gt: np.ndarray,
        spacing: Tuple[float, float, float],
        inference_s: float = 0.0,
        vram_gb: float = 0.0,
        extra: Optional[Dict] = None,
    ) -> None:
        """Record one prediction result."""
        row = {
            "model": model,
            "timepoint": timepoint,
            "dice": round(dice_coefficient(pred, gt), 4),
            "iou": round(iou_score(pred, gt), 4),
            "volume_pred_cm3": round(compute_volume_cm3(pred, spacing), 4),
            "volume_gt_cm3": round(compute_volume_cm3(gt, spacing), 4),
            "inference_s": round(inference_s, 2),
            "vram_gb": round(vram_gb, 2),
        }
        try:
            row["hd95_mm"] = round(hausdorff_distance_95(pred, gt, spacing), 2)
        except Exception:
            row["hd95_mm"] = None
        if extra:
            row.update(extra)
        self._rows.append(row)

    def add_mock(
        self,
        model: str,
        timepoint: str,
        note: str = "mock – GPU/model not available",
    ) -> None:
        """Record a placeholder row when inference was not possible."""
        self._rows.append({
            "model": model,
            "timepoint": timepoint,
            "dice": None,
            "iou": None,
            "volume_pred_cm3": None,
            "volume_gt_cm3": None,
            "inference_s": None,
            "vram_gb": None,
            "hd95_mm": None,
            "note": note,
        })

    def to_dataframe(self) -> pd.DataFrame:
        """Return all recorded results as a DataFrame."""
        return pd.DataFrame(self._rows)

    def summary(self) -> pd.DataFrame:
        """Return mean metrics grouped by model (ignoring mock rows)."""
        df = self.to_dataframe()
        numeric_cols = ["dice", "iou", "volume_pred_cm3", "inference_s", "vram_gb", "hd95_mm"]
        return (
            df[df["dice"].notna()]
            .groupby("model")[numeric_cols]
            .mean()
            .round(4)
            .sort_values("dice", ascending=False)
        )

    def __repr__(self) -> str:
        return f"BenchmarkTracker({len(self._rows)} records)"


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------

class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
        if self.label:
            print(f"[Timer] {self.label}: {self.elapsed:.3f}s")
