"""
longitudinal/metrics.py – Volume, Dice, HD95, RECIST, growth-rate metrics.

Implements Stage 4 of IMPLEMENTATION_PLAN.md Step 4.7. Every metric accepts
arrays + spacing (mm/voxel); none of them load files directly.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


Spacing = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


def compute_volume_cm3(mask: np.ndarray, spacing: Spacing) -> float:
    voxel_vol_mm3 = float(np.prod(spacing))
    n = int((mask > 0).sum())
    return n * voxel_vol_mm3 / 1000.0


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------


def compute_dice(a: np.ndarray, b: np.ndarray, smooth: float = 1e-8) -> float:
    a_b = (a > 0).astype(bool)
    b_b = (b > 0).astype(bool)
    inter = np.logical_and(a_b, b_b).sum()
    denom = a_b.sum() + b_b.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * inter + smooth) / (denom + smooth))


def compute_iou(a: np.ndarray, b: np.ndarray, smooth: float = 1e-8) -> float:
    a_b = (a > 0).astype(bool)
    b_b = (b > 0).astype(bool)
    inter = np.logical_and(a_b, b_b).sum()
    union = np.logical_or(a_b, b_b).sum()
    if union == 0:
        return 1.0
    return float((inter + smooth) / (union + smooth))


# ---------------------------------------------------------------------------
# Hausdorff-95
# ---------------------------------------------------------------------------


def compute_hausdorff_95(
    a: np.ndarray, b: np.ndarray, spacing: Spacing = (1.0, 1.0, 1.0)
) -> float:
    """95th-percentile symmetric Hausdorff distance in millimetres."""
    from scipy.ndimage import binary_erosion
    from scipy.spatial import cKDTree

    a_b = (a > 0).astype(bool)
    b_b = (b > 0).astype(bool)
    if not a_b.any() or not b_b.any():
        return float("inf")

    a_surf = a_b & ~binary_erosion(a_b)
    b_surf = b_b & ~binary_erosion(b_b)

    a_pts = np.argwhere(a_surf) * np.asarray(spacing)
    b_pts = np.argwhere(b_surf) * np.asarray(spacing)

    max_pts = 4000
    rng = np.random.default_rng(0)
    if len(a_pts) > max_pts:
        a_pts = a_pts[rng.choice(len(a_pts), max_pts, replace=False)]
    if len(b_pts) > max_pts:
        b_pts = b_pts[rng.choice(len(b_pts), max_pts, replace=False)]

    tree_a = cKDTree(a_pts)
    tree_b = cKDTree(b_pts)
    d_ab, _ = tree_b.query(a_pts)
    d_ba, _ = tree_a.query(b_pts)
    return float(np.percentile(np.concatenate([d_ab, d_ba]), 95))


# ---------------------------------------------------------------------------
# RECIST diameter
# ---------------------------------------------------------------------------


def compute_recist_diameter_mm(
    mask: np.ndarray, spacing: Spacing = (1.0, 1.0, 1.0)
) -> float:
    """
    RECIST-1.1 proxy: longest in-plane diameter (mm) of the largest connected
    component, measured axially.

    Not a full 3-D maximum-caliper diameter (that's a known post-MVP item in
    IMPLEMENTATION_PLAN.md note #9); this axial proxy is consistent with the
    plan's current implementation.
    """
    from scipy.ndimage import label

    mask_b = (mask > 0).astype(np.uint8)
    if not mask_b.any():
        return 0.0

    labeled, n = label(mask_b)
    if n == 0:
        return 0.0

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    lesion = (labeled == largest)

    sx, sy, _sz = spacing
    diameters: list = []
    for z in range(lesion.shape[2]):
        plane = lesion[:, :, z]
        if not plane.any():
            continue
        ys, xs = np.where(plane)
        if len(xs) < 2:
            continue
        # Use extent of bounding-box diagonal as a cheap longest-axis proxy.
        dx = (xs.max() - xs.min()) * sx
        dy = (ys.max() - ys.min()) * sy
        diameters.append(float(np.sqrt(dx * dx + dy * dy)))

    return float(max(diameters)) if diameters else 0.0


# ---------------------------------------------------------------------------
# Growth rate
# ---------------------------------------------------------------------------


def compute_growth_rate(
    vol_a: float, vol_b: float, date_a: Optional[date], date_b: Optional[date]
) -> float:
    """cm³/day. Returns 0.0 if dates missing or identical."""
    if date_a is None or date_b is None:
        return 0.0
    days = (date_b - date_a).days
    if days == 0:
        return 0.0
    return (vol_b - vol_a) / days


# ---------------------------------------------------------------------------
# Metrics container
# ---------------------------------------------------------------------------


@dataclass
class LongitudinalMetrics:
    """All numeric metrics produced by Stage 4 of the longitudinal algorithm."""

    volume_a_cm3: float
    volume_b_cm3: float
    delta_cm3: float
    pct_change: float
    dice_overlap: float
    hd95_mm: float
    recist_a_mm: float
    recist_b_mm: float
    recist_ratio: float
    growth_rate_cm3_per_day: float
    registration_ncc: float
    vol_delta_ci_half_cm3: float = 0.0
    registration_method: str = ""
    registration_backend: str = ""
    did_resegment: bool = False
    per_model_volumes_a_cm3: Dict[str, float] = field(default_factory=dict)
    per_model_volumes_b_cm3: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = asdict(self)
        # Round for readability; callers that need full precision can read raw fields.
        for k, v in list(out.items()):
            if isinstance(v, float):
                out[k] = round(v, 4)
        return out
