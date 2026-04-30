"""
ensemble/strategies.py – voxel-wise fusion of N binary (or soft) masks.

Implements the strategies listed in IMPLEMENTATION_PLAN.md Step 4.5:
  - majority_vote   (default; robust baseline)
  - union / intersection  (debugging / bracketing)
  - staple          (SimpleITK EM-based probabilistic fusion)
  - confidence_weighted  (mean of soft masks, threshold at 0.5)

All strategies take a dict `{model_name: mask_array}` so callers can
mix-and-match without worrying about ordering.
"""

from __future__ import annotations

import logging
from typing import Dict, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

Strategy = Literal[
    "majority_vote", "union", "intersection", "staple", "confidence_weighted"
]


# ---------------------------------------------------------------------------
# Binary voting strategies
# ---------------------------------------------------------------------------


def majority_vote(masks: Dict[str, np.ndarray]) -> np.ndarray:
    """Voxel accepted if at least half of the models vote for it."""
    if not masks:
        raise ValueError("majority_vote requires at least one mask")
    stack = np.stack([(m > 0).astype(np.uint8) for m in masks.values()], axis=0)
    threshold = len(masks) / 2.0
    return (stack.sum(axis=0) >= threshold).astype(np.uint8)


def union_vote(masks: Dict[str, np.ndarray]) -> np.ndarray:
    """Voxel accepted if ANY model votes for it. Tends to over-segment."""
    stack = np.stack([(m > 0) for m in masks.values()], axis=0)
    return stack.any(axis=0).astype(np.uint8)


def intersection_vote(masks: Dict[str, np.ndarray]) -> np.ndarray:
    """Voxel accepted only if ALL models vote for it. Tends to under-segment."""
    stack = np.stack([(m > 0) for m in masks.values()], axis=0)
    return stack.all(axis=0).astype(np.uint8)


# ---------------------------------------------------------------------------
# STAPLE (SimpleITK)
# ---------------------------------------------------------------------------


def staple_fusion(
    masks: Dict[str, np.ndarray],
    *,
    foreground: int = 1,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    EM-based probabilistic fusion (STAPLE) via SimpleITK.

    Returns a binary mask thresholded at `threshold` on the per-voxel
    foreground probability. Falls back to `majority_vote` if SimpleITK is
    unavailable or raises.
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        logger.warning("STAPLE requested but SimpleITK not available – using majority_vote")
        return majority_vote(masks)

    try:
        sitk_masks = [
            sitk.GetImageFromArray((m > 0).astype(np.uint8))
            for m in masks.values()
        ]
        staple_filter = sitk.STAPLEImageFilter()
        staple_filter.SetForegroundValue(foreground)
        prob = staple_filter.Execute(sitk_masks)
        prob_arr = sitk.GetArrayFromImage(prob).astype(np.float32)
        return (prob_arr >= threshold).astype(np.uint8)
    except Exception as exc:
        logger.warning("STAPLE failed (%s) – falling back to majority_vote", exc)
        return majority_vote(masks)


# ---------------------------------------------------------------------------
# Confidence-weighted
# ---------------------------------------------------------------------------


def confidence_weighted(
    probs: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Weighted mean of soft-mask probabilities, thresholded to a binary mask.

    `probs` maps model_name → float mask in [0, 1]. `weights` assigns a
    scalar weight per model; defaults to uniform.
    """
    if not probs:
        raise ValueError("confidence_weighted requires at least one prob map")

    names = list(probs.keys())
    w = np.array(
        [float(weights.get(n, 1.0)) if weights else 1.0 for n in names],
        dtype=np.float32,
    )
    w = w / (w.sum() + 1e-8)

    stack = np.stack(
        [np.clip(probs[n].astype(np.float32), 0.0, 1.0) for n in names], axis=0
    )
    mean = (stack * w.reshape(-1, 1, 1, 1)).sum(axis=0)
    return (mean >= threshold).astype(np.uint8)


# ---------------------------------------------------------------------------
# Strategy dispatcher
# ---------------------------------------------------------------------------


def fuse(
    strategy: Strategy,
    masks: Dict[str, np.ndarray],
    probs: Optional[Dict[str, np.ndarray]] = None,
    *,
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Dispatch to the chosen fusion strategy."""
    if strategy == "majority_vote":
        return majority_vote(masks)
    if strategy == "union":
        return union_vote(masks)
    if strategy == "intersection":
        return intersection_vote(masks)
    if strategy == "staple":
        return staple_fusion(masks)
    if strategy == "confidence_weighted":
        # Fall back gracefully if no soft masks available.
        if probs and any(p is not None for p in probs.values()):
            usable = {k: v for k, v in probs.items() if v is not None}
            if usable:
                return confidence_weighted(usable, weights=weights)
        return majority_vote(masks)
    raise ValueError(f"Unknown ensemble strategy: {strategy!r}")
