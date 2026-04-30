"""
_utils_compat.py – shim that re-exports utilities from `ml/exploration/utils/`
so we don't duplicate metric/io code.

The exploration notebooks add `ml/exploration/utils` to `sys.path` at
notebook-launch time; the inference package is importable as `ml.inference`
from the repository root, so we explicitly add the utils folder here too.
"""

from __future__ import annotations

import sys
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parent.parent / "exploration" / "utils"
if _UTILS_DIR.exists() and str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

# Re-export the bits we actually use so callers can `from ml.inference._utils_compat import ...`
try:  # pragma: no cover - import-time shim
    from metrics import (  # type: ignore
        dice_coefficient,
        iou_score,
        hausdorff_distance_95,
        compute_volume_cm3,
        volume_delta,
        pairwise_dice_matrix,
        agreement_score,
        BenchmarkTracker,
        Timer,
    )
except Exception:  # pragma: no cover
    # Fallback for environments where the exploration utils aren't present
    # (CI with only the inference package). Local implementations live in
    # ml/inference/longitudinal/metrics.py which duplicates the math where
    # necessary.
    dice_coefficient = None  # type: ignore
    iou_score = None  # type: ignore
    hausdorff_distance_95 = None  # type: ignore
    compute_volume_cm3 = None  # type: ignore
    volume_delta = None  # type: ignore
    pairwise_dice_matrix = None  # type: ignore
    agreement_score = None  # type: ignore
    BenchmarkTracker = None  # type: ignore
    Timer = None  # type: ignore
