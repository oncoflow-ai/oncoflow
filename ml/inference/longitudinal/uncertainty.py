"""
longitudinal/uncertainty.py – jackknife confidence interval on the volume delta.

Implements Stage 5 of IMPLEMENTATION_PLAN.md Step 4.7. Given per-model
volumes at timepoint A and B, we compute the leave-one-out jackknife estimate
of (vol_B - vol_A) and its 95 % half-width. Used by the interpretation flag
to surface "high model disagreement" cases.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def jackknife_volume_ci(
    vols_a: Dict[str, float],
    vols_b: Dict[str, float],
    z: float = 1.96,
) -> Tuple[float, float]:
    """
    Jackknife 95% CI on the volume delta.

    Args:
        vols_a: per-model volumes at timepoint A.
        vols_b: per-model volumes at timepoint B.
        z: Gaussian critical value (default 1.96 for 95%).

    Returns:
        (delta_mean, ci_half_width)  – both in the same units as the input.
    """
    common = [k for k in vols_a.keys() if k in vols_b]
    if len(common) < 2:
        # Need at least 2 models to compute variance.
        if common:
            mean_delta = vols_b[common[0]] - vols_a[common[0]]
        else:
            mean_delta = 0.0
        return float(mean_delta), 0.0

    a = np.array([vols_a[k] for k in common], dtype=np.float64)
    b = np.array([vols_b[k] for k in common], dtype=np.float64)
    n = len(common)

    delta_mean = float(b.mean() - a.mean())

    jk = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        jk[i] = b[mask].mean() - a[mask].mean()

    # Standard jackknife variance estimator
    jk_mean = jk.mean()
    var = ((n - 1) / n) * ((jk - jk_mean) ** 2).sum()
    se = float(np.sqrt(var))
    return delta_mean, float(z * se)
