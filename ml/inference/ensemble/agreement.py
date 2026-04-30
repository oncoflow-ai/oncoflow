"""
ensemble/agreement.py – per-model Dice vs ensemble and panel agreement flag.

Surfaces the numbers clinicians need to decide whether to trust the ensemble
(from IMPLEMENTATION_PLAN.md Step 4.5):

    >=0.90  High agreement     – report automatically
    0.75-0.89 Moderate          – flag for radiologist review
    <0.75   Low agreement     – require manual segmentation check
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np


def _dice(a: np.ndarray, b: np.ndarray, smooth: float = 1e-8) -> float:
    a_b = (a > 0).astype(bool)
    b_b = (b > 0).astype(bool)
    inter = np.logical_and(a_b, b_b).sum()
    denom = a_b.sum() + b_b.sum()
    if denom == 0:
        return 1.0  # both empty – trivially agree
    return float((2.0 * inter + smooth) / (denom + smooth))


@dataclass(frozen=True)
class PanelAgreement:
    """Structured agreement result across the model panel."""

    per_model_dice: Dict[str, float]
    mean_agreement: float
    level: Literal["high", "moderate", "low"]
    models_used: tuple

    def as_dict(self) -> dict:
        return {
            "per_model_dice_vs_ensemble": self.per_model_dice,
            "mean_agreement": self.mean_agreement,
            "agreement_level": self.level,
            "models_used": list(self.models_used),
        }


def agreement_score(
    masks: Dict[str, np.ndarray], ensemble: np.ndarray
) -> PanelAgreement:
    """Compute each model's Dice vs the ensemble + panel-mean agreement."""
    per_model = {name: _dice(mask, ensemble) for name, mask in masks.items()}
    mean_dice = float(np.mean(list(per_model.values()))) if per_model else 1.0
    return PanelAgreement(
        per_model_dice={k: round(v, 4) for k, v in per_model.items()},
        mean_agreement=round(mean_dice, 4),
        level=agreement_flag(mean_dice),
        models_used=tuple(sorted(masks.keys())),
    )


def agreement_flag(mean_dice: float) -> Literal["high", "moderate", "low"]:
    if mean_dice >= 0.90:
        return "high"
    if mean_dice >= 0.75:
        return "moderate"
    return "low"
