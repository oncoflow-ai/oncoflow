"""
longitudinal/interpretation.py – RECIST-style interpretation flags.

Turns the Stage-4 numeric metrics into the human-readable flag defined in
IMPLEMENTATION_PLAN.md Step 4.7 Stage 5:

    registration_ncc < 0.55  → Registration failed – manual review
    CI / |delta| > 15 %      → High model uncertainty
    pct_change > +25 %       → Progressive disease (PD)
    pct_change ≤ -25 %       → Partial/complete response
    |pct_change| ≤ 5 %       → Stable disease
    otherwise                → Minor change
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Level = Literal[
    "registration_failed",
    "high_uncertainty",
    "progressive_disease",
    "stable_disease",
    "response",
    "minor_change",
]


@dataclass(frozen=True)
class InterpretationFlag:
    level: Level
    label: str              # short human string, e.g. "Progressive disease"
    message: str            # longer explanation

    def as_dict(self) -> dict:
        return {"level": self.level, "label": self.label, "message": self.message}


_LABELS = {
    "registration_failed": "Registration failed",
    "high_uncertainty": "High model uncertainty",
    "progressive_disease": "Progressive disease",
    "stable_disease": "Stable disease",
    "response": "Response (>=25% reduction)",
    "minor_change": "Minor change",
}


def interpret(
    *,
    delta_cm3: float,
    pct_change: float,
    registration_ncc: float,
    ci_half_cm3: float,
    ncc_fail_threshold: float = 0.55,
    ci_threshold: float = 0.15,
) -> InterpretationFlag:
    """Apply the thresholds in order of clinical priority."""

    if registration_ncc < ncc_fail_threshold:
        return InterpretationFlag(
            level="registration_failed",
            label=_LABELS["registration_failed"],
            message=(
                f"NCC_after={registration_ncc:.3f} < {ncc_fail_threshold}; "
                "warp quality is poor – manual radiologist review required."
            ),
        )

    denom = abs(delta_cm3) + 1e-6
    if ci_half_cm3 / denom > ci_threshold and abs(delta_cm3) > 0.5:
        return InterpretationFlag(
            level="high_uncertainty",
            label=_LABELS["high_uncertainty"],
            message=(
                f"95% CI half-width {ci_half_cm3:.2f} cm^3 exceeds "
                f"{ci_threshold*100:.0f}% of |delta|={delta_cm3:.2f} – "
                "models disagree; review recommended."
            ),
        )

    if pct_change >= 25.0:
        return InterpretationFlag(
            level="progressive_disease",
            label=_LABELS["progressive_disease"],
            message=f"Volume grew by {pct_change:.1f}% – RECIST PD equivalent.",
        )

    if pct_change <= -25.0:
        return InterpretationFlag(
            level="response",
            label=_LABELS["response"],
            message=f"Volume shrank by {abs(pct_change):.1f}% – partial/complete response.",
        )

    if abs(pct_change) <= 5.0:
        return InterpretationFlag(
            level="stable_disease",
            label=_LABELS["stable_disease"],
            message=f"Volume change within +/-5% ({pct_change:+.1f}%) – stable disease.",
        )

    return InterpretationFlag(
        level="minor_change",
        label=_LABELS["minor_change"],
        message=f"Minor change ({pct_change:+.1f}%) – continue monitoring.",
    )
