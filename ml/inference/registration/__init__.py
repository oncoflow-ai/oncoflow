"""Image registration + NCC-gated resegmentation (Step 4.7 Stage 2–3)."""

from ml.inference.registration.register import (
    register_followup_to_baseline,
    warp_mask,
    warp_volume,
    RegistrationResult,
    ncc,
)
from ml.inference.registration.resegment import (
    should_resegment,
    get_followup_mask,
)

__all__ = [
    "register_followup_to_baseline",
    "warp_mask",
    "warp_volume",
    "RegistrationResult",
    "ncc",
    "should_resegment",
    "get_followup_mask",
]
