"""
registration/resegment.py – NCC-gated re-segmentation fallback (Stage 3).

If registration quality (NCC_after) falls below `ncc_resegment_threshold`,
we trust the warp less and re-run the segmentation pipeline on the REGISTERED
follow-up volume rather than simply warping the old mask. Matches Step 4.7
Stage 3 in IMPLEMENTATION_PLAN.md.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

from ml.inference.config import InferenceConfig
from ml.inference.io import Volume
from ml.inference.registration.register import RegistrationResult, warp_mask

logger = logging.getLogger(__name__)


def should_resegment(
    reg: RegistrationResult, cfg: InferenceConfig
) -> bool:
    """Return True when the NCC_after is below the configured threshold."""
    return float(reg.ncc_after) < float(cfg.ncc_resegment_threshold)


def get_followup_mask(
    followup_mask: Volume,
    registration: RegistrationResult,
    reference: Volume,
    cfg: InferenceConfig,
    *,
    resegment_fn: Optional[Callable[[Volume, InferenceConfig], Volume]] = None,
) -> Volume:
    """
    Decide whether to warp the existing follow-up mask or re-segment the
    registered volume.

    Args:
        followup_mask: follow-up segmentation in follow-up native space.
        registration: outcome of registering follow-up → baseline space.
        reference: baseline volume (defines the target voxel grid).
        cfg: InferenceConfig.
        resegment_fn: callback that accepts the *registered* follow-up volume
            and returns a fresh mask Volume. When None, we always warp.

    Returns:
        Mask Volume in the baseline coordinate space.
    """
    if should_resegment(registration, cfg) and resegment_fn is not None:
        logger.warning(
            "Registration NCC_after=%.3f < %.3f → re-segmenting registered volume",
            registration.ncc_after, cfg.ncc_resegment_threshold,
        )
        fresh = resegment_fn(registration.warped_image, cfg)
        return fresh

    return warp_mask(followup_mask, registration, reference)
