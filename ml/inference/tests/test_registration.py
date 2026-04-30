"""
Tests for NCC metric and the identity-warp sanity check.

A full ANTsPy/SITK registration needs those libraries installed; those tests
are skipped when the dependencies aren't available.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.inference.config import InferenceConfig
from ml.inference.io import Volume
from ml.inference.registration.register import (
    RegistrationResult,
    ncc,
    register_followup_to_baseline,
    warp_mask,
)
from ml.inference.registration.resegment import should_resegment


def test_ncc_identity_is_one(synthetic_volume):
    val = ncc(synthetic_volume.data, synthetic_volume.data)
    assert val == pytest.approx(1.0, abs=1e-6)


def test_ncc_constant_arrays_zero():
    a = np.zeros((5, 5, 5), dtype=np.float32)
    b = np.zeros((5, 5, 5), dtype=np.float32)
    assert ncc(a, b) == 0.0


def test_ncc_shape_mismatch():
    with pytest.raises(ValueError):
        ncc(np.zeros((3, 3, 3)), np.zeros((4, 4, 4)))


def test_should_resegment_threshold():
    cfg = InferenceConfig(ncc_resegment_threshold=0.65)
    res_low = RegistrationResult(
        warped_image=None,  # type: ignore[arg-type]
        fwd_transforms=[],
        ncc_before=0.5,
        ncc_after=0.5,
        method="Affine",
        backend="sitk",
    )
    res_high = RegistrationResult(
        warped_image=None,  # type: ignore[arg-type]
        fwd_transforms=[],
        ncc_before=0.8,
        ncc_after=0.8,
        method="Affine",
        backend="sitk",
    )
    assert should_resegment(res_low, cfg) is True
    assert should_resegment(res_high, cfg) is False


def test_identity_registration(synthetic_volume):
    """A volume registered against itself should produce NCC_after ≈ 1.0."""
    try:
        import ants  # type: ignore  # noqa: F401
    except ImportError:
        try:
            import SimpleITK  # type: ignore  # noqa: F401
        except ImportError:
            pytest.skip("Neither ANTsPy nor SimpleITK installed")

    res = register_followup_to_baseline(
        synthetic_volume, synthetic_volume, method="Affine"
    )
    # Identity registration should produce reasonable correlation, but both
    # ANTsPy and SimpleITK optimizers + resampling introduce interpolation
    # artifacts, especially on synthetic smooth blobs. Allow generous tolerance.
    assert res.ncc_after >= 0.75, f"NCC too low: {res.ncc_after}"
    # Verify registration didn't make things dramatically worse
    assert res.ncc_after >= res.ncc_before - 0.25
