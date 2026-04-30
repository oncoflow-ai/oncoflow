"""
Tests for preprocessing: RAS orientation and isotropic resample.

N4 and skull-strip need SimpleITK / antspynet respectively and are exercised
by the integration smoke test when available.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.inference.config import InferenceConfig
from ml.inference.io import Volume
from ml.inference.preprocessing import orient_to_ras, resample_isotropic


def test_orient_to_ras_shape_preserved(synthetic_volume):
    out = orient_to_ras(synthetic_volume)
    assert out.shape == synthetic_volume.shape
    assert out.meta.get("orientation") == "RAS"
    assert out.data.sum() > 0  # foreground preserved


def test_resample_isotropic_target_spacing(synthetic_volume):
    """Resample a 2 mm volume to 1 mm – size should roughly double."""
    # Artificially set non-isotropic spacing on a copy
    vol = Volume(
        data=synthetic_volume.data,
        affine=synthetic_volume.affine.copy(),
        spacing=(2.0, 2.0, 2.0),
    )
    try:
        out = resample_isotropic(vol, target_mm=1.0)
    except ImportError:
        pytest.skip("SimpleITK not installed")
    assert all(abs(s - 1.0) < 1e-6 for s in out.spacing)
    # Volume along each axis should approximately double.
    for src_sz, dst_sz in zip(vol.shape, out.shape):
        assert abs(dst_sz - 2 * src_sz) <= 2


def test_volume_foreground_preserved_after_isotropic(synthetic_volume):
    try:
        out = resample_isotropic(synthetic_volume, target_mm=1.0)
    except ImportError:
        pytest.skip("SimpleITK not installed")
    # Total intensity should be roughly preserved (linear interp of a smooth blob)
    src_sum = float(synthetic_volume.data.sum())
    dst_sum = float(out.data.sum())
    rel_err = abs(src_sum - dst_sum) / max(src_sum, 1e-6)
    assert rel_err < 0.25, f"intensity drift too large: {rel_err:.2f}"
