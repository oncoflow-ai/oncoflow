"""
Shared pytest fixtures.

All fixtures are synthetic – no models, no weights, no network. Tests that need
the real P01 dataset must guard with @pytest.mark.skipif(not P01_AVAILABLE, ...).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the repository root importable so `ml.inference` resolves.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.inference.io import Volume  # noqa: E402


P01_ROOT = _REPO_ROOT / "data" / "P01"
P01_AVAILABLE = (P01_ROOT / "BraTS" / "baseline" / "t1c.nii.gz").exists()


@pytest.fixture
def tmp_cache_dir(tmp_path) -> Path:
    d = tmp_path / "oncoflow_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def synthetic_volume() -> Volume:
    """A 40^3 float32 volume with a Gaussian blob centred in the middle."""
    shape = (40, 40, 40)
    zz, yy, xx = np.meshgrid(
        np.arange(shape[2]), np.arange(shape[1]), np.arange(shape[0]), indexing="ij"
    )
    cx, cy, cz = 20, 20, 20
    d = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    data = np.exp(-d / (2 * 5.0 ** 2)).astype(np.float32)
    affine = np.eye(4, dtype=np.float32)
    return Volume(data=data, affine=affine, spacing=(1.0, 1.0, 1.0))


@pytest.fixture
def synthetic_mask(synthetic_volume) -> np.ndarray:
    return (synthetic_volume.data > 0.3).astype(np.uint8)


@pytest.fixture
def two_overlapping_masks():
    """Two 20^3 binary masks that overlap on ~50% of voxels."""
    a = np.zeros((20, 20, 20), dtype=np.uint8)
    b = np.zeros_like(a)
    a[5:15, 5:15, 5:15] = 1
    b[8:18, 5:15, 5:15] = 1
    return a, b


@pytest.fixture
def p01_available() -> bool:
    return P01_AVAILABLE


@pytest.fixture
def p01_root() -> Path:
    return P01_ROOT
