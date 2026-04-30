"""
P01 end-to-end smoke test.

Runs `compare_studies()` on `baseline` vs `fu1` using the provided GROUND-TRUTH
masks as the segmentation stand-in – so the test runs in CI without any model
weights. Exercises:

  * Preprocessing
  * Registration (ANTsPy or SimpleITK fallback)
  * Mask warping
  * Stage-4 metrics (volume, Dice, HD95, RECIST)
  * Stage-5 interpretation flag
  * JSON artefact output

Skipped automatically when the P01 data folder is missing.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ml.inference.config import InferenceConfig
from ml.inference.pipeline.longitudinal import compare_studies


def _require_deps():
    try:
        import SimpleITK  # noqa: F401
    except ImportError:
        try:
            import ants  # type: ignore  # noqa: F401
        except ImportError:
            pytest.skip("neither SimpleITK nor ANTsPy installed – cannot register")


def test_p01_longitudinal_smoke(p01_available, p01_root, tmp_path):
    if not p01_available:
        pytest.skip("P01 data not present")
    _require_deps()

    base_vol = p01_root / "BraTS" / "baseline" / "t1c.nii.gz"
    fu_vol = p01_root / "BraTS" / "fu1" / "t1c.nii.gz"
    base_mask = p01_root / "tumor segmentation" / "P01_tumor_mask_baseline.nii.gz"
    fu_mask = p01_root / "tumor segmentation" / "P01_tumor_mask_fu1.nii.gz"

    for p in (base_vol, fu_vol, base_mask, fu_mask):
        if not p.exists():
            pytest.skip(f"missing required file: {p}")

    cfg = InferenceConfig(
        enabled_models=(),              # no segmentation models – rely on GT
        n4_bias_correction=False,       # speed
        skull_strip=False,
        isotropic_spacing_mm=1.0,
        cache_dir=tmp_path / "cache",
    )

    result = compare_studies(
        base_vol,
        fu_vol,
        cfg,
        output_dir=tmp_path / "run",
        baseline_mask=base_mask,
        followup_mask=fu_mask,
        use_cache=False,
    )

    summary = result.summary()
    assert "metrics" in summary
    m = result.metrics
    assert m.volume_a_cm3 > 0.0
    assert m.volume_b_cm3 > 0.0
    assert 0.0 <= m.dice_overlap <= 1.0
    assert np.isfinite(m.hd95_mm)
    assert m.registration_method in {"Rigid", "Affine", "SyN"}

    # Same-head pair → NCC_after should be reasonably high.
    assert result.registration_ncc_after >= 0.7, (
        f"registration quality too low: {result.registration_ncc_after}"
    )

    # JSON artefacts written
    assert (tmp_path / "run" / "comparison.json").exists()
    with open(tmp_path / "run" / "comparison.json") as f:
        loaded = json.load(f)
    assert loaded["metrics"]["volume_a_cm3"] == pytest.approx(
        round(m.volume_a_cm3, 4), abs=1e-3
    )
