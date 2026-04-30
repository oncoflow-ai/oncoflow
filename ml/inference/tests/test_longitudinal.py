"""
Tests for longitudinal metrics, uncertainty CI, and interpretation flags.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from ml.inference.longitudinal.interpretation import interpret
from ml.inference.longitudinal.metrics import (
    compute_dice,
    compute_growth_rate,
    compute_hausdorff_95,
    compute_iou,
    compute_recist_diameter_mm,
    compute_volume_cm3,
)
from ml.inference.longitudinal.uncertainty import jackknife_volume_ci


def test_volume_cm3_matches_voxel_count():
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[2:4, 2:4, 2:4] = 1  # 8 voxels
    # 1 mm spacing -> 8 mm^3 -> 0.008 cm^3
    assert compute_volume_cm3(mask, (1.0, 1.0, 1.0)) == pytest.approx(0.008)


def test_dice_iou_identity(two_overlapping_masks):
    a, _ = two_overlapping_masks
    assert compute_dice(a, a) == pytest.approx(1.0, abs=1e-6)
    assert compute_iou(a, a) == pytest.approx(1.0, abs=1e-6)


def test_dice_disjoint():
    a = np.zeros((5, 5, 5), dtype=np.uint8)
    b = np.zeros_like(a)
    a[0:2, 0:2, 0:2] = 1
    b[3:5, 3:5, 3:5] = 1
    assert compute_dice(a, b) == pytest.approx(0.0, abs=1e-6)


def test_hausdorff_identity(two_overlapping_masks):
    a, _ = two_overlapping_masks
    try:
        val = compute_hausdorff_95(a, a, (1.0, 1.0, 1.0))
    except ImportError:
        pytest.skip("scipy not available")
    assert val == pytest.approx(0.0, abs=1e-6)


def test_recist_diameter_positive():
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[5:15, 5:15, 9:11] = 1
    diameter = compute_recist_diameter_mm(mask, (1.0, 1.0, 1.0))
    assert diameter > 0.0


def test_growth_rate_positive():
    rate = compute_growth_rate(10.0, 13.0, date(2024, 1, 1), date(2024, 1, 31))
    assert rate == pytest.approx(0.1, abs=1e-6)


def test_growth_rate_zero_when_missing_date():
    assert compute_growth_rate(10.0, 13.0, None, None) == 0.0
    assert compute_growth_rate(10.0, 13.0, date(2024, 1, 1), date(2024, 1, 1)) == 0.0


def test_jackknife_ci_shrinks_with_agreement():
    vols_a = {"nnunet": 10.0, "medgemma": 10.0, "sam3": 10.0}
    vols_b = {"nnunet": 12.0, "medgemma": 12.0, "sam3": 12.0}
    delta, ci = jackknife_volume_ci(vols_a, vols_b)
    assert delta == pytest.approx(2.0, abs=1e-6)
    assert ci == pytest.approx(0.0, abs=1e-6)


def test_jackknife_ci_grows_with_disagreement():
    vols_a = {"nnunet": 10.0, "medgemma": 10.0, "sam3": 10.0}
    vols_b = {"nnunet": 11.0, "medgemma": 12.0, "sam3": 18.0}
    _, ci_high = jackknife_volume_ci(vols_a, vols_b)
    assert ci_high > 0.1


def test_interpret_progressive_disease():
    flag = interpret(
        delta_cm3=5.0,
        pct_change=30.0,
        registration_ncc=0.9,
        ci_half_cm3=0.1,
    )
    assert flag.level == "progressive_disease"


def test_interpret_response():
    flag = interpret(
        delta_cm3=-5.0,
        pct_change=-30.0,
        registration_ncc=0.9,
        ci_half_cm3=0.1,
    )
    assert flag.level == "response"


def test_interpret_stable():
    flag = interpret(
        delta_cm3=0.1,
        pct_change=2.0,
        registration_ncc=0.9,
        ci_half_cm3=0.01,
    )
    assert flag.level == "stable_disease"


def test_interpret_registration_failed_trumps_everything():
    flag = interpret(
        delta_cm3=5.0,
        pct_change=30.0,
        registration_ncc=0.3,
        ci_half_cm3=0.1,
    )
    assert flag.level == "registration_failed"


def test_interpret_high_uncertainty():
    flag = interpret(
        delta_cm3=2.0,
        pct_change=3.0,
        registration_ncc=0.9,
        ci_half_cm3=1.0,  # half-width ~= |delta| => ratio 0.5 >> 0.15
    )
    assert flag.level == "high_uncertainty"
