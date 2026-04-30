"""
Tests for ensemble fusion strategies and agreement scoring.
"""

from __future__ import annotations

import numpy as np

from ml.inference.ensemble.agreement import agreement_flag, agreement_score
from ml.inference.ensemble.postprocess import keep_largest_cc, clean_mask
from ml.inference.ensemble.strategies import (
    confidence_weighted,
    fuse,
    intersection_vote,
    majority_vote,
    union_vote,
)


def _three_masks():
    """Three 10^3 masks. Voxel (5,5,5) appears in all three; others partial."""
    shape = (10, 10, 10)
    a = np.zeros(shape, dtype=np.uint8)
    b = np.zeros(shape, dtype=np.uint8)
    c = np.zeros(shape, dtype=np.uint8)
    a[4:7, 4:7, 4:7] = 1
    b[5:8, 4:7, 4:7] = 1
    c[4:7, 5:8, 4:7] = 1
    return {"nnunet": a, "medgemma": b, "sam3": c}


def test_union_is_superset_of_each():
    masks = _three_masks()
    u = union_vote(masks)
    for m in masks.values():
        assert (u >= m).all()


def test_intersection_is_subset_of_each():
    masks = _three_masks()
    i = intersection_vote(masks)
    for m in masks.values():
        assert (i <= m).all()


def test_majority_vote_between_union_and_intersection():
    masks = _three_masks()
    mv = majority_vote(masks)
    u = union_vote(masks)
    i = intersection_vote(masks)
    assert (mv >= i).all()
    assert (mv <= u).all()
    # Must contain the shared voxel.
    assert mv[5, 5, 5] == 1


def test_fuse_dispatcher_defaults():
    masks = _three_masks()
    mv = fuse("majority_vote", masks)
    assert mv.sum() > 0


def test_confidence_weighted_weights():
    probs = {
        "nnunet": np.full((5, 5, 5), 0.9, dtype=np.float32),
        "medgemma": np.zeros((5, 5, 5), dtype=np.float32),
    }
    # Heavy weight on nnunet → all-1 mask
    out = confidence_weighted(probs, weights={"nnunet": 1.0, "medgemma": 0.0})
    assert out.sum() == out.size
    # Balanced weights (mean 0.45) < 0.5 → all zeros
    out2 = confidence_weighted(probs, weights={"nnunet": 1.0, "medgemma": 1.0})
    assert out2.sum() == 0


def test_agreement_score_levels():
    masks = _three_masks()
    mv = majority_vote(masks)
    agr = agreement_score(masks, mv)
    assert 0.0 <= agr.mean_agreement <= 1.0
    assert agr.level in {"high", "moderate", "low"}
    assert set(agr.per_model_dice.keys()) == set(masks.keys())


def test_agreement_flag_thresholds():
    assert agreement_flag(0.95) == "high"
    assert agreement_flag(0.80) == "moderate"
    assert agreement_flag(0.50) == "low"


def test_keep_largest_cc_filters_speckles():
    m = np.zeros((10, 10, 10), dtype=np.uint8)
    m[1:4, 1:4, 1:4] = 1  # large CC
    m[8, 8, 8] = 1         # speckle
    cleaned = keep_largest_cc(m, min_voxels=0, max_components=1)
    assert cleaned[8, 8, 8] == 0
    assert cleaned[2, 2, 2] == 1


def test_clean_mask_pipeline():
    m = np.zeros((10, 10, 10), dtype=np.uint8)
    m[2:6, 2:6, 2:6] = 1
    out = clean_mask(m, keep_largest=True, min_voxels=1, closing_radius=0)
    assert out.sum() == m.sum()
