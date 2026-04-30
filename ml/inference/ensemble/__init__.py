"""Ensemble fusion strategies and post-processing."""

from ml.inference.ensemble.strategies import (
    fuse,
    majority_vote,
    union_vote,
    intersection_vote,
    staple_fusion,
    confidence_weighted,
)
from ml.inference.ensemble.postprocess import clean_mask, keep_largest_cc
from ml.inference.ensemble.agreement import (
    agreement_score,
    agreement_flag,
    PanelAgreement,
)

__all__ = [
    "fuse",
    "majority_vote",
    "union_vote",
    "intersection_vote",
    "staple_fusion",
    "confidence_weighted",
    "clean_mask",
    "keep_largest_cc",
    "agreement_score",
    "agreement_flag",
    "PanelAgreement",
]
