"""
OncoFlow Inference Algorithm.

Standalone, backend-free Python package implementing:
  * 3-model panel segmentation (nnU-Net v2, MedGemma-1.5, Meta SAM3 / SAM2 fallback)
  * Ensemble fusion (majority vote / STAPLE / confidence-weighted)
  * ANTsPy-based longitudinal registration (Affine + NCC quality gate)
  * RECIST-style change metrics + jackknife uncertainty + interpretation flags

Entry points:
    >>> from ml.inference import segment_study, compare_studies, InferenceConfig

Reference: IMPLEMENTATION_PLAN.md – Phase 4.
"""

from ml.inference.config import InferenceConfig, load_config
from ml.inference.pipeline.segment import segment_study, StudySegmentation
from ml.inference.pipeline.longitudinal import compare_studies, ComparisonResult

__all__ = [
    "InferenceConfig",
    "load_config",
    "segment_study",
    "StudySegmentation",
    "compare_studies",
    "ComparisonResult",
]

__version__ = "0.1.0"
