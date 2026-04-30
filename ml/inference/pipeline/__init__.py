"""End-to-end inference pipelines: single-study segmentation + longitudinal comparison."""

from ml.inference.pipeline.segment import segment_study, StudySegmentation
from ml.inference.pipeline.longitudinal import compare_studies, ComparisonResult

__all__ = [
    "segment_study",
    "StudySegmentation",
    "compare_studies",
    "ComparisonResult",
]
