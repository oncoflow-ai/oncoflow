"""Segmentation adapters (one per model, all implementing `SegmentationAdapter`)."""

from ml.inference.adapters.base import (
    AdapterResult,
    Bbox,
    SegmentationAdapter,
    empty_result,
    build_adapter,
)

__all__ = [
    "AdapterResult",
    "Bbox",
    "SegmentationAdapter",
    "empty_result",
    "build_adapter",
]
