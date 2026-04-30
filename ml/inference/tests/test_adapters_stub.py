"""
Adapter base-class contract tests.

We don't exercise the real model weights here; we only verify the stub-mode
behaviour that every adapter MUST satisfy:

  1. `is_available()` never raises.
  2. If unavailable, `predict()` returns a zero mask with the input shape and
     `meta["stub"] is True`.
  3. Even when `_predict_impl` raises, the safe wrapper returns a stub result.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ml.inference.adapters.base import (
    AdapterResult,
    Bbox,
    SegmentationAdapter,
    empty_result,
)
from ml.inference.config import InferenceConfig
from ml.inference.io import Volume


class _UnavailableAdapter(SegmentationAdapter):
    name = "nnunet"

    def is_available(self) -> bool:
        return False

    def _predict_impl(self, vol, roi):
        raise RuntimeError("should not be called")


class _RaisingAdapter(SegmentationAdapter):
    name = "medgemma"

    def is_available(self) -> bool:
        return True

    def _predict_impl(self, vol, roi):
        raise RuntimeError("boom")


class _FakeOkAdapter(SegmentationAdapter):
    name = "sam3"

    def is_available(self) -> bool:
        return True

    def _predict_impl(self, vol, roi):
        mask = (vol.data > 0.3).astype(np.uint8)
        return {"mask": mask, "prob": None, "runtime_s": 0.0, "meta": {}}


def _cfg() -> InferenceConfig:
    return InferenceConfig(enabled_models=("nnunet", "medgemma", "sam3"))


def test_unavailable_returns_stub(synthetic_volume):
    adapter = _UnavailableAdapter(_cfg())
    res = adapter.predict(synthetic_volume)
    assert res["mask"].shape == synthetic_volume.shape
    assert res["mask"].dtype == np.uint8
    assert res["mask"].sum() == 0
    assert res["meta"]["stub"] is True


def test_raising_adapter_caught(synthetic_volume):
    adapter = _RaisingAdapter(_cfg())
    res = adapter.predict(synthetic_volume)
    assert res["mask"].sum() == 0
    assert res["meta"]["stub"] is True
    assert "boom" in res["meta"]["error"]


def test_ok_adapter_sets_runtime(synthetic_volume):
    adapter = _FakeOkAdapter(_cfg())
    res = adapter.predict(synthetic_volume)
    assert res["mask"].shape == synthetic_volume.shape
    assert res["mask"].sum() > 0
    assert res["runtime_s"] >= 0.0
    assert res["meta"]["backend"] == "local"


def test_bbox_from_mask_and_pad():
    m = np.zeros((10, 10, 10), dtype=np.uint8)
    m[3:6, 2:5, 4:7] = 1
    bbox = Bbox.from_mask(m)
    assert bbox is not None
    assert bbox.as_tuple() == (3, 2, 4, 6, 5, 7)
    padded = bbox.pad(2, m.shape)
    assert padded.x_min == 1 and padded.x_max == 8
    assert padded.y_min == 0 and padded.y_max == 7
    assert padded.z_min == 2 and padded.z_max == 9


def test_empty_result_shape(synthetic_volume):
    res = empty_result(synthetic_volume.shape, error="x", model="nnunet")
    assert res["mask"].shape == synthetic_volume.shape
    assert res["meta"]["stub"] is True
