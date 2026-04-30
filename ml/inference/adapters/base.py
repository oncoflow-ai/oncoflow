"""
adapters/base.py – shared contract for all segmentation adapters.

Every model (nnU-Net, MedGemma, SAM3) implements `SegmentationAdapter`. The
pipeline, ensemble, tests, and CLI only know this Protocol — they do not care
which backend (local vs gpu-prod) is in use.

Key contract rules
------------------
1. `is_available()` must NEVER raise. It returns False when weights/libraries
   are missing so the orchestrator can gracefully drop that model.
2. `predict()` always returns a mask with the SAME shape as the input volume.
   If the adapter is unavailable or fails, it returns a zero mask with
   `meta["stub"] = True` and `meta["error"] = <reason>` — never raises.
3. `predict()` accepts an optional `roi` bounding box so MedGemma and SAM can
   restrict slice work when bootstrapped from a fast nnU-Net pass.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, TypedDict

import numpy as np

from ml.inference.config import InferenceConfig
from ml.inference.io import Volume

logger = logging.getLogger(__name__)


ModelName = Literal["nnunet", "medgemma", "sam3"]


@dataclass(frozen=True)
class Bbox:
    """Inclusive-exclusive 3-D bounding box in voxel coordinates."""

    x_min: int
    y_min: int
    z_min: int
    x_max: int
    y_max: int
    z_max: int

    def clip(self, shape: Tuple[int, int, int]) -> "Bbox":
        return Bbox(
            max(0, self.x_min),
            max(0, self.y_min),
            max(0, self.z_min),
            min(shape[0], self.x_max),
            min(shape[1], self.y_max),
            min(shape[2], self.z_max),
        )

    def pad(self, padding: int, shape: Tuple[int, int, int]) -> "Bbox":
        return Bbox(
            self.x_min - padding,
            self.y_min - padding,
            self.z_min - padding,
            self.x_max + padding,
            self.y_max + padding,
            self.z_max + padding,
        ).clip(shape)

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> Optional["Bbox"]:
        nz = np.argwhere(mask > 0)
        if len(nz) == 0:
            return None
        mins = nz.min(axis=0)
        maxs = nz.max(axis=0) + 1
        return cls(
            int(mins[0]), int(mins[1]), int(mins[2]),
            int(maxs[0]), int(maxs[1]), int(maxs[2]),
        )

    @property
    def z_slice(self) -> slice:
        return slice(self.z_min, self.z_max)

    def as_tuple(self) -> Tuple[int, int, int, int, int, int]:
        return (
            self.x_min, self.y_min, self.z_min,
            self.x_max, self.y_max, self.z_max,
        )


class AdapterResult(TypedDict, total=False):
    """Return type of `SegmentationAdapter.predict()`."""

    mask: np.ndarray          # (H, W, D) uint8 binary – REQUIRED
    prob: Optional[np.ndarray]  # (H, W, D) float32 soft mask, optional
    runtime_s: float          # wall-clock seconds
    meta: Dict                # backend-specific info


def empty_result(
    shape: Tuple[int, int, int],
    *,
    error: str = "",
    model: str = "",
    stub: bool = True,
) -> AdapterResult:
    """Build an all-zeros AdapterResult (used for graceful fallback)."""
    return {
        "mask": np.zeros(shape, dtype=np.uint8),
        "prob": None,
        "runtime_s": 0.0,
        "meta": {"stub": stub, "error": error, "model": model},
    }


class SegmentationAdapter(ABC):
    """Abstract base class that every model adapter inherits from."""

    name: ModelName = "nnunet"  # override in subclass

    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self._loaded = False

    # ---- Required interface ------------------------------------------------

    @abstractmethod
    def is_available(self) -> bool:
        """Return True iff weights + libraries are ready. MUST NOT raise."""

    @abstractmethod
    def _predict_impl(self, vol: Volume, roi: Optional[Bbox]) -> AdapterResult:
        """Subclass-specific inference. Called only when is_available()."""

    def load(self) -> None:
        """Lazy model load. Default no-op; subclasses override if needed."""
        self._loaded = True

    # ---- Public wrapper ----------------------------------------------------

    def predict(
        self, vol: Volume, *, roi: Optional[Bbox] = None
    ) -> AdapterResult:
        """
        Safe wrapper: times the run, catches any exception, and returns a
        zero-mask AdapterResult on failure instead of propagating.
        """
        if not self.is_available():
            return empty_result(
                vol.shape,
                error="adapter unavailable (missing weights or dependencies)",
                model=self.name,
            )

        if not self._loaded:
            try:
                self.load()
            except Exception as exc:
                logger.warning("%s.load() failed: %s", self.name, exc)
                return empty_result(
                    vol.shape,
                    error=f"load failed: {exc}",
                    model=self.name,
                )

        t0 = time.perf_counter()
        try:
            result = self._predict_impl(vol, roi)
        except Exception as exc:
            logger.exception("%s.predict() raised: %s", self.name, exc)
            return empty_result(
                vol.shape,
                error=f"predict raised: {exc}",
                model=self.name,
            )
        elapsed = time.perf_counter() - t0

        mask = np.asarray(result.get("mask"))
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8)
        if mask.shape != vol.shape:
            logger.warning(
                "%s returned mask shape %s != volume shape %s – padding/cropping",
                self.name, mask.shape, vol.shape,
            )
            mask = _conform_shape(mask, vol.shape)
        result["mask"] = mask
        result["runtime_s"] = elapsed
        meta = dict(result.get("meta", {}))
        meta.setdefault("model", self.name)
        meta.setdefault("backend", self.cfg.backend)
        meta.setdefault("device", self.cfg.resolve_device())
        result["meta"] = meta
        return result


def _conform_shape(
    arr: np.ndarray, target: Tuple[int, int, int]
) -> np.ndarray:
    """Pad/crop array to match target shape (centered)."""
    out = np.zeros(target, dtype=arr.dtype)
    slicers_src = []
    slicers_dst = []
    for src_sz, dst_sz in zip(arr.shape, target):
        if src_sz >= dst_sz:
            start = (src_sz - dst_sz) // 2
            slicers_src.append(slice(start, start + dst_sz))
            slicers_dst.append(slice(0, dst_sz))
        else:
            start = (dst_sz - src_sz) // 2
            slicers_src.append(slice(0, src_sz))
            slicers_dst.append(slice(start, start + src_sz))
    out[tuple(slicers_dst)] = arr[tuple(slicers_src)]
    return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_adapter(name: str, cfg: InferenceConfig) -> SegmentationAdapter:
    """Instantiate the adapter for `name` according to cfg.backend."""
    name = name.lower()
    if name == "nnunet":
        from ml.inference.adapters.nnunet import NNUNetAdapter

        return NNUNetAdapter(cfg)
    if name == "medgemma":
        from ml.inference.adapters.medgemma import MedGemmaAdapter

        return MedGemmaAdapter(cfg)
    if name == "sam3":
        from ml.inference.adapters.sam3 import Sam3Adapter

        return Sam3Adapter(cfg)
    raise ValueError(f"Unknown adapter: {name!r}")
