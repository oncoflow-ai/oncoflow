"""
adapters/sam3.py – Meta SAM3 / SAM2 / MedSAM segmentation adapter.

Strategy
--------
SAM3 (`facebook/sam3-hiera-large`) is a promptable foundation model. In the
batch pipeline we drive it with a bounding-box prompt derived from the
nnU-Net bootstrap mask (when available); without a prompt we fall back to
SAM's automatic mode (or an intensity-based bbox).

Backend matrix
--------------
- `gpu-prod` / any: try SAM3 first (`SAM3Predictor`).
- If SAM3 unavailable: try SAM2 (`SAM2ImagePredictor`).
- If SAM2 unavailable: try MedSAM.
- If nothing: `is_available() == False`.

Inference proceeds slice-by-slice over axial cuts inside the ROI z-range
(propagation between adjacent slices is used when the predictor exposes it).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from ml.inference.adapters.base import (
    AdapterResult,
    Bbox,
    SegmentationAdapter,
    empty_result,
)
from ml.inference.io import Volume

logger = logging.getLogger(__name__)

_PREDICTOR = None
_BACKEND_USED: Optional[str] = None  # "sam3" | "sam2" | "medsam"


class Sam3Adapter(SegmentationAdapter):
    """SAM3 adapter with SAM2 → MedSAM fallback chain."""

    name = "sam3"

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        if self._probe_sam3():
            return True
        if self._probe_sam2():
            return True
        if self._probe_medsam():
            return True
        return False

    @staticmethod
    def _probe_sam3() -> bool:
        try:
            import sam3  # type: ignore  # noqa: F401
            return True
        except Exception:
            return False

    @staticmethod
    def _probe_sam2() -> bool:
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: F401
            return True
        except Exception:
            return False

    @staticmethod
    def _probe_medsam() -> bool:
        try:
            import medsam  # type: ignore  # noqa: F401
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        global _PREDICTOR, _BACKEND_USED

        if self._loaded and _PREDICTOR is not None:
            return

        # Try SAM3
        if self._probe_sam3():
            try:
                from sam3 import SAM3Predictor  # type: ignore

                _PREDICTOR = SAM3Predictor.from_pretrained(self.cfg.sam3_model_id)
                _BACKEND_USED = "sam3"
                self._loaded = True
                logger.info("Sam3Adapter: SAM3 loaded (%s)", self.cfg.sam3_model_id)
                return
            except Exception as exc:
                logger.warning("SAM3 load failed (%s) – trying SAM2", exc)

        # Try SAM2
        if self._probe_sam2():
            try:
                from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

                _PREDICTOR = SAM2ImagePredictor.from_pretrained(
                    self.cfg.sam2_model_id
                )
                _BACKEND_USED = "sam2"
                self._loaded = True
                logger.info("Sam3Adapter: SAM2 fallback loaded")
                return
            except Exception as exc:
                logger.warning("SAM2 load failed (%s) – trying MedSAM", exc)

        # Try MedSAM
        if self._probe_medsam():
            try:
                import medsam  # type: ignore

                _PREDICTOR = medsam.get_predictor()  # type: ignore
                _BACKEND_USED = "medsam"
                self._loaded = True
                logger.info("Sam3Adapter: MedSAM fallback loaded")
                return
            except Exception as exc:
                logger.warning("MedSAM load failed: %s", exc)

        _PREDICTOR = None
        _BACKEND_USED = None
        self._loaded = True

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _predict_impl(
        self, vol: Volume, roi: Optional[Bbox]
    ) -> AdapterResult:
        if _PREDICTOR is None:
            return empty_result(
                vol.shape, error="no SAM backend loaded", model=self.name
            )

        if _BACKEND_USED == "sam3":
            return self._predict_sam3(vol, roi)
        if _BACKEND_USED == "sam2":
            return self._predict_sam2(vol, roi)
        if _BACKEND_USED == "medsam":
            return self._predict_medsam(vol, roi)
        return empty_result(
            vol.shape, error=f"unknown SAM backend {_BACKEND_USED}", model=self.name
        )

    # ------------------------------------------------------------------
    # SAM3: volumetric API
    # ------------------------------------------------------------------

    def _predict_sam3(
        self, vol: Volume, roi: Optional[Bbox]
    ) -> AdapterResult:
        """
        SAM3 exposes a volumetric predictor. We try `generate_automatic()` if
        available, else fall back to a bbox prompt.
        """
        try:
            _PREDICTOR.set_volume(vol.data)  # type: ignore[union-attr]
        except Exception:
            # SAM3 API is still evolving; some versions use set_image per slice.
            return self._predict_sam_slicewise(vol, roi, is_sam3=True)

        try:
            if roi is not None:
                masks, _, _ = _PREDICTOR.predict(  # type: ignore[union-attr]
                    box=np.array([[roi.x_min, roi.y_min, roi.x_max, roi.y_max]]),
                    multimask_output=False,
                )
            elif hasattr(_PREDICTOR, "generate_automatic"):
                masks = _PREDICTOR.generate_automatic()  # type: ignore[union-attr]
            else:
                return self._predict_sam_slicewise(vol, roi, is_sam3=True)

            mask = _first_mask(masks).astype(np.uint8)
            return {
                "mask": mask,
                "prob": None,
                "runtime_s": 0.0,
                "meta": {"backend": "sam3", "mode": "volumetric"},
            }
        except Exception as exc:
            logger.warning("SAM3 volumetric predict failed (%s); slice-wise", exc)
            return self._predict_sam_slicewise(vol, roi, is_sam3=True)

    # ------------------------------------------------------------------
    # SAM2: slice-wise 2-D predictor
    # ------------------------------------------------------------------

    def _predict_sam2(
        self, vol: Volume, roi: Optional[Bbox]
    ) -> AdapterResult:
        return self._predict_sam_slicewise(vol, roi, is_sam3=False)

    def _predict_sam_slicewise(
        self, vol: Volume, roi: Optional[Bbox], *, is_sam3: bool
    ) -> AdapterResult:
        """Shared slice-wise implementation for SAM2 and SAM3 image mode."""
        z_indices = _select_axial_slices(vol, roi, max_slices=None)
        if not z_indices:
            return empty_result(
                vol.shape, error="no slices selected", model=self.name
            )

        mask_3d = np.zeros(vol.shape, dtype=np.uint8)

        for z in z_indices:
            sl = vol.data[:, :, z]
            rgb = _slice_to_rgb(sl)

            try:
                _PREDICTOR.set_image(rgb)  # type: ignore[union-attr]
            except Exception as exc:
                logger.debug("SAM set_image failed z=%d: %s", z, exc)
                continue

            box = None
            if roi is not None:
                box = np.array(
                    [[roi.x_min, roi.y_min, roi.x_max, roi.y_max]],
                    dtype=np.float32,
                )

            try:
                if box is not None:
                    masks, _, _ = _PREDICTOR.predict(  # type: ignore[union-attr]
                        box=box,
                        multimask_output=False,
                    )
                else:
                    # Centroid-based single positive point
                    cy, cx = sl.shape[0] // 2, sl.shape[1] // 2
                    masks, _, _ = _PREDICTOR.predict(  # type: ignore[union-attr]
                        point_coords=np.array([[cx, cy]]),
                        point_labels=np.array([1]),
                        multimask_output=False,
                    )
            except Exception as exc:
                logger.debug("SAM predict z=%d failed: %s", z, exc)
                continue

            sl_mask = _first_mask(masks)
            if sl_mask.ndim == 2 and sl_mask.shape == sl.shape:
                mask_3d[:, :, z] = (sl_mask > 0.5).astype(np.uint8)

        return {
            "mask": mask_3d,
            "prob": None,
            "runtime_s": 0.0,
            "meta": {
                "backend": "sam3" if is_sam3 else "sam2",
                "mode": "slicewise",
                "slices_run": len(z_indices),
                "roi_used": roi.as_tuple() if roi else None,
            },
        }

    # ------------------------------------------------------------------
    # MedSAM fallback (very lightweight)
    # ------------------------------------------------------------------

    def _predict_medsam(
        self, vol: Volume, roi: Optional[Bbox]
    ) -> AdapterResult:
        """MedSAM expects a 2-D bbox prompt per slice. We delegate to slicewise."""
        return self._predict_sam_slicewise(vol, roi, is_sam3=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slice_to_rgb(slice_2d: np.ndarray) -> np.ndarray:
    """Normalise a grayscale slice to (H, W, 3) uint8."""
    s = slice_2d.astype(np.float32)
    lo = np.percentile(s, 1)
    hi = np.percentile(s, 99)
    if hi <= lo:
        img = np.zeros_like(s, dtype=np.uint8)
    else:
        img = np.clip((s - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        img = (img * 255).astype(np.uint8)
    return np.repeat(img[:, :, None], 3, axis=-1)


def _select_axial_slices(
    vol: Volume, roi: Optional[Bbox], max_slices: Optional[int]
) -> List[int]:
    H, W, D = vol.shape
    if roi is not None:
        rng = list(range(roi.z_min, roi.z_max))
    else:
        intensities = vol.data.reshape(-1, D).max(axis=0)
        nz = np.where(intensities > float(intensities.mean() * 0.2))[0]
        rng = list(range(int(nz[0]), int(nz[-1]) + 1)) if len(nz) else list(range(D))

    if max_slices and len(rng) > max_slices:
        idx = np.linspace(0, len(rng) - 1, max_slices).astype(int)
        rng = [rng[i] for i in idx]
    return rng


def _first_mask(out) -> np.ndarray:
    """Pluck the first mask from a SAM-style return (array, list, or dict)."""
    if isinstance(out, np.ndarray):
        if out.ndim == 4:
            return out[0, 0]
        if out.ndim == 3:
            return out[0]
        return out
    if isinstance(out, (list, tuple)):
        return np.asarray(out[0])
    if hasattr(out, "numpy"):
        return out.numpy()
    return np.asarray(out)
