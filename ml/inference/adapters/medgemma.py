"""
adapters/medgemma.py – MedGemma-1.5 vision-language segmentation adapter.

Strategy
--------
MedGemma-1.5 (`google/medgemma-1.5-4b-it`) is a VLM: given a 2-D image and a
text prompt, it produces text. We treat it as a "segmentation-by-description"
model: for each axial slice we ask for box coordinates (or polygon) of tumor
regions, parse the structured JSON response, and rasterise back to a binary
mask. 3-D mask = stack of per-slice 2-D masks.

Speed knobs
-----------
1. ROI gating – we only run the VLM on slices inside the ROI bbox (either the
   nnU-Net bootstrap mask or an intensity-based brain bbox).
2. `medgemma_max_slices` cap – never exceed N slices (64 by default).
3. FP16 on MPS, BF16 on CUDA via `torch_dtype`.
4. Slice batching when the processor supports a list input (opportunistic).

Fallbacks
---------
- If `HF_TOKEN` missing, swap to `microsoft/llava-med-v1.5-mistral-7b`.
- If transformers or torch unavailable, `is_available()` returns False.
- If parsing fails on a slice, that slice contributes an empty mask (adapter
  never raises).
"""

from __future__ import annotations

import json
import logging
import os
import re
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

# Module-level singletons – avoid reloading 4B parameters per call.
_MODEL = None
_PROCESSOR = None
_MODEL_ID_LOADED: Optional[str] = None


class MedGemmaAdapter(SegmentationAdapter):
    """Slice-wise VLM segmentation adapter for MedGemma-1.5 / LLaVA-Med."""

    name = "medgemma"

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            from transformers import AutoProcessor, AutoModelForImageTextToText  # noqa: F401
        except ImportError:
            return False
        # If HF_TOKEN is present or a fallback model is fetchable, we're good.
        return True

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        global _MODEL, _PROCESSOR, _MODEL_ID_LOADED

        if self._loaded:
            return

        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        hf_token = os.environ.get("HF_TOKEN", "")
        use_medgemma = bool(hf_token)
        model_id = (
            self.cfg.medgemma_model_id
            if use_medgemma
            else self.cfg.medgemma_fallback_model_id
        )

        if _MODEL is not None and _MODEL_ID_LOADED == model_id:
            self._loaded = True
            return

        device = self.cfg.resolve_device()
        dtype_str = self.cfg.resolve_dtype()
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype_str]

        kwargs = {"token": hf_token} if hf_token else {}

        logger.info(
            "MedGemmaAdapter: loading %s on %s (%s)", model_id, device, dtype_str
        )
        try:
            _PROCESSOR = AutoProcessor.from_pretrained(model_id, **kwargs)
            _MODEL = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                **kwargs,
            ).to(device)
            _MODEL.eval()
            _MODEL_ID_LOADED = model_id
        except Exception as exc:
            logger.warning("MedGemmaAdapter: model load failed: %s", exc)
            _MODEL = None
            _PROCESSOR = None
            _MODEL_ID_LOADED = None
            raise

        self._loaded = True

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _predict_impl(
        self, vol: Volume, roi: Optional[Bbox]
    ) -> AdapterResult:
        if _MODEL is None or _PROCESSOR is None:
            return empty_result(
                vol.shape, error="MedGemma not loaded", model=self.name
            )

        # Decide which axial slices to run.
        z_indices = self._select_slices(vol, roi)
        if not z_indices:
            return empty_result(
                vol.shape, error="no slices selected", model=self.name
            )

        mask = np.zeros(vol.shape, dtype=np.uint8)
        n_tumor_found = 0

        for z in z_indices:
            sl = vol.data[:, :, z]
            try:
                slice_mask = self._infer_slice(sl, roi)
            except Exception as exc:
                logger.debug("MedGemma slice %d failed: %s", z, exc)
                continue
            mask[:, :, z] = slice_mask
            if slice_mask.any():
                n_tumor_found += 1

        return {
            "mask": mask,
            "prob": None,
            "runtime_s": 0.0,
            "meta": {
                "slices_run": len(z_indices),
                "slices_with_tumor": n_tumor_found,
                "model_id": _MODEL_ID_LOADED,
                "roi_used": roi.as_tuple() if roi else None,
            },
        }

    # ------------------------------------------------------------------
    # Slice selection + 2-D inference helpers
    # ------------------------------------------------------------------

    def _select_slices(
        self, vol: Volume, roi: Optional[Bbox]
    ) -> List[int]:
        """Pick the z-slice indices to feed MedGemma."""
        H, W, D = vol.shape

        if roi is not None:
            z_range = list(range(roi.z_min, roi.z_max))
        else:
            # Fallback: skip empty slices based on intensity.
            intensities = vol.data.reshape(-1, D).max(axis=0)
            nonempty = np.where(intensities > float(intensities.mean() * 0.2))[0]
            if len(nonempty) > 0:
                z_range = list(range(int(nonempty[0]), int(nonempty[-1]) + 1))
            else:
                z_range = list(range(D))

        cap = max(1, self.cfg.medgemma_max_slices)
        if len(z_range) > cap:
            # Uniformly stride-sample inside the range.
            idx = np.linspace(0, len(z_range) - 1, cap).astype(int)
            z_range = [z_range[i] for i in idx]
        return z_range

    def _infer_slice(
        self, slice_2d: np.ndarray, roi: Optional[Bbox]
    ) -> np.ndarray:
        """Run MedGemma on one 2-D slice. Returns binary (H, W) uint8 mask."""
        import torch

        h, w = slice_2d.shape
        pil = _array_to_pil(slice_2d)

        prompt = self.cfg.medgemma_prompt
        inputs = _build_inputs(_PROCESSOR, pil, prompt).to(_MODEL.device)  # type: ignore[union-attr]

        with torch.no_grad():
            out = _MODEL.generate(  # type: ignore[union-attr]
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        text = _PROCESSOR.decode(out[0], skip_special_tokens=True)  # type: ignore[union-attr]

        boxes = _parse_boxes(text)
        mask = np.zeros((h, w), dtype=np.uint8)
        for box in boxes:
            y1, x1, y2, x2 = box
            # MedGemma normalises coords to 0-1000 over the PIL image.
            y1 = int(round(y1 * h / 1000.0))
            x1 = int(round(x1 * w / 1000.0))
            y2 = int(round(y2 * h / 1000.0))
            x2 = int(round(x2 * w / 1000.0))
            y1, y2 = sorted((max(0, y1), min(h, y2)))
            x1, x2 = sorted((max(0, x1), min(w, x2)))
            mask[y1:y2, x1:x2] = 1

        # Restrict to ROI in-plane if provided
        if roi is not None:
            roi_mask = np.zeros_like(mask)
            roi_mask[roi.x_min:roi.x_max, roi.y_min:roi.y_max] = 1
            mask = mask & roi_mask

        return mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _array_to_pil(slice_2d: np.ndarray):
    """Normalise to 0-255 uint8 and convert to a PIL image."""
    from PIL import Image

    s = slice_2d.astype(np.float32)
    lo = np.percentile(s, 1)
    hi = np.percentile(s, 99)
    if hi <= lo:
        s = np.zeros_like(s)
    else:
        s = np.clip((s - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    img = (s * 255).astype(np.uint8)
    return Image.fromarray(img).convert("RGB")


def _build_inputs(processor, pil_img, prompt: str):
    """Build processor inputs compatible with Gemma3 / MedGemma / LLaVA-Med."""
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return processor(images=[pil_img], text=text, return_tensors="pt")

    img_token = getattr(processor, "image_token", "<image>")
    return processor(
        images=pil_img, text=f"{img_token}\n{prompt}", return_tensors="pt"
    )


_BOX_JSON_RE = re.compile(r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]")


def _parse_boxes(text: str) -> List[Tuple[int, int, int, int]]:
    """
    Extract bounding boxes from MedGemma's response text.

    MedGemma-1.5 tends to return structured JSON like:
        {"box_2d": [y1, x1, y2, x2], "label": "tumor"}
    We also accept raw 4-integer arrays as a fallback.
    """
    boxes: List[Tuple[int, int, int, int]] = []

    # First try to parse as JSON (possibly multiple objects)
    for jstart in (0,):
        try:
            decoded = json.loads(text)
            boxes.extend(_extract_boxes_from_json(decoded))
            if boxes:
                return boxes
        except Exception:
            pass

    # Fallback: regex-find any [y,x,y,x]-looking array
    for m in _BOX_JSON_RE.finditer(text):
        try:
            vals = json.loads(m.group(0))
            if len(vals) == 4 and all(isinstance(v, (int, float)) for v in vals):
                boxes.append(tuple(int(v) for v in vals))  # type: ignore[arg-type]
        except Exception:
            continue
    return boxes


def _extract_boxes_from_json(obj) -> List[Tuple[int, int, int, int]]:
    out: List[Tuple[int, int, int, int]] = []
    if isinstance(obj, dict):
        if "box_2d" in obj and isinstance(obj["box_2d"], (list, tuple)):
            b = obj["box_2d"]
            if len(b) == 4:
                out.append(tuple(int(v) for v in b))  # type: ignore[arg-type]
        for v in obj.values():
            out.extend(_extract_boxes_from_json(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_extract_boxes_from_json(v))
    return out
