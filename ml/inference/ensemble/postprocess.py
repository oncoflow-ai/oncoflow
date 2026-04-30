"""
ensemble/postprocess.py – morphological + connected-component cleanup.

Applied to the fused ensemble mask (and optionally to individual model masks
before fusion) to remove isolated speckles and close small holes.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def keep_largest_cc(
    mask: np.ndarray,
    *,
    min_voxels: int = 0,
    max_components: int = 1,
) -> np.ndarray:
    """
    Keep the `max_components` largest connected components above `min_voxels`.
    Uses scipy.ndimage.label with 3-D 26-connectivity.
    """
    from scipy.ndimage import label

    if not mask.any():
        return mask.astype(np.uint8)

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, n = label(mask > 0, structure=structure)
    if n == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    order = np.argsort(sizes)[::-1]

    out = np.zeros_like(mask, dtype=np.uint8)
    kept = 0
    for idx in order:
        if idx == 0:
            continue
        if sizes[idx] < min_voxels:
            break
        if kept >= max_components:
            break
        out[labeled == idx] = 1
        kept += 1
    return out


def morph_close(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary 3-D morphological closing with a spherical structuring element."""
    if radius <= 0:
        return mask.astype(np.uint8)
    try:
        from scipy.ndimage import binary_closing, generate_binary_structure, iterate_structure

        base = generate_binary_structure(3, 1)
        struct = iterate_structure(base, radius)
        return binary_closing(mask > 0, structure=struct).astype(np.uint8)
    except Exception as exc:  # pragma: no cover
        logger.warning("morph_close failed (%s) – returning input", exc)
        return mask.astype(np.uint8)


def clean_mask(
    mask: np.ndarray,
    *,
    keep_largest: bool = True,
    min_voxels: int = 20,
    closing_radius: int = 0,
    max_components: int = 1,
) -> np.ndarray:
    """Convenience pipeline: closing → keep largest CC(s) with min voxel count."""
    out = mask.astype(np.uint8)
    if closing_radius > 0:
        out = morph_close(out, closing_radius)
    if keep_largest:
        out = keep_largest_cc(
            out, min_voxels=min_voxels, max_components=max_components
        )
    return out
