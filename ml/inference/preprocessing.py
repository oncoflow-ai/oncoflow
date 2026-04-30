"""
preprocessing.py – Volume preprocessing: RAS orientation, N4 bias correction,
isotropic resampling, optional skull-stripping.

Used once per timepoint before any adapter runs. The same preprocessed volume
is fed to registration and all enabled segmentation adapters so their outputs
live in the same voxel grid.

Implements Stage 1 of the longitudinal algorithm in IMPLEMENTATION_PLAN.md.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np

from ml.inference.config import InferenceConfig
from ml.inference.io import Volume, save_nifti, cache_dir_for

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SITK helpers (lazy import so CPU-only tests don't require SITK at import time)
# ---------------------------------------------------------------------------


def _volume_to_sitk(vol: Volume):
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(np.ascontiguousarray(vol.data.transpose(2, 1, 0)))
    img.SetSpacing(tuple(float(s) for s in vol.spacing))
    # Use identity direction + origin from the affine's translation.
    origin = tuple(float(x) for x in vol.affine[:3, 3])
    img.SetOrigin(origin)
    return img


def _sitk_to_volume(img, affine_ref: np.ndarray) -> Volume:
    import SimpleITK as sitk

    arr = sitk.GetArrayFromImage(img).transpose(2, 1, 0).astype(np.float32)
    spacing = tuple(float(s) for s in img.GetSpacing())

    # Rebuild affine with updated spacing while keeping the original rotation
    # direction (we do not touch cosines in N4/resample).
    new_affine = affine_ref.copy()
    rot = new_affine[:3, :3]
    norms = np.linalg.norm(rot, axis=0)
    norms[norms == 0] = 1.0
    direction = rot / norms
    new_affine[:3, :3] = direction * np.asarray(spacing)
    new_affine[:3, 3] = np.asarray(img.GetOrigin())
    return Volume(data=arr, affine=new_affine, spacing=spacing)


# ---------------------------------------------------------------------------
# Preprocessing steps
# ---------------------------------------------------------------------------


def orient_to_ras(vol: Volume) -> Volume:
    """Reorient a nibabel-compatible Volume to canonical RAS+."""
    img = nib.Nifti1Image(vol.data, vol.affine)
    ras_img = nib.as_closest_canonical(img)
    data = np.asarray(ras_img.dataobj, dtype=np.float32)
    zooms = ras_img.header.get_zooms()[:3]
    return Volume(
        data=data,
        affine=ras_img.affine.copy(),
        spacing=tuple(float(z) for z in zooms),  # type: ignore[arg-type]
        source_path=vol.source_path,
        meta={**vol.meta, "orientation": "RAS"},
    )


def n4_bias_correction(vol: Volume, iterations: int = 50) -> Volume:
    """
    Apply N4 bias-field correction. Corrects low-frequency scanner inhomogeneity
    which otherwise biases registration and intensity-based segmentation.
    """
    import SimpleITK as sitk

    img = _volume_to_sitk(vol)
    img_f = sitk.Cast(img, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([iterations] * 3)
    corrected = corrector.Execute(img_f)
    out = _sitk_to_volume(corrected, vol.affine)
    out.meta = {**vol.meta, "n4_bias_corrected": True}
    out.source_path = vol.source_path
    return out


def resample_isotropic(vol: Volume, target_mm: float = 1.0) -> Volume:
    """Resample to isotropic voxel spacing using linear interpolation."""
    import SimpleITK as sitk

    img = _volume_to_sitk(vol)
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    new_spacing = (target_mm, target_mm, target_mm)
    new_size = [
        int(round(sz * spc / target_mm))
        for sz, spc in zip(original_size, original_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled = resampler.Execute(img)
    out = _sitk_to_volume(resampled, vol.affine)
    out.meta = {
        **vol.meta,
        "resampled_to_mm": target_mm,
        "orig_spacing": vol.spacing,
    }
    out.source_path = vol.source_path
    return out


def skull_strip(vol: Volume, cfg: InferenceConfig) -> Volume:
    """
    Optional skull-stripping. Tries antspynet (CPU) first; silently skips if
    unavailable. Returns vol unchanged if skipping so the pipeline remains
    idempotent.
    """
    try:
        import ants  # type: ignore
        from antspynet import brain_extraction  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dep
        logger.warning("skull_strip: antspynet unavailable (%s) – skipping", exc)
        return vol

    try:
        ants_img = ants.from_numpy(
            vol.data.astype(np.float32),
            origin=tuple(float(x) for x in vol.affine[:3, 3]),
            spacing=tuple(float(s) for s in vol.spacing),
        )
        mask = brain_extraction(ants_img, modality="t1")
        mask_arr = mask.numpy().astype(bool)
        out_data = np.where(mask_arr, vol.data, 0.0).astype(np.float32)
        return Volume(
            data=out_data,
            affine=vol.affine.copy(),
            spacing=vol.spacing,
            source_path=vol.source_path,
            meta={**vol.meta, "skull_stripped": True},
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("skull_strip failed: %s – returning original volume", exc)
        return vol


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------


def preprocess_volume(
    vol: Volume,
    cfg: InferenceConfig,
    *,
    cache_out_path: Path | None = None,
    force: bool = False,
) -> Volume:
    """
    Apply the configured preprocessing chain once per timepoint.

    Order is fixed: RAS → skull-strip (optional) → N4 → isotropic resample.
    """
    if (
        cache_out_path is not None
        and cache_out_path.exists()
        and not force
    ):
        from ml.inference.io import load_nifti

        logger.info("preprocess: cache hit – %s", cache_out_path)
        cached = load_nifti(cache_out_path)
        cached.meta["from_cache"] = True
        return cached

    out = vol
    if cfg.orient_to_ras:
        out = orient_to_ras(out)

    if cfg.skull_strip:
        out = skull_strip(out, cfg)

    if cfg.n4_bias_correction:
        try:
            out = n4_bias_correction(out)
        except Exception as exc:  # pragma: no cover
            logger.warning("N4 failed (%s) – continuing without bias correction", exc)

    try:
        out = resample_isotropic(out, cfg.isotropic_spacing_mm)
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Isotropic resample failed (%s) – continuing at original spacing", exc
        )

    if cache_out_path is not None:
        save_nifti(out, cache_out_path)

    return out


def preprocess_from_path(
    nifti_path: Path, cfg: InferenceConfig, *, force: bool = False
) -> Tuple[Volume, Path]:
    """Load → preprocess → cache. Returns the preprocessed Volume and its cached path."""
    from ml.inference.io import load_nifti

    cdir = cache_dir_for(
        cfg.cache_dir,
        nifti_path,
        "preproc",
        f"iso{cfg.isotropic_spacing_mm}",
        "n4" if cfg.n4_bias_correction else "no-n4",
        "ras" if cfg.orient_to_ras else "no-ras",
        "bet" if cfg.skull_strip else "no-bet",
    )
    cached = cdir / "preprocessed.nii.gz"
    raw = load_nifti(nifti_path)
    pre = preprocess_volume(raw, cfg, cache_out_path=cached, force=force)
    pre.source_path = str(nifti_path)
    return pre, cached
