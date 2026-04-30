"""
registration/register.py – Rigid / Affine / SyN image registration via ANTsPy.

Implements Stage 2 of IMPLEMENTATION_PLAN.md Step 4.7. The follow-up volume
is registered into the baseline coordinate space; the same transform is then
applied to the follow-up segmentation mask (with nearest-neighbour
interpolation to preserve binary labels).

Quality is reported as normalised cross-correlation (NCC) before/after.
Production fallback: if ANTsPy is unavailable we use a pure-SimpleITK ITKv4
registration with the same Mattes-MI metric; quality gates work identically.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np

from ml.inference.io import Volume

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RegistrationResult:
    """Outcome of one image-to-image registration."""

    warped_image: Volume          # moving warped into fixed space
    fwd_transforms: List[str]     # filesystem paths to transform files (ANTs .mat / .h5)
    ncc_before: float             # normalised cross-correlation of fixed vs raw moving
    ncc_after: float              # normalised cross-correlation of fixed vs warped moving
    method: str                   # "Rigid" | "Affine" | "SyN" | "SimpleITK-Affine"
    backend: str                  # "ants" | "sitk"
    runtime_s: float = 0.0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------


def ncc(a: np.ndarray, b: np.ndarray, *, mask: Optional[np.ndarray] = None) -> float:
    """
    Normalised cross-correlation on numpy arrays (unit variance).

    Values in [-1, 1]; higher is better. If either array is constant, returns 0.
    """
    if a.shape != b.shape:
        raise ValueError(f"NCC shape mismatch: {a.shape} vs {b.shape}")
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    if mask is not None:
        idx = mask > 0
        af = af[idx]
        bf = bf[idx]
    else:
        af = af.ravel()
        bf = bf.ravel()
    af -= af.mean()
    bf -= bf.mean()
    denom = np.sqrt((af ** 2).sum() * (bf ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float(np.clip((af * bf).sum() / denom, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Public registration entry point
# ---------------------------------------------------------------------------


def register_followup_to_baseline(
    fixed: Volume,
    moving: Volume,
    *,
    method: str = "Affine",
) -> RegistrationResult:
    """
    Register `moving` into `fixed` space. Uses ANTsPy if installed, otherwise
    falls back to a pure-SimpleITK pipeline with the same metric.
    """
    try:
        import ants  # type: ignore  # noqa: F401
        return _register_ants(fixed, moving, method=method)
    except ImportError:
        logger.info("ANTsPy not installed – falling back to SimpleITK")
        return _register_sitk(fixed, moving)


# ---------------------------------------------------------------------------
# ANTsPy backend
# ---------------------------------------------------------------------------


def _register_ants(
    fixed: Volume, moving: Volume, *, method: str
) -> RegistrationResult:
    import ants
    import time as _time

    t0 = _time.perf_counter()
    fixed_ants = _volume_to_ants(fixed)
    moving_ants = _volume_to_ants(moving)

    ncc_before_val = float(
        ants.image_similarity(fixed_ants, moving_ants, metric_type="Correlation")
    )
    # ANTs' Correlation metric is negative NCC (lower = better). Convert to NCC.
    ncc_before_val = -ncc_before_val

    type_map = {"Rigid": "Rigid", "Affine": "Affine", "SyN": "SyN"}
    if method not in type_map:
        raise ValueError(f"Unsupported registration method: {method!r}")

    result = ants.registration(
        fixed=fixed_ants,
        moving=moving_ants,
        type_of_transform=type_map[method],
        aff_metric="mattes",
        aff_sampling=32,
        verbose=False,
    )

    warped = result["warpedmovout"]
    ncc_after_val = -float(
        ants.image_similarity(fixed_ants, warped, metric_type="Correlation")
    )

    warped_vol = _ants_to_volume(warped, ref_affine=fixed.affine)
    warped_vol.meta["warped"] = True
    warped_vol.meta["registration_method"] = method
    return RegistrationResult(
        warped_image=warped_vol,
        fwd_transforms=list(result.get("fwdtransforms", [])),
        ncc_before=ncc_before_val,
        ncc_after=ncc_after_val,
        method=method,
        backend="ants",
        runtime_s=_time.perf_counter() - t0,
        extra={"ants_keys": list(result.keys())},
    )


def _volume_to_ants(vol: Volume):
    import ants

    return ants.from_numpy(
        vol.data.astype(np.float32),
        origin=tuple(float(x) for x in vol.affine[:3, 3]),
        spacing=tuple(float(s) for s in vol.spacing),
    )


def _ants_to_volume(ants_img, ref_affine: np.ndarray) -> Volume:
    arr = ants_img.numpy().astype(np.float32)
    spacing = tuple(float(s) for s in ants_img.spacing)
    affine = ref_affine.copy()
    rot = affine[:3, :3]
    norms = np.linalg.norm(rot, axis=0)
    norms[norms == 0] = 1.0
    affine[:3, :3] = (rot / norms) * np.asarray(spacing)
    affine[:3, 3] = np.asarray(ants_img.origin)
    return Volume(data=arr, affine=affine, spacing=spacing)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SimpleITK fallback backend
# ---------------------------------------------------------------------------


def _register_sitk(fixed: Volume, moving: Volume) -> RegistrationResult:
    import SimpleITK as sitk
    import time as _time

    t0 = _time.perf_counter()

    fixed_sitk = _volume_to_sitk(fixed)
    moving_sitk = _volume_to_sitk(moving)

    fixed_f = sitk.Cast(fixed_sitk, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving_sitk, sitk.sitkFloat32)

    ncc_before_val = ncc(
        fixed.data, _resample_to(moving, fixed).data
    )

    initial = sitk.CenteredTransformInitializer(
        fixed_f, moving_f, sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.1, seed=1)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=200,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetInitialTransform(initial, inPlace=False)
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])

    transform = registration.Execute(fixed_f, moving_f)

    warped = sitk.Resample(
        moving_f,
        fixed_f,
        transform,
        sitk.sitkLinear,
        0.0,
        moving_f.GetPixelID(),
    )
    warped_arr = sitk.GetArrayFromImage(warped).transpose(2, 1, 0).astype(np.float32)
    warped_vol = Volume(
        data=warped_arr,
        affine=fixed.affine.copy(),
        spacing=fixed.spacing,
    )
    ncc_after_val = ncc(fixed.data, warped_vol.data)

    # Persist the transform to a tmp file so mask warping can reload it later.
    tx_path = Path(tempfile.mkstemp(suffix=".tfm", prefix="oncoflow_sitk_")[1])
    sitk.WriteTransform(transform, str(tx_path))

    return RegistrationResult(
        warped_image=warped_vol,
        fwd_transforms=[str(tx_path)],
        ncc_before=ncc_before_val,
        ncc_after=ncc_after_val,
        method="Affine",
        backend="sitk",
        runtime_s=_time.perf_counter() - t0,
    )


def _volume_to_sitk(vol: Volume):
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(np.ascontiguousarray(vol.data.transpose(2, 1, 0)))
    img.SetSpacing(tuple(float(s) for s in vol.spacing))
    img.SetOrigin(tuple(float(x) for x in vol.affine[:3, 3]))
    return img


def _resample_to(src: Volume, ref: Volume) -> Volume:
    """Trivial nearest-spacing resample used only for the pre-registration NCC."""
    if src.shape == ref.shape:
        return src
    import SimpleITK as sitk

    src_img = _volume_to_sitk(src)
    ref_img = _volume_to_sitk(ref)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    out = resampler.Execute(sitk.Cast(src_img, sitk.sitkFloat32))
    arr = sitk.GetArrayFromImage(out).transpose(2, 1, 0).astype(np.float32)
    return Volume(data=arr, affine=ref.affine.copy(), spacing=ref.spacing)


# ---------------------------------------------------------------------------
# Mask / volume warping (post-registration)
# ---------------------------------------------------------------------------


def warp_mask(
    mask: Volume,
    result: RegistrationResult,
    reference: Volume,
) -> Volume:
    """
    Apply the forward transform to a binary mask. Nearest-neighbour is used to
    preserve label values (critical – do not interpolate binary masks).
    """
    if result.backend == "ants":
        try:
            import ants

            mask_img = _volume_to_ants(mask)
            ref_img = _volume_to_ants(reference)
            warped = ants.apply_transforms(
                fixed=ref_img,
                moving=mask_img,
                transformlist=result.fwd_transforms,
                interpolator="nearestNeighbor",
            )
            return _ants_to_volume(warped, ref_affine=reference.affine)
        except Exception as exc:
            logger.warning("ANTs mask warp failed (%s) – attempting SITK", exc)

    import SimpleITK as sitk

    tx_path = result.fwd_transforms[0]
    transform = sitk.ReadTransform(tx_path)
    mask_sitk = _volume_to_sitk(mask)
    ref_sitk = _volume_to_sitk(reference)
    warped = sitk.Resample(
        mask_sitk,
        ref_sitk,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        mask_sitk.GetPixelID(),
    )
    arr = sitk.GetArrayFromImage(warped).transpose(2, 1, 0).astype(np.uint8)
    return Volume(data=arr, affine=reference.affine.copy(), spacing=reference.spacing)


def warp_volume(
    moving: Volume,
    result: RegistrationResult,
    reference: Volume,
) -> Volume:
    """Apply the forward transform to an intensity volume with linear interpolation."""
    if result.backend == "ants":
        try:
            import ants

            moving_img = _volume_to_ants(moving)
            ref_img = _volume_to_ants(reference)
            warped = ants.apply_transforms(
                fixed=ref_img,
                moving=moving_img,
                transformlist=result.fwd_transforms,
                interpolator="linear",
            )
            return _ants_to_volume(warped, ref_affine=reference.affine)
        except Exception as exc:
            logger.warning("ANTs volume warp failed (%s) – attempting SITK", exc)

    import SimpleITK as sitk

    tx_path = result.fwd_transforms[0]
    transform = sitk.ReadTransform(tx_path)
    moving_sitk = sitk.Cast(_volume_to_sitk(moving), sitk.sitkFloat32)
    ref_sitk = _volume_to_sitk(reference)
    warped = sitk.Resample(
        moving_sitk,
        ref_sitk,
        transform,
        sitk.sitkLinear,
        0.0,
        moving_sitk.GetPixelID(),
    )
    arr = sitk.GetArrayFromImage(warped).transpose(2, 1, 0).astype(np.float32)
    return Volume(data=arr, affine=reference.affine.copy(), spacing=reference.spacing)
