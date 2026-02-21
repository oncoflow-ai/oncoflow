"""
dicom_utils.py – DICOM → NIfTI conversion utilities for OncoFlow exploration.

Provides two conversion paths for benchmarking:
  1. SimpleITK-based (in-process, fine-grained control)
  2. dcm2niix-based (subprocess, clinical gold-standard)

Also provides preprocessing helpers:
  - N4 bias field correction (SimpleITK)
  - Z-score + percentile intensity normalisation
  - Isotropic resampling
"""

import os
import time
import subprocess
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import nibabel as nib
import pydicom
import SimpleITK as sitk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def find_dcm_series(dicom_dir: str | Path) -> List[str]:
    """Return sorted list of .dcm file paths inside a directory."""
    p = Path(dicom_dir)
    dcm_files = sorted(p.glob("*.dcm"))
    if not dcm_files:
        dcm_files = sorted(p.rglob("*.dcm"))  # recursive fallback
    return [str(f) for f in dcm_files]


# ---------------------------------------------------------------------------
# Option A1: SimpleITK conversion
# ---------------------------------------------------------------------------

def dicom_to_nifti_sitk(
    dicom_dir: str | Path,
    out_path: str | Path,
    normalize: bool = False,
) -> Dict:
    """
    Convert a DICOM series to NIfTI using SimpleITK.

    Args:
        dicom_dir: Directory containing .dcm files for ONE series.
        out_path:  Output .nii.gz file path.
        normalize: If True, apply z-score normalisation after conversion.

    Returns:
        dict with keys: 'size', 'spacing', 'duration_s', 'out_path'
    """
    t0 = time.perf_counter()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not dicom_names:
        raise ValueError(f"No DICOM series found in {dicom_dir}")

    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    if normalize:
        arr = sitk.GetArrayFromImage(image).astype(np.float32)
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        norm_img = sitk.GetImageFromArray(arr)
        norm_img.CopyInformation(image)
        image = norm_img

    sitk.WriteImage(image, str(out_path))
    duration = time.perf_counter() - t0

    return {
        "method": "SimpleITK",
        "out_path": str(out_path),
        "size": image.GetSize(),
        "spacing_mm": image.GetSpacing(),
        "duration_s": round(duration, 3),
        "file_size_mb": round(out_path.stat().st_size / 1e6, 2),
    }


# ---------------------------------------------------------------------------
# Option A2: dcm2niix conversion
# ---------------------------------------------------------------------------

def dicom_to_nifti_dcm2niix(
    dicom_dir: str | Path,
    out_dir: str | Path,
    filename: str = "converted",
) -> Dict:
    """
    Convert a DICOM series to NIfTI using dcm2niix (subprocess).

    Requires dcm2niix to be installed: brew install dcm2niix

    Returns:
        dict with keys: 'out_path', 'duration_s', 'file_size_mb', 'stdout'
    """
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("dcm2niix") is None:
        return {
            "method": "dcm2niix",
            "error": "dcm2niix not found. Install with: brew install dcm2niix",
            "available": False,
        }

    cmd = [
        "dcm2niix",
        "-z", "y",          # gzip output
        "-f", filename,     # output filename
        "-o", str(out_dir),
        str(dicom_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.perf_counter() - t0

    # Find output file
    nifti_files = list(out_dir.glob(f"{filename}*.nii.gz"))
    out_path = str(nifti_files[0]) if nifti_files else None

    return {
        "method": "dcm2niix",
        "out_path": out_path,
        "duration_s": round(duration, 3),
        "file_size_mb": round(Path(out_path).stat().st_size / 1e6, 2) if out_path else None,
        "returncode": result.returncode,
        "stdout": result.stdout[-500:],  # last 500 chars
        "available": True,
    }


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def apply_n4_bias_correction(image: sitk.Image, n_iterations: int = 50) -> sitk.Image:
    """
    Apply N4 bias field correction via SimpleITK.
    Recommended for T1 MRI to remove scanner inhomogeneity.
    """
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([n_iterations] * 3)
    # N4 requires float
    image_float = sitk.Cast(image, sitk.sitkFloat32)
    return corrector.Execute(image_float)


def normalise_intensity_zscore(arr: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Z-score normalise intensity. If mask provided, compute stats only inside mask."""
    arr = arr.astype(np.float32)
    if mask is not None and mask.sum() > 0:
        vals = arr[mask > 0]
    else:
        vals = arr[arr > 0]
    mean, std = vals.mean(), vals.std()
    return (arr - mean) / (std + 1e-8)


def normalise_intensity_percentile(
    arr: np.ndarray,
    low: float = 1.0,
    high: float = 99.0,
) -> np.ndarray:
    """Clip and rescale to [0, 1] using percentile clipping."""
    arr = arr.astype(np.float32)
    p_low = np.percentile(arr[arr > 0], low)
    p_high = np.percentile(arr[arr > 0], high)
    arr = np.clip(arr, p_low, p_high)
    return (arr - p_low) / (p_high - p_low + 1e-8)


def resample_to_isotropic(image: sitk.Image, target_spacing: float = 1.0) -> sitk.Image:
    """
    Resample a SimpleITK image to isotropic voxel spacing.

    Args:
        image: Input SimpleITK image.
        target_spacing: Target spacing in mm (e.g. 1.0 = 1mm isotropic).

    Returns:
        Resampled SimpleITK image.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_spacing = [target_spacing] * 3
    new_size = [
        int(round(sz * spc / target_spacing))
        for sz, spc in zip(original_size, original_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(image)


# ---------------------------------------------------------------------------
# NIfTI loading helpers
# ---------------------------------------------------------------------------

def load_nifti(path: str | Path) -> Tuple[np.ndarray, np.ndarray, object]:
    """
    Load a NIfTI file.

    Returns:
        (data_array, affine, header)
    """
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32), img.affine, img.header


def save_nifti(arr: np.ndarray, affine: np.ndarray, out_path: str | Path) -> None:
    """Save numpy array as NIfTI."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr, affine), str(out_path))


def get_nifti_info(path: str | Path) -> Dict:
    """Return shape, spacing, and file size for a NIfTI file."""
    img = nib.load(str(path))
    zooms = img.header.get_zooms()
    return {
        "path": str(path),
        "shape": img.shape,
        "spacing_mm": tuple(round(float(z), 3) for z in zooms[:3]),
        "file_size_mb": round(Path(path).stat().st_size / 1e6, 2),
        "dtype": str(img.get_data_dtype()),
    }


# ---------------------------------------------------------------------------
# P01 data helpers
# ---------------------------------------------------------------------------

TIMEPOINTS = ["baseline", "fu1", "fu2", "fu3", "fu4"]
MODALITIES = ["t1", "t1c", "t2", "fla"]


def get_p01_brats_paths(data_root: str | Path) -> Dict[str, Dict[str, str]]:
    """
    Return a nested dict of all BraTS NIfTI paths for P01.

    Usage:
        paths = get_p01_brats_paths("/path/to/P01/BraTS")
        paths["baseline"]["t1c"]  # → "/path/to/P01/BraTS/baseline/t1c.nii.gz"
    """
    root = Path(data_root)
    result = {}
    for tp in TIMEPOINTS:
        tp_dir = root / tp
        if tp_dir.exists():
            result[tp] = {
                mod: str(tp_dir / f"{mod}.nii.gz")
                for mod in MODALITIES
                if (tp_dir / f"{mod}.nii.gz").exists()
            }
    return result


def get_p01_mask_paths(data_root: str | Path) -> Dict[str, str]:
    """
    Return ground-truth tumor mask paths keyed by timepoint name.

    Usage:
        masks = get_p01_mask_paths("/path/to/P01/tumor segmentation")
        masks["baseline"]  # → ".../P01_tumor_mask_baseline.nii.gz"
    """
    root = Path(data_root)
    result = {}
    for tp in TIMEPOINTS:
        fpath = root / f"P01_tumor_mask_{tp}.nii.gz"
        if fpath.exists():
            result[tp] = str(fpath)
    return result


def get_p01_dicom_series(data_root: str | Path) -> Dict[str, str]:
    """
    Return dict mapping series name → directory for all DICOM series in P01.

    Usage:
        series = get_p01_dicom_series("/path/to/P01/DICOM")
        series["T1C_2020-01-07"]  # → "/path/to/P01/DICOM/T1C_2020-01-07"
    """
    root = Path(data_root)
    return {
        d.name: str(d)
        for d in sorted(root.iterdir())
        if d.is_dir() and not d.name.startswith(".")
    }
