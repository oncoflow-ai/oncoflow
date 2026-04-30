"""
io.py – NIfTI I/O, Volume dataclass, path helpers, and disk caching.

Thin wrappers around nibabel so the rest of the package deals only with the
`Volume` dataclass (raw array + affine + spacing) and doesn't re-open files.

Caching: content-addressed via SHA-256 of the file bytes + a config-derived
suffix. Preprocessed volumes and per-model masks are stored under
`cfg.cache_dir/<sha>/<kind>.nii.gz`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np


# ---------------------------------------------------------------------------
# Volume dataclass
# ---------------------------------------------------------------------------


@dataclass
class Volume:
    """In-memory 3-D medical image with voxel-to-world affine."""

    data: np.ndarray  # (H, W, D) float32 or uint8
    affine: np.ndarray  # (4, 4)
    spacing: Tuple[float, float, float]
    source_path: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.data.shape)  # type: ignore[return-value]

    @property
    def voxel_volume_mm3(self) -> float:
        return float(np.prod(self.spacing))

    def volume_cm3(self, threshold: float = 0.5) -> float:
        n_vox = int((self.data > threshold).sum())
        return n_vox * self.voxel_volume_mm3 / 1000.0

    def as_binary(self, threshold: float = 0.5) -> np.ndarray:
        return (self.data > threshold).astype(np.uint8)

    def copy_with(self, data: np.ndarray, meta: dict | None = None) -> "Volume":
        new_meta = dict(self.meta)
        if meta:
            new_meta.update(meta)
        return Volume(
            data=data,
            affine=self.affine.copy(),
            spacing=self.spacing,
            source_path=self.source_path,
            meta=new_meta,
        )


# ---------------------------------------------------------------------------
# NIfTI read/write
# ---------------------------------------------------------------------------


def load_nifti(path: str | Path) -> Volume:
    """Load a .nii.gz into a Volume. Data returned as float32."""
    p = Path(path)
    img = nib.load(str(p))
    data = np.asarray(img.dataobj, dtype=np.float32)
    zooms = img.header.get_zooms()[:3]
    return Volume(
        data=data,
        affine=img.affine.copy(),
        spacing=tuple(float(z) for z in zooms),  # type: ignore[arg-type]
        source_path=str(p),
    )


def load_nifti_mask(path: str | Path) -> Volume:
    """Load a mask NIfTI as uint8 binary."""
    p = Path(path)
    img = nib.load(str(p))
    data = np.asarray(img.dataobj)
    # BraTS masks can have label values {0,1,2,3,4}; we treat any non-zero as tumor.
    binary = (data > 0).astype(np.uint8)
    zooms = img.header.get_zooms()[:3]
    return Volume(
        data=binary,
        affine=img.affine.copy(),
        spacing=tuple(float(z) for z in zooms),  # type: ignore[arg-type]
        source_path=str(p),
    )


def save_nifti(vol: Volume, path: str | Path) -> Path:
    """Save a Volume to a NIfTI file. Returns the output Path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(vol.data, vol.affine)
    nib.save(img, str(out))
    return out


# ---------------------------------------------------------------------------
# Content-addressed caching
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_key(input_path: str | Path, *parts: str) -> str:
    """Build a cache subdirectory name from input SHA + arbitrary string parts."""
    sha = sha256_file(input_path)
    suffix = "_".join(p.replace("/", "-") for p in parts if p)
    return f"{sha[:16]}__{suffix}" if suffix else sha[:16]


def cache_dir_for(
    cache_root: Path, input_path: str | Path, *parts: str
) -> Path:
    d = cache_root / cache_key(input_path, *parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_cache_metadata(cache_dir: Path, metadata: dict) -> None:
    with open(cache_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
