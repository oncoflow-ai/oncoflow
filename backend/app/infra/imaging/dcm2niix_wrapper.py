from __future__ import annotations

import gzip
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from app.infra.imaging.dicom_inventory import DicomSeriesRecord
from app.infra.imaging.geometry import extract_geometry_metadata


def convert_dicom_series(
    series_record: DicomSeriesRecord,
    output_dir: str | Path,
    *,
    filename_stem: str,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    nifti_path = output_path / f"{filename_stem}.nii.gz"
    sidecar_path = output_path / f"{filename_stem}.json"
    log_path = output_path / f"{filename_stem}.log"
    geometry = extract_geometry_metadata(series_record)

    if shutil.which("dcm2niix"):
        cmd = [
            "dcm2niix",
            "-z",
            "y",
            "-f",
            filename_stem,
            "-o",
            str(output_path),
            str(series_record.files[0].parent),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        log_path.write_text(result.stdout + "\n" + result.stderr)
        if result.returncode != 0:
            raise RuntimeError("dcm2niix conversion failed")
        derived_nifti = next(output_path.glob(f"{filename_stem}*.nii.gz"), None)
        derived_sidecar = next(output_path.glob(f"{filename_stem}*.json"), None)
        return {
            "nifti_path": str(derived_nifti or nifti_path),
            "sidecar_path": str(derived_sidecar or sidecar_path),
            "log_path": str(log_path),
            "geometry": geometry,
            "converter": "dcm2niix",
        }

    with gzip.open(nifti_path, "wb") as nifti_file:
        nifti_file.write(b"ONCOFLOW_NIFTI_PLACEHOLDER")
    sidecar_path.write_text(json.dumps({"geometry": geometry, "converter": "placeholder"}, indent=2))
    log_path.write_text("dcm2niix unavailable; wrote placeholder NIfTI artifact for contract testing\n")
    return {
        "nifti_path": str(nifti_path),
        "sidecar_path": str(sidecar_path),
        "log_path": str(log_path),
        "geometry": geometry,
        "converter": "placeholder",
    }
