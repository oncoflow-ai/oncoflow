"""
adapters/nnunet.py – nnU-Net v2 segmentation adapter.

Two backends:

  * `local`  – nnU-Net v2 Python API, preferred config is `3d_lowres` for speed
               on CPU/MPS. If no usable checkpoint is found, falls back to a
               MONAI BraTS bundle when available. Otherwise reports
               `is_available() == False`.
  * `gpu-prod` – invokes the `nnUNetv2_predict` CLI in a subprocess with the
               configured dataset ID and `3d_fullres` + TTA (per
               IMPLEMENTATION_PLAN.md Step 4.2).

Environment variables honoured:
    nnUNet_results   – directory containing trained Dataset fingerprints
    nnUNet_raw       – required by nnUNetv2_predict CLI
    nnUNet_preprocessed – required by nnUNetv2_predict CLI
    OFLOW_NNUNET_CHECKPOINT_DIR – override local checkpoint directory
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

from ml.inference.adapters.base import (
    AdapterResult,
    Bbox,
    SegmentationAdapter,
    empty_result,
)
from ml.inference.io import Volume

logger = logging.getLogger(__name__)


class NNUNetAdapter(SegmentationAdapter):
    """nnU-Net v2 adapter with local (Python API) and GPU-prod (CLI) backends."""

    name = "nnunet"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._predictor = None          # nnU-Net Python API
        self._monai_bundle = None       # MONAI BraTS bundle inferer
        self._mode: str = ""            # "nnunet_api" | "nnunet_cli" | "monai" | "none"

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        if self.cfg.backend == "gpu-prod":
            if shutil.which("nnUNetv2_predict") is None:
                return False
            if os.environ.get("nnUNet_results") is None:
                return False
            return True

        # Local backend: probe nnunetv2 or MONAI bundle.
        try:
            import nnunetv2  # noqa: F401
        except ImportError:
            nnunetv2_ok = False
        else:
            nnunetv2_ok = True

        if nnunetv2_ok and self._find_local_checkpoint() is not None:
            return True

        # Fallback probe: MONAI BraTS bundle.
        try:
            from monai.bundle import ConfigParser  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            return False
        return self._find_monai_bundle() is not None

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self._loaded:
            return

        if self.cfg.backend == "gpu-prod":
            self._mode = "nnunet_cli"
            self._loaded = True
            return

        # Local: prefer nnunetv2 Python API if checkpoint exists.
        ckpt = self._find_local_checkpoint()
        if ckpt is not None:
            try:
                import torch
                from nnunetv2.inference.predict_from_raw_data import (
                    nnUNetPredictor,
                )

                device = torch.device(self.cfg.resolve_device())
                predictor = nnUNetPredictor(
                    tile_step_size=0.5,
                    use_gaussian=True,
                    use_mirroring=self.cfg.nnunet_use_tta_local,
                    perform_everything_on_device=(device.type != "cpu"),
                    device=device,
                    verbose=False,
                    verbose_preprocessing=False,
                    allow_tqdm=False,
                )
                predictor.initialize_from_trained_model_folder(
                    str(ckpt),
                    use_folds=("all",),
                    checkpoint_name="checkpoint_final.pth",
                )
                self._predictor = predictor
                self._mode = "nnunet_api"
                self._loaded = True
                logger.info("NNUNetAdapter: loaded nnU-Net v2 from %s", ckpt)
                return
            except Exception as exc:
                logger.warning(
                    "NNUNetAdapter: nnunetv2 Python API load failed (%s); trying MONAI bundle",
                    exc,
                )

        bundle = self._find_monai_bundle()
        if bundle is not None:
            self._monai_bundle = bundle
            self._mode = "monai"
            self._loaded = True
            logger.info("NNUNetAdapter: loaded MONAI BraTS bundle at %s", bundle)
            return

        self._mode = "none"
        self._loaded = True

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _predict_impl(
        self, vol: Volume, roi: Optional[Bbox]
    ) -> AdapterResult:
        if self._mode == "nnunet_cli":
            return self._predict_cli(vol)
        if self._mode == "nnunet_api":
            return self._predict_api(vol)
        if self._mode == "monai":
            return self._predict_monai(vol)
        return empty_result(vol.shape, error="no backend loaded", model=self.name)

    # ---- CLI backend (gpu-prod) ---------------------------------------

    def _predict_cli(self, vol: Volume) -> AdapterResult:
        """Call `nnUNetv2_predict` CLI on a temp directory (Step 4.2)."""
        dataset_id = self.cfg.nnunet_dataset_id
        config = self.cfg.nnunet_config_gpu

        with tempfile.TemporaryDirectory(prefix="oncoflow_nnunet_") as td:
            tmp = Path(td)
            in_dir = tmp / "input"
            out_dir = tmp / "output"
            in_dir.mkdir()
            out_dir.mkdir()

            stem = "case000"
            in_file = in_dir / f"{stem}_0000.nii.gz"
            img = nib.Nifti1Image(vol.data, vol.affine)
            nib.save(img, str(in_file))

            cmd = [
                "nnUNetv2_predict",
                "-i", str(in_dir),
                "-o", str(out_dir),
                "-d", dataset_id,
                "-c", config,
            ]
            if not self.cfg.nnunet_use_tta_gpu:
                cmd += ["--disable_tta"]

            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                return empty_result(
                    vol.shape,
                    error=f"nnUNetv2_predict failed rc={res.returncode}: {res.stderr[-300:]}",
                    model=self.name,
                )

            pred_file = out_dir / f"{stem}.nii.gz"
            if not pred_file.exists():
                # nnU-Net sometimes writes without the trailing _0000
                candidates = list(out_dir.glob("*.nii.gz"))
                if not candidates:
                    return empty_result(
                        vol.shape,
                        error="nnU-Net produced no output NIfTI",
                        model=self.name,
                    )
                pred_file = candidates[0]

            pred_img = nib.load(str(pred_file))
            pred = np.asarray(pred_img.dataobj)
            binary = (pred > 0).astype(np.uint8)
            return {
                "mask": binary,
                "prob": None,
                "runtime_s": 0.0,
                "meta": {
                    "mode": "nnunet_cli",
                    "dataset": dataset_id,
                    "config": config,
                    "tta": self.cfg.nnunet_use_tta_gpu,
                    "labels_seen": sorted(set(np.unique(pred).tolist())),
                },
            }

    # ---- Python API backend (local) -----------------------------------

    def _predict_api(self, vol: Volume) -> AdapterResult:
        """Call nnunetv2 Python predictor directly (no subprocess)."""
        # Save volume to a temp file and use predict_from_files (safest API).
        with tempfile.TemporaryDirectory(prefix="oncoflow_nnunet_") as td:
            tmp = Path(td)
            in_dir = tmp / "input"
            out_dir = tmp / "output"
            in_dir.mkdir()
            out_dir.mkdir()
            stem = "case000"
            in_file = in_dir / f"{stem}_0000.nii.gz"
            nib.save(nib.Nifti1Image(vol.data, vol.affine), str(in_file))

            self._predictor.predict_from_files(  # type: ignore[union-attr]
                list_of_lists_or_source_folder=str(in_dir),
                output_folder_or_list_of_truncated_output_files=str(out_dir),
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
                folder_with_segs_from_prev_stage=None,
                num_parts=1,
                part_id=0,
            )

            # Find produced file
            candidates = list(out_dir.glob("*.nii.gz"))
            if not candidates:
                return empty_result(
                    vol.shape,
                    error="nnU-Net API wrote no output",
                    model=self.name,
                )
            pred = np.asarray(nib.load(str(candidates[0])).dataobj)
            binary = (pred > 0).astype(np.uint8)
            return {
                "mask": binary,
                "prob": None,
                "runtime_s": 0.0,
                "meta": {
                    "mode": "nnunet_api",
                    "config": self.cfg.nnunet_config_local,
                    "tta": self.cfg.nnunet_use_tta_local,
                    "labels_seen": sorted(set(np.unique(pred).tolist())),
                },
            }

    # ---- MONAI bundle backend (local fallback) ------------------------

    def _predict_monai(self, vol: Volume) -> AdapterResult:
        """
        Run the MONAI BraTS segmentation bundle if present in
        `weights_dir/monai_brats_bundle`. This is a recognised open fallback
        when no community nnU-Net BraTS checkpoint is available.
        """
        try:
            import torch
            from monai.bundle import ConfigParser
        except ImportError:
            return empty_result(
                vol.shape, error="MONAI not installed", model=self.name
            )

        assert self._monai_bundle is not None

        parser = ConfigParser()
        parser.read_config(str(self._monai_bundle / "configs" / "inference.json"))
        parser.parse()

        # Best-effort: we load the network and run with a single modality.
        # Production MONAI BraTS bundles expect 4 modalities; we replicate the
        # single modality to 4 channels as a degraded fallback.
        try:
            network = parser.get_parsed_content("network", instantiate=True)
            device = torch.device(self.cfg.resolve_device())
            network = network.to(device).eval()

            x = torch.from_numpy(vol.data.astype(np.float32))[None, None]
            x = x.repeat(1, 4, 1, 1, 1).to(device)

            with torch.no_grad():
                y = network(x)
                if isinstance(y, (list, tuple)):
                    y = y[0]
                y = torch.sigmoid(y).cpu().numpy()

            # Take channel 0 (whole tumor) if multi-channel
            prob = y[0, 0] if y.ndim == 5 else y[0]
            binary = (prob > 0.5).astype(np.uint8)
            return {
                "mask": binary,
                "prob": prob.astype(np.float32),
                "runtime_s": 0.0,
                "meta": {"mode": "monai_bundle", "bundle": str(self._monai_bundle)},
            }
        except Exception as exc:
            return empty_result(
                vol.shape, error=f"MONAI bundle inference failed: {exc}",
                model=self.name,
            )

    # ------------------------------------------------------------------
    # Checkpoint discovery
    # ------------------------------------------------------------------

    def _find_local_checkpoint(self) -> Optional[Path]:
        """Look for a usable nnU-Net trained model folder."""
        override = os.environ.get("OFLOW_NNUNET_CHECKPOINT_DIR")
        if override:
            p = Path(override).expanduser()
            if (p / "checkpoint_final.pth").exists() or (
                p / "fold_all" / "checkpoint_final.pth"
            ).exists():
                return p

        results_root = os.environ.get("nnUNet_results")
        if results_root:
            root = Path(results_root)
            if root.exists():
                for d in root.rglob("checkpoint_final.pth"):
                    return d.parent.parent  # folder containing fold_x

        candidate = self.cfg.weights_dir / "nnunet_brats_lowres"
        if (candidate / "fold_all" / "checkpoint_final.pth").exists():
            return candidate

        return None

    def _find_monai_bundle(self) -> Optional[Path]:
        candidate = self.cfg.weights_dir / "monai_brats_bundle"
        if (candidate / "configs" / "inference.json").exists():
            return candidate
        return None
