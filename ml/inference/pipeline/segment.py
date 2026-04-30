"""
pipeline/segment.py – single-study segmentation pipeline.

Orchestration steps
-------------------
1. Preprocess (RAS + N4 + 1 mm isotropic), cached on disk.
2. Run nnU-Net adapter first (fast in local backend). Its mask bootstraps the
   ROI bounding-box that MedGemma and SAM consume — the 10x speed win on
   Mac/MPS.
3. Run the remaining enabled adapters (MedGemma, SAM3) in parallel.
4. Post-process each mask (optional largest-CC + closing).
5. Fuse via the configured ensemble strategy.
6. Post-process the ensemble mask.
7. Compute agreement score.
8. Persist per-model masks + ensemble mask + metadata to the cache.

This module is import-safe: it does not eagerly load any heavy model.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ml.inference.adapters import (
    AdapterResult,
    Bbox,
    build_adapter,
    empty_result,
)
from ml.inference.config import InferenceConfig
from ml.inference.ensemble import (
    agreement_score,
    clean_mask,
    fuse,
    PanelAgreement,
)
from ml.inference.io import Volume, load_nifti, save_nifti, cache_dir_for
from ml.inference.preprocessing import preprocess_from_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class StudySegmentation:
    """Output of the single-study segmentation pipeline."""

    # Inputs / IO
    source_path: str
    preprocessed_path: str
    output_dir: str

    # Masks (same shape/spacing as the preprocessed volume)
    ensemble_mask: np.ndarray
    per_model_masks: Dict[str, np.ndarray]

    # Metrics
    ensemble_volume_cm3: float
    per_model_volumes_cm3: Dict[str, float]
    per_model_runtime_s: Dict[str, float]
    panel_agreement: PanelAgreement

    # Pipeline metadata
    preprocessed_spacing: Tuple[float, float, float]
    shape: Tuple[int, int, int]
    config_snapshot: dict
    adapter_meta: Dict[str, dict] = field(default_factory=dict)
    roi_bbox: Optional[Tuple[int, int, int, int, int, int]] = None
    elapsed_s: float = 0.0

    def summary(self) -> dict:
        return {
            "source_path": self.source_path,
            "output_dir": self.output_dir,
            "shape": list(self.shape),
            "spacing": list(self.preprocessed_spacing),
            "ensemble_volume_cm3": round(float(self.ensemble_volume_cm3), 4),
            "per_model_volumes_cm3": {
                k: round(float(v), 4)
                for k, v in self.per_model_volumes_cm3.items()
            },
            "per_model_runtime_s": {
                k: round(float(v), 3)
                for k, v in self.per_model_runtime_s.items()
            },
            "panel_agreement": self.panel_agreement.as_dict(),
            "roi_bbox": list(self.roi_bbox) if self.roi_bbox else None,
            "adapter_meta": self.adapter_meta,
            "elapsed_s": round(self.elapsed_s, 3),
            "config_snapshot": self.config_snapshot,
        }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def segment_study(
    nifti_path: str | Path,
    cfg: Optional[InferenceConfig] = None,
    *,
    output_dir: Optional[Path] = None,
    use_cache: bool = True,
    save_outputs: bool = True,
) -> StudySegmentation:
    """
    Run the full 3-model segmentation panel on a single NIfTI input.

    Args:
        nifti_path: path to a .nii/.nii.gz intensity volume.
        cfg: InferenceConfig; defaults to `InferenceConfig()`.
        output_dir: where to write masks + metadata.json; defaults to a
            cache subdirectory.
        use_cache: read preprocessed volume + existing masks from the cache
            when available (skips re-running expensive adapters).
        save_outputs: write NIfTI masks to disk.
    """
    cfg = cfg or InferenceConfig()
    nifti_path = Path(nifti_path).expanduser().resolve()
    if not nifti_path.exists():
        raise FileNotFoundError(nifti_path)

    pipeline_t0 = time.perf_counter()

    # ---- Preprocess (cached) ---------------------------------------
    preproc, preproc_path = preprocess_from_path(
        nifti_path, cfg, force=not use_cache
    )

    # ---- Output directory -----------------------------------------
    run_dir = output_dir or cache_dir_for(
        cfg.cache_dir,
        nifti_path,
        "segment",
        ",".join(cfg.enabled_models),
        cfg.backend,
        cfg.ensemble_strategy,
    )
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- nnU-Net first (bootstrap ROI) ----------------------------
    per_model_results: Dict[str, AdapterResult] = {}
    roi: Optional[Bbox] = None

    enabled = [m for m in cfg.enabled_models if m]
    nnunet_first = "nnunet" in enabled

    if nnunet_first:
        nnunet_result = _run_adapter("nnunet", preproc, None, cfg, run_dir, use_cache)
        per_model_results["nnunet"] = nnunet_result
        if cfg.use_roi_bootstrap:
            roi = _derive_roi(
                nnunet_result["mask"], preproc.shape, cfg.roi_padding_voxels
            )
            logger.info("ROI bbox from nnU-Net: %s", roi)

    # ---- Fan out MedGemma + SAM (parallel where safe) -------------
    remaining = [m for m in enabled if m != "nnunet"]
    if remaining:
        results = _run_adapters_parallel(
            remaining, preproc, roi, cfg, run_dir, use_cache,
            parallel=cfg.parallel_adapters,
        )
        per_model_results.update(results)

    # ---- Post-process each mask -----------------------------------
    cleaned: Dict[str, np.ndarray] = {}
    probs: Dict[str, Optional[np.ndarray]] = {}
    for name, res in per_model_results.items():
        m = res["mask"].astype(np.uint8)
        if res.get("meta", {}).get("stub"):
            # Don't post-process empty/stub masks
            cleaned[name] = m
        else:
            cleaned[name] = clean_mask(
                m,
                keep_largest=cfg.keep_largest_cc,
                min_voxels=cfg.min_component_voxels,
                closing_radius=cfg.morph_closing_radius,
            )
        probs[name] = res.get("prob")

    # ---- Fuse -----------------------------------------------------
    # Only fuse non-stub masks; if everything is stub, the ensemble is zeros.
    usable = {n: m for n, m in cleaned.items()
              if not per_model_results[n].get("meta", {}).get("stub")}
    if not usable:
        ensemble = np.zeros(preproc.shape, dtype=np.uint8)
        logger.warning("All adapters returned stub masks – ensemble is empty")
    else:
        ensemble = fuse(
            cfg.ensemble_strategy,  # type: ignore[arg-type]
            usable,
            probs={n: probs[n] for n in usable if probs[n] is not None} or None,
        )
        ensemble = clean_mask(
            ensemble,
            keep_largest=cfg.keep_largest_cc,
            min_voxels=cfg.min_component_voxels,
            closing_radius=cfg.morph_closing_radius,
        )

    # ---- Agreement ------------------------------------------------
    agreement = agreement_score(usable, ensemble) if usable else PanelAgreement(
        per_model_dice={}, mean_agreement=0.0, level="low", models_used=()
    )

    # ---- Volumes --------------------------------------------------
    spacing = preproc.spacing
    per_model_vol = {
        name: _volume_cm3(m, spacing) for name, m in cleaned.items()
    }
    ens_vol = _volume_cm3(ensemble, spacing)

    # ---- Persist --------------------------------------------------
    if save_outputs:
        for name, m in cleaned.items():
            save_nifti(preproc.copy_with(m.astype(np.uint8)), run_dir / f"{name}_mask.nii.gz")
        save_nifti(
            preproc.copy_with(ensemble.astype(np.uint8)),
            run_dir / "ensemble_mask.nii.gz",
        )

    meta_snapshot = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in asdict(cfg).items()
    }

    seg = StudySegmentation(
        source_path=str(nifti_path),
        preprocessed_path=str(preproc_path),
        output_dir=str(run_dir),
        ensemble_mask=ensemble,
        per_model_masks=cleaned,
        ensemble_volume_cm3=ens_vol,
        per_model_volumes_cm3=per_model_vol,
        per_model_runtime_s={
            n: float(r.get("runtime_s", 0.0)) for n, r in per_model_results.items()
        },
        panel_agreement=agreement,
        preprocessed_spacing=spacing,
        shape=preproc.shape,
        config_snapshot=meta_snapshot,
        adapter_meta={n: dict(r.get("meta", {})) for n, r in per_model_results.items()},
        roi_bbox=roi.as_tuple() if roi else None,
        elapsed_s=time.perf_counter() - pipeline_t0,
    )

    if save_outputs:
        with open(run_dir / "segmentation.json", "w") as f:
            json.dump(seg.summary(), f, indent=2, default=str)

    return seg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_adapter(
    name: str,
    vol: Volume,
    roi: Optional[Bbox],
    cfg: InferenceConfig,
    run_dir: Path,
    use_cache: bool,
) -> AdapterResult:
    """Run one adapter (with optional disk cache)."""
    cache_file = run_dir / f"{name}_raw_mask.npz"
    if use_cache and cache_file.exists():
        try:
            data = np.load(cache_file, allow_pickle=True)
            mask = data["mask"]
            meta = (
                json.loads(str(data["meta"])) if "meta" in data else {}
            )
            logger.info("Adapter %s: cache hit", name)
            return {
                "mask": mask,
                "prob": None,
                "runtime_s": float(data["runtime_s"]) if "runtime_s" in data else 0.0,
                "meta": {**meta, "from_cache": True},
            }
        except Exception as exc:
            logger.warning("Cache read failed for %s: %s", name, exc)

    adapter = build_adapter(name, cfg)
    if not adapter.is_available():
        logger.info("Adapter %s unavailable – using stub", name)
        return empty_result(
            vol.shape,
            error="adapter unavailable",
            model=name,
        )

    result = adapter.predict(vol, roi=roi)

    # Best-effort cache write
    try:
        np.savez_compressed(
            cache_file,
            mask=result["mask"],
            runtime_s=np.float32(result.get("runtime_s", 0.0)),
            meta=json.dumps(result.get("meta", {})),
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Cache write failed for %s: %s", name, exc)
    return result


def _run_adapters_parallel(
    names: List[str],
    vol: Volume,
    roi: Optional[Bbox],
    cfg: InferenceConfig,
    run_dir: Path,
    use_cache: bool,
    *,
    parallel: bool,
) -> Dict[str, AdapterResult]:
    """Run several adapters. Thread-based; heavy model frameworks release the GIL."""
    if not parallel or len(names) <= 1:
        return {n: _run_adapter(n, vol, roi, cfg, run_dir, use_cache) for n in names}

    results: Dict[str, AdapterResult] = {}
    with ThreadPoolExecutor(max_workers=min(len(names), cfg.max_workers)) as ex:
        futures = {
            ex.submit(_run_adapter, n, vol, roi, cfg, run_dir, use_cache): n
            for n in names
        }
        for fut in as_completed(futures):
            n = futures[fut]
            try:
                results[n] = fut.result()
            except Exception as exc:
                logger.exception("Adapter %s failed in pool: %s", n, exc)
                results[n] = empty_result(
                    vol.shape, error=f"pool failure: {exc}", model=n
                )
    return results


def _derive_roi(
    mask: np.ndarray, shape: Tuple[int, int, int], padding: int
) -> Optional[Bbox]:
    bbox = Bbox.from_mask(mask)
    if bbox is None:
        return None
    return bbox.pad(padding, shape)


def _volume_cm3(mask: np.ndarray, spacing: Tuple[float, float, float]) -> float:
    n = int((mask > 0).sum())
    return n * float(np.prod(spacing)) / 1000.0
