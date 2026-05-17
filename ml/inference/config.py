"""
config.py – Inference configuration.

Single source of truth for backend selection, device routing, ensemble strategy,
and cache/weight paths. Loaded from environment variables (prefix `OFLOW_`) or a
YAML file (`oncoflow.yaml` in CWD or the path passed to `load_config`).

Design goals:
  * Immutable (frozen dataclass) so adapters can rely on it across processes
  * One flag (`backend`) flips every adapter between local-Mac and GPU-prod
  * All threshold knobs from IMPLEMENTATION_PLAN.md live here
"""

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

Backend = Literal["local", "gpu-prod"]
Device = Literal["auto", "cuda", "mps", "cpu"]
EnsembleStrategy = Literal[
    "majority_vote", "union", "intersection", "staple", "confidence_weighted"
]

DEFAULT_WEIGHTS_DIR = Path("~/.oncoflow/weights").expanduser()
DEFAULT_CACHE_DIR = Path("~/.oncoflow/cache").expanduser()


@dataclass(frozen=True)
class InferenceConfig:
    """Inference-time configuration. Immutable so it can cross process boundaries."""

    # Backend + device
    backend: Backend = "local"
    device: Device = "auto"

    # Models
    enabled_models: tuple[str, ...] = ("nnunet", "medgemma", "sam3")
    modality: str = "t1c"  # which MRI modality to feed adapters by default

    # Ensemble
    ensemble_strategy: EnsembleStrategy = "majority_vote"
    keep_largest_cc: bool = True
    min_component_voxels: int = 20
    morph_closing_radius: int = 0  # 0 = disabled

    # Registration / longitudinal thresholds (Step 4.7)
    registration_type: Literal["Rigid", "Affine", "SyN"] = "Affine"
    ncc_resegment_threshold: float = 0.65
    ncc_fail_threshold: float = 0.55
    agreement_review_threshold: float = 0.75
    agreement_auto_threshold: float = 0.90

    # Speed knobs
    use_roi_bootstrap: bool = True  # use nnU-Net mask to crop MedGemma/SAM work
    roi_padding_voxels: int = 8
    parallel_adapters: bool = True
    max_workers: int = 2  # MedGemma + SAM fan-out; nnU-Net runs first for ROI

    # Preprocessing
    orient_to_ras: bool = True
    n4_bias_correction: bool = True
    isotropic_spacing_mm: float = 1.0
    skull_strip: bool = True  # Opt-in; antspynet on CPU, HD-BET on CUDA

    # Paths
    weights_dir: Path = field(default_factory=lambda: DEFAULT_WEIGHTS_DIR)
    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)

    # Vertex AI
    vertex_project_id: str = "oncoflow-496517"
    vertex_region: str = "us-central1"
    vertex_endpoint_nnunet: str = ""
    vertex_endpoint_sam3: str = ""
    vertex_endpoint_medgemma: str = ""

    # MedGemma specifics
    medgemma_model_id: str = "google/medgemma-1.5-4b-it"
    medgemma_fallback_model_id: str = "microsoft/llava-med-v1.5-mistral-7b"
    medgemma_max_slices: int = 64  # hard cap so we never run the full 150+
    medgemma_prompt: str = (
        "Segment all tumor regions in this brain MRI slice. "
        "Output a binary mask where 1=tumor, 0=background."
    )

    # SAM specifics
    sam3_model_id: str = "facebook/sam3-hiera-large"
    sam2_model_id: str = "facebook/sam2-hiera-large"
    sam_point_prompts_per_slice: int = 1

    # nnU-Net specifics
    nnunet_config_local: Literal["3d_lowres", "2d", "3d_fullres"] = "3d_lowres"
    nnunet_config_gpu: Literal["3d_fullres", "3d_lowres", "2d"] = "3d_fullres"
    nnunet_task_id: str = "Task001_BrainTumour"
    nnunet_use_tta_local: bool = False
    nnunet_use_tta_gpu: bool = True

    def resolve_device(self) -> str:
        """Resolve 'auto' to the best available torch device name."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                return "mps"
        except Exception:
            pass
        return "cpu"

    def resolve_dtype(self) -> str:
        dev = self.resolve_device()
        if dev == "cuda":
            return "bfloat16"
        if dev == "mps":
            return "float16"
        return "float32"

    def with_(self, **overrides) -> "InferenceConfig":
        """Return a new config with overrides applied (immutable-friendly)."""
        return replace(self, **overrides)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

_ENV_PREFIX = "OFLOW_"


def _env(key: str, default=None):
    return os.environ.get(_ENV_PREFIX + key, default)


def load_config(yaml_path: str | Path | None = None) -> InferenceConfig:
    """
    Build an InferenceConfig from (in priority order):
      1. The YAML at `yaml_path` if provided and exists.
      2. `./oncoflow.yaml` if present.
      3. Environment variables prefixed `OFLOW_`.
      4. Defaults.
    """
    data: dict = {}

    candidate_paths = []
    if yaml_path is not None:
        candidate_paths.append(Path(yaml_path))
    candidate_paths.append(Path.cwd() / "oncoflow.yaml")

    for p in candidate_paths:
        if p.exists():
            try:
                import yaml  # type: ignore

                with open(p, "r") as f:
                    data = yaml.safe_load(f) or {}
                break
            except ImportError:
                break

    def _coerce(field_name, default, cast):
        env_val = _env(field_name.upper())
        if env_val is not None:
            try:
                return cast(env_val)
            except Exception:
                return default
        if field_name in data:
            return data[field_name]
        return default

    defaults = InferenceConfig()

    return InferenceConfig(
        backend=_coerce("backend", defaults.backend, str),  # type: ignore
        device=_coerce("device", defaults.device, str),  # type: ignore
        enabled_models=tuple(
            _coerce(
                "enabled_models",
                list(defaults.enabled_models),
                lambda s: [m.strip() for m in s.split(",") if m.strip()],
            )
        ),
        modality=_coerce("modality", defaults.modality, str),
        ensemble_strategy=_coerce(
            "ensemble_strategy", defaults.ensemble_strategy, str
        ),  # type: ignore
        keep_largest_cc=_coerce(
            "keep_largest_cc",
            defaults.keep_largest_cc,
            lambda s: str(s).lower() in {"1", "true", "yes"},
        ),
        min_component_voxels=_coerce(
            "min_component_voxels", defaults.min_component_voxels, int
        ),
        morph_closing_radius=_coerce(
            "morph_closing_radius", defaults.morph_closing_radius, int
        ),
        registration_type=_coerce(
            "registration_type", defaults.registration_type, str
        ),  # type: ignore
        ncc_resegment_threshold=_coerce(
            "ncc_resegment_threshold", defaults.ncc_resegment_threshold, float
        ),
        ncc_fail_threshold=_coerce(
            "ncc_fail_threshold", defaults.ncc_fail_threshold, float
        ),
        agreement_review_threshold=_coerce(
            "agreement_review_threshold",
            defaults.agreement_review_threshold,
            float,
        ),
        agreement_auto_threshold=_coerce(
            "agreement_auto_threshold",
            defaults.agreement_auto_threshold,
            float,
        ),
        use_roi_bootstrap=_coerce(
            "use_roi_bootstrap",
            defaults.use_roi_bootstrap,
            lambda s: str(s).lower() in {"1", "true", "yes"},
        ),
        roi_padding_voxels=_coerce(
            "roi_padding_voxels", defaults.roi_padding_voxels, int
        ),
        parallel_adapters=_coerce(
            "parallel_adapters",
            defaults.parallel_adapters,
            lambda s: str(s).lower() in {"1", "true", "yes"},
        ),
        max_workers=_coerce("max_workers", defaults.max_workers, int),
        orient_to_ras=_coerce(
            "orient_to_ras",
            defaults.orient_to_ras,
            lambda s: str(s).lower() in {"1", "true", "yes"},
        ),
        n4_bias_correction=_coerce(
            "n4_bias_correction",
            defaults.n4_bias_correction,
            lambda s: str(s).lower() in {"1", "true", "yes"},
        ),
        isotropic_spacing_mm=_coerce(
            "isotropic_spacing_mm", defaults.isotropic_spacing_mm, float
        ),
        skull_strip=_coerce(
            "skull_strip",
            defaults.skull_strip,
            lambda s: str(s).lower() in {"1", "true", "yes"},
        ),
        weights_dir=Path(_coerce("weights_dir", str(defaults.weights_dir), str)).expanduser(),
        cache_dir=Path(_coerce("cache_dir", str(defaults.cache_dir), str)).expanduser(),
        medgemma_model_id=_coerce(
            "medgemma_model_id", defaults.medgemma_model_id, str
        ),
        medgemma_fallback_model_id=_coerce(
            "medgemma_fallback_model_id",
            defaults.medgemma_fallback_model_id,
            str,
        ),
        medgemma_max_slices=_coerce(
            "medgemma_max_slices", defaults.medgemma_max_slices, int
        ),
        medgemma_prompt=_coerce(
            "medgemma_prompt", defaults.medgemma_prompt, str
        ),
        sam3_model_id=_coerce("sam3_model_id", defaults.sam3_model_id, str),
        sam2_model_id=_coerce("sam2_model_id", defaults.sam2_model_id, str),
        sam_point_prompts_per_slice=_coerce(
            "sam_point_prompts_per_slice",
            defaults.sam_point_prompts_per_slice,
            int,
        ),
        nnunet_config_local=_coerce(
            "nnunet_config_local", defaults.nnunet_config_local, str
        ),  # type: ignore
        nnunet_config_gpu=_coerce(
            "nnunet_config_gpu", defaults.nnunet_config_gpu, str
        ),  # type: ignore
        nnunet_task_id=_coerce(
            "nnunet_task_id", defaults.nnunet_task_id, str
        ),
        nnunet_use_tta_local=_coerce(
            "nnunet_use_tta_local",
            defaults.nnunet_use_tta_local,
            lambda s: str(s).lower() in {"1", "true", "yes"},
        ),
        nnunet_use_tta_gpu=_coerce(
            "nnunet_use_tta_gpu",
            defaults.nnunet_use_tta_gpu,
            lambda s: str(s).lower() in {"1", "true", "yes"},
        ),
        vertex_project_id=_coerce("vertex_project_id", defaults.vertex_project_id, str),
        vertex_region=_coerce("vertex_region", defaults.vertex_region, str),
        vertex_endpoint_nnunet=_coerce("vertex_endpoint_nnunet", defaults.vertex_endpoint_nnunet, str),
        vertex_endpoint_sam3=_coerce("vertex_endpoint_sam3", defaults.vertex_endpoint_sam3, str),
        vertex_endpoint_medgemma=_coerce("vertex_endpoint_medgemma", defaults.vertex_endpoint_medgemma, str),
    )
