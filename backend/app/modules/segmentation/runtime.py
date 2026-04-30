from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.util import find_spec
from pathlib import Path

from app.core.config import Settings, get_settings
from app.modules.benchmark.model_registry import get_model_spec

SUPPORTED_MODEL_IDS = {"nnunet-v2-resenc"}
SUPPORTED_DEVICES = {"cpu", "mps", "cuda"}
RUNTIME_DEPENDENCIES = ("numpy", "nibabel", "torch", "nnunetv2")


class SegmentationRuntimeError(RuntimeError):
    pass


class UnsupportedRunnerModelError(SegmentationRuntimeError):
    pass


class ModelPackageMissingError(SegmentationRuntimeError):
    pass


class RuntimeDependencyMissingError(SegmentationRuntimeError):
    pass


class UnsupportedRuntimeDeviceError(SegmentationRuntimeError):
    pass


class RuntimeDeviceUnavailableError(SegmentationRuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeReadiness:
    model_id: str
    model_directory: str
    checkpoint_relative_path: str
    package_manifest_relative_path: str
    weights_digest: str
    device: str
    execution_backend: str

    @property
    def runner_version(self) -> str:
        return self.weights_digest[:12]


def is_real_runner_configured(*, settings: Settings | None = None, model_id: str = "nnunet-v2-resenc") -> bool:
    settings = settings or get_settings()
    return model_id in SUPPORTED_MODEL_IDS and bool(settings.nnunet_model_dir)


def _first_existing(path: Path, *candidates: str) -> Path | None:
    for candidate in candidates:
        found = path / candidate
        if found.exists():
            return found
    return None


def _find_checkpoint(model_root: Path) -> Path:
    explicit = _first_existing(
        model_root,
        "checkpoint_final.pth",
        "checkpoint_best.pth",
        "fold_0/checkpoint_final.pth",
        "fold_0/checkpoint_best.pth",
    )
    if explicit is not None:
        return explicit

    candidates = sorted(model_root.rglob("*.pth"))
    if not candidates:
        raise ModelPackageMissingError(
            "nnU-Net model package must contain at least one checkpoint (.pth) file"
        )
    return candidates[0]


def _find_manifest(model_root: Path) -> Path:
    manifest = _first_existing(
        model_root,
        "dataset.json",
        "plans.json",
        "plans.pkl",
        "nnUNetPlans.json",
    )
    if manifest is not None:
        return manifest

    for filename in ("dataset.json", "plans.json", "plans.pkl", "nnUNetPlans.json"):
        candidates = sorted(model_root.rglob(filename))
        if candidates:
            return candidates[0]
    raise ModelPackageMissingError(
        "nnU-Net model package must include dataset/plans metadata such as dataset.json or plans.json"
    )


def _dependency_gaps() -> tuple[str, ...]:
    return tuple(name for name in RUNTIME_DEPENDENCIES if find_spec(name) is None)


def _ensure_device_available(device: str) -> None:
    if device == "cpu":
        return

    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeDeviceUnavailableError(
            "ONCOFLOW_NNUNET_DEVICE=cuda but CUDA is not available in the active runtime"
        )

    if device == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeDeviceUnavailableError(
                "ONCOFLOW_NNUNET_DEVICE=mps but MPS is not available in the active runtime"
            )


def resolve_runtime_readiness(
    *,
    model_id: str = "nnunet-v2-resenc",
    settings: Settings | None = None,
) -> RuntimeReadiness:
    settings = settings or get_settings()
    spec = get_model_spec(model_id)
    if spec.selection_role != "baseline" or model_id not in SUPPORTED_MODEL_IDS:
        raise UnsupportedRunnerModelError(
            f"Real automatic inference is only supported for baseline model ids: {sorted(SUPPORTED_MODEL_IDS)}"
        )

    device = settings.nnunet_device.strip().lower()
    if device not in SUPPORTED_DEVICES:
        raise UnsupportedRuntimeDeviceError(
            f"Unsupported nnU-Net device '{settings.nnunet_device}'. Expected one of {sorted(SUPPORTED_DEVICES)}"
        )

    model_dir = (settings.nnunet_model_dir or "").strip()
    if not model_dir:
        raise ModelPackageMissingError(
            "ONCOFLOW_NNUNET_MODEL_DIR is required to enable real nnU-Net inference"
        )

    model_root = Path(model_dir).expanduser().resolve()
    if not model_root.exists():
        raise ModelPackageMissingError(f"Configured nnU-Net model package does not exist: {model_root}")
    if not model_root.is_dir():
        raise ModelPackageMissingError(f"Configured nnU-Net model package is not a directory: {model_root}")

    dependency_gaps = _dependency_gaps()
    if dependency_gaps:
        raise RuntimeDependencyMissingError(
            "Real nnU-Net runtime dependencies are missing: " + ", ".join(dependency_gaps)
        )
    _ensure_device_available(device)

    checkpoint = _find_checkpoint(model_root)
    manifest = _find_manifest(model_root)
    weights_digest = sha256(checkpoint.read_bytes()).hexdigest()

    return RuntimeReadiness(
        model_id=model_id,
        model_directory=str(model_root),
        checkpoint_relative_path=checkpoint.relative_to(model_root).as_posix(),
        package_manifest_relative_path=manifest.relative_to(model_root).as_posix(),
        weights_digest=weights_digest,
        device=device,
        execution_backend="nnunetv2",
    )
