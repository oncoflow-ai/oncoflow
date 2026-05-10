from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings
from app.modules.segmentation.runtime import (
    ModelPackageMissingError,
    RuntimeDeviceUnavailableError,
    RuntimeDependencyMissingError,
    RuntimeReadiness,
    UnsupportedRunnerModelError,
    UnsupportedRuntimeDeviceError,
    is_real_runner_configured,
    resolve_runtime_readiness,
)


def _settings(tmp_path: Path, *, model_dir: str | None, device: str = "cpu") -> Settings:
    return Settings(
        database_url="sqlite+pysqlite:///:memory:",
        storage_root=str(tmp_path / "storage"),
        nnunet_model_dir=model_dir,
        nnunet_device=device,  # type: ignore[arg-type]
    )


def _seed_model_package(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset.json").write_text("{}")
    (root / "checkpoint_final.pth").write_bytes(b"weights")


def test_runtime_readiness_accepts_valid_model_package(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    package = tmp_path / "model"
    _seed_model_package(package)
    monkeypatch.setattr("app.modules.segmentation.runtime._dependency_gaps", lambda: ())

    readiness = resolve_runtime_readiness(
        model_id="nnunet-v2-resenc",
        settings=_settings(tmp_path, model_dir=str(package)),
    )

    assert isinstance(readiness, RuntimeReadiness)
    assert readiness.model_directory == str(package.resolve())
    assert readiness.checkpoint_relative_path == "checkpoint_final.pth"
    assert readiness.package_manifest_relative_path == "dataset.json"
    assert readiness.device == "cpu"
    assert readiness.runner_version


def test_runtime_readiness_rejects_missing_or_unreadable_packages(tmp_path: Path) -> None:
    with pytest.raises(ModelPackageMissingError, match="required"):
        resolve_runtime_readiness(
            model_id="nnunet-v2-resenc",
            settings=_settings(tmp_path, model_dir=None),
        )

    with pytest.raises(ModelPackageMissingError, match="does not exist"):
        resolve_runtime_readiness(
            model_id="nnunet-v2-resenc",
            settings=_settings(tmp_path, model_dir=str(tmp_path / "missing")),
        )


def test_runtime_readiness_rejects_non_baseline_models_devices_and_missing_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    package = tmp_path / "model"
    _seed_model_package(package)

    with pytest.raises(UnsupportedRunnerModelError):
        resolve_runtime_readiness(
            model_id="nnunet-2d",
            settings=_settings(tmp_path, model_dir=str(package)),
        )

    with pytest.raises(UnsupportedRuntimeDeviceError):
        resolve_runtime_readiness(
            model_id="nnunet-v2-resenc",
            settings=_settings(tmp_path, model_dir=str(package), device="tpu"),
        )

    monkeypatch.setattr("app.modules.segmentation.runtime._dependency_gaps", lambda: ("torch", "nnunetv1"))
    with pytest.raises(RuntimeDependencyMissingError, match="torch, nnunetv1"):
        resolve_runtime_readiness(
            model_id="nnunet-v2-resenc",
            settings=_settings(tmp_path, model_dir=str(package)),
        )


def test_runtime_readiness_rejects_unavailable_accelerator_devices(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    package = tmp_path / "model"
    _seed_model_package(package)
    monkeypatch.setattr("app.modules.segmentation.runtime._dependency_gaps", lambda: ())

    class FakeMpsBackend:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class backends:
            mps = FakeMpsBackend()

    monkeypatch.setattr("app.modules.segmentation.runtime.__import__", __import__, raising=False)
    monkeypatch.setitem(__import__("sys").modules, "torch", FakeTorch)

    with pytest.raises(RuntimeDeviceUnavailableError, match="CUDA is not available"):
        resolve_runtime_readiness(
            model_id="nnunet-v2-resenc",
            settings=_settings(tmp_path, model_dir=str(package), device="cuda"),
        )

    with pytest.raises(RuntimeDeviceUnavailableError, match="MPS is not available"):
        resolve_runtime_readiness(
            model_id="nnunet-v2-resenc",
            settings=_settings(tmp_path, model_dir=str(package), device="mps"),
        )


def test_real_runner_configuration_is_opt_in(tmp_path: Path) -> None:
    assert is_real_runner_configured(settings=_settings(tmp_path, model_dir=None)) is False
    assert is_real_runner_configured(settings=_settings(tmp_path, model_dir=str(tmp_path / "model"))) is True
