from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal


@dataclass(frozen=True)
class Settings:
    app_name: str = "OncoFlow Backend"
    environment: Literal["development", "test", "staging", "production"] = "development"
    api_prefix: str = "/api/v1"
    frontend_origin_regex: str = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"
    job_execution_mode: Literal["deferred", "threaded"] = "deferred"
    log_level: str = "INFO"
    verbose_worker_logs: bool = False
    allow_phi_logging: bool = False
    database_url: str = "postgresql+psycopg://oncoflow:oncoflow@localhost:5432/oncoflow"
    storage_root: str = "./var/oncoflow"
    storage_staging_dir: str = "staging"
    artifact_bucket: str = "oncoflow-local"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None
    nnunet_model_dir: str | None = None
    nnunet_device: Literal["cpu", "mps", "cuda"] = "cpu"

    @classmethod
    def from_env(cls) -> "Settings":
        def get_str(name: str, default: str) -> str:
            return os.getenv(name, default)

        def get_int(name: str, default: int) -> int:
            return int(os.getenv(name, str(default)))

        def get_bool(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "on"}

        def get_optional_str(name: str) -> str | None:
            raw = os.getenv(name)
            if raw is None:
                return None
            cleaned = raw.strip()
            return cleaned or None

        return cls(
            app_name=get_str("ONCOFLOW_APP_NAME", cls.app_name),
            environment=get_str("ONCOFLOW_ENVIRONMENT", cls.environment),  # type: ignore[arg-type]
            api_prefix=get_str("ONCOFLOW_API_PREFIX", cls.api_prefix),
            frontend_origin_regex=get_str("ONCOFLOW_FRONTEND_ORIGIN_REGEX", cls.frontend_origin_regex),
            job_execution_mode=get_str("ONCOFLOW_JOB_EXECUTION_MODE", cls.job_execution_mode),  # type: ignore[arg-type]
            log_level=get_str("ONCOFLOW_LOG_LEVEL", cls.log_level),
            verbose_worker_logs=get_bool("ONCOFLOW_VERBOSE_WORKER_LOGS", cls.verbose_worker_logs),
            allow_phi_logging=get_bool("ONCOFLOW_ALLOW_PHI_LOGGING", cls.allow_phi_logging),
            database_url=get_str("ONCOFLOW_DATABASE_URL", cls.database_url),
            storage_root=get_str("ONCOFLOW_STORAGE_ROOT", cls.storage_root),
            storage_staging_dir=get_str(
                "ONCOFLOW_STORAGE_STAGING_DIR",
                cls.storage_staging_dir,
            ),
            artifact_bucket=get_str("ONCOFLOW_ARTIFACT_BUCKET", cls.artifact_bucket),
            redis_host=get_str("ONCOFLOW_REDIS_HOST", cls.redis_host),
            redis_port=get_int("ONCOFLOW_REDIS_PORT", cls.redis_port),
            redis_db=get_int("ONCOFLOW_REDIS_DB", cls.redis_db),
            redis_password=os.getenv("ONCOFLOW_REDIS_PASSWORD"),
            nnunet_model_dir=get_optional_str("ONCOFLOW_NNUNET_MODEL_DIR"),
            nnunet_device=get_str("ONCOFLOW_NNUNET_DEVICE", cls.nnunet_device),  # type: ignore[arg-type]
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
