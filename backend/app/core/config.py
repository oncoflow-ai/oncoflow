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
    log_level: str = "INFO"
    allow_phi_logging: bool = False
    database_url: str = "postgresql+psycopg://oncoflow:oncoflow@localhost:5432/oncoflow"
    storage_root: str = "./var/oncoflow"
    storage_staging_dir: str = "staging"
    artifact_bucket: str = "oncoflow-local"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None
    broker_url: str | None = None
    result_backend: str | None = None

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

        return cls(
            app_name=get_str("ONCOFLOW_APP_NAME", cls.app_name),
            environment=get_str("ONCOFLOW_ENVIRONMENT", cls.environment),  # type: ignore[arg-type]
            api_prefix=get_str("ONCOFLOW_API_PREFIX", cls.api_prefix),
            log_level=get_str("ONCOFLOW_LOG_LEVEL", cls.log_level),
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
            broker_url=os.getenv("ONCOFLOW_BROKER_URL"),
            result_backend=os.getenv("ONCOFLOW_RESULT_BACKEND"),
        )

    @property
    def broker_dsn(self) -> str:
        if self.broker_url:
            return self.broker_url

        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def result_backend_dsn(self) -> str:
        return self.result_backend or self.broker_dsn


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
