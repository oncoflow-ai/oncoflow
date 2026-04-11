from __future__ import annotations

from typing import Any

from app.core.config import get_settings

try:
    from celery import Celery
except ModuleNotFoundError:
    class Celery:  # pragma: no cover - exercised via import test
        def __init__(self, main: str, broker: str, backend: str) -> None:
            self.main = main
            self.conf = {
                "broker_url": broker,
                "result_backend": backend,
            }

        def autodiscover_tasks(self, *_args: Any, **_kwargs: Any) -> None:
            return None


def create_celery_app() -> Celery:
    settings = get_settings()
    celery_app = Celery(
        "oncoflow",
        broker=settings.broker_dsn,
        backend=settings.result_backend_dsn,
    )

    if hasattr(celery_app, "conf") and hasattr(celery_app.conf, "update"):
        celery_app.conf.update(
            task_default_queue="mri-processing",
            task_track_started=True,
            worker_hijack_root_logger=False,
        )

    celery_app.autodiscover_tasks(["app.modules"])
    return celery_app


celery_app = create_celery_app()
