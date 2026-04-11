from __future__ import annotations

try:
    from fastapi import APIRouter
except ModuleNotFoundError:
    class APIRouter:  # pragma: no cover - exercised via app boot tests
        def __init__(self) -> None:
            self.routes: list[tuple[str, str, object]] = []

        def get(self, path: str, tags: list[str] | None = None):
            def decorator(func):
                self.routes.append(("GET", path, func))
                return func

            return decorator

from app.core.config import get_settings
from app.api.routes.jobs import router as jobs_router

router = APIRouter()


@router.get("/health", tags=["system"])
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", tags=["system"])
def readiness() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ready",
        "environment": settings.environment,
        "queue": "configured",
    }


router.include_router(jobs_router)
