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

from fastapi import Depends

from app.core.config import get_settings
from app.api.routes.jobs import router as jobs_router
from app.api.routes.results import router as results_router
from app.api.routes.auth import router as auth_router
from app.api.routes.patients import router as patients_router
from app.api.routes.audit import router as audit_router
from app.api.routes.rag import router as rag_router
from app.api.routes.agents import router as agents_router
from app.modules.segmentation.runtime import get_inference_readiness
from app.api.deps import get_current_user

router = APIRouter()


@router.get("/health", tags=["system"])
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", tags=["system"])
def readiness() -> dict[str, object]:
    settings = get_settings()
    return {
        "status": "ready",
        "environment": settings.environment,
        "queue": settings.job_execution_mode,
        "inference": get_inference_readiness(settings=settings),
    }


router.include_router(auth_router)
router.include_router(patients_router)
router.include_router(audit_router, dependencies=[Depends(get_current_user)])
router.include_router(jobs_router, dependencies=[Depends(get_current_user)])
router.include_router(results_router, dependencies=[Depends(get_current_user)])
router.include_router(rag_router, dependencies=[Depends(get_current_user)])
router.include_router(agents_router, dependencies=[Depends(get_current_user)])
