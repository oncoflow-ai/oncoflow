from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
except ModuleNotFoundError:
    class CORSMiddleware:  # pragma: no cover - test stub only
        pass

    @dataclass
    class Response:
        status_code: int
        body: dict[str, Any]

        def json(self) -> dict[str, Any]:
            return self.body

    class FastAPI:  # pragma: no cover - exercised via app boot tests
        def __init__(self, title: str) -> None:
            self.title = title
            self._routes: dict[tuple[str, str], Any] = {}
            self._event_handlers: dict[str, list[Any]] = {}

        def add_middleware(self, _middleware_class: Any, **_kwargs: Any) -> None:
            return None

        def include_router(self, router: Any, prefix: str = "") -> None:
            for method, path, endpoint in getattr(router, "routes", []):
                self._routes[(method, f"{prefix}{path}")] = endpoint

        def add_event_handler(self, event_type: str, handler: Any) -> None:
            self._event_handlers.setdefault(event_type, []).append(handler)

        def handle_request(self, method: str, path: str) -> Response:
            endpoint = self._routes.get((method.upper(), path))
            if endpoint is None:
                return Response(status_code=404, body={"detail": "Not Found"})
            return Response(status_code=200, body=endpoint())

from app.api.router import router as api_router
from app.core.config import get_settings
from app.modules.jobs.worker_tasks import shutdown_background_workers


_PATH_PATTERN = re.compile(r"([A-Za-z]:)?[/\\][^\s]+")
_DICOM_TAG_PATTERN = re.compile(r"\([0-9A-Fa-f]{4},[0-9A-Fa-f]{4}\)")


class PhiSafeLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "allow_phi", False):
            return True

        message = record.getMessage()
        sanitized = _PATH_PATTERN.sub("[redacted-path]", message)
        sanitized = _DICOM_TAG_PATTERN.sub("[redacted-dicom-tag]", sanitized)
        record.msg = sanitized
        record.args = ()
        return True


def configure_logging() -> None:
    settings = get_settings()
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level.upper())

    filter_present = any(
        isinstance(existing_filter, PhiSafeLogFilter)
        for existing_filter in root_logger.filters
    )
    if not filter_present and not settings.allow_phi_logging:
        root_logger.addFilter(PhiSafeLogFilter())

    if not root_logger.handlers:
        logging.basicConfig(
            level=settings.log_level.upper(),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )

    audit_logger = logging.getLogger("oncoflow.audit")
    audit_logger.propagate = False

    if not audit_logger.handlers:
        from logging.handlers import RotatingFileHandler
        from pathlib import Path
        from pythonjsonlogger.jsonlogger import JsonFormatter

        storage_path = Path(settings.storage_root).expanduser().resolve()
        storage_path.mkdir(parents=True, exist_ok=True)
        
        audit_file = storage_path / "audit.log"
        handler = RotatingFileHandler(
            audit_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        
        formatter = JsonFormatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        audit_logger.addHandler(handler)
        audit_logger.setLevel(logging.INFO)


from app.infra.db.session import create_session_factory
from app.infra.db.models import User
from app.core.security import get_password_hash

def bootstrap_users() -> None:
    session_factory = create_session_factory()
    demo_users = [
        {"email": "admin@oncoflow.local", "name": "Maya Administrator", "password": "admin123", "role": "admin"},
        {"email": "dr.cohen@ichilov.gov.il", "name": "Dr. D. Cohen", "password": "password", "role": "doctor"},
        {"email": "radiology@oncoflow.local", "name": "Alex Rahman", "password": "password", "role": "radiologist"},
        {"email": "clinician@oncoflow.local", "name": "Noa Clinical", "password": "password", "role": "clinician"},
        {"email": "sarah.jenkins@example.test", "name": "Sarah Jenkins", "password": "patient123", "role": "patient"}
    ]
    with session_factory() as session:
        for u in demo_users:
            if not session.query(User).filter(User.email == u["email"]).first():
                user = User(
                    email=u["email"],
                    name=u["name"],
                    hashed_password=get_password_hash(u["password"]),
                    role=u["role"]
                )
                session.add(user)
        session.commit()

def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=settings.frontend_origin_regex,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if hasattr(app, "add_event_handler"):
        app.add_event_handler("startup", bootstrap_users)
        app.add_event_handler("shutdown", shutdown_background_workers)
    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()
