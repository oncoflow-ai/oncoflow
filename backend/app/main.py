from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

try:
    from fastapi import FastAPI
except ModuleNotFoundError:
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

        def include_router(self, router: Any, prefix: str = "") -> None:
            for method, path, endpoint in getattr(router, "routes", []):
                self._routes[(method, f"{prefix}{path}")] = endpoint

        def handle_request(self, method: str, path: str) -> Response:
            endpoint = self._routes.get((method.upper(), path))
            if endpoint is None:
                return Response(status_code=404, body={"detail": "Not Found"})
            return Response(status_code=200, body=endpoint())

from app.api.router import router as api_router
from app.core.config import get_settings


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


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()
