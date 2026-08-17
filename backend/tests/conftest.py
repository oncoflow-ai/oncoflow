from __future__ import annotations

from collections.abc import Iterator

import pytest

from app.core.config import get_settings
from app.infra.db.session import get_engine
from app.main import create_app


class AppClient:
    def __init__(self, app) -> None:
        self.app = app

    def get(self, path: str):
        if hasattr(self.app, "handle_request"):
            return self.app.handle_request("GET", path)

        from fastapi.testclient import TestClient

        with TestClient(self.app) as test_client:
            return test_client.get(path)

    def post(self, path: str, **kwargs):
        from fastapi.testclient import TestClient

        with TestClient(self.app) as test_client:
            return test_client.post(path, **kwargs)


@pytest.fixture(autouse=True)
def clear_settings_cache(monkeypatch) -> Iterator[None]:
    monkeypatch.setenv("ONCOFLOW_DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("ONCOFLOW_ENVIRONMENT", "test")
    get_settings.cache_clear()
    get_engine.cache_clear()
    yield
    get_settings.cache_clear()
    get_engine.cache_clear()


@pytest.fixture
def client() -> Iterator[AppClient]:
    app = create_app()
    yield AppClient(app)
