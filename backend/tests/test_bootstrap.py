from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.infra.db.models import Patient, User
from app.infra.db.session import create_session_factory, get_engine
from app.main import bootstrap_users


@pytest.mark.parametrize("environment", ["test", "staging", "production"])
def test_bootstrap_never_seeds_demo_data_outside_development(
    environment: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONCOFLOW_ENVIRONMENT", environment)
    monkeypatch.setenv("ONCOFLOW_SEED_DEMO_DATA", "true")
    get_settings.cache_clear()
    get_engine.cache_clear()

    bootstrap_users()

    with create_session_factory()() as session:
        assert session.query(User).count() == 0
        assert session.query(Patient).count() == 0


def test_bootstrap_requires_explicit_demo_seed_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONCOFLOW_ENVIRONMENT", "development")
    monkeypatch.delenv("ONCOFLOW_SEED_DEMO_DATA", raising=False)
    get_settings.cache_clear()
    get_engine.cache_clear()

    bootstrap_users()

    with create_session_factory()() as session:
        assert session.query(User).count() == 0
        assert session.query(Patient).count() == 0
