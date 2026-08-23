from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.core.security import get_password_hash, verify_password
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


def test_bootstrap_adds_david_to_a_legacy_sarah_demo_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restarted development backend must converge an old demo database."""
    monkeypatch.setenv("ONCOFLOW_DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("ONCOFLOW_ENVIRONMENT", "development")
    monkeypatch.setenv("ONCOFLOW_SEED_DEMO_DATA", "true")
    get_settings.cache_clear()
    get_engine.cache_clear()

    with create_session_factory()() as session:
        session.add(
            User(
                email="sarah.jenkins@example.test",
                name="Sarah Jenkins",
                hashed_password=get_password_hash("patient123"),
                role="patient",
            )
        )
        session.commit()

    bootstrap_users()

    with create_session_factory()() as session:
        david = session.query(User).filter(User.email == "david.levi@example.test").one()
        assert david.name == "David Levi"
        assert david.role == "patient"
        assert verify_password("patient123", david.hashed_password)
