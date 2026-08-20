import pytest
from app.main import create_app, bootstrap_users
from fastapi.testclient import TestClient
from app.core.config import get_settings


def _enable_demo_seed(monkeypatch) -> None:
    monkeypatch.setenv("ONCOFLOW_ENVIRONMENT", "development")
    monkeypatch.setenv("ONCOFLOW_SEED_DEMO_DATA", "true")
    get_settings.cache_clear()

def test_login_success(monkeypatch):
    monkeypatch.setenv("ONCOFLOW_DATABASE_URL", "sqlite:///:memory:")
    _enable_demo_seed(monkeypatch)
    app = create_app()
    bootstrap_users()
    with TestClient(app) as client:

        response = client.post(
            "/api/v1/auth/login",
            data={"username": "admin@oncoflow.local", "password": "admin123"},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["email"] == "admin@oncoflow.local"
        assert data["user"]["role"] == "admin"

def test_login_failure(monkeypatch):
    monkeypatch.setenv("ONCOFLOW_DATABASE_URL", "sqlite:///:memory:")
    _enable_demo_seed(monkeypatch)
    app = create_app()
    bootstrap_users()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/auth/login",
            data={"username": "admin@oncoflow.local", "password": "wrongpassword"},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        assert response.status_code == 401

def test_protected_route(monkeypatch):
    monkeypatch.setenv("ONCOFLOW_DATABASE_URL", "sqlite:///:memory:")
    _enable_demo_seed(monkeypatch)
    app = create_app()
    bootstrap_users()
    with TestClient(app) as client:
        # Attempting to access jobs without a token should fail
        response = client.get("/api/v1/jobs/fake-id")
        assert response.status_code == 401
        
        # Login to get token
        login_response = client.post(
            "/api/v1/auth/login",
            data={"username": "admin@oncoflow.local", "password": "admin123"},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        token = login_response.json()["access_token"]
        
        # Access jobs with token should succeed (or at least bypass auth and hit 404 Not Found for fake-id)
        response = client.get(
            "/api/v1/jobs/fake-id",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 404
