from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.infra.db.session import get_engine
from app.main import bootstrap_users, create_app


@pytest.fixture
def auth_client(monkeypatch):
    monkeypatch.setenv("ONCOFLOW_DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("ONCOFLOW_ENVIRONMENT", "development")
    monkeypatch.setenv("ONCOFLOW_SEED_DEMO_DATA", "true")
    get_settings.cache_clear()
    get_engine.cache_clear()
    app = create_app()
    bootstrap_users()

    with TestClient(app) as client:
        # Login as admin to get bearer token
        login_res = client.post(
            "/api/v1/auth/login",
            data={"username": "admin@oncoflow.local", "password": "admin123"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert login_res.status_code == 200
        token = login_res.json()["access_token"]
        client.headers.update({"Authorization": f"Bearer {token}"})
        yield client


def test_rag_ingest_and_query_api(auth_client):
    # Ingest document
    ingest_res = auth_client.post(
        "/api/v1/rag/documents",
        json={
            "patient_id": "P-9999",
            "document_type": "prior_summary",
            "title": "Historical Brain MRI Report",
            "content": "Tumor volume at baseline was 11.4 cm3 with 32 mm diameter. Patient was on radiation protocol.",
            "metadata": {"source": "hospital-archive"},
        },
    )
    assert ingest_res.status_code == 200
    data = ingest_res.json()
    assert data["patient_id"] == "P-9999"
    assert data["status"] == "indexed"

    # List documents
    list_res = auth_client.get("/api/v1/rag/documents/P-9999")
    assert list_res.status_code == 200
    docs = list_res.json()
    assert len(docs) >= 1
    assert docs[0]["title"] == "Historical Brain MRI Report"

    # Query RAG
    query_res = auth_client.post(
        "/api/v1/rag/query",
        json={
            "patient_id": "P-9999",
            "query": "tumor volume baseline radiation",
            "top_k": 3,
        },
    )
    assert query_res.status_code == 200
    q_data = query_res.json()
    assert q_data["patient_id"] == "P-9999"
    assert len(q_data["chunks"]) >= 1
    assert "11.4 cm3" in q_data["formatted_context"]


def test_orchestrate_summary_api(auth_client):
    # Seed prior summary in RAG
    auth_client.post(
        "/api/v1/rag/documents",
        json={
            "patient_id": "P-1029",
            "document_type": "prior_summary",
            "title": "Baseline Scan Summary",
            "content": "Baseline lesion volume 12.92 cm3 and longest diameter 35.8 mm.",
        },
    )

    # Orchestrate summary
    orch_res = auth_client.post(
        "/api/v1/agents/orchestrate-summary",
        json={
            "patient_id": "P-1029",
            "override_metrics": {
                "total_volume_cm3": 14.815,
                "longest_diameter_mm": 39.1,
                "prior_volume_cm3": 12.92,
                "prior_diameter_mm": 35.8,
                "volume_change_pct": 14.7,
                "diameter_change_mm": 3.3,
                "lesion_count": 1,
            },
            "custom_query": "prior baseline volume diameter change",
            "persist": True,
        },
    )
    assert orch_res.status_code == 200
    res_data = orch_res.json()
    assert res_data["patient_id"] == "P-1029"
    assert "findings" in res_data["summary"]
    assert "impression" in res_data["summary"]
    assert len(res_data["summary"]["recommendations"]) >= 3
    assert res_data["summary"]["validation"]["is_valid"] is True
    assert len(res_data["agent_logs"]) >= 5

    # Check that summary can be listed
    list_summaries_res = auth_client.get("/api/v1/agents/summaries/P-1029")
    assert list_summaries_res.status_code == 200
    summaries = list_summaries_res.json()
    assert len(summaries) >= 1
    assert summaries[0]["patient_id"] == "P-1029"
