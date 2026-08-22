from __future__ import annotations

from datetime import datetime, timezone
import pytest

from app.modules.agents.contracts import (
    ImageStreamPayload,
    SynthesizedSummary,
    TextStreamPayload,
)
from app.modules.agents.image_agent import ImageStreamAgent
from app.modules.agents.orchestrator import MultiAgentOrchestrator
from app.modules.agents.summary_agent import ClinicalSummaryAgent
from app.modules.agents.text_agent import TextStreamAgent
from app.modules.agents.validation_agent import SafetyValidationAgent
from app.modules.rag.contracts import PatientDocumentPayload
from app.modules.rag.service import ingest_patient_document


def test_image_stream_agent_with_override_metrics():
    agent = ImageStreamAgent()
    payload, log = agent.process(
        study_id="study-001",
        override_metrics={
            "total_volume_cm3": 15.5,
            "longest_diameter_mm": 41.2,
            "prior_volume_cm3": 13.0,
            "volume_change_pct": 19.2,
            "diameter_change_mm": 3.4,
            "lesion_count": 1,
        },
    )

    assert payload.total_volume_cm3 == 15.5
    assert payload.longest_diameter_mm == 41.2
    assert payload.is_longitudinal is True
    assert payload.volume_change_pct == 19.2
    assert log.action == "extract_image_metrics"
    assert log.status == "completed"


def test_text_stream_agent_retrieval(monkeypatch):
    monkeypatch.setenv("ONCOFLOW_DATABASE_URL", "sqlite:///:memory:")
    from app.infra.db.base import Base
    from app.infra.db.session import create_session_factory

    session_factory = create_session_factory()
    with session_factory() as session:
        Base.metadata.create_all(session.get_bind())

    ingest_patient_document(
        PatientDocumentPayload(
            patient_id="patient-xyz",
            document_type="prior_summary",
            title="Scan Summary 2025-06",
            content="Historical baseline volume: 10.5 cm3 with stable post-radiation changes.",
        )
    )

    agent = TextStreamAgent()
    payload, log = agent.process(patient_id="patient-xyz")

    assert payload.retrieved_summaries_count >= 1
    assert "10.5 cm3" in payload.context_text
    assert log.action == "retrieve_rag_context"
    assert log.status == "completed"


def test_summary_agent_and_safety_validation():
    summary_agent = ClinicalSummaryAgent()
    validation_agent = SafetyValidationAgent()

    img_data = ImageStreamPayload(
        study_id="study-001",
        lesion_count=1,
        primary_volume_cm3=14.82,
        total_volume_cm3=14.82,
        longest_diameter_mm=39.1,
        is_longitudinal=True,
        prior_volume_cm3=12.92,
        prior_diameter_mm=35.8,
        volume_change_pct=14.7,
        diameter_change_mm=3.3,
        recist_category="Progressive Disease (PD)",
    )

    txt_data = TextStreamPayload(
        patient_id="patient-xyz",
        retrieved_summaries_count=1,
        retrieved_notes_count=0,
        context_text="Prior summary: 12.92 cm3 tumor volume.",
        referenced_sources=[{"title": "Prior Summary", "relevance_score": 0.95}],
    )

    summary, sum_log = summary_agent.process(image_data=img_data, text_data=txt_data)

    assert "14.82 cm³" in summary.findings
    assert "14.7%" in summary.comparison
    assert len(summary.recommendations) >= 3

    val_res, val_log = validation_agent.process(summary=summary, ground_truth_image=img_data)

    assert val_res.is_valid is True
    assert val_res.hallucination_detected is False
    assert val_res.confidence_score >= 0.8
    assert val_res.metric_checks["current_volume_consistent"] is True


def test_safety_validation_flags_hallucinations():
    validation_agent = SafetyValidationAgent()

    img_data = ImageStreamPayload(
        study_id="study-001",
        lesion_count=1,
        primary_volume_cm3=14.82,
        total_volume_cm3=14.82,
        longest_diameter_mm=39.1,
    )

    # Fabricated summary stating multiple lesions and wrong volume
    hallucinated_summary = SynthesizedSummary(
        title="Report",
        technique="MRI",
        findings="Analysis reveals multifocal enhancing lesions with multiple lesions in bilateral lobes.",
        impression="Multiple metastases.",
        comparison="None",
        recommendations=["Followup"],
        quantitative={},
    )

    val_res, _ = validation_agent.process(summary=hallucinated_summary, ground_truth_image=img_data)

    assert val_res.hallucination_detected is True
    assert val_res.is_valid is False
    assert any("multiple lesions" in w.lower() for w in val_res.warnings)


def test_multi_agent_orchestrator_end_to_end(monkeypatch):
    monkeypatch.setenv("ONCOFLOW_DATABASE_URL", "sqlite:///:memory:")
    from app.infra.db.base import Base
    from app.infra.db.session import create_session_factory

    session_factory = create_session_factory()
    with session_factory() as session:
        Base.metadata.create_all(session.get_bind())

    # Pre-seed RAG with older summary for patient
    ingest_patient_document(
        PatientDocumentPayload(
            patient_id="patient-1029",
            document_type="prior_summary",
            title="Baseline MRI Scan",
            content="Baseline study: Solitary parietal tumor measuring 12.92 cm3 and 35.8 mm.",
        )
    )

    orchestrator = MultiAgentOrchestrator()
    result = orchestrator.orchestrate(
        patient_id="patient-1029",
        override_metrics={
            "total_volume_cm3": 14.815,
            "longest_diameter_mm": 39.1,
            "prior_volume_cm3": 12.92,
            "prior_diameter_mm": 35.8,
            "volume_change_pct": 14.7,
            "diameter_change_mm": 3.3,
            "lesion_count": 1,
        },
    )

    assert result.patient_id == "patient-1029"
    assert result.summary.title == "AI Brain MRI Longitudinal Segmentation Report"
    assert "14.81" in result.summary.findings or "14.82" in result.summary.findings
    assert result.summary.validation is not None
    assert result.summary.validation.is_valid is True
    assert len(result.agent_logs) >= 5
