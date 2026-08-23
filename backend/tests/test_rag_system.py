from __future__ import annotations

from datetime import datetime, timezone
import pytest

from app.modules.rag.contracts import PatientDocumentPayload
from app.modules.rag.service import (
    chunk_text,
    ingest_patient_document,
    retrieve_patient_context,
    retrieve_prior_summaries_context,
)
from app.modules.rag.vector_store import InMemoryVectorStore


def test_chunk_text_splits_paragraphs_correctly():
    doc_id = "doc-123"
    content = "Paragraph 1 with baseline tumor facts.\n\nParagraph 2 with RECIST 1.1 progression details."
    chunks = chunk_text(
        document_id=doc_id,
        patient_id="P-001",
        document_type="prior_summary",
        title="Baseline Summary",
        content=content,
        created_at=datetime.now(timezone.utc),
    )
    assert len(chunks) == 1
    assert "Baseline Summary" in chunks[0].title
    assert "RECIST 1.1" in chunks[0].content


def test_rag_vector_store_strict_patient_isolation():
    store = InMemoryVectorStore()
    now = datetime.now(timezone.utc)

    # Ingest for Patient A
    chunks_a = chunk_text(
        document_id="doc-A",
        patient_id="patient-A",
        document_type="prior_summary",
        title="Patient A Summary",
        content="Patient A has a right temporal glioblastoma measuring 14.8 cm3.",
        created_at=now,
    )
    store.add_chunks(chunks_a)

    # Ingest for Patient B
    chunks_b = chunk_text(
        document_id="doc-B",
        patient_id="patient-B",
        document_type="prior_summary",
        title="Patient B Summary",
        content="Patient B has a left frontal meningioma measuring 8.2 cm3.",
        created_at=now,
    )
    store.add_chunks(chunks_b)

    # Query for Patient A
    res_a = store.query(patient_id="patient-A", query_text="temporal glioblastoma")
    assert len(res_a) == 1
    assert res_a[0].chunk.patient_id == "patient-A"
    assert "temporal" in res_a[0].chunk.content

    # Query for Patient B - should NEVER return Patient A documents
    res_b = store.query(patient_id="patient-B", query_text="temporal glioblastoma")
    assert len(res_b) == 0  # No temporal glioblastoma for Patient B

    # Query non-existent patient
    res_c = store.query(patient_id="patient-C", query_text="temporal glioblastoma")
    assert len(res_c) == 0


def test_retrieve_prior_summaries_context(monkeypatch, tmp_path):
    monkeypatch.setenv("ONCOFLOW_DATABASE_URL", "sqlite:///:memory:")
    from app.infra.db.base import Base
    from app.infra.db.session import create_session_factory

    session_factory = create_session_factory()
    with session_factory() as session:
        Base.metadata.create_all(session.get_bind())

    # Ingest prior summary
    ingest_patient_document(
        PatientDocumentPayload(
            patient_id="patient-demo-01",
            document_type="prior_summary",
            title="Baseline MRI Brain Summary",
            content="Baseline MRI scan from 2025-08-01: Single enhancing intra-axial parietal lesion measuring 12.92 cm3 and 35.8 mm diameter. Stable appearance.",
        )
    )

    # Ingest clinical note
    ingest_patient_document(
        PatientDocumentPayload(
            patient_id="patient-demo-01",
            document_type="clinical_note",
            title="Oncology Progress Note",
            content="Patient completed cycle 2 of temozolomide chemotherapy. Reports mild headache, no focal neurological deficits.",
        )
    )

    # Retrieve prior summaries context
    context = retrieve_prior_summaries_context(
        patient_id="patient-demo-01",
        query="lesion volume chemotherapy baseline",
    )

    assert context.older_summaries_count >= 1
    assert "HISTORICAL SUMMARY" in context.formatted_context
    assert "12.92 cm3" in context.formatted_context
    assert len(context.retrieved_chunks) >= 1
