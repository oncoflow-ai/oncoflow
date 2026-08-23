from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

from app.infra.db.models import PatientDocument, PatientSummary
from app.infra.db.session import create_session_factory
from app.modules.rag.contracts import (
    DocumentChunk,
    PatientDocumentPayload,
    RAGContext,
    RetrievedChunk,
)
from app.modules.rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def chunk_text(
    *,
    document_id: str,
    patient_id: str,
    document_type: str,
    title: str,
    content: str,
    created_at: datetime,
    metadata: dict | None = None,
    chunk_size_chars: int = 600,
    overlap_chars: int = 80,
) -> list[DocumentChunk]:
    """Splits a document text into overlapping chunks or structured paragraphs."""
    metadata = metadata or {}
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    chunks: list[DocumentChunk] = []
    chunk_idx = 0

    if not paragraphs:
        paragraphs = [content.strip()]

    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) <= chunk_size_chars:
            current_chunk = f"{current_chunk}\n\n{para}".strip()
        else:
            if current_chunk:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document_id}_chunk_{chunk_idx}",
                        document_id=document_id,
                        patient_id=patient_id,
                        document_type=document_type,
                        title=title,
                        content=current_chunk,
                        created_at=created_at,
                        metadata={**metadata, "chunk_index": chunk_idx},
                    )
                )
                chunk_idx += 1
            current_chunk = para

    if current_chunk:
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document_id}_chunk_{chunk_idx}",
                document_id=document_id,
                patient_id=patient_id,
                document_type=document_type,
                title=title,
                content=current_chunk,
                created_at=created_at,
                metadata={**metadata, "chunk_index": chunk_idx},
            )
        )

    return chunks


def ingest_patient_document(payload: PatientDocumentPayload) -> str:
    """Persists a clinical document in DB and indexes it in the isolated patient vector store."""
    session_factory = create_session_factory()
    doc_uuid = uuid4()
    now = datetime.now(timezone.utc)

    # Validate or parse patient UUID
    try:
        patient_uuid = UUID(payload.patient_id)
    except ValueError:
        # Generate deterministic UUID from patient string if not already a UUID (e.g. "P-1029" or "P01")
        import hashlib
        patient_uuid = UUID(bytes=hashlib.md5(payload.patient_id.encode("utf-8")).digest())

    with session_factory() as session:
        doc = PatientDocument(
            public_id=doc_uuid,
            patient_public_id=patient_uuid,
            document_type=payload.document_type,
            title=payload.title,
            content=payload.content,
            doc_metadata=payload.metadata,
            created_at=now,
        )
        session.add(doc)
        session.commit()

    # Index in vector store
    chunks = chunk_text(
        document_id=str(doc_uuid),
        patient_id=payload.patient_id,
        document_type=payload.document_type,
        title=payload.title,
        content=payload.content,
        created_at=now,
        metadata=payload.metadata,
    )
    vector_store = get_vector_store()
    vector_store.add_chunks(chunks)

    logger.info(
        "Ingested and indexed patient document %s for patient %s (%d chunks)",
        doc_uuid,
        payload.patient_id,
        len(chunks),
    )
    return str(doc_uuid)


def index_summary_in_rag(
    *,
    summary_id: str,
    patient_id: str,
    title: str,
    summary_text: str,
    created_at: datetime,
    metadata: dict | None = None,
) -> None:
    """Directly indexes a newly generated summary into RAG for future longitudinal retrieval."""
    chunks = chunk_text(
        document_id=summary_id,
        patient_id=patient_id,
        document_type="prior_summary",
        title=title,
        content=summary_text,
        created_at=created_at,
        metadata=metadata or {},
    )
    get_vector_store().add_chunks(chunks)


def retrieve_prior_summaries_context(
    *,
    patient_id: str,
    query: str = "tumor volume lesion progression response prior baseline",
    top_k: int = 4,
) -> RAGContext:
    """Retrieves older summaries and reports for this client to provide historical longitudinal context."""
    vector_store = get_vector_store()

    # 1. Fetch chronological prior summaries
    chrono_chunks = vector_store.get_chronological_summaries(patient_id=patient_id, limit=top_k)

    # 2. Also run semantic query for specific matching points
    semantic_results = vector_store.query(
        patient_id=patient_id,
        query_text=query,
        top_k=top_k,
        document_types=["prior_summary", "radiology_report", "clinical_summary", "clinical_note"],
    )

    # Deduplicate chunks
    seen_chunk_ids = set()
    combined_retrieved: list[RetrievedChunk] = []

    for chunk in chrono_chunks:
        if chunk.chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk.chunk_id)
            combined_retrieved.append(RetrievedChunk(chunk=chunk, score=0.95))

    for item in semantic_results:
        if item.chunk.chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(item.chunk.chunk_id)
            combined_retrieved.append(item)

    # Format structured context
    summary_count = 0
    note_count = 0
    formatted_parts: list[str] = []

    for item in combined_retrieved:
        chunk = item.chunk
        date_str = (
            chunk.created_at.strftime("%Y-%m-%d")
            if hasattr(chunk.created_at, "strftime")
            else str(chunk.created_at)[:10]
        )
        if chunk.document_type in {"prior_summary", "radiology_report", "clinical_summary"}:
            summary_count += 1
            formatted_parts.append(
                f"[HISTORICAL SUMMARY · {date_str} · {chunk.title}]\n{chunk.content}"
            )
        else:
            note_count += 1
            formatted_parts.append(
                f"[CLINICAL NOTE · {date_str} · {chunk.title}]\n{chunk.content}"
            )

    formatted_context = "\n\n---\n\n".join(formatted_parts) if formatted_parts else "No prior summaries or records found for this patient."

    return RAGContext(
        patient_id=patient_id,
        query=query,
        retrieved_chunks=combined_retrieved,
        formatted_context=formatted_context,
        older_summaries_count=summary_count,
        clinical_notes_count=note_count,
    )


def retrieve_patient_context(
    *,
    patient_id: str,
    query_text: str,
    top_k: int = 5,
) -> RAGContext:
    """General semantic retrieval of patient context across all document types."""
    vector_store = get_vector_store()
    results = vector_store.query(
        patient_id=patient_id,
        query_text=query_text,
        top_k=top_k,
    )

    formatted_parts = []
    summary_count = 0
    note_count = 0

    for item in results:
        chunk = item.chunk
        date_str = (
            chunk.created_at.strftime("%Y-%m-%d")
            if hasattr(chunk.created_at, "strftime")
            else str(chunk.created_at)[:10]
        )
        if chunk.document_type in {"prior_summary", "radiology_report"}:
            summary_count += 1
        else:
            note_count += 1

        formatted_parts.append(
            f"[{chunk.document_type.upper()} · {date_str} · {chunk.title} (Relevance: {item.score})]\n{chunk.content}"
        )

    formatted_context = "\n\n---\n\n".join(formatted_parts) if formatted_parts else "No matching records found for this patient."

    return RAGContext(
        patient_id=patient_id,
        query=query_text,
        retrieved_chunks=results,
        formatted_context=formatted_context,
        older_summaries_count=summary_count,
        clinical_notes_count=note_count,
    )
