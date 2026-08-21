from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    patient_id: str
    document_type: str
    title: str
    content: str
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    score: float


@dataclass(frozen=True)
class RAGContext:
    patient_id: str
    query: str
    retrieved_chunks: list[RetrievedChunk]
    formatted_context: str
    older_summaries_count: int = 0
    clinical_notes_count: int = 0


@dataclass(frozen=True)
class PatientDocumentPayload:
    patient_id: str
    document_type: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
