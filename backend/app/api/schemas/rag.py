from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class IngestDocumentRequest(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier")
    document_type: str = Field(default="clinical_note", description="Document type (prior_summary, clinical_note, pathology_report, etc.)")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document textual content")
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestDocumentResponse(BaseModel):
    document_id: str
    patient_id: str
    status: str = "indexed"
    chunks_created: int


class RAGQueryRequest(BaseModel):
    patient_id: str = Field(..., description="Patient identifier to scope the search")
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, ge=1, le=20)
    document_types: list[str] | None = None


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    patient_id: str
    document_type: str
    title: str
    snippet: str
    score: float
    created_at: str


class RAGQueryResponse(BaseModel):
    patient_id: str
    query: str
    older_summaries_count: int
    clinical_notes_count: int
    formatted_context: str
    chunks: list[RetrievedChunkResponse]
