from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.api.schemas.rag import (
    IngestDocumentRequest,
    IngestDocumentResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    RetrievedChunkResponse,
)
from app.infra.db.models import PatientDocument
from app.infra.db.session import create_session_factory
from app.modules.rag.contracts import PatientDocumentPayload
from app.modules.rag.service import (
    ingest_patient_document,
    retrieve_patient_context,
    retrieve_prior_summaries_context,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/documents", response_model=IngestDocumentResponse)
def ingest_document_endpoint(req: IngestDocumentRequest) -> IngestDocumentResponse:
    try:
        payload = PatientDocumentPayload(
            patient_id=req.patient_id,
            document_type=req.document_type,
            title=req.title,
            content=req.content,
            metadata=req.metadata,
        )
        doc_id = ingest_patient_document(payload)
        return IngestDocumentResponse(
            document_id=doc_id,
            patient_id=req.patient_id,
            status="indexed",
            chunks_created=1,
        )
    except Exception as exc:
        logger.error("Error ingesting document: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/documents/{patient_id}")
def list_patient_documents(patient_id: str) -> list[dict]:
    session_factory = create_session_factory()
    with session_factory() as session:
        try:
            patient_uuid = UUID(patient_id)
        except ValueError:
            import hashlib
            patient_uuid = UUID(bytes=hashlib.md5(patient_id.encode("utf-8")).digest())

        docs = (
            session.query(PatientDocument)
            .filter(PatientDocument.patient_public_id == patient_uuid)
            .order_by(PatientDocument.created_at.desc())
            .all()
        )
        return [
            {
                "document_id": str(d.public_id),
                "patient_id": patient_id,
                "document_type": d.document_type,
                "title": d.title,
                "content_preview": d.content[:200],
                "created_at": d.created_at.isoformat(),
            }
            for d in docs
        ]


@router.post("/query", response_model=RAGQueryResponse)
def query_rag_endpoint(req: RAGQueryRequest) -> RAGQueryResponse:
    try:
        if req.document_types and "prior_summary" in req.document_types:
            ctx = retrieve_prior_summaries_context(
                patient_id=req.patient_id,
                query=req.query,
                top_k=req.top_k,
            )
        else:
            ctx = retrieve_patient_context(
                patient_id=req.patient_id,
                query_text=req.query,
                top_k=req.top_k,
            )

        chunks_resp = [
            RetrievedChunkResponse(
                chunk_id=rc.chunk.chunk_id,
                document_id=rc.chunk.document_id,
                patient_id=rc.chunk.patient_id,
                document_type=rc.chunk.document_type,
                title=rc.chunk.title,
                snippet=rc.chunk.content[:240] + ("..." if len(rc.chunk.content) > 240 else ""),
                score=rc.score,
                created_at=(
                    rc.chunk.created_at.isoformat()
                    if hasattr(rc.chunk.created_at, "isoformat")
                    else str(rc.chunk.created_at)
                ),
            )
            for rc in ctx.retrieved_chunks
        ]

        return RAGQueryResponse(
            patient_id=req.patient_id,
            query=req.query,
            older_summaries_count=ctx.older_summaries_count,
            clinical_notes_count=ctx.clinical_notes_count,
            formatted_context=ctx.formatted_context,
            chunks=chunks_resp,
        )
    except Exception as exc:
        logger.error("Error querying RAG: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
