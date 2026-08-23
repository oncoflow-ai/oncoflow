from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.modules.agents.contracts import AgentStepLog, TextStreamPayload
from app.modules.rag.service import retrieve_prior_summaries_context

logger = logging.getLogger(__name__)


class TextStreamAgent:
    """Agent responsible for retrieving historical clinical summaries and patient records via RAG."""

    name = "TextStreamAgent"

    def process(
        self,
        *,
        patient_id: str,
        custom_query: str = "tumor volume interval change prior summary response",
        top_k: int = 5,
    ) -> tuple[TextStreamPayload, AgentStepLog]:
        start_time = datetime.now(timezone.utc).isoformat()

        # Retrieve RAG context specifically targeting prior summaries for this client
        rag_context = retrieve_prior_summaries_context(
            patient_id=patient_id,
            query=custom_query,
            top_k=top_k,
        )

        sources = []
        for rc in rag_context.retrieved_chunks:
            sources.append(
                {
                    "chunk_id": rc.chunk.chunk_id,
                    "title": rc.chunk.title,
                    "document_type": rc.chunk.document_type,
                    "created_at": (
                        rc.chunk.created_at.isoformat()
                        if hasattr(rc.chunk.created_at, "isoformat")
                        else str(rc.chunk.created_at)
                    ),
                    "relevance_score": rc.score,
                    "snippet": rc.chunk.content[:160] + "..." if len(rc.chunk.content) > 160 else rc.chunk.content,
                }
            )

        payload = TextStreamPayload(
            patient_id=patient_id,
            retrieved_summaries_count=rag_context.older_summaries_count,
            retrieved_notes_count=rag_context.clinical_notes_count,
            context_text=rag_context.formatted_context,
            referenced_sources=sources,
        )

        step_log = AgentStepLog(
            agent_name=self.name,
            action="retrieve_rag_context",
            status="completed",
            timestamp=start_time,
            details={
                "patient_id": patient_id,
                "retrieved_summaries_count": payload.retrieved_summaries_count,
                "retrieved_notes_count": payload.retrieved_notes_count,
                "sources_count": len(sources),
            },
        )

        return payload, step_log
