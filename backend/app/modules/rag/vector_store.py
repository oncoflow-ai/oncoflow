from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Sequence

from app.modules.rag.contracts import DocumentChunk, RetrievedChunk


class InMemoryVectorStore:
    """Per-tenant isolated in-memory vector store for patient clinical documents and summaries.

    Adheres strictly to the security requirement: queries for patient A are strictly isolated
    from patient B's indices.
    """

    def __init__(self) -> None:
        # patient_id -> list of DocumentChunk
        self._patient_indices: dict[str, list[DocumentChunk]] = {}

    def clear(self) -> None:
        self._patient_indices.clear()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [w.lower() for w in re.findall(r"\b[A-Za-z0-9_-]+\b", text)]

    @classmethod
    def _compute_sparse_embedding(cls, text: str, dim: int = 128) -> list[float]:
        tokens = cls._tokenize(text)
        if not tokens:
            return [0.0] * dim

        counts = Counter(tokens)
        vec = [0.0] * dim
        for word, count in counts.items():
            # Hash the word into the fixed dimension
            idx = hash(word) % dim
            # Simple TF-IDF inspired log-frequency
            vec[idx] += 1.0 + math.log(count)

        # Normalize to unit length
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 1e-9:
            vec = [v / norm for v in vec]
        return vec

    @staticmethod
    def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        if len(vec_a) != len(vec_b) or not vec_a:
            return 0.0
        return sum(a * b for a, b in zip(vec_a, vec_b))

    def add_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        for chunk in chunks:
            if chunk.embedding is None:
                # Compute embedding
                computed_emb = self._compute_sparse_embedding(
                    f"{chunk.title} {chunk.document_type} {chunk.content}"
                )
                chunk = DocumentChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    patient_id=chunk.patient_id,
                    document_type=chunk.document_type,
                    title=chunk.title,
                    content=chunk.content,
                    created_at=chunk.created_at,
                    metadata=chunk.metadata,
                    embedding=computed_emb,
                )

            patient_chunks = self._patient_indices.setdefault(chunk.patient_id, [])
            # Replace existing chunk if exists, else append
            self._patient_indices[chunk.patient_id] = [
                c for c in patient_chunks if c.chunk_id != chunk.chunk_id
            ] + [chunk]

    def query(
        self,
        *,
        patient_id: str,
        query_text: str,
        top_k: int = 5,
        document_types: Sequence[str] | None = None,
        min_score: float = 0.01,
    ) -> list[RetrievedChunk]:
        """Query RAG store strictly for a single patient."""
        chunks = self._patient_indices.get(patient_id, [])
        if not chunks:
            return []

        if document_types is not None:
            type_set = set(document_types)
            chunks = [c for c in chunks if c.document_type in type_set]

        query_vec = self._compute_sparse_embedding(query_text)
        query_tokens = set(self._tokenize(query_text))

        scored: list[RetrievedChunk] = []
        now = datetime.now(timezone.utc)

        for chunk in chunks:
            # 1. Cosine similarity of embedding
            sim = 0.0
            if chunk.embedding is not None:
                sim = self._cosine_similarity(query_vec, chunk.embedding)

            # 2. Token overlap bonus (lexical matching)
            chunk_tokens = set(self._tokenize(f"{chunk.title} {chunk.content}"))
            token_overlap = (
                len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)
                if query_tokens
                else 0.0
            )

            # 3. Recency factor (up to 10% boost for fresher summaries)
            age_days = max(
                (now - (chunk.created_at if chunk.created_at.tzinfo else chunk.created_at.replace(tzinfo=timezone.utc))).total_seconds() / 86400.0,
                0.0,
            )
            recency_decay = 1.0 / (1.0 + age_days * 0.01)

            final_score = (sim * 0.6 + token_overlap * 0.4) * recency_decay

            if final_score >= min_score:
                scored.append(RetrievedChunk(chunk=chunk, score=round(final_score, 4)))


        # Sort descending by score
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def get_chronological_summaries(
        self,
        patient_id: str,
        limit: int = 10,
    ) -> list[DocumentChunk]:
        """Fetch all prior summaries for a patient in chronological order."""
        chunks = self._patient_indices.get(patient_id, [])
        summary_chunks = [
            c for c in chunks if c.document_type in {"prior_summary", "radiology_report", "clinical_summary"}
        ]
        # Sort chronologically (oldest to newest or newest to oldest)
        summary_chunks.sort(key=lambda c: c.created_at, reverse=False)
        return summary_chunks[:limit]


_GLOBAL_VECTOR_STORE = InMemoryVectorStore()


def get_vector_store() -> InMemoryVectorStore:
    return _GLOBAL_VECTOR_STORE
