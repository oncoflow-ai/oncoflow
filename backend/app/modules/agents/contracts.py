from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class AgentStepLog:
    agent_name: str
    action: str
    status: str
    timestamp: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageStreamPayload:
    study_id: str
    lesion_count: int
    primary_volume_cm3: float
    total_volume_cm3: float
    longest_diameter_mm: float
    is_longitudinal: bool = False
    prior_volume_cm3: float | None = None
    prior_diameter_mm: float | None = None
    volume_change_pct: float | None = None
    diameter_change_mm: float | None = None
    recist_category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextStreamPayload:
    patient_id: str
    retrieved_summaries_count: int
    retrieved_notes_count: int
    context_text: str
    referenced_sources: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    hallucination_detected: bool
    confidence_score: float
    metric_checks: dict[str, bool]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SynthesizedSummary:
    title: str
    technique: str
    findings: str
    impression: str
    comparison: str
    recommendations: list[str]
    quantitative: dict[str, Any]
    rag_context_used: list[dict[str, Any]] = field(default_factory=list)
    validation: ValidationResult | None = None


@dataclass(frozen=True)
class OrchestrationResult:
    orchestration_id: str
    patient_id: str
    study_id: str | None
    summary: SynthesizedSummary
    agent_logs: list[AgentStepLog]
    completed_at: str
