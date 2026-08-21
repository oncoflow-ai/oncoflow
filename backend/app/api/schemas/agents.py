from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class OrchestrateSummaryRequest(BaseModel):
    patient_id: str = Field(..., description="Target patient identifier")
    study_id: str | None = Field(default=None, description="Current study UUID to extract image stream metrics from")
    override_metrics: dict[str, Any] | None = Field(default=None, description="Optional manual/simulated image stream metrics")
    custom_query: str = Field(default="tumor volume interval change prior summary response", description="RAG query prompt")
    persist: bool = Field(default=True, description="Whether to persist and index the synthesized summary")


class AgentStepLogResponse(BaseModel):
    agent_name: str
    action: str
    status: str
    timestamp: str
    details: dict[str, Any] = Field(default_factory=dict)


class ValidationResponse(BaseModel):
    is_valid: bool
    hallucination_detected: bool
    confidence_score: float
    metric_checks: dict[str, bool]
    warnings: list[str] = Field(default_factory=list)


class SynthesizedSummaryResponse(BaseModel):
    title: str
    technique: str
    findings: str
    impression: str
    comparison: str
    recommendations: list[str]
    quantitative: dict[str, Any]
    rag_context_used: list[dict[str, Any]]
    validation: ValidationResponse | None = None


class OrchestrationResponse(BaseModel):
    orchestration_id: str
    patient_id: str
    study_id: str | None
    summary: SynthesizedSummaryResponse
    agent_logs: list[AgentStepLogResponse]
    completed_at: str


class PatientSummaryListItem(BaseModel):
    summary_id: str
    patient_id: str
    study_id: str | None
    title: str
    model_name: str
    findings: str
    impression: str
    comparison: str
    recommendations: list[str]
    quantitative_summary: dict[str, Any]
    created_at: str
