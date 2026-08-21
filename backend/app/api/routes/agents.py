from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.api.schemas.agents import (
    AgentStepLogResponse,
    OrchestrateSummaryRequest,
    OrchestrationResponse,
    PatientSummaryListItem,
    SynthesizedSummaryResponse,
    ValidationResponse,
)
from app.infra.db.models import PatientSummary
from app.infra.db.session import create_session_factory
from app.modules.agents.orchestrator import MultiAgentOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])
_orchestrator = MultiAgentOrchestrator()


@router.post("/orchestrate-summary", response_model=OrchestrationResponse)
def orchestrate_summary_endpoint(req: OrchestrateSummaryRequest) -> OrchestrationResponse:
    try:
        res = _orchestrator.orchestrate(
            patient_id=req.patient_id,
            study_id=req.study_id,
            override_metrics=req.override_metrics,
            custom_query=req.custom_query,
            persist=req.persist,
        )

        validation_resp = None
        if res.summary.validation:
            v = res.summary.validation
            validation_resp = ValidationResponse(
                is_valid=v.is_valid,
                hallucination_detected=v.hallucination_detected,
                confidence_score=v.confidence_score,
                metric_checks=v.metric_checks,
                warnings=v.warnings,
            )

        summary_resp = SynthesizedSummaryResponse(
            title=res.summary.title,
            technique=res.summary.technique,
            findings=res.summary.findings,
            impression=res.summary.impression,
            comparison=res.summary.comparison,
            recommendations=res.summary.recommendations,
            quantitative=res.summary.quantitative,
            rag_context_used=res.summary.rag_context_used,
            validation=validation_resp,
        )

        logs_resp = [
            AgentStepLogResponse(
                agent_name=l.agent_name,
                action=l.action,
                status=l.status,
                timestamp=l.timestamp,
                details=l.details,
            )
            for l in res.agent_logs
        ]

        return OrchestrationResponse(
            orchestration_id=res.orchestration_id,
            patient_id=res.patient_id,
            study_id=res.study_id,
            summary=summary_resp,
            agent_logs=logs_resp,
            completed_at=res.completed_at,
        )
    except Exception as exc:
        logger.error("Error orchestrating summary: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/summaries/{patient_id}", response_model=list[PatientSummaryListItem])
def list_patient_summaries_endpoint(patient_id: str) -> list[PatientSummaryListItem]:
    session_factory = create_session_factory()
    with session_factory() as session:
        try:
            patient_uuid = UUID(patient_id)
        except ValueError:
            import hashlib
            patient_uuid = UUID(bytes=hashlib.md5(patient_id.encode("utf-8")).digest())

        summaries = (
            session.query(PatientSummary)
            .filter(PatientSummary.patient_public_id == patient_uuid)
            .order_by(PatientSummary.created_at.desc())
            .all()
        )

        return [
            PatientSummaryListItem(
                summary_id=str(s.public_id),
                patient_id=patient_id,
                study_id=str(s.study.public_id) if s.study else None,
                title=s.title,
                model_name=s.model_name,
                findings=s.findings,
                impression=s.impression,
                comparison=s.comparison,
                recommendations=s.recommendations or [],
                quantitative_summary=s.quantitative_summary or {},
                created_at=s.created_at.isoformat(),
            )
            for s in summaries
        ]
