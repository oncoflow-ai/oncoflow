from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from app.infra.db.models import PatientSummary, Study
from app.infra.db.session import create_session_factory
from app.modules.agents.contracts import (
    AgentStepLog,
    OrchestrationResult,
    SynthesizedSummary,
)
from app.modules.agents.image_agent import ImageStreamAgent
from app.modules.agents.summary_agent import ClinicalSummaryAgent
from app.modules.agents.text_agent import TextStreamAgent
from app.modules.agents.validation_agent import SafetyValidationAgent
from app.modules.rag.service import index_summary_in_rag

logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """Orchestrates dual-stream coordination (Image Stream + Text RAG Stream) to generate verified clinical summaries."""

    def __init__(
        self,
        *,
        image_agent: ImageStreamAgent | None = None,
        text_agent: TextStreamAgent | None = None,
        summary_agent: ClinicalSummaryAgent | None = None,
        validation_agent: SafetyValidationAgent | None = None,
    ) -> None:
        self.image_agent = image_agent or ImageStreamAgent()
        self.text_agent = text_agent or TextStreamAgent()
        self.summary_agent = summary_agent or ClinicalSummaryAgent()
        self.validation_agent = validation_agent or SafetyValidationAgent()

    def orchestrate(
        self,
        *,
        patient_id: str,
        study_id: str | None = None,
        override_metrics: dict[str, Any] | None = None,
        custom_query: str = "tumor volume interval change prior summary response",
        persist: bool = True,
    ) -> OrchestrationResult:
        orchestration_id = str(uuid4())
        logs: list[AgentStepLog] = []

        start_log = AgentStepLog(
            agent_name="MultiAgentOrchestrator",
            action="start_coordination",
            status="running",
            timestamp=datetime.now(timezone.utc).isoformat(),
            details={"orchestration_id": orchestration_id, "patient_id": patient_id, "study_id": study_id},
        )
        logs.append(start_log)

        # 1. Coordinate Image Stream
        image_payload, img_log = self.image_agent.process(
            study_id=study_id,
            override_metrics=override_metrics,
        )
        logs.append(img_log)

        # 2. Coordinate Text Stream via RAG (retrieving older summaries for context)
        text_payload, txt_log = self.text_agent.process(
            patient_id=patient_id,
            custom_query=custom_query,
        )
        logs.append(txt_log)

        # 3. Synthesize Summary
        pre_summary, sum_log = self.summary_agent.process(
            image_data=image_payload,
            text_data=text_payload,
        )
        logs.append(sum_log)

        # 4. Safety & Validation Agent
        validation_result, val_log = self.validation_agent.process(
            summary=pre_summary,
            ground_truth_image=image_payload,
        )
        logs.append(val_log)

        # Attach validation
        final_summary = SynthesizedSummary(
            title=pre_summary.title,
            technique=pre_summary.technique,
            findings=pre_summary.findings,
            impression=pre_summary.impression,
            comparison=pre_summary.comparison,
            recommendations=pre_summary.recommendations,
            quantitative=pre_summary.quantitative,
            rag_context_used=pre_summary.rag_context_used,
            validation=validation_result,
        )

        completed_at = datetime.now(timezone.utc)
        completed_iso = completed_at.isoformat()

        finish_log = AgentStepLog(
            agent_name="MultiAgentOrchestrator",
            action="finish_coordination",
            status="completed" if validation_result.is_valid else "warning",
            timestamp=completed_iso,
            details={
                "validation_status": validation_result.is_valid,
                "confidence": validation_result.confidence_score,
            },
        )
        logs.append(finish_log)

        # 5. Persist and index back into RAG for future longitudinal comparison
        if persist:
            self._persist_and_index_summary(
                patient_id=patient_id,
                study_id=study_id,
                summary=final_summary,
                logs=logs,
                created_at=completed_at,
            )

        return OrchestrationResult(
            orchestration_id=orchestration_id,
            patient_id=patient_id,
            study_id=study_id,
            summary=final_summary,
            agent_logs=logs,
            completed_at=completed_iso,
        )

    def _persist_and_index_summary(
        self,
        *,
        patient_id: str,
        study_id: str | None,
        summary: SynthesizedSummary,
        logs: list[AgentStepLog],
        created_at: datetime,
    ) -> None:
        try:
            session_factory = create_session_factory()
            summary_uuid = uuid4()

            try:
                patient_uuid = UUID(patient_id)
            except ValueError:
                import hashlib
                patient_uuid = UUID(bytes=hashlib.md5(patient_id.encode("utf-8")).digest())

            study_db_id = None
            if study_id:
                try:
                    s_uuid = UUID(study_id)
                    with session_factory() as session:
                        st = session.query(Study).filter(Study.public_id == s_uuid).one_or_none()
                        if st:
                            study_db_id = st.id
                except ValueError:
                    study_db_id = None

            with session_factory() as session:
                record = PatientSummary(
                    public_id=summary_uuid,
                    patient_public_id=patient_uuid,
                    study_id=study_db_id,
                    title=summary.title,
                    model_name="oncoflow-multiagent-v1",
                    technique=summary.technique,
                    findings=summary.findings,
                    impression=summary.impression,
                    comparison=summary.comparison,
                    recommendations=summary.recommendations,
                    quantitative_summary=summary.quantitative,
                    rag_context_used=summary.rag_context_used,
                    agent_trace={
                        "logs": [
                            {
                                "agent": l.agent_name,
                                "action": l.action,
                                "status": l.status,
                                "timestamp": l.timestamp,
                                "details": l.details,
                            }
                            for l in logs
                        ],
                        "validation": {
                            "is_valid": summary.validation.is_valid if summary.validation else True,
                            "confidence_score": summary.validation.confidence_score if summary.validation else 1.0,
                            "warnings": summary.validation.warnings if summary.validation else [],
                        },
                    },
                    created_at=created_at,
                )
                session.add(record)
                session.commit()

            # Index this new summary into RAG so future scans retrieve it
            combined_summary_text = (
                f"{summary.findings}\n\n{summary.impression}\n\n{summary.comparison}"
            )
            index_summary_in_rag(
                summary_id=str(summary_uuid),
                patient_id=patient_id,
                title=summary.title,
                summary_text=combined_summary_text,
                created_at=created_at,
                metadata={
                    "study_id": study_id,
                    "volume_cm3": summary.quantitative.get("current_volume_cm3"),
                    "longest_diameter_mm": summary.quantitative.get("longest_diameter_mm"),
                },
            )
            logger.info("Persisted and indexed summary %s for patient %s", summary_uuid, patient_id)
        except Exception as exc:
            logger.warning("Failed to persist summary: %s", exc)
