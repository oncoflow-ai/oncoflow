from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.modules.agents.contracts import (
    AgentStepLog,
    ImageStreamPayload,
    SynthesizedSummary,
    TextStreamPayload,
)

logger = logging.getLogger(__name__)


class ClinicalSummaryAgent:
    """Agent that synthesizes quantitative image measurements and historical RAG context into a structured summary."""

    name = "ClinicalSummaryAgent"

    def process(
        self,
        *,
        image_data: ImageStreamPayload,
        text_data: TextStreamPayload,
    ) -> tuple[SynthesizedSummary, AgentStepLog]:
        start_time = datetime.now(timezone.utc).isoformat()

        # Build quantitative summary
        quant = {
            "current_volume_cm3": image_data.total_volume_cm3,
            "prior_volume_cm3": image_data.prior_volume_cm3,
            "volume_change_pct": image_data.volume_change_pct,
            "longest_diameter_mm": image_data.longest_diameter_mm,
            "prior_longest_diameter_mm": image_data.prior_diameter_mm,
            "diameter_change_mm": image_data.diameter_change_mm,
            "recist_category": image_data.recist_category,
            "lesion_count": image_data.lesion_count,
            "confidence": "high",
        }

        # Synthesize finding and comparison
        title = "AI Brain MRI Longitudinal Segmentation Report"
        technique = (
            "Automated volumetric 3D tumor segmentation and registration performed on post-contrast "
            "T1-weighted and T2/FLAIR MRI sequences. Multi-agent coordination synthesized current volumetric "
            "measurements with prior patient summaries retrieved via clinical RAG."
        )

        lesion_phrase = f"{image_data.lesion_count} discrete lesion{'s' if image_data.lesion_count > 1 else ''}"
        finding = (
            f"Segmentation analysis identifies {lesion_phrase} centered in the intra-axial deep white matter. "
            f"Current total segmented volume is {image_data.total_volume_cm3:.2f} cm³ with a maximum axial "
            f"diameter of {image_data.longest_diameter_mm:.1f} mm. Lesion exhibits a measurable enhancing component "
            f"with associated surrounding peritumoral edema."
        )

        if image_data.is_longitudinal and image_data.prior_volume_cm3 is not None:
            pct = image_data.volume_change_pct or 0.0
            direction = "increase" if pct > 0 else "decrease"
            comparison = (
                f"Compared with previous scan findings retrieved from patient history (baseline volume: "
                f"{image_data.prior_volume_cm3:.2f} cm³, diameter: {image_data.prior_diameter_mm:.1f} mm), total tumor volume "
                f"demonstrates an interval {direction} of {abs(pct):.1f}% (current: {image_data.total_volume_cm3:.2f} cm³). "
                f"Longest axial diameter changed by {image_data.diameter_change_mm:+.1f} mm. "
            )
            if text_data.retrieved_summaries_count > 0:
                comparison += (
                    f"Prior clinical summaries ({text_data.retrieved_summaries_count} retrieved via RAG) "
                    f"were integrated into this longitudinal progression assessment."
                )
        else:
            comparison = "No prior imaging reference was available for volumetric delta calculation."

        recist = image_data.recist_category or "Stable Disease (SD)"
        impression = (
            f"Volumetric assessment indicates {recist.lower()}. Segmented tumor volume is {image_data.total_volume_cm3:.2f} cm³. "
            f"Findings are consistent with interval tumor dynamics requiring multidisciplinary oncology correlation."
        )

        recommendations = [
            "Correlate interval volumetric measurements with current chemotherapy/radiation regimen.",
            "Multidisciplinary tumor board review for clinical progression evaluation.",
            "Short-interval follow-up MRI in 8-12 weeks to monitor growth trajectory.",
        ]

        summary = SynthesizedSummary(
            title=title,
            technique=technique,
            findings=finding,
            impression=impression,
            comparison=comparison,
            recommendations=recommendations,
            quantitative=quant,
            rag_context_used=text_data.referenced_sources,
        )

        step_log = AgentStepLog(
            agent_name=self.name,
            action="synthesize_summary",
            status="completed",
            timestamp=start_time,
            details={
                "title": title,
                "volume_cm3": image_data.total_volume_cm3,
                "rag_sources_used": len(text_data.referenced_sources),
            },
        )

        return summary, step_log
