from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from app.modules.agents.contracts import (
    AgentStepLog,
    ImageStreamPayload,
    SynthesizedSummary,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class SafetyValidationAgent:
    """Agent that performs safety checks, hallucination validation, and ground-truth numerical consistency checks."""

    name = "SafetyValidationAgent"

    def process(
        self,
        *,
        summary: SynthesizedSummary,
        ground_truth_image: ImageStreamPayload,
    ) -> tuple[ValidationResult, AgentStepLog]:
        start_time = datetime.now(timezone.utc).isoformat()

        metric_checks = {}
        warnings: list[str] = []
        hallucination_detected = False

        # 1. Check volume consistency in text vs image stream
        vol_str = f"{ground_truth_image.total_volume_cm3:.2f}"
        vol_found = (
            vol_str in summary.findings
            or vol_str in summary.impression
            or vol_str in summary.comparison
            or f"{ground_truth_image.total_volume_cm3:.1f}" in summary.findings
        )
        metric_checks["current_volume_consistent"] = vol_found
        if not vol_found:
            warnings.append(
                f"Warning: Current volume {vol_str} cm³ not explicitly confirmed in narrative text."
            )

        # 2. Check diameter consistency
        diam_str = f"{ground_truth_image.longest_diameter_mm:.1f}"
        diam_found = diam_str in summary.findings or diam_str in summary.comparison
        metric_checks["diameter_consistent"] = diam_found

        # 3. Check longitudinal delta consistency if longitudinal
        if ground_truth_image.is_longitudinal and ground_truth_image.volume_change_pct is not None:
            pct_val = abs(ground_truth_image.volume_change_pct)
            pct_str = f"{pct_val:.1f}%"
            pct_found = pct_str in summary.comparison or f"{int(pct_val)}%" in summary.comparison
            metric_checks["volume_delta_consistent"] = pct_found
            if not pct_found:
                warnings.append(
                    f"Warning: Stated volume delta {pct_str} missing in comparison narrative."
                )

        # 4. Check for hallucination patterns (e.g. fabricated second lesion when lesion_count==1)
        if ground_truth_image.lesion_count == 1:
            if re.search(r"multiple lesions|multifocal enhancing lesions|several separate lesions", summary.findings, re.I):
                hallucination_detected = True
                warnings.append("Potential hallucination: Text states multiple lesions but image stream found 1.")
                metric_checks["lesion_count_consistent"] = False
            else:
                metric_checks["lesion_count_consistent"] = True

        # Calculate confidence score
        passed_count = sum(1 for v in metric_checks.values() if v)
        total_count = max(len(metric_checks), 1)
        confidence = round(passed_count / total_count, 2)
        if hallucination_detected:
            confidence = min(confidence, 0.4)

        is_valid = (not hallucination_detected) and (confidence >= 0.6)

        result = ValidationResult(
            is_valid=is_valid,
            hallucination_detected=hallucination_detected,
            confidence_score=confidence,
            metric_checks=metric_checks,
            warnings=warnings,
        )

        step_log = AgentStepLog(
            agent_name=self.name,
            action="validate_summary_safety",
            status="completed" if is_valid else "warning",
            timestamp=start_time,
            details={
                "is_valid": is_valid,
                "confidence_score": confidence,
                "hallucination_detected": hallucination_detected,
                "warnings": warnings,
            },
        )

        return result, step_log
