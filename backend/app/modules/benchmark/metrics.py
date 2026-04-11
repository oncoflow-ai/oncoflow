from __future__ import annotations

from dataclasses import dataclass

REQUIRED_SEGMENTATION_METRICS = (
    "lesion_recall",
    "false_positives_per_scan",
    "small_lesion_sensitivity",
    "volume_agreement",
)
REQUIRED_RUNTIME_METRICS = (
    "runtime_seconds",
    "failure_count",
)


@dataclass(frozen=True)
class MetricContract:
    segmentation_metrics: tuple[str, ...]
    runtime_metrics: tuple[str, ...]

    def __post_init__(self) -> None:
        segmentation = set(self.segmentation_metrics)
        runtime = set(self.runtime_metrics)

        if segmentation == {"dice"}:
            raise ValueError("Dice-only metric sets are incomplete for benchmark selection")

        missing_segmentation = set(REQUIRED_SEGMENTATION_METRICS) - segmentation
        if missing_segmentation:
            raise ValueError(
                "Benchmark metric contract is missing required lesion-centric metrics: "
                + ", ".join(sorted(missing_segmentation))
            )

        missing_runtime = set(REQUIRED_RUNTIME_METRICS) - runtime
        if missing_runtime:
            raise ValueError(
                "Benchmark metric contract is missing required runtime metadata: "
                + ", ".join(sorted(missing_runtime))
            )
