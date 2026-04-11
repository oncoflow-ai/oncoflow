from __future__ import annotations

from dataclasses import dataclass

from app.modules.benchmark.metrics import MetricContract


@dataclass(frozen=True)
class BenchmarkReportContract:
    manifest_id: str
    model_id: str
    metrics: MetricContract
    runtime_seconds: float
    failure_count: int

    def __post_init__(self) -> None:
        if self.runtime_seconds < 0:
            raise ValueError("runtime_seconds must be non-negative")
        if self.failure_count < 0:
            raise ValueError("failure_count must be non-negative")
