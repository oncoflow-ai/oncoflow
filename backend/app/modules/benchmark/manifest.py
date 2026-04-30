from __future__ import annotations

from dataclasses import dataclass

from app.modules.benchmark.model_registry import get_model_spec


@dataclass(frozen=True)
class DatasetSplit:
    training_case_ids: tuple[str, ...]
    validation_case_ids: tuple[str, ...]
    test_case_ids: tuple[str, ...]
    split_provenance: str

    def __post_init__(self) -> None:
        if not self.split_provenance.strip():
            raise ValueError("Benchmark dataset split provenance is required")

        if not self.training_case_ids or not self.validation_case_ids or not self.test_case_ids:
            raise ValueError("Benchmark dataset split must include train, validation, and test cases")


@dataclass(frozen=True)
class PreprocessingProvenance:
    pipeline_id: str
    pipeline_version: str
    steps: tuple[str, ...]
    source_artifact_uri: str

    def __post_init__(self) -> None:
        if not self.pipeline_id.strip() or not self.pipeline_version.strip():
            raise ValueError("Benchmark preprocessing provenance requires pipeline identity")
        if not self.steps or not self.source_artifact_uri.strip():
            raise ValueError("Benchmark preprocessing provenance is required")


@dataclass(frozen=True)
class BenchmarkManifest:
    manifest_id: str
    manifest_version: str
    models: list[str]
    dataset_split: DatasetSplit
    preprocessing: PreprocessingProvenance

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("Benchmark manifests must declare at least one model")
        for model_id in self.models:
            get_model_spec(model_id)
