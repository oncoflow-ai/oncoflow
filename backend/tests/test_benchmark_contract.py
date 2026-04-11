from __future__ import annotations

from uuid import uuid4

import pytest

from app.modules.benchmark.contracts import BenchmarkReportContract
from app.modules.benchmark.metrics import MetricContract
from app.modules.benchmark.manifest import BenchmarkManifest, DatasetSplit, PreprocessingProvenance
from app.modules.benchmark.model_registry import (
    BENCHMARK_MODELS,
    BenchmarkModelSpec,
    get_model_spec,
)


def test_model_registry_includes_phase_one_baseline_and_challengers() -> None:
    registry_ids = {spec.model_id for spec in BENCHMARK_MODELS}

    assert "nnunet-v2-resenc" in registry_ids
    assert "nnunet-2d" in registry_ids
    assert "nnunet-25d" in registry_ids
    assert "mednext" in registry_ids
    assert "monai-dynunet" in registry_ids
    assert "monai-segresnetds" in registry_ids
    assert "swinunetr-v2" in registry_ids

    baseline = get_model_spec("nnunet-v2-resenc")
    assert baseline.selection_role == "baseline"
    assert baseline.family == "nnU-Net v2 residual encoder"


def test_medsam2_is_registered_for_interactive_qc_only() -> None:
    medsam2 = get_model_spec("medsam2")

    assert medsam2.selection_role == "interactive-qc"
    assert medsam2.automation_mode == "assisted"


def test_manifest_rejects_unknown_model_ids_and_missing_provenance() -> None:
    with pytest.raises(ValueError, match="Unknown benchmark model id"):
        BenchmarkManifest(
            manifest_id="phase1-contract",
            manifest_version="1.0",
            models=["unknown-model"],
            dataset_split=DatasetSplit(
                training_case_ids=("case-001",),
                validation_case_ids=("case-002",),
                test_case_ids=("case-003",),
                split_provenance="frozen split",
            ),
            preprocessing=PreprocessingProvenance(
                pipeline_id="ingest-v1",
                pipeline_version="2026.04",
                steps=("dicom-to-nifti",),
                source_artifact_uri="s3://oncoflow/ingest.json",
            ),
        )


def test_metric_contract_requires_lesion_and_runtime_signals() -> None:
    contract = MetricContract(
        segmentation_metrics=(
            "lesion_recall",
            "false_positives_per_scan",
            "small_lesion_sensitivity",
            "volume_agreement",
            "dice",
        ),
        runtime_metrics=("runtime_seconds", "failure_count"),
    )

    report = BenchmarkReportContract(
        manifest_id="manifest-01",
        model_id="nnunet-v2-resenc",
        metrics=contract,
        runtime_seconds=123.4,
        failure_count=0,
    )

    assert report.model_id == "nnunet-v2-resenc"


def test_metric_contract_rejects_dice_only_or_missing_runtime_metadata() -> None:
    with pytest.raises(ValueError, match="Dice-only"):
        MetricContract(
            segmentation_metrics=("dice",),
            runtime_metrics=("runtime_seconds", "failure_count"),
        )

    with pytest.raises(ValueError, match="runtime metadata"):
        MetricContract(
            segmentation_metrics=(
                "lesion_recall",
                "false_positives_per_scan",
                "small_lesion_sensitivity",
                "volume_agreement",
            ),
            runtime_metrics=("runtime_seconds",),
        )

    with pytest.raises(ValueError, match="dataset split provenance"):
        BenchmarkManifest(
            manifest_id="phase1-contract",
            manifest_version="1.0",
            models=["nnunet-v2-resenc"],
            dataset_split=DatasetSplit(
                training_case_ids=("case-001",),
                validation_case_ids=("case-002",),
                test_case_ids=("case-003",),
                split_provenance="",
            ),
            preprocessing=PreprocessingProvenance(
                pipeline_id="ingest-v1",
                pipeline_version="2026.04",
                steps=("dicom-to-nifti",),
                source_artifact_uri="s3://oncoflow/ingest.json",
            ),
        )

    with pytest.raises(ValueError, match="preprocessing provenance"):
        BenchmarkManifest(
            manifest_id="phase1-contract",
            manifest_version="1.0",
            models=["nnunet-v2-resenc"],
            dataset_split=DatasetSplit(
                training_case_ids=("case-001",),
                validation_case_ids=("case-002",),
                test_case_ids=("case-003",),
                split_provenance="frozen split",
            ),
            preprocessing=PreprocessingProvenance(
                pipeline_id="ingest-v1",
                pipeline_version="2026.04",
                steps=(),
                source_artifact_uri="",
            ),
        )
