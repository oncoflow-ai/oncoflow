from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkModelSpec:
    model_id: str
    family: str
    selection_role: str
    automation_mode: str
    notes: str


BENCHMARK_MODELS = (
    BenchmarkModelSpec(
        model_id="nnunet-v2-resenc",
        family="nnU-Net v2 residual encoder",
        selection_role="baseline",
        automation_mode="automatic",
        notes="Primary supervised production baseline for Phase 2.",
    ),
    BenchmarkModelSpec(
        model_id="nnunet-2d",
        family="nnU-Net v2 2D",
        selection_role="challenger",
        automation_mode="automatic",
        notes="Benchmark for anisotropic MRI protocols.",
    ),
    BenchmarkModelSpec(
        model_id="nnunet-25d",
        family="nnU-Net v2 2.5D",
        selection_role="challenger",
        automation_mode="automatic",
        notes="Alternative for MRI series with limited through-plane resolution.",
    ),
    BenchmarkModelSpec(
        model_id="mednext",
        family="MedNeXt",
        selection_role="challenger",
        automation_mode="automatic",
        notes="CNN challenger with strong medical segmentation performance.",
    ),
    BenchmarkModelSpec(
        model_id="monai-dynunet",
        family="MONAI DynUNet",
        selection_role="challenger",
        automation_mode="automatic",
        notes="MONAI baseline with strong volumetric segmentation support.",
    ),
    BenchmarkModelSpec(
        model_id="monai-segresnetds",
        family="MONAI SegResNetDS",
        selection_role="challenger",
        automation_mode="automatic",
        notes="Residual CNN challenger for robust medical segmentation comparisons.",
    ),
    BenchmarkModelSpec(
        model_id="swinunetr-v2",
        family="SwinUNETR-V2",
        selection_role="challenger",
        automation_mode="automatic",
        notes="Transformer challenger to test beyond CNN baselines.",
    ),
    BenchmarkModelSpec(
        model_id="medsam2",
        family="MedSAM2",
        selection_role="interactive-qc",
        automation_mode="assisted",
        notes="Use for annotation/QC assistance rather than automatic production baseline.",
    ),
)

_REGISTRY = {spec.model_id: spec for spec in BENCHMARK_MODELS}


def get_model_spec(model_id: str) -> BenchmarkModelSpec:
    try:
        return _REGISTRY[model_id]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark model id: {model_id}") from exc
