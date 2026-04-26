from __future__ import annotations

from pathlib import Path

import pytest

from app.modules.segmentation.contracts import BoundingBox3D, ManagedArtifactRef
from app.modules.segmentation.packaging import package_predictions
from app.modules.segmentation.review import determine_case_review, make_review_artifact_ref
from app.modules.segmentation.runner import NormalizedLesionPrediction


@pytest.fixture(autouse=True)
def configure_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONCOFLOW_DATABASE_URL", f"sqlite+pysqlite:///{tmp_path / 'segmentation-qc.sqlite3'}")
    monkeypatch.setenv("ONCOFLOW_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("ONCOFLOW_STORAGE_STAGING_DIR", "raw")


def _prediction(*, z_min: int, confidence: float, reasons: tuple[str, ...] = ()) -> NormalizedLesionPrediction:
    return NormalizedLesionPrediction(
        mask_artifact=ManagedArtifactRef(
            storage_root="derived",
            relative_path=f"studies/study-001/lesions/component-{z_min}.nii.gz",
        ),
        bounding_box=BoundingBox3D(x_min=1, x_max=4, y_min=2, y_max=6, z_min=z_min, z_max=z_min + 2),
        confidence_score=confidence,
        occupied_voxels_ijk=((0, 0, 0), (1, 1, 1)),
        warning_reasons=reasons,
    )


def test_packaging_preserves_low_confidence_components_for_review() -> None:
    review_artifact = make_review_artifact_ref(
        artifact_kind="overlay",
        relative_path="studies/study-001/review/lesion-001-overlay.png",
        provenance_label="qc-overlay",
    )
    lesions = package_predictions(
        study_id="study-001",
        predictions=(
            _prediction(z_min=5, confidence=0.91),
            _prediction(z_min=1, confidence=0.22, reasons=("low confidence logits",)),
        ),
        review_artifacts_by_index={0: (review_artifact.artifact,), 1: ()},
    )

    assert [lesion.lesion_id for lesion in lesions] == ["study-001:lesion-001", "study-001:lesion-002"]
    assert lesions[0].qc.flagged_for_review is True
    assert lesions[0].review_artifacts[0].artifact_kind == "overlay"
    assert lesions[1].qc.flagged_for_review is False


def test_packaging_synthesizes_qc_reason_for_low_confidence_without_runner_warning() -> None:
    lesions = package_predictions(
        study_id="study-001",
        predictions=(
            _prediction(z_min=2, confidence=0.22, reasons=()),
        ),
    )

    assert lesions[0].qc.flagged_for_review is True
    assert lesions[0].qc.reasons == ("low confidence score",)


def test_case_review_combines_bundle_degradation_and_lesion_flags() -> None:
    needs_review, reasons = determine_case_review(
        bundle_degradation_reasons=("missing canonical slots: t1_post_or_fs",),
        lesion_flagged_for_review=(False, True),
        runner_warnings=("runner used fallback execution backend",),
    )

    assert needs_review is True
    assert "missing canonical slots: t1_post_or_fs" in reasons
    assert "one or more lesions were flagged for review" in reasons
    assert "runner used fallback execution backend" in reasons


def test_review_artifact_descriptor_rejects_phi_bearing_metadata() -> None:
    with pytest.raises(ValueError, match="PHI-bearing keys"):
        make_review_artifact_ref(
            artifact_kind="thumbnail",
            relative_path="studies/study-001/review/thumb.png",
            provenance_label="thumb",
            metadata={"patient_name": "Alice Example"},
        )
