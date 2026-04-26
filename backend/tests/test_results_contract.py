from __future__ import annotations

import pytest

from app.modules.results.contracts import (
    StoredArtifactRef,
    StoredCaseResult,
    StoredLesionMeasurement,
    StoredLesionResult,
)


def test_stored_case_result_requires_stable_ids_measurements_and_safe_artifact_refs() -> None:
    lesion = StoredLesionResult(
        lesion_id="study-001:lesion-001",
        bounding_box={"x_min": 1, "x_max": 4, "y_min": 2, "y_max": 5, "z_min": 0, "z_max": 3},
        measurements=StoredLesionMeasurement(volume_mm3=12.5, longest_diameter_mm=5.1),
        mask_artifact=StoredArtifactRef(
            artifact_kind="segmentation-mask",
            storage_root="derived",
            relative_path="studies/study-001/lesions/lesion-001-mask.nii.gz",
        ),
    )
    case = StoredCaseResult(
        study_id="study-001",
        result_artifact=StoredArtifactRef(
            artifact_kind="study-result-bundle",
            storage_root="derived",
            relative_path="studies/study-001/results/study-result.json",
        ),
        lesions=(lesion,),
        needs_review=False,
    )

    assert case.lesions[0].lesion_id == "study-001:lesion-001"
    assert case.lesions[0].measurements.volume_mm3 == 12.5


def test_results_contract_rejects_raw_path_leakage_or_missing_ids() -> None:
    with pytest.raises(ValueError, match="retrieval-safe"):
        StoredArtifactRef(
            artifact_kind="mask",
            storage_root="derived",
            relative_path="/tmp/raw-path.nii.gz",
        )

    with pytest.raises(ValueError, match="lesion_id is required"):
        StoredLesionResult(
            lesion_id="",
            bounding_box={"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1, "z_min": 0, "z_max": 1},
            measurements=StoredLesionMeasurement(volume_mm3=1.0, longest_diameter_mm=1.0),
            mask_artifact=StoredArtifactRef(
                artifact_kind="segmentation-mask",
                storage_root="derived",
                relative_path="studies/study/results/mask.nii.gz",
            ),
        )
