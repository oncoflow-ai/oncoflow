from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from app.infra.db.models import Artifact, StoredLesionResult, Study
from app.infra.db.session import create_session_factory
from app.modules.results.materialize import materialize_study_results


@pytest.fixture(autouse=True)
def configure_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONCOFLOW_DATABASE_URL", f"sqlite+pysqlite:///{tmp_path / 'materialize.sqlite3'}")
    monkeypatch.setenv("ONCOFLOW_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("ONCOFLOW_STORAGE_STAGING_DIR", "raw")


def test_materialize_study_results_persists_result_rows_and_bundle_artifact() -> None:
    session_factory = create_session_factory()
    with session_factory() as session:
        study = Study(
            public_id=uuid4(),
            study_instance_uid="1.2.3.4.5",
            source_kind="dicom-study",
            source_metadata={},
            staging_status="processed",
        )
        session.add(study)
        session.flush()
        session.add(
            Artifact(
                study_id=study.id,
                artifact_kind="segmentation-case-result",
                storage_root="derived",
                relative_path=f"studies/{study.public_id}/segmentation/result.json",
                source_metadata={"needs_review": True, "case_qc_reasons": ["review"], "lesion_count": 1},
            )
        )
        session.add(
            Artifact(
                study_id=study.id,
                artifact_kind="segmentation-mask",
                storage_root="derived",
                relative_path=f"studies/{study.public_id}/lesions/lesion-001-mask.nii.gz",
                source_metadata={
                    "lesion_id": "study-001:lesion-001",
                    "bounding_box": {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1, "z_min": 0, "z_max": 1},
                    "slot_provenance": [],
                    "runner": {"model_id": "nnunet-v2-resenc"},
                    "geometry": {"spacing_mm": [1.0, 1.0, 1.0]},
                    "occupied_voxels_ijk": [[0, 0, 0], [1, 1, 1]],
                },
            )
        )
        session.add(
            Artifact(
                study_id=study.id,
                artifact_kind="review-overlay",
                storage_root="derived",
                relative_path=f"studies/{study.public_id}/review/lesion-001-overlay.png",
                source_metadata={"lesion_id": "study-001:lesion-001"},
            )
        )
        session.commit()

        result = materialize_study_results(session=session, study_public_id=study.public_id)
        session.commit()
        assert result.lesion_count == 1

    with session_factory() as session:
        artifacts = session.query(Artifact).filter(Artifact.artifact_kind == "study-result-bundle").all()
        lesion = session.query(StoredLesionResult).one()
        assert len(artifacts) == 1
        assert artifacts[0].relative_path.endswith("study-result.json")
        assert lesion.measurement_payload["volume_mm3"] == 2.0
        assert round(lesion.measurement_payload["longest_diameter_mm"], 5) == round(3 ** 0.5, 5)


def test_materialize_study_results_scopes_artifacts_to_latest_segmentation_run() -> None:
    session_factory = create_session_factory()
    with session_factory() as session:
        study = Study(
            public_id=uuid4(),
            study_instance_uid="1.2.3.4.5",
            source_kind="dicom-study",
            source_metadata={},
            staging_status="processed",
        )
        session.add(study)
        session.flush()
        session.add(
            Artifact(
                study_id=study.id,
                artifact_kind="segmentation-case-result",
                storage_root="derived",
                relative_path=f"studies/{study.public_id}/segmentation/old-result.json",
                source_metadata={
                    "segmentation_run_id": "run-old",
                    "needs_review": False,
                    "case_qc_reasons": [],
                    "lesion_count": 1,
                },
            )
        )
        session.add(
            Artifact(
                study_id=study.id,
                artifact_kind="segmentation-mask",
                storage_root="derived",
                relative_path=f"studies/{study.public_id}/lesions/old-mask.nii.gz",
                source_metadata={
                    "segmentation_run_id": "run-old",
                    "lesion_id": "study-001:lesion-001",
                    "bounding_box": {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1, "z_min": 0, "z_max": 1},
                    "slot_provenance": [],
                    "runner": {"model_id": "old"},
                    "geometry": {"spacing_mm": [1.0, 1.0, 1.0]},
                    "occupied_voxels_ijk": [[0, 0, 0]],
                },
            )
        )
        session.add(
            Artifact(
                study_id=study.id,
                artifact_kind="review-overlay",
                storage_root="derived",
                relative_path=f"studies/{study.public_id}/review/old-overlay.png",
                source_metadata={"segmentation_run_id": "run-old", "lesion_id": "study-001:lesion-001"},
            )
        )
        session.add(
            Artifact(
                study_id=study.id,
                artifact_kind="segmentation-case-result",
                storage_root="derived",
                relative_path=f"studies/{study.public_id}/segmentation/new-result.json",
                source_metadata={
                    "segmentation_run_id": "run-new",
                    "needs_review": True,
                    "case_qc_reasons": ["review"],
                    "lesion_count": 1,
                },
            )
        )
        session.add(
            Artifact(
                study_id=study.id,
                artifact_kind="segmentation-mask",
                storage_root="derived",
                relative_path=f"studies/{study.public_id}/lesions/new-mask.nii.gz",
                source_metadata={
                    "segmentation_run_id": "run-new",
                    "lesion_id": "study-001:lesion-002",
                    "bounding_box": {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1, "z_min": 0, "z_max": 1},
                    "slot_provenance": [],
                    "runner": {"model_id": "new"},
                    "geometry": {"spacing_mm": [1.0, 1.0, 1.0]},
                    "occupied_voxels_ijk": [[0, 0, 0]],
                },
            )
        )
        session.add(
            Artifact(
                study_id=study.id,
                artifact_kind="review-overlay",
                storage_root="derived",
                relative_path=f"studies/{study.public_id}/review/new-overlay.png",
                source_metadata={"segmentation_run_id": "run-new", "lesion_id": "study-001:lesion-002"},
            )
        )
        session.commit()

        result = materialize_study_results(session=session, study_public_id=study.public_id)
        session.commit()
        assert result.lesion_count == 1

    with session_factory() as session:
        lesions = session.query(StoredLesionResult).all()
        assert len(lesions) == 1
        assert lesions[0].lesion_id == "study-001:lesion-002"
        assert lesions[0].artifact_refs["mask"]["relative_path"].endswith("new-mask.nii.gz")
