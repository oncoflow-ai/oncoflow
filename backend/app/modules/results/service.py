from __future__ import annotations

from uuid import UUID

from app.infra.db.models import Artifact, StoredLesionResult, Study, StudyResult
from app.infra.db.session import create_session_factory
from app.modules.results.contracts import (
    StoredArtifactRef,
    StoredCaseResult,
    StoredLesionMeasurement,
    StoredLesionResult as StoredLesionContract,
)


class ResultNotFoundError(Exception):
    pass


class InvalidResultRequestError(Exception):
    pass


def get_case_result_payload(*, study_id: str) -> StoredCaseResult:
    session_factory = create_session_factory()
    with session_factory() as session:
        try:
            parsed = UUID(study_id)
        except ValueError as exc:
            raise InvalidResultRequestError("study_id must be a valid UUID") from exc
        study = session.query(Study).filter(Study.public_id == parsed).one_or_none()
        if study is None:
            raise ResultNotFoundError("study not found")

        bundle_artifact = None
        for artifact in (
            session.query(Artifact)
            .filter(
                Artifact.study_id == study.id,
                Artifact.artifact_kind == "study-result-bundle",
            )
            .order_by(Artifact.id.desc())
            .all()
        ):
            candidate_study_result_id = artifact.source_metadata.get("study_result_id")
            if isinstance(candidate_study_result_id, int):
                bundle_artifact = artifact
                break
        if bundle_artifact is None:
            raise ResultNotFoundError("result bundle artifact not found")

        study_result_id = bundle_artifact.source_metadata.get("study_result_id")

        study_result = (
            session.query(StudyResult)
            .filter(StudyResult.study_id == study.id, StudyResult.id == study_result_id)
            .one_or_none()
        )
        if study_result is None:
            raise ResultNotFoundError("result not found")

        lesions = (
            session.query(StoredLesionResult)
            .filter(StoredLesionResult.study_result_id == study_result.id)
            .order_by(StoredLesionResult.id.asc())
            .all()
        )
        lesion_payloads = tuple(
            StoredLesionContract(
                lesion_id=lesion.lesion_id,
                bounding_box=dict(lesion.bounding_box),
                measurements=StoredLesionMeasurement(
                    volume_mm3=float(lesion.measurement_payload["volume_mm3"]),
                    longest_diameter_mm=float(lesion.measurement_payload["longest_diameter_mm"]),
                ),
                mask_artifact=StoredArtifactRef(**lesion.artifact_refs["mask"]),
                review_artifacts=tuple(StoredArtifactRef(**artifact) for artifact in lesion.artifact_refs.get("review", [])),
                metadata=dict(lesion.result_metadata),
            )
            for lesion in lesions
        )
        return StoredCaseResult(
            study_id=str(study.public_id),
            result_artifact=StoredArtifactRef(
                artifact_kind=bundle_artifact.artifact_kind,
                storage_root=bundle_artifact.storage_root,
                relative_path=bundle_artifact.relative_path,
            ),
            lesions=lesion_payloads,
            needs_review=bool(study_result.needs_review),
            case_qc_reasons=tuple(study_result.summary_metadata.get("case_qc_reasons", [])),
            metadata=dict(study_result.summary_metadata),
        )
