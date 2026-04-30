---
phase: 01-backend-foundation-mri-ingestion
plan: 03
subsystem: api
tags: [fastapi, jobs, uploads, staging, async]
requires:
  - phase: 01-01
    provides: backend entrypoint, router, shared settings, worker seam
  - phase: 01-02
    provides: persistence schema and managed storage roots
provides:
  - Multipart MRI study submission endpoint
  - Durable staged-study and queued-job creation flow
  - Status polling with structured failure payloads
affects: [api, jobs, ingestion, storage]
tech-stack:
  added: []
  patterns: [submit-and-poll, durable-job-status, staged-archive-extraction]
key-files:
  created:
    - backend/app/api/schemas/jobs.py
    - backend/app/api/routes/jobs.py
    - backend/app/modules/jobs/service.py
    - backend/app/modules/jobs/state_machine.py
    - backend/app/modules/jobs/worker_tasks.py
    - backend/app/modules/ingestion/contracts.py
  modified:
    - backend/app/api/router.py
    - backend/tests/conftest.py
    - backend/tests/test_jobs_api.py
key-decisions:
  - "Persist the uploaded archive unchanged and also extract it into a managed raw-study directory before any worker processing."
  - "Return public UUIDs for both job and study while keeping storage details internal."
  - "Drive worker dispatch with persisted identifiers and extracted-study references only, never raw upload bytes."
patterns-established:
  - "Job submission is multipart upload plus immediate queued response."
  - "Status polling exposes durable public status and internal stage separately."
  - "Failure payloads are structured objects, not freeform log strings."
requirements-completed: [ING-01, API-01]
duration: 20min
completed: 2026-04-11
---

# Phase 01: Plan 03 Summary

**MRI study submission and status polling flow with managed archive staging, extracted-study roots, and durable queued-job creation**

## Performance

- **Duration:** 20 min
- **Started:** 2026-04-11T15:16:00Z
- **Completed:** 2026-04-11T15:36:07Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments

- Added `POST /api/v1/jobs/mri-ingestion` for multipart MRI archive submission and immediate queued responses.
- Added `GET /api/v1/jobs/{jobId}` for durable status polling with structured failure details.
- Implemented managed archive persistence, extracted-study-root creation, and identifier-only worker dispatch.

## Task Commits

Atomic per-task commits were not created because this plan was completed as part of the ongoing phase execution and will be recorded in the phase continuation commit on `backend-test`.

## Files Created/Modified

- `backend/app/api/schemas/jobs.py` - Pydantic response schemas with camelCase API output.
- `backend/app/api/routes/jobs.py` - Submit and status endpoints for MRI ingestion jobs.
- `backend/app/modules/jobs/service.py` - Submission, archive extraction, job creation, polling, and failure-marking service.
- `backend/app/modules/jobs/state_machine.py` - Valid status-transition rules and structured error state.
- `backend/app/modules/jobs/worker_tasks.py` - Worker dispatch envelope and execution entrypoint contract.
- `backend/app/modules/ingestion/contracts.py` - Extracted-study and raw-archive reference contract.
- `backend/tests/test_jobs_api.py` - Coverage for staging, polling, validation failures, and identifier-only dispatch.

## Decisions Made

- Treated `application/octet-stream` as acceptable only when it is still a valid zip payload, which keeps the endpoint tolerant to imperfect client content types without dropping zip validation.
- Kept the ingestion job worker handoff lightweight and synchronous-at-the-boundary for now, returning a dispatch envelope so later queue integration can plug in without changing the API contract.
- Used public study IDs in the durable schema because Phase 2 and later clinical flows will need stable external identifiers as well as job IDs.

## Deviations from Plan

None - plan executed exactly as written.

---

**Total deviations:** 0 auto-fixed
**Impact on plan:** None. The implementation stayed within the planned submit-and-poll scope.

## Issues Encountered

- The first draft left one broken test helper (`pytest.run`) in place; it was corrected to `asyncio.run` during execution and the full jobs suite now passes.

## User Setup Required

None - no external service configuration required for this plan.

## Next Phase Readiness

- The worker boundary now has a concrete extracted-study-root contract to feed into the MRI profiling and conversion pipeline.
- Phase 1 can proceed directly into profiling, validation, and NIfTI artifact generation without redefining the upload contract.

## Self-Check: PASSED

Verified `pytest -q tests/test_jobs_api.py` passes in `backend/`.

---
*Phase: 01-backend-foundation-mri-ingestion*
*Completed: 2026-04-11*
