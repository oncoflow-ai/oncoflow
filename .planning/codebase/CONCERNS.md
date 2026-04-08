# Codebase Concerns

**Analysis Date:** 2026-04-08

## Tech Debt

**Mock Data Architecture:**
- Issue: Entire application runs on hardcoded mock data without backend integration
- Files: `frontend/src/data/mockData.ts` (220 lines), `frontend/src/api/patients.ts`, `frontend/src/api/scans.ts`, `frontend/src/api/reports.ts`, `frontend/src/api/mri.ts`
- Impact: Cannot process real patient data, upload scans, or generate actual reports. All API calls are stubs with artificial delays. No persistence of any user actions.
- Fix approach: Implement FastAPI backend per `IMPLEMENTATION_PLAN.md` Phase 3-6. Replace mock functions with actual HTTP requests via axios client in `frontend/src/api/client.ts`.

**Disabled JWT Authentication:**
- Issue: JWT interceptor is commented out and authentication accepts any non-empty credentials
- Files: `frontend/src/api/client.ts` (lines 7-13), `frontend/src/store/authStore.ts` (lines 16-24)
- Impact: No actual authentication or authorization. Session storage persists physician object but has no backend validation. Any user can access all patient records.
- Fix approach: Implement RBAC backend (IMPLEMENTATION_PLAN.md Phase 2), uncomment JWT interceptor, integrate with real auth endpoint, implement proper session management with token refresh.

**MRI Viewer is Non-Functional Mockup:**
- Issue: MRI workspace renders static CSS gradient circles, not actual DICOM/NIfTI data
- Files: `frontend/src/components/shared/MriWorkspace.tsx` (lines 44-80), `frontend/src/api/mri.ts` (returns placeholder URL)
- Impact: Cannot view actual medical images, perform annotations, or validate segmentation results. All visual tumor overlays are hardcoded CSS shapes.
- Fix approach: Integrate NIfTI.js or similar DICOM viewer library. Implement actual slice navigation, windowing/leveling controls, and overlay rendering from backend segmentation masks. Requires S3 presigned URL implementation and CORS configuration.

**Missing Backend Infrastructure:**
- Issue: No backend, database, ML pipeline, or cloud infrastructure exists
- Files: Referenced in `IMPLEMENTATION_PLAN.md` but not implemented. No `backend/`, `ml/`, or `infra/` directories.
- Impact: Application is purely frontend prototype. Cannot upload scans, process images, run ML models, store data, or generate reports.
- Fix approach: Follow IMPLEMENTATION_PLAN.md phases 1-6 sequentially. Requires significant backend development effort (~3000+ lines of Python code, Docker orchestration, cloud infrastructure).

**Hardcoded Physician Identity:**
- Issue: All logins result in same physician (DR-001, Dr. D. Cohen)
- Files: `frontend/src/store/authStore.ts` (line 23)
- Impact: Cannot test multi-user workflows, RBAC, or patient assignment logic.
- Fix approach: Implement proper user management system with physician registration, credential verification, and role-based access control.

**No Configuration Management:**
- Issue: Single hardcoded environment variable with fallback
- Files: `frontend/src/api/client.ts` (line 4), `frontend/.env.local` (existence noted, not read)
- Impact: Cannot configure different API endpoints for dev/staging/prod environments. No validation of required configuration.
- Fix approach: Implement proper .env validation using Zod or similar, environment-specific config files, configuration loader at app startup.

**Large Mock Data File:**
- Issue: 220-line mock data file with embedded clinical narratives
- Files: `frontend/src/data/mockData.ts`
- Impact: Difficult to maintain, search, and update. Clinical text mixed with structured data. Not realistic for production scale.
- Fix approach: Move to database-backed API responses. For continued mocking during development, extract to JSON fixtures or use MSW (Mock Service Worker) with realistic response generators.

**ML Utilities Without Integration:**
- Issue: Complete ML preprocessing pipeline exists but is disconnected from application
- Files: `ml/exploration/utils/dicom_utils.py` (314 lines), `ml/exploration/utils/metrics.py` (307 lines), `ml/exploration/utils/visualisation.py` (347 lines)
- Impact: Research code exists but no integration path. No way to invoke from backend, no API wrapper, no containerization.
- Fix approach: Wrap utilities in FastAPI services per IMPLEMENTATION_PLAN.md Phase 4. Containerize each ML model (nnU-Net, MedGemma-1.5, SAM3), implement ensemble service, integrate with Celery worker queue.

## Known Bugs

**Patient Registration Form Non-Functional:**
- Symptoms: "Request Access" registration form collects inputs but submit button does nothing
- Files: `frontend/src/pages/AuthPage.tsx` (lines 118-138)
- Trigger: Click "Request access from admin" link, fill form, click "Submit Request"
- Workaround: Use sign-in flow which accepts any credentials

**PDF Generation Buttons Non-Functional:**
- Symptoms: "Generate PDF Report" and "↓ PDF" buttons have no handlers
- Files: `frontend/src/pages/PatientDetailPage.tsx` (line 92), `frontend/src/components/shared/MriWorkspace.tsx` (line 116)
- Trigger: Click any PDF export button
- Workaround: None - PDF generation not implemented

**Missing Error Handling for Query Failures:**
- Symptoms: React Query errors in patient/scan queries are not displayed to user in most views
- Files: `frontend/src/pages/PatientDetailPage.tsx` (error only handled for patient query, not scans/summary)
- Trigger: Network failure or API error during scan/summary fetch
- Workaround: Refresh page

## Security Considerations

**Session Storage of Authentication:**
- Risk: Physician credentials and session persisted in sessionStorage are vulnerable to XSS
- Files: `frontend/src/store/authStore.ts` (line 32 - sessionStorage)
- Current mitigation: None - relies on browser same-origin policy only
- Recommendations: Use httpOnly cookies for token storage when backend is implemented. Implement CSRF protection. Add Content Security Policy headers. Consider secure, httpOnly cookie + CSRF token pattern.

**No Input Validation:**
- Risk: All form inputs accepted without validation or sanitization
- Files: `frontend/src/pages/AuthPage.tsx` (accepts any non-empty string), patient/scan forms not implemented yet
- Current mitigation: Mock phase - inputs don't reach backend
- Recommendations: Implement Zod schemas for all form inputs. Add server-side validation when backend exists. Sanitize all user inputs before display to prevent XSS.

**Hardcoded API URLs:**
- Risk: Default localhost:8000 endpoint could leak in production build
- Files: `frontend/src/api/client.ts` (line 4)
- Current mitigation: Environment variable override available
- Recommendations: Fail-fast if VITE_API_URL not set in production builds. Add build-time validation. Use different base URLs per environment with no fallback.

**Missing RBAC Frontend Checks:**
- Risk: No UI-level authorization checks for physician vs admin actions
- Files: All pages and components assume single role
- Current mitigation: Backend doesn't exist yet
- Recommendations: Implement role-based UI rendering. Hide/disable admin actions for non-admin users. Add authorization checks to all sensitive components.

**Exposed Patient Data in Mock Files:**
- Risk: Realistic patient names, diagnoses, and medical histories committed to repository
- Files: `frontend/src/data/mockData.ts` (lines 3-220)
- Current mitigation: Data appears synthetic but uses realistic medical terminology
- Recommendations: Replace with clearly synthetic data (e.g., "Patient A", "John Doe"). Add disclaimer in file header. Never commit real patient data. Implement data anonymization pipeline for development datasets.

## Performance Bottlenecks

**Large Mock Data Loaded Synchronously:**
- Problem: 220-line mockData module imported and parsed on every page load
- Files: `frontend/src/data/mockData.ts`, imported by `frontend/src/api/patients.ts`, `frontend/src/api/scans.ts`, `frontend/src/api/reports.ts`
- Cause: All mock data loaded into memory at module resolution time
- Improvement path: Lazy-load mock data only when needed. Use dynamic imports. Consider IndexedDB for larger mock datasets. This will be resolved when backend API replaces mocks.

**React Query Cache Not Configured:**
- Problem: No custom cache time or stale-while-revalidate settings for patient/scan queries
- Files: `frontend/src/pages/PatientDetailPage.tsx` (lines 22-38), `frontend/src/pages/DashboardPage.tsx`
- Cause: Using default React Query settings (5min stale time)
- Improvement path: Configure query-specific cache times based on data volatility. Patient demographics: long cache. Scan results: shorter cache. Implement optimistic updates for mutations.

**No Code Splitting:**
- Problem: All routes bundled into single JavaScript chunk
- Files: `frontend/src/router.tsx` (static imports of all pages)
- Cause: React Router configured with static imports instead of React.lazy
- Improvement path: Implement route-based code splitting with React.lazy and Suspense. Split heavy dependencies (Recharts, Victory charts) into separate chunks. Measure bundle size with vite-bundle-visualizer.

**Recharts Bundle Size:**
- Problem: Recharts library is heavy (~120KB) and loaded upfront
- Files: `frontend/package.json` (recharts dependency), `frontend/src/components/scan/VolumeChart.tsx` (uses Recharts)
- Cause: Full Recharts library imported for single line chart
- Improvement path: Lazy-load chart component. Consider lighter alternatives (Chart.js, uPlot). Tree-shake unused Recharts components.

## Fragile Areas

**Patient ID Handling:**
- Files: `frontend/src/pages/PatientDetailPage.tsx` (lines 19-26), `frontend/src/api/patients.ts`, `frontend/src/api/scans.ts`
- Why fragile: Patient IDs used as object keys in mock data structures. No validation of ID format. Assumes ! assertion is safe.
- Safe modification: Always validate patient ID format before queries. Add type guard for patient ID. Handle undefined/null cases explicitly. When backend exists, return proper 404 responses.
- Test coverage: Basic test exists in `frontend/src/test/api.test.ts` but doesn't cover all edge cases (empty string, special characters, SQL injection patterns)

**Mock Delay Timing:**
- Files: `frontend/src/api/patients.ts` (delay 400/300ms), `frontend/src/api/scans.ts` (delay 350/200ms), `frontend/src/api/reports.ts` (delay 600ms), `frontend/src/api/mri.ts` (delay 200ms)
- Why fragile: Hardcoded delays can cause tests to be flaky or slow. Inconsistent across different API calls.
- Safe modification: Extract delay duration to configuration constant. Make configurable via environment variable. Mock time in tests using vi.useFakeTimers().
- Test coverage: Tests don't validate timing behavior. Could break if delays become async state dependencies.

**Scan Sorting Logic:**
- Files: `frontend/src/pages/PatientDetailPage.tsx` (lines 41-44)
- Why fragile: Creates new sorted array using useMemo but depends on scan dates being valid ISO strings. No error handling for malformed dates.
- Safe modification: Add date validation before sort. Handle timezone edge cases. Consider backend-sorted responses. Add null/undefined guards.
- Test coverage: No test coverage for date sorting edge cases.

**Volume Delta Calculations:**
- Files: `frontend/src/pages/PatientDetailPage.tsx` (lines 48-54), `frontend/src/lib/utils.ts` (calcVolumeDeltaPct function)
- Why fragile: Assumes at least 2 scans exist. No handling of division by zero or negative volumes. Percentage calculation could overflow for large deltas.
- Safe modification: Add null checks. Validate volume values are positive numbers. Cap percentage display at reasonable bounds. Handle edge case of zero baseline volume.
- Test coverage: Utility function tests exist in `frontend/src/test/utils.test.ts` but unclear if they cover edge cases.

**React Router Param Extraction:**
- Files: `frontend/src/pages/PatientDetailPage.tsx` (line 19)
- Why fragile: useParams returns optional string, but code enables queries assuming ID is defined
- Safe modification: Add explicit ID validation after extraction. Redirect to 404 or dashboard if undefined. Use type guard before query enablement.
- Test coverage: No test coverage for missing ID parameter scenario.

**Zustand Store Persistence:**
- Files: `frontend/src/store/authStore.ts` (lines 30-34)
- Why fragile: Session storage can throw exceptions if full or blocked. No error handling for storage failures. No migration strategy if state shape changes.
- Safe modification: Wrap persist middleware in try/catch. Add version field to persisted state. Implement migration function for schema changes. Handle quota exceeded errors gracefully.
- Test coverage: Basic auth store tests exist in `frontend/src/test/authStore.test.ts` but don't test persistence layer failures.

## Scaling Limits

**Mock Data Structure:**
- Current capacity: 8 patients, 33 total scans across all patients
- Limit: Object lookup by patient ID is O(1), but scan filtering across all timepoints is O(n). Structure doesn't support pagination.
- Scaling path: When backend exists, implement cursor-based pagination for scan lists. Add filtering/search indexes in database. Consider separate endpoints for scan metadata vs full details.

**Frontend State Management:**
- Current capacity: All patient/scan data loaded into React Query cache
- Limit: For 1000+ patients with dozens of scans each, query cache could exceed memory limits. No cache eviction strategy defined.
- Scaling path: Implement virtual scrolling for patient table. Load scans on-demand per patient. Configure React Query cache size limits and eviction policy. Consider server-side filtering/sorting.

**MRI Volume Rendering:**
- Current capacity: Mock CSS gradients only
- Limit: Real DICOM volumes can be 100MB+ per scan. Browser memory limits ~2GB. Rendering full 3D volumes in browser is impractical.
- Scaling path: Implement server-side volume rendering with streaming. Use progressive loading (slice-by-slice). Consider WebGL-based rendering (VTK.js, AMI.js). Implement region-of-interest cropping. Use lower-resolution previews with full-resolution on-demand.

**TypeScript Compilation:**
- Current capacity: 28 source files, 325 test lines, builds in <5s
- Limit: No issue now, but project plan suggests 3000+ lines of backend code. Monorepo with backend TypeScript could slow compilation.
- Scaling path: Already using Vite with esbuild (fast). If adding backend: use separate tsconfig for backend. Implement incremental builds. Consider project references for monorepo.

## Dependencies at Risk

**React Router v6:**
- Risk: Active development but frequent breaking changes between minor versions
- Impact: Routing is core to navigation. Breaking changes require refactoring all route definitions.
- Migration plan: Pin to specific minor version. Monitor release notes carefully. React Router DOM stable but consider TanStack Router for type-safe alternative.

**Zustand v4:**
- Risk: Lightweight library but single maintainer
- Impact: Auth state management depends on it. Small API surface reduces risk.
- Migration plan: Easy to migrate to Redux Toolkit or Jotai if needed. State structure is simple (single physician object).

**@tanstack/react-query v5:**
- Risk: New major version (v5) released recently, ecosystem still catching up
- Impact: Core data fetching mechanism. Breaking changes could affect all API integrations.
- Migration plan: Well-maintained library with active community. Migration guides available. Consider staying on v4 LTS if stability critical.

**Recharts v2:**
- Risk: Infrequent updates, known performance issues with large datasets
- Impact: Volume chart rendering. Only used in one component.
- Migration plan: Easy to swap for Chart.js, Victory, or uPlot. Component is isolated. Consider switch if performance becomes issue.

**Axios v1.6:**
- Risk: Native fetch API now widely supported and may deprecate need for Axios
- Impact: HTTP client used throughout API layer. Migration would touch all API files.
- Migration plan: Modern alternative is TanStack Query + native fetch. Axios still widely used and maintained. Low urgency but consider native fetch for new features.

**Testing Library v15:**
- Risk: Major version bump (v14 → v15) changed some APIs
- Impact: Test suite uses @testing-library/react for component tests
- Migration plan: Well-documented migration guides. Breaking changes minimal. Keep updated to latest minor versions.

## Missing Critical Features

**Upload Functionality:**
- Problem: No way to upload DICOM files or create new scans
- Blocks: Core value proposition of the application. Cannot test end-to-end workflow.
- Priority: High - Required for Phase 3 of IMPLEMENTATION_PLAN.md

**ML Pipeline Integration:**
- Problem: nnU-Net, MedGemma-1.5, SAM3 models referenced but not integrated
- Blocks: Automated segmentation, volumetric analysis, treatment response prediction
- Priority: High - Core differentiator. Required for Phase 4 of IMPLEMENTATION_PLAN.md

**RAG Text Pipeline:**
- Problem: AI-generated summaries are hardcoded mock strings, no actual LLM integration
- Blocks: Clinical narrative generation, context-aware reporting
- Priority: Medium - Phase 5 feature, required for production but not MVP

**Real-time Collaboration:**
- Problem: No multi-user annotation or concurrent editing support
- Blocks: Radiologist + oncologist workflows where multiple physicians review same scan
- Priority: Low - Not in current implementation plan but mentioned in HLD

**Audit Logging:**
- Problem: No tracking of who viewed/edited patient records
- Blocks: HIPAA compliance, security forensics, clinical audit requirements
- Priority: High - Required for production deployment in medical setting

**DICOM Metadata Extraction:**
- Problem: No parsing of DICOM headers for patient demographics, study details, acquisition parameters
- Blocks: Automated patient record creation, study validation, modality-specific processing
- Priority: High - Required for Phase 3

**Encryption at Rest:**
- Problem: No encryption of stored patient data or medical images
- Blocks: HIPAA compliance, data protection regulations
- Priority: High - Required before any real patient data can be stored

## Test Coverage Gaps

**API Layer:**
- What's not tested: Error handling, network timeouts, retry logic, malformed responses
- Files: `frontend/src/test/api.test.ts` only tests happy paths
- Risk: API integration failures could crash application or show cryptic errors to users
- Priority: High

**Component Integration:**
- What's not tested: PatientDetailPage with empty scans, loading states, query error states
- Files: No integration tests for `frontend/src/pages/PatientDetailPage.tsx`
- Risk: Component crashes or renders incorrectly when data is missing
- Priority: Medium

**Router Guards:**
- What's not tested: ProtectedRoute behavior when auth changes, navigation after logout
- Files: `frontend/src/router.test.tsx` exists but minimal coverage
- Risk: Authentication bypass or redirect loops
- Priority: High

**MRI Workspace:**
- What's not tested: Tool selection, slice navigation (when implemented)
- Files: `frontend/src/test/MriWorkspace.test.tsx` likely minimal (file exists)
- Risk: Annotation tools could malfunction without detection
- Priority: Medium

**Mock Data Validity:**
- What's not tested: Date formats, volume calculations, patient ID uniqueness
- Files: `frontend/src/test/mockData.test.ts` exists but unknown coverage depth
- Risk: Mock data inconsistencies could cause runtime errors
- Priority: Low - mock data is temporary

**E2E User Flows:**
- What's not tested: Sign in → view patient → view scan → generate report flow
- Files: No E2E tests detected (no Playwright/Cypress config)
- Risk: Integration issues between components not caught until manual testing
- Priority: Medium - Important for regression testing

**ML Utilities:**
- What's not tested: No Python tests found for ML utility functions
- Files: `ml/exploration/utils/` contains no test files
- Risk: DICOM conversion, metrics calculation, volume rendering could have bugs
- Priority: High - Medical accuracy depends on these calculations

**Accessibility:**
- What's not tested: Screen reader support, keyboard navigation, ARIA labels
- Files: No accessibility tests detected
- Risk: Application unusable for physicians with disabilities, potential ADA compliance issue
- Priority: Medium - Important for inclusive design

---

*Concerns audit: 2026-04-08*
