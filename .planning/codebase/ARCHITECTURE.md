# Architecture

**Analysis Date:** 2024-04-08

## Pattern Overview

**Overall:** Dual-Stream Processing Architecture with Research/Production Separation

**Key Characteristics:**
- Frontend SPA with client-side state management and data fetching
- ML exploration notebooks isolated from production frontend
- Mock-driven development with clear backend integration points
- Component-based UI architecture with atomic design principles
- Async data flow using React Query with optimistic updates

## Layers

**Presentation Layer:**
- Purpose: User interface and client-side interactions
- Location: `frontend/src/`
- Contains: React components, pages, routing, UI state
- Depends on: API layer, shared utilities, type definitions
- Used by: End users (physicians viewing patient data)

**API Client Layer:**
- Purpose: HTTP communication and data transformation
- Location: `frontend/src/api/`
- Contains: Axios client configuration, endpoint functions, mock implementations
- Depends on: Type definitions, axios library
- Used by: React Query hooks in components and pages

**State Management Layer:**
- Purpose: Global application state (authentication, physician session)
- Location: `frontend/src/store/`
- Contains: Zustand stores with persistence middleware
- Depends on: Type definitions, zustand library
- Used by: Components and pages requiring auth state

**Type System Layer:**
- Purpose: Shared TypeScript contracts across application
- Location: `frontend/src/types/`
- Contains: Domain model interfaces (Patient, Scan, Summary, Physician)
- Depends on: Nothing (pure type definitions)
- Used by: All other layers

**Utility Layer:**
- Purpose: Shared helper functions and class name utilities
- Location: `frontend/src/lib/`
- Contains: Date formatting, volume calculations, Tailwind utilities
- Depends on: External utilities (clsx, tailwind-merge)
- Used by: Components and pages

**ML Research Layer:**
- Purpose: Exploratory data analysis and model evaluation (isolated)
- Location: `ml/exploration/`
- Contains: Jupyter notebooks, Python utilities, model outputs
- Depends on: PyTorch, SimpleITK, nnU-Net, HuggingFace transformers
- Used by: Data scientists and ML engineers (not integrated with frontend)

## Data Flow

**Authentication Flow:**

1. User submits credentials via `AuthPage` component
2. `useAuthStore.login()` validates (mock: any non-empty credentials)
3. Store persists physician object to sessionStorage
4. Router redirects to `/dashboard` via `ProtectedRoute` guard
5. All subsequent pages read `physician` from store

**State Management:**
- Zustand store with persist middleware handles auth state
- sessionStorage backend (cleared on browser close)
- No JWT tokens in current mock implementation

**Patient Dashboard Flow:**

1. `DashboardPage` mounts and triggers React Query `useQuery(['patients'])`
2. `getPatients()` API function simulates 400ms delay, returns mock data
3. Parallel query fetches scans for all patients: `useQuery(['all-scans', patientIds])`
4. Component filters patients client-side based on search input
5. `PatientTable` renders rows with skeleton loaders during fetch

**Patient Detail Flow:**

1. User navigates to `/patients/:id` 
2. `PatientDetailPage` triggers three parallel queries:
   - `useQuery(['patient', id])` → `getPatient(id)`
   - `useQuery(['scans', id])` → `getScans(id)`
   - `useQuery(['summary', id])` → `getSummary(id)`
3. Scans sorted chronologically, volume deltas calculated
4. Latest scan triggers MRI workspace URL fetch: `getMriUrl(scanId)`
5. Charts and history render from computed data

**Error Handling:**
- React Query manages loading, error, and success states
- `ErrorBanner` component displays retry options
- `EmptyState` component handles zero-data scenarios

## Key Abstractions

**Patient Domain Model:**
- Purpose: Core entity representing a patient and their metadata
- Examples: `frontend/src/types/index.ts` (Patient interface)
- Pattern: Flat DTO with computed properties (`scanCount`, `lastScanDate`)

**Scan Domain Model:**
- Purpose: Represents a single imaging study with volumetric measurements
- Examples: `frontend/src/types/index.ts` (Scan interface)
- Pattern: Time-series data point with `volumeMm3`, `maxDiameterMm`, `isAnnotated`

**API Client Functions:**
- Purpose: Encapsulate backend communication with consistent interface
- Examples: `frontend/src/api/patients.ts`, `frontend/src/api/scans.ts`
- Pattern: Async functions returning typed promises, mock delay simulation

**Component Composition:**
- Purpose: Reusable UI elements with single responsibility
- Examples: `frontend/src/components/shared/StatBlock.tsx`, `frontend/src/components/scan/VolumeChart.tsx`
- Pattern: Props-based configuration, Tailwind styling, TypeScript props interface

**Route Protection:**
- Purpose: Authorization guard for authenticated routes
- Examples: `frontend/src/router.tsx` (`ProtectedRoute`, `AuthGuard`)
- Pattern: Wrapper components checking auth store, conditional navigation

## Entry Points

**Frontend Application:**
- Location: `frontend/src/main.tsx`
- Triggers: Browser loads `frontend/index.html`
- Responsibilities: Initialize React app, wrap with QueryClientProvider and RouterProvider, render to DOM

**Router Configuration:**
- Location: `frontend/src/router.tsx`
- Triggers: RouterProvider in main.tsx
- Responsibilities: Define route hierarchy, apply auth guards, lazy-load pages

**ML Exploration Notebooks:**
- Location: `ml/exploration/notebooks/*.ipynb`
- Triggers: Manual execution via Jupyter Lab
- Responsibilities: Data exploration, preprocessing pipeline testing, model benchmarking

## Error Handling

**Strategy:** Declarative error boundaries with user-facing retry mechanisms

**Patterns:**
- React Query `isError` state triggers `ErrorBanner` component
- API functions throw on 404/500, caught by React Query
- Auth errors (`login()`) display inline validation messages
- Router handles invalid routes with redirects (not 404 pages in current implementation)

## Cross-Cutting Concerns

**Logging:** Client-side console logging (no structured logging framework)

**Validation:** 
- Form validation in `AuthPage` (non-empty credential check)
- Type safety via TypeScript interfaces
- No runtime schema validation (Zod/Yup not detected)

**Authentication:** 
- Mock implementation in `frontend/src/store/authStore.ts`
- Session persistence via Zustand middleware
- JWT interceptor stubbed in `frontend/src/api/client.ts` (commented out)
- Backend integration ready via `VITE_API_URL` environment variable

---

*Architecture analysis: 2024-04-08*
