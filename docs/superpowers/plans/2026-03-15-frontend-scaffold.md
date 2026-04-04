# OncoFlow Frontend Scaffold Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold the OncoFlow React SPA — Auth, Physician Dashboard, and Patient Detail pages — wired to mock data but architected for a real FastAPI backend.

**Architecture:** Vite + React 18 + TypeScript SPA living at `frontend/` in the monorepo root. All data flows through a typed `src/api/` layer (currently returning mock data); TanStack Query manages server state; Zustand holds auth session only. When the backend is ready, only the function bodies in `src/api/` change — nothing in components or stores.

**Tech Stack:** React 18 · TypeScript · Vite · Tailwind CSS v3 · shadcn/ui · Lucide React · Recharts · TanStack Query v5 · Zustand · Axios · react-router-dom v6 · Geist + Geist Mono fonts · Vitest + React Testing Library

> **Commits:** All individual commit steps have been removed. Make one final commit at the end covering all changes (see Task 23).

---

## File Map

| File | Responsibility |
|---|---|
| `frontend/src/api/client.ts` | Axios instance + JWT interceptor stub |
| `frontend/src/api/patients.ts` | `getPatients()`, `getPatient(id)` |
| `frontend/src/api/scans.ts` | `getScans(patientId)`, `getScan(id)` |
| `frontend/src/api/reports.ts` | `getSummary(patientId)` |
| `frontend/src/api/mri.ts` | `getMriUrl(scanId)` |
| `frontend/src/types/index.ts` | `Patient`, `Scan`, `Summary`, `MriUrl` TypeScript interfaces |
| `frontend/src/data/mockData.ts` | 8 mock patients + scans + AI narratives — consumed only by `src/api/` |
| `frontend/src/store/authStore.ts` | Zustand auth store — physician session, persisted to sessionStorage |
| `frontend/src/lib/utils.ts` | shadcn `cn()` helper + `formatVolume`, `formatDate` |
| `frontend/src/router.tsx` | Route definitions + `<ProtectedRoute>` guard |
| `frontend/src/components/layout/TopNav.tsx` | Shared top navigation bar (wordmark, search, avatar, CTA) |
| `frontend/src/components/shared/StatBlock.tsx` | Single stat card (label + value + delta tag) |
| `frontend/src/components/shared/DeltaTag.tsx` | Color-coded delta percentage/value badge |
| `frontend/src/components/shared/SkeletonRow.tsx` | Shimmer skeleton row for table loading state |
| `frontend/src/components/shared/ErrorBanner.tsx` | Inline error banner with retry button |
| `frontend/src/components/shared/EmptyState.tsx` | Centered empty state with icon + message |
| `frontend/src/components/shared/AIInsightsPanel.tsx` | AI narrative panel with teal left-border accent |
| `frontend/src/components/shared/MriWorkspace.tsx` | Dark MRI sidebar — cosmetic viewer + toolbar |
| `frontend/src/components/patient/PatientRow.tsx` | Single patient table row |
| `frontend/src/components/patient/PatientTable.tsx` | Full patient table with header + rows |
| `frontend/src/components/scan/VolumeChart.tsx` | Recharts line+area chart for tumor trajectory |
| `frontend/src/components/scan/ScanRow.tsx` | Single imaging history row |
| `frontend/src/components/scan/ImagingHistory.tsx` | List of ScanRow items |
| `frontend/src/pages/AuthPage.tsx` | Split-panel sign-in + register-request page |
| `frontend/src/pages/DashboardPage.tsx` | Patient list with search filter |
| `frontend/src/pages/PatientDetailPage.tsx` | Full patient detail — stats, chart, history, AI, MRI sidebar |
| `frontend/src/main.tsx` | App entry point |
| `frontend/tailwind.config.ts` | Theme tokens — colors, fonts, radius |
| `frontend/vite.config.ts` | Vite config with path alias `@/` |
| `frontend/.env.local` | `VITE_API_URL=http://localhost:8000` |
| `frontend/src/test/setup.ts` | Vitest + Testing Library global setup |

---

## Chunk 1: Project Setup & Theme

### Task 1: Bootstrap Vite project

**Files:**
- Create: `frontend/` (directory)

- [ ] **Step 1: Scaffold Vite + React + TypeScript**

From the repo root:
```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
```

- [ ] **Step 2: Verify dev server starts**

```bash
npm run dev
```
Expected: Vite dev server at `http://localhost:5173` with the default React page.

- [ ] **Step 3: Remove boilerplate**

Delete `src/App.css`, `src/assets/react.svg`, clear `src/App.tsx` to a minimal placeholder:
```tsx
export default function App() {
  return <div>OncoFlow</div>
}
```
Clear `src/index.css` to empty.

---

### Task 2: Install all dependencies

**Files:**
- Modify: `frontend/package.json`

- [ ] **Step 1: Install runtime dependencies**

```bash
cd frontend
npm install \
  react-router-dom \
  @tanstack/react-query \
  zustand \
  axios \
  recharts \
  lucide-react \
  class-variance-authority \
  clsx \
  tailwind-merge \
  @radix-ui/react-slot \
  @radix-ui/react-dialog \
  @radix-ui/react-dropdown-menu \
  @radix-ui/react-tooltip
```

- [ ] **Step 2: Install dev dependencies**

```bash
npm install -D \
  tailwindcss \
  postcss \
  autoprefixer \
  @tailwindcss/typography \
  vitest \
  @vitest/ui \
  jsdom \
  @testing-library/react \
  @testing-library/jest-dom \
  @testing-library/user-event
```

- [ ] **Step 3: Initialise Tailwind**

```bash
npx tailwindcss init -p
```
Expected: `tailwind.config.js` and `postcss.config.js` created.

- [ ] **Step 4: Rename config to TypeScript**

```bash
mv tailwind.config.js tailwind.config.ts
```

- [ ] **Step 5: Verify no install errors**

```bash
npm run dev
```
Expected: dev server still starts cleanly.


---

### Task 3: Configure Tailwind theme, fonts, and CSS variables

**Files:**
- Create: `frontend/tailwind.config.ts`
- Modify: `frontend/src/index.css`
- Modify: `frontend/index.html`

- [ ] **Step 1: Write Tailwind config**

Replace `frontend/tailwind.config.ts`:
```ts
import type { Config } from 'tailwindcss'

export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Geist', 'sans-serif'],
        mono: ['Geist Mono', 'monospace'],
      },
      colors: {
        bg:       '#0B0D12',
        surface:  '#12151F',
        surface2: '#191D2A',
        surface3: '#21263A',
        border:   '#252A3A',
        border2:  '#323850',
        text1:    '#EAE6DC',
        text2:    '#7A8499',
        text3:    '#4E566A',
        teal:     '#0DC5A0',
        'teal-dim':'#0A8A70',
        amber:    '#E8935A',
        danger:   '#E05252',
        positive: '#3DBE8C',
      },
      borderRadius: {
        DEFAULT: '2px',
        sm: '2px',
        md: '4px',
        lg: '6px',
      },
    },
  },
  plugins: [],
} satisfies Config
```

- [ ] **Step 2: Write global CSS**

Replace `frontend/src/index.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  body {
    @apply bg-bg text-text1 font-sans antialiased;
  }
  * {
    @apply border-border;
  }
}
```

- [ ] **Step 3: Add Geist fonts to index.html**

Add inside `<head>` in `frontend/index.html`:
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@400;700&display=swap" rel="stylesheet">
```

- [ ] **Step 4: Verify dark background renders**

```bash
npm run dev
```
Expected: browser shows dark `#0B0D12` background.


---

### Task 4: Configure Vite path alias and test setup

**Files:**
- Modify: `frontend/vite.config.ts`
- Create: `frontend/src/test/setup.ts`

- [ ] **Step 1: Add `@/` path alias to Vite config**

Replace `frontend/vite.config.ts`:
```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
  },
})
```

- [ ] **Step 2: Create test setup file**

Create `frontend/src/test/setup.ts`:
```ts
import '@testing-library/jest-dom'
```

- [ ] **Step 3: Add test script to package.json**

In `frontend/package.json`, add to `"scripts"`:
```json
"test": "vitest",
"test:ui": "vitest --ui"
```

- [ ] **Step 4: Add `@types/node` for path resolution**

```bash
cd frontend && npm install -D @types/node
```

- [ ] **Step 5: Verify build still passes**

```bash
npm run build
```
Expected: build completes without errors.


---

### Task 5: Create utility helpers and `.env.local`

**Files:**
- Create: `frontend/src/lib/utils.ts`
- Create: `frontend/.env.local`

- [ ] **Step 1: Write utils**

Create `frontend/src/lib/utils.ts`:
```ts
import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatVolume(mm3: number): string {
  return mm3.toLocaleString('en-US')
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  })
}
```

- [ ] **Step 2: Write test for utils**

Create `frontend/src/lib/utils.test.ts`:
```ts
import { describe, it, expect } from 'vitest'
import { formatVolume, formatDate, cn } from './utils'

describe('formatVolume', () => {
  it('formats large numbers with commas', () => {
    expect(formatVolume(12480)).toBe('12,480')
  })
})

describe('formatDate', () => {
  it('formats ISO date to dd Mon yyyy', () => {
    expect(formatDate('2026-03-08')).toBe('08 Mar 2026')
  })
})

describe('cn', () => {
  it('merges tailwind classes', () => {
    expect(cn('text-red-500', 'text-blue-500')).toBe('text-blue-500')
  })
})
```

- [ ] **Step 3: Run tests**

```bash
cd frontend && npm test -- --run
```
Expected: 3 tests pass.

- [ ] **Step 4: Create `.env.local`**

Create `frontend/.env.local`:
```
VITE_API_URL=http://localhost:8000
```

- [ ] **Step 5: Add `.env.local` to root `.gitignore`**

Open the repo-root `.gitignore` (not `frontend/.gitignore`) and add:
```
frontend/.env.local
```

---

## Chunk 2: Types, Mock Data & API Layer

### Task 6: Define TypeScript types

**Files:**
- Create: `frontend/src/types/index.ts`

- [ ] **Step 1: Write all types**

Create `frontend/src/types/index.ts`:
```ts
export type PatientStatus = 'active' | 'review'

export interface Patient {
  id: string
  name: string
  dob: string
  diagnosis: string
  diagnosisLocation: string
  assignedPhysicianId: string
  status: PatientStatus
  scanCount: number
  lastScanDate: string
}

export interface Scan {
  id: string
  patientId: string
  studyLabel: string
  date: string
  modality: string
  sequence: string
  plane: string
  sliceCount: number
  resolution: string
  volumeMm3: number
  maxDiameterMm: number
  isAnnotated: boolean
}

export interface Summary {
  patientId: string
  generatedAt: string
  model: string
  text: string
}

export interface MriUrl {
  url: string
  expiresAt: string
}

export interface Physician {
  id: string
  name: string
  initials: string
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd frontend && npx tsc --noEmit
```
Expected: no errors.


---

### Task 7: Write mock data

**Files:**
- Create: `frontend/src/data/mockData.ts`

- [ ] **Step 1: Write patients and scans mock data**

Create `frontend/src/data/mockData.ts`:
```ts
import type { Patient, Scan, Summary } from '@/types'

export const mockPatients: Patient[] = [
  {
    id: 'P-1029', name: 'Sarah Jenkins', dob: '1994-07-22',
    diagnosis: 'Osteosarcoma', diagnosisLocation: 'Distal Left Femur',
    assignedPhysicianId: 'DR-001', status: 'active', scanCount: 3, lastScanDate: '2026-03-08',
  },
  {
    id: 'P-1031', name: 'David Levi', dob: '1968-11-03',
    diagnosis: 'Glioblastoma', diagnosisLocation: 'Right Temporal Lobe',
    assignedPhysicianId: 'DR-001', status: 'review', scanCount: 5, lastScanDate: '2026-03-03',
  },
  {
    id: 'P-1044', name: 'Miriam Cohen', dob: '1979-04-15',
    diagnosis: 'Breast Carcinoma', diagnosisLocation: 'Stage III, Left',
    assignedPhysicianId: 'DR-001', status: 'active', scanCount: 2, lastScanDate: '2026-02-28',
  },
  {
    id: 'P-1051', name: 'Jonathan Weiss', dob: '1955-09-29',
    diagnosis: 'Non-Hodgkin Lymphoma', diagnosisLocation: 'Mediastinal',
    assignedPhysicianId: 'DR-001', status: 'active', scanCount: 4, lastScanDate: '2026-02-21',
  },
  {
    id: 'P-1062', name: 'Noa Shapiro', dob: '2001-02-07',
    diagnosis: 'Renal Cell Carcinoma', diagnosisLocation: 'Right Kidney',
    assignedPhysicianId: 'DR-001', status: 'review', scanCount: 1, lastScanDate: '2026-02-14',
  },
  {
    id: 'P-1073', name: 'Yosef Mizrahi', dob: '1962-06-18',
    diagnosis: 'Colorectal Adenocarcinoma', diagnosisLocation: 'Sigmoid',
    assignedPhysicianId: 'DR-001', status: 'active', scanCount: 6, lastScanDate: '2026-02-07',
  },
  {
    id: 'P-1081', name: 'Rachel Ben-David', dob: '1971-12-01',
    diagnosis: 'Pancreatic Ductal Adenocarcinoma', diagnosisLocation: 'Head of Pancreas',
    assignedPhysicianId: 'DR-001', status: 'review', scanCount: 2, lastScanDate: '2026-01-30',
  },
  {
    id: 'P-1094', name: 'Eitan Goldberg', dob: '1983-03-25',
    diagnosis: 'Lung Adenocarcinoma', diagnosisLocation: 'Left Lower Lobe',
    assignedPhysicianId: 'DR-001', status: 'active', scanCount: 3, lastScanDate: '2026-01-22',
  },
]

export const mockScans: Scan[] = [
  // Sarah Jenkins P-1029
  { id: 'SCN-0039', patientId: 'P-1029', studyLabel: 'MRI Study #1', date: '2025-01-22', modality: 'MRI', sequence: 'T1W', plane: 'AXIAL', sliceCount: 128, resolution: '1.2mm iso', volumeMm3: 18900, maxDiameterMm: 36.1, isAnnotated: true },
  { id: 'SCN-0040', patientId: 'P-1029', studyLabel: 'MRI Study #2', date: '2025-08-14', modality: 'MRI', sequence: 'T1W', plane: 'AXIAL', sliceCount: 128, resolution: '1.2mm iso', volumeMm3: 15230, maxDiameterMm: 31.6, isAnnotated: true },
  { id: 'SCN-0041', patientId: 'P-1029', studyLabel: 'MRI Study #3', date: '2026-03-08', modality: 'MRI', sequence: 'T1W', plane: 'AXIAL', sliceCount: 128, resolution: '1.2mm iso', volumeMm3: 12480, maxDiameterMm: 28.4, isAnnotated: true },
  // David Levi P-1031
  { id: 'SCN-0020', patientId: 'P-1031', studyLabel: 'MRI Study #1', date: '2024-06-10', modality: 'MRI', sequence: 'T1W+Gd', plane: 'AXIAL', sliceCount: 160, resolution: '1.0mm iso', volumeMm3: 18400, maxDiameterMm: 42.0, isAnnotated: true },
  { id: 'SCN-0021', patientId: 'P-1031', studyLabel: 'MRI Study #2', date: '2024-09-20', modality: 'MRI', sequence: 'T1W+Gd', plane: 'AXIAL', sliceCount: 160, resolution: '1.0mm iso', volumeMm3: 20100, maxDiameterMm: 44.5, isAnnotated: true },
  { id: 'SCN-0022', patientId: 'P-1031', studyLabel: 'MRI Study #3', date: '2025-01-05', modality: 'MRI', sequence: 'T1W+Gd', plane: 'AXIAL', sliceCount: 160, resolution: '1.0mm iso', volumeMm3: 21800, maxDiameterMm: 46.2, isAnnotated: true },
  { id: 'SCN-0023', patientId: 'P-1031', studyLabel: 'MRI Study #4', date: '2025-07-14', modality: 'MRI', sequence: 'T1W+Gd', plane: 'AXIAL', sliceCount: 160, resolution: '1.0mm iso', volumeMm3: 23500, maxDiameterMm: 48.8, isAnnotated: true },
  { id: 'SCN-0024', patientId: 'P-1031', studyLabel: 'MRI Study #5', date: '2026-03-03', modality: 'MRI', sequence: 'T1W+Gd', plane: 'AXIAL', sliceCount: 160, resolution: '1.0mm iso', volumeMm3: 24910, maxDiameterMm: 50.1, isAnnotated: false },
  // Miriam Cohen P-1044
  { id: 'SCN-0060', patientId: 'P-1044', studyLabel: 'MRI Study #1', date: '2025-09-10', modality: 'MRI', sequence: 'DCE', plane: 'AXIAL', sliceCount: 96, resolution: '1.5mm iso', volumeMm3: 10700, maxDiameterMm: 29.0, isAnnotated: true },
  { id: 'SCN-0061', patientId: 'P-1044', studyLabel: 'MRI Study #2', date: '2026-02-28', modality: 'MRI', sequence: 'DCE', plane: 'AXIAL', sliceCount: 96, resolution: '1.5mm iso', volumeMm3: 8340, maxDiameterMm: 25.2, isAnnotated: true },
  // Jonathan Weiss P-1051
  { id: 'SCN-0070', patientId: 'P-1051', studyLabel: 'MRI Study #1', date: '2025-03-01', modality: 'MRI', sequence: 'T2W', plane: 'CORONAL', sliceCount: 80, resolution: '2.0mm iso', volumeMm3: 29800, maxDiameterMm: 55.0, isAnnotated: true },
  { id: 'SCN-0071', patientId: 'P-1051', studyLabel: 'MRI Study #2', date: '2025-07-10', modality: 'MRI', sequence: 'T2W', plane: 'CORONAL', sliceCount: 80, resolution: '2.0mm iso', volumeMm3: 30400, maxDiameterMm: 55.9, isAnnotated: true },
  { id: 'SCN-0072', patientId: 'P-1051', studyLabel: 'MRI Study #3', date: '2025-11-22', modality: 'MRI', sequence: 'T2W', plane: 'CORONAL', sliceCount: 80, resolution: '2.0mm iso', volumeMm3: 31000, maxDiameterMm: 56.3, isAnnotated: true },
  { id: 'SCN-0073', patientId: 'P-1051', studyLabel: 'MRI Study #4', date: '2026-02-21', modality: 'MRI', sequence: 'T2W', plane: 'CORONAL', sliceCount: 80, resolution: '2.0mm iso', volumeMm3: 31200, maxDiameterMm: 56.6, isAnnotated: false },
  // Noa Shapiro P-1062 (baseline only)
  { id: 'SCN-0080', patientId: 'P-1062', studyLabel: 'MRI Study #1', date: '2026-02-14', modality: 'MRI', sequence: 'T1W', plane: 'AXIAL', sliceCount: 112, resolution: '1.2mm iso', volumeMm3: 19650, maxDiameterMm: 44.0, isAnnotated: false },
  // Yosef Mizrahi P-1073
  { id: 'SCN-0090', patientId: 'P-1073', studyLabel: 'MRI Study #1', date: '2024-02-01', modality: 'MRI', sequence: 'T2W', plane: 'AXIAL', sliceCount: 64, resolution: '2.5mm iso', volumeMm3: 10400, maxDiameterMm: 32.0, isAnnotated: true },
  { id: 'SCN-0091', patientId: 'P-1073', studyLabel: 'MRI Study #2', date: '2024-05-15', modality: 'MRI', sequence: 'T2W', plane: 'AXIAL', sliceCount: 64, resolution: '2.5mm iso', volumeMm3: 8900, maxDiameterMm: 29.5, isAnnotated: true },
  { id: 'SCN-0092', patientId: 'P-1073', studyLabel: 'MRI Study #3', date: '2024-09-10', modality: 'MRI', sequence: 'T2W', plane: 'AXIAL', sliceCount: 64, resolution: '2.5mm iso', volumeMm3: 7200, maxDiameterMm: 26.1, isAnnotated: true },
  { id: 'SCN-0093', patientId: 'P-1073', studyLabel: 'MRI Study #4', date: '2025-01-20', modality: 'MRI', sequence: 'T2W', plane: 'AXIAL', sliceCount: 64, resolution: '2.5mm iso', volumeMm3: 5800, maxDiameterMm: 23.4, isAnnotated: true },
  { id: 'SCN-0094', patientId: 'P-1073', studyLabel: 'MRI Study #5', date: '2025-08-30', modality: 'MRI', sequence: 'T2W', plane: 'AXIAL', sliceCount: 64, resolution: '2.5mm iso', volumeMm3: 6300, maxDiameterMm: 24.0, isAnnotated: true },
  { id: 'SCN-0095', patientId: 'P-1073', studyLabel: 'MRI Study #6', date: '2026-02-07', modality: 'MRI', sequence: 'T2W', plane: 'AXIAL', sliceCount: 64, resolution: '2.5mm iso', volumeMm3: 6100, maxDiameterMm: 23.7, isAnnotated: true },
  // Rachel Ben-David P-1081
  { id: 'SCN-0100', patientId: 'P-1081', studyLabel: 'MRI Study #1', date: '2025-10-05', modality: 'MRI', sequence: 'T1W+Gd', plane: 'AXIAL', sliceCount: 96, resolution: '1.5mm iso', volumeMm3: 22000, maxDiameterMm: 47.5, isAnnotated: true },
  { id: 'SCN-0101', patientId: 'P-1081', studyLabel: 'MRI Study #2', date: '2026-01-30', modality: 'MRI', sequence: 'T1W+Gd', plane: 'AXIAL', sliceCount: 96, resolution: '1.5mm iso', volumeMm3: 25400, maxDiameterMm: 51.2, isAnnotated: false },
  // Eitan Goldberg P-1094
  { id: 'SCN-0110', patientId: 'P-1094', studyLabel: 'MRI Study #1', date: '2025-06-14', modality: 'MRI', sequence: 'T1W', plane: 'AXIAL', sliceCount: 120, resolution: '1.2mm iso', volumeMm3: 14200, maxDiameterMm: 38.0, isAnnotated: true },
  { id: 'SCN-0111', patientId: 'P-1094', studyLabel: 'MRI Study #2', date: '2025-10-01', modality: 'MRI', sequence: 'T1W', plane: 'AXIAL', sliceCount: 120, resolution: '1.2mm iso', volumeMm3: 13800, maxDiameterMm: 37.2, isAnnotated: true },
  { id: 'SCN-0112', patientId: 'P-1094', studyLabel: 'MRI Study #3', date: '2026-01-22', modality: 'MRI', sequence: 'T1W', plane: 'AXIAL', sliceCount: 120, resolution: '1.2mm iso', volumeMm3: 13500, maxDiameterMm: 36.8, isAnnotated: true },
]

export const mockSummaries: Summary[] = [
  {
    patientId: 'P-1029', generatedAt: '2026-03-09T08:14:00Z', model: 'Gemini 1.5 (RAG-augmented)',
    text: 'Comparing the most recent study (08 Mar 2026) with the prior acquisition (14 Aug 2025), the distal left femoral lesion demonstrates an 18.2% reduction in total segmented volume (15,230 mm³ → 12,480 mm³) and a decrease in maximum axial diameter from 31.6 mm to 28.4 mm. Morphological margins appear increasingly well-defined on T1-weighted sequences, consistent with a <strong>partial response</strong> by RECIST 1.1 criteria. No new satellite lesions or skip metastases are identified within the visualized field. The longitudinal trajectory over 14 months shows a consistent downward trend, suggesting the current neo-adjuvant chemotherapy protocol is achieving meaningful cytoreduction ahead of the planned surgical resection.',
  },
  {
    patientId: 'P-1031', generatedAt: '2026-03-04T10:22:00Z', model: 'Gemini 1.5 (RAG-augmented)',
    text: 'The right temporal lesion continues to demonstrate progressive enlargement across all five acquired studies. The most recent acquisition (03 Mar 2026) shows a 6.4% increase in volume compared to the prior study (14 Jul 2025), with maximum diameter now 50.1 mm. Enhancement pattern on T1+Gd sequences remains heterogeneous with areas of central necrosis. <strong>Progressive disease</strong> by RANO criteria. Urgent multidisciplinary team review is recommended to reassess the current treatment regimen.',
  },
  {
    patientId: 'P-1044', generatedAt: '2026-03-01T09:05:00Z', model: 'Gemini 1.5 (RAG-augmented)',
    text: 'The left breast lesion has decreased from 10,700 mm³ to 8,340 mm³ (22.1% reduction) over the five-month interval between studies. Maximum diameter decreased from 29.0 mm to 25.2 mm. DCE-MRI kinetic analysis shows a reduction in early-phase enhancement, consistent with a <strong>good partial response</strong> to the current neoadjuvant chemotherapy. Axillary lymph node involvement appears stable. Surgical planning may be considered at the next multidisciplinary review.',
  },
  {
    patientId: 'P-1051', generatedAt: '2026-02-22T11:30:00Z', model: 'Gemini 1.5 (RAG-augmented)',
    text: 'The mediastinal lymphomatous mass has shown minimal interval change across four studies spanning 12 months. Volume has increased marginally from 29,800 mm³ to 31,200 mm³ (+4.7% overall), within the range of measurement variability on T2-weighted coronal sequences. No new mediastinal or hilar adenopathy identified. Current findings are consistent with <strong>stable disease</strong>. Continued monitoring with the current regimen is appropriate; re-staging PET-CT may provide additional metabolic activity data.',
  },
  {
    patientId: 'P-1062', generatedAt: '2026-02-15T14:00:00Z', model: 'Gemini 1.5 (RAG-augmented)',
    text: 'This is the first acquired study for this patient, establishing a <strong>baseline volumetric measurement</strong> of 19,650 mm³ with maximum diameter of 44.0 mm for the right renal lesion. T1-weighted axial imaging shows a heterogeneous mass with areas of internal complexity. A follow-up MRI in 3 months is recommended to assess treatment response following initiation of the planned targeted therapy. No distant metastatic lesions identified on the imaged field.',
  },
  {
    patientId: 'P-1073', generatedAt: '2026-02-08T08:50:00Z', model: 'Gemini 1.5 (RAG-augmented)',
    text: 'Across six studies over 24 months, the sigmoid lesion has demonstrated a sustained and clinically significant volumetric response, declining from 10,400 mm³ to 6,100 mm³ — a <strong>41.3% total reduction</strong>. A transient increase was noted between studies 5 and 6 (Aug 2025: 6,300 mm³ → Feb 2026: 6,100 mm³), but this is within measurement variance and does not represent a change in overall trajectory. Maximum diameter has reduced from 32.0 mm to 23.7 mm. Continued response to current chemotherapy protocol is confirmed.',
  },
  {
    patientId: 'P-1081', generatedAt: '2026-01-31T15:20:00Z', model: 'Gemini 1.5 (RAG-augmented)',
    text: 'Interval comparison between the two available studies (Oct 2025 → Jan 2026) reveals a 15.5% increase in tumor volume (22,000 mm³ → 25,400 mm³) and progression in maximum diameter from 47.5 mm to 51.2 mm. T1+Gd sequences demonstrate increased peri-tumoral enhancement. These findings are consistent with <strong>progressive disease</strong>. The pancreatic duct remains obstructed. Palliative care consultation and reassessment of systemic therapy options are recommended at the next multidisciplinary oncology board.',
  },
  {
    patientId: 'P-1094', generatedAt: '2026-01-23T10:10:00Z', model: 'Gemini 1.5 (RAG-augmented)',
    text: 'Longitudinal assessment across three studies over seven months demonstrates a gradual reduction in the left lower lobe lesion: 14,200 mm³ → 13,800 mm³ → 13,500 mm³, representing a cumulative 4.9% decrease. While the absolute change is modest, the consistent downward trajectory is encouraging. Maximum diameter has decreased from 38.0 mm to 36.8 mm. Findings are consistent with <strong>minimal response / stable disease</strong> by RECIST criteria. Continuation of current targeted therapy with reassessment at the next scheduled imaging is recommended.',
  },
]
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd frontend && npx tsc --noEmit
```
Expected: no errors.


---

### Task 8: Build the API layer

**Files:**
- Create: `frontend/src/api/client.ts`
- Create: `frontend/src/api/patients.ts`
- Create: `frontend/src/api/scans.ts`
- Create: `frontend/src/api/reports.ts`
- Create: `frontend/src/api/mri.ts`

- [ ] **Step 1: Write Axios client**

Create `frontend/src/api/client.ts`:
```ts
import axios from 'axios'

export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? 'http://localhost:8000',
})

// JWT interceptor stub — no-op in mock phase.
// When backend is ready, replace with:
// import { useAuthStore } from '@/store/authStore'
// apiClient.interceptors.request.use(cfg => {
//   const token = useAuthStore.getState().token
//   if (token) cfg.headers.Authorization = `Bearer ${token}`
//   return cfg
// })
```

- [ ] **Step 2: Write patients API**

Create `frontend/src/api/patients.ts`:
```ts
import type { Patient } from '@/types'
import { mockPatients } from '@/data/mockData'

export async function getPatients(): Promise<Patient[]> {
  // Mock: return all patients assigned to the current physician.
  // Backend: return apiClient.get<Patient[]>('/api/patients').then(r => r.data)
  return mockPatients
}

export async function getPatient(id: string): Promise<Patient> {
  const patient = mockPatients.find(p => p.id === id)
  if (!patient) throw new Error(`Patient ${id} not found`)
  return patient
  // Backend: return apiClient.get<Patient>(`/api/patients/${id}`).then(r => r.data)
}
```

- [ ] **Step 3: Write scans API**

Create `frontend/src/api/scans.ts`:
```ts
import type { Scan } from '@/types'
import { mockScans } from '@/data/mockData'

export async function getScans(patientId: string): Promise<Scan[]> {
  return mockScans
    .filter(s => s.patientId === patientId)
    .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
  // Backend: return apiClient.get<Scan[]>(`/api/patients/${patientId}/scans`).then(r => r.data)
}

export async function getScan(id: string): Promise<Scan> {
  const scan = mockScans.find(s => s.id === id)
  if (!scan) throw new Error(`Scan ${id} not found`)
  return scan
  // Backend: return apiClient.get<Scan>(`/api/scans/${id}`).then(r => r.data)
}
```

- [ ] **Step 4: Write reports API**

Create `frontend/src/api/reports.ts`:
```ts
import type { Summary } from '@/types'
import { mockSummaries } from '@/data/mockData'

export async function getSummary(patientId: string): Promise<Summary> {
  const summary = mockSummaries.find(s => s.patientId === patientId)
  if (!summary) throw new Error(`Summary for patient ${patientId} not found`)
  return summary
  // Backend: return apiClient.get<Summary>(`/api/patients/${patientId}/summary`).then(r => r.data)
}
```

- [ ] **Step 5: Write MRI URL API**

Create `frontend/src/api/mri.ts`:
```ts
import type { MriUrl } from '@/types'

export async function getMriUrl(_scanId: string): Promise<MriUrl> {
  // Mock: return a placeholder — no real viewer in this phase.
  return { url: '', expiresAt: '' }
  // Backend: return apiClient.get<MriUrl>(`/api/scans/${_scanId}/mri-url`).then(r => r.data)
}
```

- [ ] **Step 6: Verify TypeScript compiles**

```bash
cd frontend && npx tsc --noEmit
```
Expected: no errors.


---

### Task 9: Auth store

**Files:**
- Create: `frontend/src/store/authStore.ts`

- [ ] **Step 1: Write Zustand auth store**

Create `frontend/src/store/authStore.ts`:
```ts
import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import type { Physician } from '@/types'

interface AuthState {
  physician: Physician | null
  isAuthenticated: boolean
  login: (id: string, password: string) => Promise<void>
  logout: () => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      physician: null,
      isAuthenticated: false,
      login: async (id: string, password: string) => {
        if (!id.trim() || !password.trim()) {
          throw new Error('Physician ID and password are required')
        }
        // Mock: accept any non-empty credentials.
        // Backend: const { data } = await apiClient.post('/api/auth/login', { physicianId: id, password })
        //          set({ physician: data.physician, isAuthenticated: true })
        set({
          physician: { id: 'DR-001', name: 'Dr. D. Cohen', initials: 'DC' },
          isAuthenticated: true,
        })
      },
      logout: () => set({ physician: null, isAuthenticated: false }),
    }),
    {
      name: 'oncoflow-auth',
      storage: createJSONStorage(() => sessionStorage),
    }
  )
)
```

- [ ] **Step 2: Write auth store tests**

Create `frontend/src/store/authStore.test.ts`:
```ts
import { describe, it, expect, beforeEach } from 'vitest'
import { useAuthStore } from './authStore'

beforeEach(() => {
  useAuthStore.setState({ physician: null, isAuthenticated: false })
})

describe('authStore', () => {
  it('starts unauthenticated', () => {
    expect(useAuthStore.getState().isAuthenticated).toBe(false)
    expect(useAuthStore.getState().physician).toBeNull()
  })

  it('sets physician on login with valid credentials', async () => {
    await useAuthStore.getState().login('DR-001', 'password123')
    expect(useAuthStore.getState().isAuthenticated).toBe(true)
    expect(useAuthStore.getState().physician?.initials).toBe('DC')
  })

  it('throws on empty credentials', async () => {
    await expect(useAuthStore.getState().login('', '')).rejects.toThrow('required')
  })

  it('clears state on logout', async () => {
    await useAuthStore.getState().login('DR-001', 'password123')
    useAuthStore.getState().logout()
    expect(useAuthStore.getState().isAuthenticated).toBe(false)
    expect(useAuthStore.getState().physician).toBeNull()
  })
})
```

- [ ] **Step 3: Run tests**

```bash
cd frontend && npm test -- --run
```
Expected: 4 auth store tests pass + 3 utils tests = 7 total.


---

## Chunk 3: Routing & Shell

### Task 10: Router with ProtectedRoute

**Files:**
- Create: `frontend/src/router.tsx`
- Modify: `frontend/src/main.tsx`

- [ ] **Step 1: Write router**

Create `frontend/src/router.tsx`:
```tsx
import { createBrowserRouter, Navigate, Outlet } from 'react-router-dom'
import { useAuthStore } from '@/store/authStore'
import AuthPage from '@/pages/AuthPage'
import DashboardPage from '@/pages/DashboardPage'
import PatientDetailPage from '@/pages/PatientDetailPage'

function ProtectedRoute() {
  const isAuthenticated = useAuthStore(s => s.isAuthenticated)
  return isAuthenticated ? <Outlet /> : <Navigate to="/auth" replace />
}

function AuthGuard() {
  const isAuthenticated = useAuthStore(s => s.isAuthenticated)
  return isAuthenticated ? <Navigate to="/dashboard" replace /> : <AuthPage />
}

export const router = createBrowserRouter([
  { path: '/', element: <Navigate to="/dashboard" replace /> },
  { path: '/auth', element: <AuthGuard /> },
  {
    element: <ProtectedRoute />,
    children: [
      { path: '/dashboard', element: <DashboardPage /> },
      { path: '/patients/:id', element: <PatientDetailPage /> },
    ],
  },
])
```

- [ ] **Step 2: Create page stubs** (needed for router to compile)

Create `frontend/src/pages/AuthPage.tsx`:
```tsx
export default function AuthPage() {
  return <div className="text-text1 p-8">Auth Page</div>
}
```

Create `frontend/src/pages/DashboardPage.tsx`:
```tsx
export default function DashboardPage() {
  return <div className="text-text1 p-8">Dashboard</div>
}
```

Create `frontend/src/pages/PatientDetailPage.tsx`:
```tsx
export default function PatientDetailPage() {
  return <div className="text-text1 p-8">Patient Detail</div>
}
```

- [ ] **Step 3: Wire router in main.tsx**

Replace `frontend/src/main.tsx`:
```tsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import { RouterProvider } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { router } from './router'
import './index.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, staleTime: 1000 * 60 * 5 },
  },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </React.StrictMode>
)
```

- [ ] **Step 4: Write route guard test**

Create `frontend/src/router.test.tsx`:
```tsx
import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter, Routes, Route } from 'react-router-dom'
import { useAuthStore } from '@/store/authStore'

function Protected() {
  const isAuthenticated = useAuthStore(s => s.isAuthenticated)
  if (!isAuthenticated) return <div>Redirected to auth</div>
  return <div>Protected content</div>
}

beforeEach(() => {
  useAuthStore.setState({ physician: null, isAuthenticated: false })
})

describe('route protection', () => {
  it('shows redirect message when not authenticated', () => {
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <Routes>
          <Route path="/dashboard" element={<Protected />} />
        </Routes>
      </MemoryRouter>
    )
    expect(screen.getByText('Redirected to auth')).toBeInTheDocument()
  })

  it('shows content when authenticated', async () => {
    await useAuthStore.getState().login('DR-001', 'pw')
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <Routes>
          <Route path="/dashboard" element={<Protected />} />
        </Routes>
      </MemoryRouter>
    )
    expect(screen.getByText('Protected content')).toBeInTheDocument()
  })
})
```

- [ ] **Step 5: Run tests**

```bash
cd frontend && npm test -- --run
```
Expected: all previous tests + 2 new route tests pass.

- [ ] **Step 6: Verify app loads in browser**

```bash
npm run dev
```
Navigate to `http://localhost:5173` — expected: redirects to `/auth` and shows "Auth Page".


---

### Task 11: TopNav component

**Files:**
- Create: `frontend/src/components/layout/TopNav.tsx`

- [ ] **Step 1: Write TopNav**

Create `frontend/src/components/layout/TopNav.tsx`:
```tsx
import { LogOut } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '@/store/authStore'
import { cn } from '@/lib/utils'

interface TopNavProps {
  searchValue?: string
  onSearchChange?: (value: string) => void
  showSearch?: boolean
  cta?: React.ReactNode
}

export default function TopNav({ searchValue, onSearchChange, showSearch = false, cta }: TopNavProps) {
  const { physician, logout } = useAuthStore()
  const navigate = useNavigate()

  function handleLogout() {
    logout()
    navigate('/auth')
  }

  return (
    <header className="h-[52px] bg-bg border-b border-border flex items-center gap-5 px-5 shrink-0">
      <span className="font-sans font-semibold text-[17px] text-text1 tracking-tight whitespace-nowrap flex items-center gap-2">
        <span className="w-[7px] h-[7px] rounded-full bg-teal shadow-[0_0_8px_#0DC5A0]" />
        OncoFlow
      </span>

      {showSearch && (
        <div className="flex-1 max-w-[360px] bg-surface border border-border2 h-[34px] flex items-center px-3 gap-2">
          <svg className="w-3.5 h-3.5 text-text3 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            value={searchValue ?? ''}
            onChange={e => onSearchChange?.(e.target.value)}
            placeholder="Search by patient name or ID..."
            className="bg-transparent border-none outline-none text-text1 text-[13px] placeholder-text3 font-sans w-full"
          />
        </div>
      )}

      <div className="ml-auto flex items-center gap-2.5">
        {cta}
        {physician && (
          <div className="w-[30px] h-[30px] bg-surface2 border border-border2 flex items-center justify-center font-mono text-[10px] text-text2">
            {physician.initials}
          </div>
        )}
        <button
          onClick={handleLogout}
          className="w-[30px] h-[30px] bg-surface2 border border-border2 flex items-center justify-center text-text3 hover:text-danger transition-colors"
          title="Sign out"
        >
          <LogOut size={13} />
        </button>
      </div>
    </header>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd frontend && npx tsc --noEmit
```
Expected: no errors.


---

## Chunk 4: Auth Page

### Task 12: Build AuthPage

**Files:**
- Modify: `frontend/src/pages/AuthPage.tsx`

- [ ] **Step 1: Write the full AuthPage**

Replace `frontend/src/pages/AuthPage.tsx`:
```tsx
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '@/store/authStore'

type Mode = 'signin' | 'register'

export default function AuthPage() {
  const [mode, setMode] = useState<Mode>('signin')
  const [physicianId, setPhysicianId] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const login = useAuthStore(s => s.login)
  const navigate = useNavigate()

  async function handleSignIn(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await login(physicianId, password)
      navigate('/dashboard')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Sign in failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-bg flex items-stretch">
      {/* Left brand panel */}
      <div className="hidden lg:flex flex-col justify-between flex-1 bg-surface border-r border-border p-14 relative overflow-hidden">
        <div className="absolute -top-10 -left-10 w-72 h-72 rounded-full bg-teal/10 blur-[80px] pointer-events-none" />
        <div>
          <div className="text-[28px] font-sans font-bold text-text1 tracking-tight flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-teal shadow-[0_0_8px_#0DC5A0]" />
            OncoFlow
          </div>
          <div className="text-[11px] font-mono text-text3 tracking-widest uppercase mt-1.5 ml-4">
            Longitudinal Tumor Intelligence
          </div>
        </div>

        <div>
          <h1 className="text-[44px] font-sans font-bold text-text1 leading-[1.15] mb-5">
            Precision<br />tracking for<br />
            <span className="italic text-teal">every scan.</span>
          </h1>
          <p className="text-[14px] text-text2 leading-relaxed max-w-sm">
            Automated tumor segmentation, volumetric comparison, and AI-generated clinical narratives — built for oncologists and radiologists.
          </p>
        </div>

        <div className="flex gap-8">
          {[
            { num: '98.4%', label: 'Seg. accuracy' },
            { num: '312', label: 'Reports generated' },
            { num: '3', label: 'AI models (ensemble)' },
          ].map(stat => (
            <div key={stat.label}>
              <div className="font-mono text-[26px] font-bold text-text1">{stat.num}</div>
              <div className="text-[11px] font-mono text-text3 uppercase tracking-widest mt-0.5">{stat.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Right form panel */}
      <div className="w-full lg:w-[380px] flex flex-col justify-center px-10 py-16 bg-bg">
        {mode === 'signin' ? (
          <form onSubmit={handleSignIn} className="space-y-4">
            <div className="text-[12px] font-mono font-bold tracking-widest uppercase text-text2 mb-8">
              Clinical Sign In
            </div>
            <div>
              <label className="block text-[11px] font-mono font-bold tracking-widest uppercase text-text3 mb-2">
                Physician ID / Email
              </label>
              <input
                value={physicianId}
                onChange={e => setPhysicianId(e.target.value)}
                placeholder="dr.cohen@ichilov.gov.il"
                className="w-full bg-surface border border-border2 text-text1 px-3.5 py-[10px] text-[14px] font-sans placeholder-text3 focus:outline-none focus:border-teal transition-colors"
              />
            </div>
            <div>
              <label className="block text-[11px] font-mono font-bold tracking-widest uppercase text-text3 mb-2">
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full bg-surface border border-border2 text-text1 px-3.5 py-[10px] text-[14px] font-sans placeholder-text3 focus:outline-none focus:border-teal transition-colors"
              />
            </div>
            {error && (
              <p className="text-danger text-[12px] font-mono">{error}</p>
            )}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-teal text-black font-mono font-bold text-[13px] tracking-widest uppercase py-3 mt-2 hover:bg-teal/90 transition-colors disabled:opacity-50"
            >
              {loading ? 'Signing in…' : 'Access Patient Records →'}
            </button>
            <hr className="border-border my-6" />
            <p className="text-[12px] text-text2 text-center font-sans">
              New clinician?{' '}
              <button type="button" onClick={() => setMode('register')} className="text-teal font-semibold">
                Request access from admin →
              </button>
            </p>
          </form>
        ) : (
          <div>
            <div className="text-[12px] font-mono font-bold tracking-widest uppercase text-text2 mb-8">
              Request Access
            </div>
            <p className="text-[13px] text-text2 mb-6 leading-relaxed">
              Submit your details and a medical administrator will grant access within 24 hours.
            </p>
            {['Full Name', 'Hospital Email', 'Hospital ID', 'Role'].map(field => (
              <div key={field} className="mb-4">
                <label className="block text-[11px] font-mono font-bold tracking-widest uppercase text-text3 mb-2">
                  {field}
                </label>
                <input
                  placeholder=""
                  className="w-full bg-surface border border-border2 text-text1 px-3.5 py-[10px] text-[14px] font-sans focus:outline-none focus:border-teal transition-colors"
                />
              </div>
            ))}
            <button className="w-full bg-surface border border-teal text-teal font-mono font-bold text-[13px] tracking-widest uppercase py-3 hover:bg-teal/5 transition-colors">
              Submit Request
            </button>
            <hr className="border-border my-6" />
            <p className="text-[12px] text-text2 text-center font-sans">
              Already have access?{' '}
              <button onClick={() => setMode('signin')} className="text-teal font-semibold">
                Sign in →
              </button>
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Verify app renders AuthPage**

```bash
cd frontend && npm run dev
```
Navigate to `http://localhost:5173/auth`. Expected: split dark auth page with brand panel and sign-in form.

- [ ] **Step 3: Test sign-in flow manually**

Enter any non-empty credentials and click "Access Patient Records". Expected: redirects to `/dashboard` (stub page).

- [ ] **Step 4: Test toggle manually**

Click "Request access from admin". Expected: form switches to registration fields.


---

## Chunk 5: Dashboard Page

### Task 13: Shared components — DeltaTag, SkeletonRow, ErrorBanner, EmptyState

**Files:**
- Create: `frontend/src/components/shared/DeltaTag.tsx`
- Create: `frontend/src/components/shared/SkeletonRow.tsx`
- Create: `frontend/src/components/shared/ErrorBanner.tsx`
- Create: `frontend/src/components/shared/EmptyState.tsx`

- [ ] **Step 1: Write DeltaTag**

Create `frontend/src/components/shared/DeltaTag.tsx`:
```tsx
import { cn } from '@/lib/utils'

interface DeltaTagProps {
  value: number | null
  unit?: string
  className?: string
}

export default function DeltaTag({ value, unit = '%', className }: DeltaTagProps) {
  if (value === null) {
    return (
      <span className={cn('inline-flex items-center font-mono text-[10px] bg-surface2 text-text2 px-1.5 py-0.5', className)}>
        — Baseline
      </span>
    )
  }
  const isPositive = value > 0
  return (
    <span className={cn(
      'inline-flex items-center font-mono text-[10px] px-1.5 py-0.5',
      isPositive
        ? 'bg-danger/10 text-danger'
        : 'bg-positive/10 text-positive',
      className
    )}>
      {isPositive ? '▲' : '▼'} {Math.abs(value)}{unit}
    </span>
  )
}
```

- [ ] **Step 2: Write SkeletonRow**

Create `frontend/src/components/shared/SkeletonRow.tsx`:
```tsx
export default function SkeletonRow() {
  return (
    <tr className="border-b border-border animate-pulse">
      {Array.from({ length: 7 }).map((_, i) => (
        <td key={i} className="px-3 py-4">
          <div className="h-3 bg-surface3 rounded-sm w-full" />
        </td>
      ))}
    </tr>
  )
}
```

- [ ] **Step 3: Write ErrorBanner**

Create `frontend/src/components/shared/ErrorBanner.tsx`:
```tsx
import { AlertCircle } from 'lucide-react'

interface ErrorBannerProps {
  message: string
  onRetry?: () => void
}

export default function ErrorBanner({ message, onRetry }: ErrorBannerProps) {
  return (
    <div className="flex items-center gap-3 bg-danger/10 border border-danger/30 px-4 py-3 text-danger text-[13px] font-sans">
      <AlertCircle size={15} />
      <span>{message}</span>
      {onRetry && (
        <button onClick={onRetry} className="ml-auto text-[11px] font-mono underline underline-offset-2 hover:no-underline">
          Retry
        </button>
      )}
    </div>
  )
}
```

- [ ] **Step 4: Write EmptyState**

Create `frontend/src/components/shared/EmptyState.tsx`:
```tsx
import { cn } from '@/lib/utils'

interface EmptyStateProps {
  icon?: React.ReactNode
  title: string
  description?: string
  action?: React.ReactNode
  className?: string
}

export default function EmptyState({ icon, title, description, action, className }: EmptyStateProps) {
  return (
    <div className={cn('flex flex-col items-center justify-center text-center py-16 gap-3', className)}>
      {icon && <div className="text-text3 mb-1">{icon}</div>}
      <p className="text-[14px] font-sans font-semibold text-text2">{title}</p>
      {description && <p className="text-[12px] text-text3 font-sans max-w-xs">{description}</p>}
      {action}
    </div>
  )
}
```


---

### Task 14: PatientRow and PatientTable

**Files:**
- Create: `frontend/src/components/patient/PatientRow.tsx`
- Create: `frontend/src/components/patient/PatientTable.tsx`

- [ ] **Step 1: Write PatientRow**

Create `frontend/src/components/patient/PatientRow.tsx`:
```tsx
import { useNavigate } from 'react-router-dom'
import type { Patient, Scan } from '@/types'
import DeltaTag from '@/components/shared/DeltaTag'
import { formatDate, formatVolume } from '@/lib/utils'

interface PatientRowProps {
  patient: Patient
  latestScan?: Scan
  previousScan?: Scan
}

export default function PatientRow({ patient, latestScan, previousScan }: PatientRowProps) {
  const navigate = useNavigate()

  const volumeDelta = latestScan && previousScan
    ? Math.round(((latestScan.volumeMm3 - previousScan.volumeMm3) / previousScan.volumeMm3) * 1000) / 10
    : null

  return (
    <tr
      className="border-b border-border cursor-pointer hover:bg-surface2 transition-colors"
      onClick={() => navigate(`/patients/${patient.id}`)}
    >
      <td className="px-3 py-3.5">
        <div className="font-sans font-semibold text-[15px] text-text1">{patient.name}</div>
        <div className="font-mono text-[10px] text-text3 mt-0.5">{patient.id}</div>
      </td>
      <td className="px-3 py-3.5">
        <div className="text-[12px] text-text2 font-sans leading-snug">
          {patient.diagnosis}<br />{patient.diagnosisLocation}
        </div>
      </td>
      <td className="px-3 py-3.5">
        <span className="inline-flex items-center gap-1 bg-surface3 border border-border2 text-teal font-mono text-[10px] font-bold px-2.5 py-1">
          ▣ {patient.scanCount} {patient.scanCount === 1 ? 'SCAN' : 'SCANS'}
        </span>
      </td>
      <td className="px-3 py-3.5">
        <div className="font-mono text-[11px] text-text2">{formatDate(patient.lastScanDate)}</div>
      </td>
      <td className="px-3 py-3.5">
        {latestScan ? (
          <>
            <div className="font-mono text-[13px] text-text1">
              {formatVolume(latestScan.volumeMm3)} <span className="text-[10px] text-text3">mm³</span>
            </div>
            <div className="mt-0.5">
              <DeltaTag value={volumeDelta} />
            </div>
          </>
        ) : (
          <span className="font-mono text-[10px] text-text3">No scans</span>
        )}
      </td>
      <td className="px-3 py-3.5">
        <div className="flex items-center gap-1.5">
          <span className={`w-1.5 h-1.5 rounded-full ${
            patient.status === 'active'
              ? 'bg-teal shadow-[0_0_6px_#0DC5A0]'
              : 'bg-amber'
          }`} />
          <span className="text-[11px] text-text2 font-sans capitalize">{patient.status}</span>
        </div>
      </td>
      <td className="px-3 py-3.5 text-text3 text-[13px]">›</td>
    </tr>
  )
}
```

- [ ] **Step 2: Write PatientTable**

Create `frontend/src/components/patient/PatientTable.tsx`:
```tsx
import type { Patient, Scan } from '@/types'
import PatientRow from './PatientRow'
import SkeletonRow from '@/components/shared/SkeletonRow'

interface PatientTableProps {
  patients: Patient[]
  scansMap: Record<string, Scan[]>
  loading?: boolean
}

const COLUMNS = ['Patient', 'Diagnosis', 'Scans', 'Last MRI', 'Volume (latest)', 'Status', '']

export default function PatientTable({ patients, scansMap, loading = false }: PatientTableProps) {
  return (
    <table className="w-full border-collapse">
      <thead>
        <tr className="border-b border-border">
          {COLUMNS.map(col => (
            <th
              key={col}
              className="text-left px-3 py-2 text-[10px] font-mono font-bold tracking-widest uppercase text-text3"
            >
              {col}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {loading
          ? Array.from({ length: 5 }).map((_, i) => <SkeletonRow key={i} />)
          : patients.map(patient => {
              const scans = (scansMap[patient.id] ?? []).sort(
                (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
              )
              const latest = scans[scans.length - 1]
              const previous = scans[scans.length - 2]
              return (
                <PatientRow
                  key={patient.id}
                  patient={patient}
                  latestScan={latest}
                  previousScan={previous}
                />
              )
            })}
      </tbody>
    </table>
  )
}
```


---

### Task 15: Build DashboardPage

**Files:**
- Modify: `frontend/src/pages/DashboardPage.tsx`

- [ ] **Step 1: Write the full DashboardPage**

Replace `frontend/src/pages/DashboardPage.tsx`:
```tsx
import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import TopNav from '@/components/layout/TopNav'
import PatientTable from '@/components/patient/PatientTable'
import ErrorBanner from '@/components/shared/ErrorBanner'
import EmptyState from '@/components/shared/EmptyState'
import { getPatients } from '@/api/patients'
import { getScans } from '@/api/scans'
import type { Scan } from '@/types'
import { Users } from 'lucide-react'

export default function DashboardPage() {
  const [search, setSearch] = useState('')

  const { data: patients = [], isLoading, isError, refetch } = useQuery({
    queryKey: ['patients'],
    queryFn: getPatients,
  })

  // Fetch scans for all patients in parallel
  const scanQueries = useQuery({
    queryKey: ['all-scans', patients.map(p => p.id)],
    queryFn: async () => {
      const results = await Promise.all(patients.map(p => getScans(p.id)))
      const map: Record<string, Scan[]> = {}
      patients.forEach((p, i) => { map[p.id] = results[i] })
      return map
    },
    enabled: patients.length > 0,
  })

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return patients
    return patients.filter(p =>
      p.name.toLowerCase().includes(q) || p.id.toLowerCase().includes(q)
    )
  }, [patients, search])

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      <TopNav
        showSearch
        searchValue={search}
        onSearchChange={setSearch}
        cta={
          <button className="border border-teal text-teal font-mono font-bold text-[12px] tracking-widest uppercase px-3.5 py-1.5 hover:bg-teal/5 transition-colors">
            + New Patient
          </button>
        }
      />

      <main className="flex-1 px-5 py-6">
        <div className="bg-surface border border-border">
          {isError && (
            <ErrorBanner
              message="Failed to load patients."
              onRetry={() => refetch()}
            />
          )}

          <div className="px-5 py-3.5 border-b border-border flex items-center justify-between">
            <span className="text-[11px] font-mono text-text3 uppercase tracking-widest">
              {filtered.length} patient{filtered.length !== 1 ? 's' : ''} · Dr. D. Cohen · Oncology
            </span>
            <span className="text-[11px] font-mono text-text3">Sorted by last scan ↓</span>
          </div>

          {!isLoading && filtered.length === 0 ? (
            <EmptyState
              icon={<Users size={28} />}
              title="No patients found"
              description={search ? 'Try a different name or ID.' : 'No patients assigned to your account yet.'}
            />
          ) : (
            <PatientTable
              patients={filtered}
              scansMap={scanQueries.data ?? {}}
              loading={isLoading}
            />
          )}
        </div>
      </main>
    </div>
  )
}
```

- [ ] **Step 2: Write search filter test**

Create `frontend/src/pages/DashboardPage.test.tsx`:
```tsx
import { describe, it, expect } from 'vitest'
import { mockPatients } from '@/data/mockData'

describe('dashboard patient filter', () => {
  function filterPatients(patients: typeof mockPatients, query: string) {
    const q = query.trim().toLowerCase()
    if (!q) return patients
    return patients.filter(p =>
      p.name.toLowerCase().includes(q) || p.id.toLowerCase().includes(q)
    )
  }

  it('returns all patients when query is empty', () => {
    expect(filterPatients(mockPatients, '').length).toBe(mockPatients.length)
  })

  it('filters by patient name case-insensitively', () => {
    const result = filterPatients(mockPatients, 'sarah')
    expect(result.length).toBe(1)
    expect(result[0].name).toBe('Sarah Jenkins')
  })

  it('filters by patient ID', () => {
    const result = filterPatients(mockPatients, 'P-1031')
    expect(result.length).toBe(1)
    expect(result[0].id).toBe('P-1031')
  })

  it('returns empty when no match', () => {
    expect(filterPatients(mockPatients, 'zzz').length).toBe(0)
  })
})
```

- [ ] **Step 3: Run tests**

```bash
cd frontend && npm test -- --run
```
Expected: all previous tests + 4 new filter tests pass.

- [ ] **Step 4: Verify dashboard in browser**

Navigate to `http://localhost:5173`. Sign in with any credentials. Expected: dark patient table with all 8 patients. Type "sarah" in search — expected: 1 row shown.


---

## Chunk 6: Patient Detail Page

### Task 16: StatBlock component

**Files:**
- Create: `frontend/src/components/shared/StatBlock.tsx`

- [ ] **Step 1: Write StatBlock**

Create `frontend/src/components/shared/StatBlock.tsx`:
```tsx
import DeltaTag from './DeltaTag'
import { cn } from '@/lib/utils'

interface StatBlockProps {
  label: string
  value: string
  delta?: number | null
  deltaUnit?: string
  badge?: React.ReactNode
  className?: string
}

export default function StatBlock({ label, value, delta, deltaUnit, badge, className }: StatBlockProps) {
  return (
    <div className={cn('bg-surface px-4 py-[18px]', className)}>
      <div className="text-[10px] font-mono font-bold tracking-widest uppercase text-text3 mb-2.5">
        {label}
      </div>
      <div className="font-mono text-[26px] font-bold text-text1 leading-none">
        {value}
      </div>
      <div className="mt-2">
        {badge ?? (delta !== undefined ? (
          <DeltaTag value={delta ?? null} unit={deltaUnit} />
        ) : null)}
      </div>
    </div>
  )
}
```


---

### Task 17: VolumeChart component

**Files:**
- Create: `frontend/src/components/scan/VolumeChart.tsx`

- [ ] **Step 1: Write VolumeChart**

Create `frontend/src/components/scan/VolumeChart.tsx`:
```tsx
import {
  ResponsiveContainer,
  LineChart,
  Line,
  Area,
  AreaChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  TooltipProps,
} from 'recharts'
import type { Scan } from '@/types'
import { formatDate, formatVolume } from '@/lib/utils'

interface VolumeChartProps {
  scans: Scan[]
}

function CustomTooltip({ active, payload, label }: TooltipProps<number, string>) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-surface2 border border-border2 px-3 py-2 font-mono text-[11px]">
      <div className="text-text3 mb-1">{label}</div>
      <div className="text-teal font-bold">{formatVolume(payload[0].value as number)} mm³</div>
    </div>
  )
}

export default function VolumeChart({ scans }: VolumeChartProps) {
  const data = scans.map(s => ({
    date: formatDate(s.date),
    volume: s.volumeMm3,
  }))

  return (
    <div className="bg-surface border border-border p-5">
      <div className="flex items-baseline justify-between mb-1">
        <span className="text-[12px] font-mono font-bold tracking-widest uppercase text-text2">
          Tumor Volume Trajectory
        </span>
        <span className="text-[11px] font-mono text-text3">
          mm³ · {scans.length} data point{scans.length !== 1 ? 's' : ''}
        </span>
      </div>
      <div className="mt-4 h-[120px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 8, right: 4, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="tealGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#0DC5A0" stopOpacity={0.15} />
                <stop offset="100%" stopColor="#0DC5A0" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#252A3A" strokeDasharray="0" vertical={false} />
            <XAxis
              dataKey="date"
              tick={{ fill: '#4E566A', fontSize: 10, fontFamily: 'Geist Mono' }}
              axisLine={{ stroke: '#252A3A' }}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: '#4E566A', fontSize: 10, fontFamily: 'Geist Mono' }}
              axisLine={false}
              tickLine={false}
              tickFormatter={v => `${Math.round(v / 1000)}k`}
              width={32}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="volume"
              stroke="#0DC5A0"
              strokeWidth={2}
              fill="url(#tealGrad)"
              dot={{ fill: '#0B0D12', stroke: '#0DC5A0', strokeWidth: 2, r: 4 }}
              activeDot={{ fill: '#0DC5A0', r: 5 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
```


---

### Task 18: ScanRow and ImagingHistory components

**Files:**
- Create: `frontend/src/components/scan/ScanRow.tsx`
- Create: `frontend/src/components/scan/ImagingHistory.tsx`

- [ ] **Step 1: Write ScanRow**

Create `frontend/src/components/scan/ScanRow.tsx`:
```tsx
import type { Scan } from '@/types'
import { formatDate, formatVolume } from '@/lib/utils'
import { ScanLine } from 'lucide-react'

interface ScanRowProps {
  scan: Scan
  index: number
}

export default function ScanRow({ scan, index }: ScanRowProps) {
  return (
    <div className="flex items-center gap-3.5 py-3 border-b border-border last:border-b-0 last:pb-0 first:pt-0">
      <span className="font-mono text-[10px] text-text3 w-5 shrink-0 text-right">
        {String(index).padStart(2, '0')}
      </span>
      <div className="w-8 h-8 bg-surface3 border border-border2 flex items-center justify-center shrink-0 text-teal">
        <ScanLine size={14} />
      </div>
      <div className="flex-1 min-w-0">
        <div className="font-sans font-medium text-[13px] text-text1">{formatDate(scan.date)}</div>
        <div className="font-mono text-[10px] text-text3 mt-0.5 truncate">
          {scan.studyLabel} · {scan.sequence} · {scan.plane} · {scan.sliceCount} slices · {scan.resolution}
        </div>
      </div>
      <div className="text-right shrink-0">
        <div className="font-mono text-[13px] text-text1">{formatVolume(scan.volumeMm3)} mm³</div>
        <div className="font-mono text-[10px] text-text3 mt-0.5">Ø {scan.maxDiameterMm} mm</div>
      </div>
      <span className={`inline-flex items-center gap-1 font-mono text-[10px] font-bold px-2 py-0.5 shrink-0 ${
        scan.isAnnotated
          ? 'bg-teal/10 border border-teal/25 text-teal'
          : 'bg-surface3 border border-border2 text-text3'
      }`}>
        {scan.isAnnotated ? '✓ ANNOTATED' : '○ PENDING'}
      </span>
    </div>
  )
}
```

- [ ] **Step 2: Write ImagingHistory**

Create `frontend/src/components/scan/ImagingHistory.tsx`:
```tsx
import type { Scan } from '@/types'
import ScanRow from './ScanRow'

interface ImagingHistoryProps {
  scans: Scan[]
}

export default function ImagingHistory({ scans }: ImagingHistoryProps) {
  const sorted = [...scans].sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
  return (
    <div className="bg-surface border border-border p-5">
      <div className="text-[12px] font-mono font-bold tracking-widest uppercase text-text2 mb-4">
        Imaging History
      </div>
      {sorted.map((scan, i) => (
        <ScanRow key={scan.id} scan={scan} index={sorted.length - i} />
      ))}
    </div>
  )
}
```


---

### Task 19: AIInsightsPanel component

**Files:**
- Create: `frontend/src/components/shared/AIInsightsPanel.tsx`

- [ ] **Step 1: Write AIInsightsPanel**

Create `frontend/src/components/shared/AIInsightsPanel.tsx`:
```tsx
import type { Summary } from '@/types'
import { formatDate } from '@/lib/utils'
import { Sparkles } from 'lucide-react'

interface AIInsightsPanelProps {
  summary: Summary
}

export default function AIInsightsPanel({ summary }: AIInsightsPanelProps) {
  return (
    <div className="bg-surface border border-border2 border-l-[3px] border-l-teal p-5">
      <div className="flex items-center gap-2 mb-3">
        <Sparkles size={14} className="text-teal" />
        <span className="font-mono text-[11px] font-bold tracking-widest uppercase text-teal">
          AI Clinical Narrative
        </span>
        <span className="font-mono text-[10px] text-text3 ml-1">
          · {summary.model} · Generated {formatDate(summary.generatedAt.split('T')[0])}
        </span>
      </div>
      <p
        className="font-sans text-[13px] text-text2 leading-[1.8]"
        dangerouslySetInnerHTML={{ __html: summary.text }}
      />
    </div>
  )
}
```


---

### Task 20: MriWorkspace sidebar

**Files:**
- Create: `frontend/src/components/shared/MriWorkspace.tsx`

- [ ] **Step 1: Write MriWorkspace**

Create `frontend/src/components/shared/MriWorkspace.tsx`:
```tsx
import { useState } from 'react'
import type { Scan } from '@/types'
import { Pencil, Ruler, Trash2 } from 'lucide-react'
import { cn } from '@/lib/utils'

interface MriWorkspaceProps {
  scan: Scan
}

type Tool = 'brush' | 'ruler' | 'delete'

export default function MriWorkspace({ scan }: MriWorkspaceProps) {
  const [activeTool, setActiveTool] = useState<Tool>('brush')

  const tools: { id: Tool; icon: React.ReactNode; label: string }[] = [
    { id: 'brush', icon: <Pencil size={13} />, label: 'Brush annotation' },
    { id: 'ruler', icon: <Ruler size={13} />, label: 'Measure' },
    { id: 'delete', icon: <Trash2 size={13} />, label: 'Delete annotation' },
  ]

  return (
    <aside className="w-[280px] shrink-0 bg-[#060810] border-l border-border flex flex-col">
      {/* Header */}
      <div className="px-4 py-3.5 border-b border-border">
        <div className="text-[10px] font-mono font-bold tracking-widest uppercase text-text3 mb-1">
          MRI Workspace
        </div>
        <div className="text-[13px] font-sans font-medium text-text1">{scan.studyLabel}</div>
        <div className="font-mono text-[10px] text-text3 mt-0.5">
          {scan.sequence} · {scan.plane} · {scan.sliceCount} SL · {scan.resolution}
        </div>
      </div>

      {/* Viewer */}
      <div className="flex-1 flex items-center justify-center relative">
        {/* Crosshair lines */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="absolute left-0 right-0 h-px bg-teal/10" />
          <div className="absolute top-0 bottom-0 w-px bg-teal/10" />
        </div>

        <div className="flex flex-col items-center gap-3">
          {/* MRI slice mockup */}
          <div className="relative">
            <div
              className="w-[180px] h-[180px] rounded-full border border-[#1a2030]"
              style={{
                background: `
                  radial-gradient(ellipse at 30% 40%, #2a3548 0%, transparent 50%),
                  radial-gradient(ellipse at 65% 55%, #1e2a3a 0%, transparent 45%),
                  radial-gradient(ellipse at 50% 50%, #141e2f 0%, #080c14 100%)
                `,
                boxShadow: '0 0 40px rgba(0,0,0,0.8), inset 0 0 30px rgba(0,0,0,0.5)',
              }}
            >
              {/* Tumor overlay */}
              <div
                className="absolute"
                style={{
                  width: 34, height: 28,
                  top: 68, left: 73,
                  borderRadius: '50%',
                  background: 'rgba(224, 82, 82, 0.22)',
                  border: '1.5px solid rgba(224, 82, 82, 0.5)',
                  boxShadow: '0 0 12px rgba(224, 82, 82, 0.2)',
                }}
              />
              {/* Ruler annotation */}
              <div
                className="absolute"
                style={{ top: 66, left: 69, width: 40, borderTop: '1px dashed rgba(13,197,160,0.7)' }}
              />
              <div
                className="absolute font-mono text-[8px] text-teal"
                style={{ top: 56, left: 70 }}
              >
                {scan.maxDiameterMm} mm
              </div>
            </div>
          </div>

          {/* Slice navigation — cosmetic only */}
          <div className="flex items-center gap-2.5">
            <button className="w-6 h-6 bg-surface2 border border-border2 flex items-center justify-center text-text2 text-[11px]">
              ‹
            </button>
            <span className="font-mono text-[10px] text-text3">
              {Math.floor(scan.sliceCount / 2).toString().padStart(3, '0')} / {scan.sliceCount}
            </span>
            <button className="w-6 h-6 bg-surface2 border border-border2 flex items-center justify-center text-text2 text-[11px]">
              ›
            </button>
          </div>
        </div>
      </div>

      {/* Toolbar */}
      <div className="px-4 py-3 border-t border-border flex items-center justify-between">
        <div className="flex gap-1.5">
          {tools.map(tool => (
            <button
              key={tool.id}
              onClick={() => setActiveTool(tool.id)}
              title={tool.label}
              className={cn(
                'w-8 h-8 flex items-center justify-center border transition-colors',
                activeTool === tool.id
                  ? 'bg-teal/10 border-teal text-teal'
                  : 'bg-surface2 border-border2 text-text2 hover:border-text3'
              )}
            >
              {tool.icon}
            </button>
          ))}
        </div>
        <button className="border border-teal text-teal font-mono text-[11px] font-bold tracking-widest uppercase px-3 py-1.5 hover:bg-teal/5 transition-colors">
          ↓ PDF
        </button>
      </div>
    </aside>
  )
}
```


---

### Task 21: Assemble PatientDetailPage

**Files:**
- Modify: `frontend/src/pages/PatientDetailPage.tsx`

- [ ] **Step 1: Write the full PatientDetailPage**

Replace `frontend/src/pages/PatientDetailPage.tsx`:
```tsx
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import TopNav from '@/components/layout/TopNav'
import StatBlock from '@/components/shared/StatBlock'
import VolumeChart from '@/components/scan/VolumeChart'
import ImagingHistory from '@/components/scan/ImagingHistory'
import AIInsightsPanel from '@/components/shared/AIInsightsPanel'
import MriWorkspace from '@/components/shared/MriWorkspace'
import ErrorBanner from '@/components/shared/ErrorBanner'
import EmptyState from '@/components/shared/EmptyState'
import { getPatient } from '@/api/patients'
import { getScans } from '@/api/scans'
import { getSummary } from '@/api/reports'
import { formatDate, formatVolume } from '@/lib/utils'
import { ScanLine } from 'lucide-react'

export default function PatientDetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

  const patientQuery = useQuery({
    queryKey: ['patient', id],
    queryFn: () => getPatient(id!),
    enabled: !!id,
  })

  const scansQuery = useQuery({
    queryKey: ['scans', id],
    queryFn: () => getScans(id!),
    enabled: !!id,
  })

  const summaryQuery = useQuery({
    queryKey: ['summary', id],
    queryFn: () => getSummary(id!),
    enabled: !!id,
  })

  const patient = patientQuery.data
  const scans = (scansQuery.data ?? []).sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
  const latestScan = scans[scans.length - 1]
  const previousScan = scans[scans.length - 2]

  const volumeDelta = latestScan && previousScan
    ? Math.round(((latestScan.volumeMm3 - previousScan.volumeMm3) / previousScan.volumeMm3) * 1000) / 10
    : null

  const diameterDelta = latestScan && previousScan
    ? Math.round((latestScan.maxDiameterMm - previousScan.maxDiameterMm) * 10) / 10
    : null

  const allAnnotated = scans.length > 0 && scans.every(s => s.isAnnotated)

  if (patientQuery.isError) {
    return (
      <div className="min-h-screen bg-bg">
        <TopNav />
        <div className="p-5">
          <ErrorBanner message="Patient not found." onRetry={() => navigate('/dashboard')} />
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      {/* Top bar */}
      <div className="h-[52px] bg-bg border-b border-border flex items-center gap-4 px-5 shrink-0">
        <button
          onClick={() => navigate('/dashboard')}
          className="text-[11px] font-mono text-text3 uppercase tracking-widest flex items-center gap-1.5 hover:text-text2 transition-colors"
        >
          ← Dashboard
        </button>
        <span className="text-border2">|</span>
        {patient ? (
          <div className="flex items-baseline gap-2 min-w-0">
            <span className="font-sans font-bold text-[18px] text-text1 truncate">{patient.name}</span>
            <span className="font-mono text-[11px] text-teal">{patient.id}</span>
            <span className="text-[12px] text-text2 truncate hidden sm:block">
              · {patient.diagnosis}, {patient.diagnosisLocation} · DOB {formatDate(patient.dob)}
            </span>
          </div>
        ) : (
          <div className="h-4 w-48 bg-surface3 animate-pulse" />
        )}
        <div className="ml-auto shrink-0">
          <button className="border border-teal text-teal font-mono text-[11px] font-bold tracking-widest uppercase px-3.5 py-1.5 hover:bg-teal/5 transition-colors">
            ↓ Generate PDF Report
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Main column */}
        <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-4 min-w-0">
          {/* Stats */}
          <div className="grid grid-cols-4 gap-px bg-border border border-border">
            <StatBlock
              label="Total Scans"
              value={scans.length.toString()}
              badge={
                <span className="inline-flex items-center font-mono text-[10px] bg-surface2 text-text2 px-1.5 py-0.5">
                  {scans.length > 1 ? `${scans.length} studies` : '1 study'}
                </span>
              }
            />
            <StatBlock
              label="Current Volume"
              value={latestScan ? `${formatVolume(latestScan.volumeMm3)} mm³` : '—'}
              delta={volumeDelta}
              deltaUnit="%"
            />
            <StatBlock
              label="Max Diameter"
              value={latestScan ? `${latestScan.maxDiameterMm} mm` : '—'}
              delta={diameterDelta}
              deltaUnit=" mm"
            />
            <StatBlock
              label="Annotated"
              value=""
              badge={
                <span className={`inline-flex items-center font-mono text-[11px] font-bold px-2 py-1 ${
                  allAnnotated
                    ? 'bg-teal/10 border border-teal/25 text-teal'
                    : 'bg-surface3 border border-border2 text-amber'
                }`}>
                  {allAnnotated ? '✓ ALL SCANS' : '○ PARTIAL'}
                </span>
              }
            />
          </div>

          {/* Chart */}
          {scansQuery.isLoading ? (
            <div className="h-[164px] bg-surface border border-border animate-pulse" />
          ) : scans.length > 0 ? (
            <VolumeChart scans={scans} />
          ) : null}

          {/* History */}
          {scansQuery.isLoading ? (
            <div className="h-32 bg-surface border border-border animate-pulse" />
          ) : scans.length === 0 ? (
            <EmptyState
              icon={<ScanLine size={24} />}
              title="No imaging studies uploaded yet"
              description="Upload a DICOM study to begin longitudinal tracking."
            />
          ) : (
            <ImagingHistory scans={scans} />
          )}

          {/* AI Insights */}
          {summaryQuery.isLoading ? (
            <div className="h-28 bg-surface border border-border animate-pulse" />
          ) : summaryQuery.data ? (
            <AIInsightsPanel summary={summaryQuery.data} />
          ) : null}
        </div>

        {/* MRI sidebar — hidden on small screens */}
        {latestScan && (
          <div className="hidden lg:flex">
            <MriWorkspace scan={latestScan} />
          </div>
        )}
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Verify the full app flow in browser**

```bash
cd frontend && npm run dev
```
- Sign in → lands on `/dashboard`
- Patient table shows 8 patients with volumes and deltas
- Click "Sarah Jenkins" → navigates to `/patients/P-1029`
- Stats row shows 3 scans, volume, diameter, annotated
- Chart shows teal line chart with 3 data points
- Imaging history shows 3 scan rows
- AI narrative panel shows clinical text
- Dark MRI sidebar visible on right with ruler annotation

- [ ] **Step 3: Run all tests**

```bash
npm test -- --run
```
Expected: all tests pass (utils, auth store, route guard, dashboard filter).


---

## Chunk 7: Polish, Responsiveness & Final Wiring

### Task 22: Responsive breakpoints

**Files:**
- Verify responsive behaviour in browser (no code changes needed beyond what's already in classes)

- [ ] **Step 1: Verify auth page responsiveness**

Resize browser to < 640px. Expected: left brand panel hidden, form takes full width.

- [ ] **Step 2: Verify dashboard responsiveness**

Resize to < 768px. Expected: table still visible but compressed. Patient name/ID columns remain readable.

- [ ] **Step 3: Verify patient detail responsiveness**

Resize to < 1024px. Expected: MRI sidebar hidden (the `hidden lg:flex` class hides it). Main column takes full width.

---

### Task 23: Add `.superpowers/` to `.gitignore` and final commit

**Files:**
- Modify: `.gitignore` (repo root)

- [ ] **Step 1: Add brainstorm session files to gitignore**

Open the repo-root `.gitignore` and add:
```
.superpowers/
```

- [ ] **Step 2: Run full test suite one final time**

```bash
cd frontend && npm test -- --run
```
Expected: all tests pass with 0 failures.

- [ ] **Step 3: Run TypeScript check**

```bash
npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 4: Run build**

```bash
npm run build
```
Expected: `dist/` created with no build errors.

- [ ] **Step 5: Final commit — spec, plan, and frontend**

From the repo root:
```bash
git add \
  docs/superpowers/specs/2026-03-15-frontend-design.md \
  docs/superpowers/plans/2026-03-15-frontend-scaffold.md \
  frontend/ \
  .gitignore
git commit -m "feat: Phase 7 frontend scaffold — OncoFlow React SPA

- Vite + React 18 + TypeScript + Tailwind CSS
- Clinical dark aesthetic (Geist fonts, teal accent, 2px radius)
- Auth page with sign-in and register-request forms
- Dashboard with patient table, search filter, volume deltas
- Patient detail with stats, Recharts volume chart, imaging history, AI narrative, MRI workspace sidebar
- API layer backed by mock data — ready for backend swap
- TanStack Query for server state, Zustand for auth session
- Responsive: sidebar collapses < 1024px, table compresses < 768px"
```
