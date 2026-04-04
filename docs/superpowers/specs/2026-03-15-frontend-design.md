# OncoFlow Frontend – Design Spec

**Date:** 2026-03-15
**Phase:** 7 – Frontend Scaffold
**Scope:** React SPA for longitudinal tumor tracking — Auth, Physician Dashboard, Patient Detail

---

## 1. Overview

A single-page application serving one user role (Physician/Oncologist) across three pages. Built to be backend-ready from day one: all data access is isolated in an `src/api/` layer so swapping mock data for real API calls requires no changes to components or stores.

---

## 2. Tech Stack

| Concern | Choice | Notes |
|---|---|---|
| Framework | React 18 + TypeScript | Strict mode |
| Build tool | Vite | Fast HMR, ESM-first |
| Styling | Tailwind CSS v3 | Utility-first |
| Component primitives | shadcn/ui | Owned, Tailwind-based, no runtime dependency |
| Icons | Lucide React | |
| Charts | Recharts | Tumor volume trajectory |
| Server state | TanStack Query (React Query v5) | Loading/error/cache handling |
| Client state | Zustand | Auth session only |
| Routing | react-router-dom v6 | |
| HTTP client | Axios | Configured in `src/api/client.ts` |
| Fonts | Geist + Geist Mono | UI text + all numeric/data values |

---

## 3. Design Aesthetic

**Direction:** Clinical Instrument — dark, precise, high-stakes.

- **Background:** `#0B0D12` (near-black)
- **Surfaces:** `#12151F` / `#191D2A` / `#21263A` (layered elevation)
- **Text:** `#EAE6DC` (warm off-white primary) / `#7A8499` (secondary) / `#4E566A` (muted)
- **Accent:** `#0DC5A0` (teal) — used for active state, data highlights, glows
- **Positive delta:** `#3DBE8C` (tumor reduction — good)
- **Negative delta:** `#E05252` (tumor growth — review)
- **Warning:** `#E8935A` (amber — review status)
- **Border radius:** 2px throughout — architectural, not bubbly
- **Typography:** Geist for all UI; Geist Mono for all IDs, volumes, dates, numeric data

---

## 4. File Structure

```
frontend/
├── src/
│   ├── api/
│   │   ├── client.ts          # Axios instance; reads VITE_API_URL
│   │   ├── patients.ts        # getPatients(), getPatient(id)
│   │   ├── scans.ts           # getScans(patientId), getScan(id)
│   │   ├── reports.ts         # getSummary(patientId)
│   │   └── mri.ts             # getMriUrl(scanId) — S3 presigned URL later
│   ├── components/
│   │   ├── ui/                # shadcn/ui primitives (Button, Badge, Input…)
│   │   ├── layout/            # AppShell, TopNav
│   │   ├── patient/           # PatientTable, PatientRow
│   │   ├── scan/              # ScanRow, VolumeChart, ImagingHistory
│   │   └── shared/            # StatBlock, AIInsightsPanel, MriWorkspace
│   ├── pages/
│   │   ├── AuthPage.tsx
│   │   ├── DashboardPage.tsx
│   │   └── PatientDetailPage.tsx
│   ├── store/
│   │   └── authStore.ts       # Zustand: physician session, persisted to sessionStorage
│   ├── data/
│   │   └── mockData.ts        # Typed mock patients + scans; used only inside src/api/
│   ├── hooks/
│   │   └── usePatient.ts      # Convenience hook wrapping TanStack Query calls
│   ├── lib/
│   │   └── utils.ts           # shadcn cn() helper + shared formatters
│   ├── router.tsx
│   └── main.tsx
├── .env.local                 # VITE_API_URL=http://localhost:8000
├── index.html
├── tailwind.config.ts
├── vite.config.ts
└── package.json
```

---

## 5. Routing

| Path | Component | Guard |
|---|---|---|
| `/` | Redirect → `/dashboard` if authenticated, else `/auth` | — |
| `/auth` | `AuthPage` | Redirect to `/dashboard` if already authenticated |
| `/dashboard` | `DashboardPage` | Requires auth |
| `/patients/:id` | `PatientDetailPage` | Requires auth |

Route guard implemented as a `<ProtectedRoute>` wrapper component that reads from `authStore`.

---

## 6. Pages

### 6.1 AuthPage (`/auth`)

Split layout — left brand panel, right form panel.

- **Left panel:** OncoFlow wordmark, editorial headline, three ambient stats (seg. accuracy, reports generated, ensemble model count)
- **Right panel:** "Clinical Sign In" form with Physician ID/Email + Password fields; "Access Patient Records" CTA; divider; "Request access from admin" toggle that swaps to a registration request form
- Auth action calls `authStore.login()` — for the mock phase this accepts any non-empty credentials

### 6.2 DashboardPage (`/dashboard`)

Top navigation bar + patient table.

**TopNav:**
- OncoFlow wordmark (left)
- Search input — filters patient table by name or ID (client-side, no API call)
- Physician avatar initials + "New Patient" button (right) — button renders but is non-functional in this phase (no create-patient flow); explicitly out of scope

**Patient Table columns:** Patient (name + ID) · Diagnosis · Scans · Last MRI · Volume (latest + delta %) · Status · Chevron

- Name rendered in Geist, all IDs and volumes in Geist Mono
- Volume delta color-coded: teal/green for reduction, red for growth, muted for baseline
- Status: pulsing teal dot = Active; amber dot = Review
- Row click navigates to `/patients/:id`
- Data loaded via TanStack Query calling `getPatients()`

### 6.3 PatientDetailPage (`/patients/:id`)

Two-column layout: scrollable main column (left) + fixed MRI workspace sidebar (right, 280px, dark).

**TopNav strip:** Back link · Patient name · Patient ID · Diagnosis · DOB · Generate PDF Report button

**Main column sections (top to bottom):**

1. **Stats Row** — 4 blocks in a CSS grid separated by 1px borders (no individual card shadows): Total Scans · Current Volume (mm³) · Max Diameter (mm) · Annotated status. Each block shows value + delta tag (color-coded).

2. **Tumor Volume Trajectory** — Recharts `LineChart` with `Area` fill. X-axis: scan dates. Y-axis: volume in mm³. Teal line + gradient fill. Grid lines in border color. All axis labels in Geist Mono. Data from `getScans(patientId)`.

3. **Imaging History** — List of scan rows. Each row: scan number · icon · date · study metadata · volume · diameter · Annotated badge. Data from `getScans(patientId)`.

4. **AI Clinical Narrative** — Panel with left teal border accent. Header shows "✦ AI Clinical Narrative" + model label + generation date. Body text in Geist 13px, line-height 1.8, with key findings bolded. Data from `getSummary(patientId)`.

**MRI Workspace Sidebar:**
- Dark background (`#060810`)
- Header: study label, study name, scan metadata (T1W · AXIAL · slices · resolution)
- Viewer area: styled placeholder div with a circular MRI slice mock (radial gradients), red tumor overlay, teal crosshair lines, ruler annotation with measurement label. Slice navigation (‹ 064/128 ›) at bottom. **All interactions are cosmetic/non-functional in this phase** — slice buttons, tool selection, and annotations are visual only; no real viewer is integrated.
- Toolbar: Brush (active/teal) · Ruler · Delete annotation tool buttons + PDF button — tool buttons toggle visual active state only
- Responsive: sidebar moves below main column on screens < 1024px

---

## 7. State Management

### 7.1 Auth Store (Zustand)

```ts
interface AuthStore {
  physician: { id: string; name: string; initials: string } | null;
  isAuthenticated: boolean;
  login: (id: string, password: string) => Promise<void>;
  logout: () => void;
}
```

Persisted to `sessionStorage`. Mock `login()` accepts any non-empty credentials and sets a hardcoded physician (Dr. D. Cohen).

### 7.2 Server State (TanStack Query)

All server data fetched via query hooks. No Zustand for server state.

```ts
useQuery({ queryKey: ['patients'], queryFn: getPatients })
useQuery({ queryKey: ['patient', id], queryFn: () => getPatient(id) })
useQuery({ queryKey: ['scans', patientId], queryFn: () => getScans(patientId) })
useQuery({ queryKey: ['summary', patientId], queryFn: () => getSummary(patientId) })
useQuery({ queryKey: ['mri-url', scanId], queryFn: () => getMriUrl(scanId) })
```

---

## 8. API Layer

### 8.1 Client (`src/api/client.ts`)

```ts
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? 'http://localhost:8000',
});

// JWT interceptor stub — no-op in mock phase.
// When backend is ready, replace with:
// apiClient.interceptors.request.use(cfg => {
//   cfg.headers.Authorization = `Bearer ${getToken()}`;
//   return cfg;
// });
```

During the mock phase, each API function returns mock data directly (with optional `setTimeout` delay to simulate latency). When the backend is ready, each function body is replaced with `apiClient.get(...)` and the interceptor stub is activated — no component or store changes needed.

### 8.2 Mock Data (`src/data/mockData.ts`)

Eight patients with realistic clinical profiles (oncology diagnoses, Israeli names). Each patient has 1–6 scans with realistic volume progressions. One AI narrative string per patient.

Sample patients:
- Sarah Jenkins (P-1029) — Osteosarcoma, Distal Left Femur — 3 scans, responding
- David Levi (P-1031) — Glioblastoma, Right Temporal Lobe — 5 scans, progressing
- Miriam Cohen (P-1044) — Breast Carcinoma Stage III — 2 scans, responding
- Jonathan Weiss (P-1051) — Non-Hodgkin Lymphoma, Mediastinal — 4 scans, stable
- Noa Shapiro (P-1062) — Renal Cell Carcinoma, Right Kidney — 1 scan, baseline
- Yosef Mizrahi (P-1073) — Colorectal Adenocarcinoma, Sigmoid — 6 scans, strong response
- Rachel Ben-David (P-1081) — Pancreatic Ductal Adenocarcinoma — 2 scans, progressing
- Eitan Goldberg (P-1094) — Lung Adenocarcinoma, Left Lower Lobe — 3 scans, stable

---

## 9. Loading & Empty States

- **Dashboard table:** Skeleton shimmer rows (5) while `getPatients()` resolves
- **Patient detail:** Skeleton blocks for stats, chart, and history panels
- **AI narrative:** Animated ellipsis "Generating narrative…" placeholder
- **No scans:** Centered empty state with "No imaging studies uploaded yet" + upload CTA
- **Error:** Inline error banner with retry button at each section level (not full-page errors)

---

## 10. Responsiveness

| Breakpoint | Behavior |
|---|---|
| ≥ 1024px | Two-column Patient Detail (main + MRI sidebar side-by-side) |
| < 1024px | MRI sidebar moves below main column; full width |
| < 768px | Dashboard table collapses to condensed patient cards |
| < 640px | Auth page stacks (left panel hidden, form full-width) |

---

## 11. Backend Contract (Required Endpoints)

The frontend is designed around these REST endpoints. The backend team must implement them for the mock→live swap to work.

### Patients

```
GET  /api/patients
     → Patient[]

GET  /api/patients/:id
     → Patient
```

**`Patient` object:**
```json
{
  "id": "P-1029",
  "name": "Sarah Jenkins",
  "dob": "1994-07-22",
  "diagnosis": "Osteosarcoma",
  "diagnosisLocation": "Distal Left Femur",
  "assignedPhysicianId": "DR-001",
  "status": "active",
  "scanCount": 3,
  "lastScanDate": "2026-03-08"
}
```

### Scans

```
GET  /api/patients/:id/scans
     → Scan[]

GET  /api/scans/:scanId
     → Scan
```

**`Scan` object:**
```json
{
  "id": "SCN-0041",
  "patientId": "P-1029",
  "studyLabel": "MRI Study #3",
  "date": "2026-03-08",
  "modality": "MRI",
  "sequence": "T1W",
  "plane": "AXIAL",
  "sliceCount": 128,
  "resolution": "1.2mm iso",
  "volumeMm3": 12480,
  "maxDiameterMm": 28.4,
  "isAnnotated": true
}
```

### AI Summary

```
GET  /api/patients/:id/summary
     → { patientId, generatedAt, model, text }
```

**Response:**
```json
{
  "patientId": "P-1029",
  "generatedAt": "2026-03-09T08:14:00Z",
  "model": "Gemini 1.5 (RAG-augmented)",
  "text": "Comparing the most recent study..."
}
```

### MRI Viewer

```
GET  /api/scans/:scanId/mri-url
     → { url, expiresAt }
```

Returns a presigned S3 URL for the DICOM/NIfTI viewer to load. The `MriWorkspace` component calls this and passes the URL to the viewer (OHIF or equivalent) when integrated.

### Auth

```
POST /api/auth/login
     Body: { physicianId, password }
     → { token, physician: { id, name, initials } }

POST /api/auth/register-request
     Body: { name, email, hospitalId, role }
     → { message }
```

The Axios client will attach the JWT token as a `Bearer` header via a request interceptor once auth is real.

---

## 12. Out of Scope (This Phase)

- Real DICOM/NIfTI MRI viewer (OHIF integration) — sidebar is a styled placeholder
- PDF report generation — button renders, endpoint not wired
- File upload / DICOM ingestion UI
- Admin dashboard
- 2FA
- Multi-role views (Radiologist, Clinician, Admin)
