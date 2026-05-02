# OncoFlow Demo Walkthrough

End-to-end UI demo using the live FastAPI backend, the bundled `data/P01/`
sample data, and the lightweight base-models inference path
(`OFLOW_ENABLED_MODELS=nnunet`).

For this demo, segmentation is sourced from the pre-computed tumor masks
under `data/P01/tumor segmentation/`. They are uploaded alongside each scan
and treated as the segmentation result, so no nnU-Net checkpoint is required
to see real volumes, RECIST diameters, and growth metrics.

---

## 0. Prerequisites

- Python 3.12+
- Node 18+
- Repo cloned locally; you are at the repo root.

The sample data lives under `data/P01/`:

| Timepoint | Scan (T1c) | Pre-computed mask |
|---|---|---|
| Baseline | `data/P01/BraTS/baseline/t1c.nii.gz` | `data/P01/tumor segmentation/P01_tumor_mask_baseline.nii.gz` |
| FU1 | `data/P01/BraTS/fu1/t1c.nii.gz` | `data/P01/tumor segmentation/P01_tumor_mask_fu1.nii.gz` |
| FU2 | `data/P01/BraTS/fu2/t1c.nii.gz` | `data/P01/tumor segmentation/P01_tumor_mask_fu2.nii.gz` |

---

## 1. One-time setup

### 1a. Backend

From the repo root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e "backend[dev,ml]"
.venv/bin/python -m pip install -r ml/inference/requirements.txt
```

Configure environment for a self-contained local demo (SQLite, threaded
workers, base models only):

```bash
export ONCOFLOW_DATABASE_URL="sqlite+pysqlite:///./var/oncoflow/dev.sqlite3"
export ONCOFLOW_STORAGE_ROOT="./var/oncoflow"
export ONCOFLOW_JOB_EXECUTION_MODE="threaded"
export OFLOW_BACKEND="local"
export OFLOW_DEVICE="auto"
export OFLOW_ENABLED_MODELS="nnunet"
export OFLOW_CACHE_DIR="./var/oncoflow/cache"
```

Create the storage root and apply DB migrations:

```bash
mkdir -p var/oncoflow
cd backend
PYTHONPATH=".:.." ../.venv/bin/alembic upgrade head
```

Start the API (still inside `backend/`):

```bash
PYTHONPATH=".:.." ../.venv/bin/uvicorn app.main:app --reload
```

You should see the server listening on `http://localhost:8000`.

### 1b. Frontend

The SPA defaults to `http://localhost:8000` as the API base URL ([`frontend/src/api/client.ts`](frontend/src/api/client.ts)). If your API runs elsewhere, copy [`frontend/.env.example`](frontend/.env.example) to `frontend/.env.local` and set `VITE_API_URL`.

In a second terminal, from the repo root:

```bash
cd frontend
npm install
npm run dev
```

Vite serves the app on `http://localhost:5173`.

### Sanity check

From the repo root you can run:

```bash
./scripts/check-demo-ready.sh
```

Or manually open `http://localhost:8000/api/v1/ready` — you should see a JSON object with `"status": "ready"` and the configured backend (`"local"`) and enabled models (`"nnunet"`).

---

## 2. UI walkthrough (roles)

The app uses **mock sign-in with roles**. Open `http://localhost:5173/auth`.

| Role | User ID example | Password | Where you land |
|------|-----------------|----------|----------------|
| **Radiologist** | any non-empty text | any non-empty | `/radiologist` — roster + upload workspace |
| **Doctor** | any non-empty text | any non-empty | `/doctor` — patient roster |
| **Patient** | `P-9001` or `P-1029` (must match `P-` + digits) | any non-empty | `/patient` — read-only portal |

> Auth is mock-only; credentials are not validated beyond non-empty fields (patient role checks the ID pattern).

**Recommended narrative for judges:** Radiologist uploads P01 volumes → longitudinal comparison → Doctor opens a chart → Patient portal as optional fourth beat.

---

### Path A — Radiologist (upload + pipeline)

Use this path for Steps A2–A6 below.

#### Step A1 — Sign in as Radiologist

1. Choose **Radiologist — upload & segmentation**.
2. Enter any User ID / Email and password → **Continue**.
3. You should be on **`/radiologist`**.

#### Step A2 — Select demo roster patient

Click the row **Demo Patient P01** (`P-9001`). The upload workspace appears below with **Source Label** prefilled (`P-9001 · Demo Patient P01`). Clear or edit **Source Label** when pasting demo strings below so the field matches exactly (see Troubleshooting if labels concatenate).

#### Step A3 — Upload the baseline scan

1. Confirm **Scan Format** is **NIfTI (recommended)**.
2. **NIfTI Scan**: `data/P01/BraTS/baseline/t1c.nii.gz`
3. **Tumor Mask**: `data/P01/tumor segmentation/P01_tumor_mask_baseline.nii.gz`
4. **Source Label**: `Patient P01 - Baseline` (clear the prefilled line first if needed).
5. **Acquisition Date**: `2024-01-15`
6. **Upload And Start**.

The **Active run status** panel moves through `queued` → `running` → `completed` (typically under a second with the bundled mask).

#### Step A4 — Inspect the baseline result

In **Results and lesion packaging**:

- Summary block: study ID, lesion count, review state, QC reasons.
- **Lesion** card: `lesion-001`, volume, longest diameter, bounding box, mask path under `derived/studies/<studyId>/lesions/`.

Note the **Study ID** for the comparison dropdowns.

#### Step A5 — Upload the follow-up scan

Repeat Step A3 pattern with:

- **Scan**: `data/P01/BraTS/fu1/t1c.nii.gz`
- **Mask**: `data/P01/tumor segmentation/P01_tumor_mask_fu1.nii.gz`
- **Source Label**: `Patient P01 - FU1`
- **Acquisition Date**: `2024-04-10`

Wait until status is `completed`.

#### Step A6 — Run the longitudinal comparison

On the same page, scroll to **Longitudinal Comparison** (“Compare two scans, see tumor change”).

1. Dropdowns list studies by backend labels/dates (refresh automatically). Pick **Baseline** `Patient P01 - Baseline · 2024-01-15` and **Follow-up** `Patient P01 - FU1 · 2024-04-10`.
2. **Run Comparison**.

Metrics populate: volumes + Δ + % change, RECIST-style diameters and ratio, Dice / HD95 / registration NCC, **Interpretation** banner, and raw JSON payload.

After segmentation completes, the UI may also trigger an automatic comparison when ≥2 studies have results (optional toast).

#### Step A7 — (Optional) Third timepoint

Upload FU2 (`data/P01/BraTS/fu2/t1c.nii.gz`, mask `P01_tumor_mask_fu2.nii.gz`, date `2024-07-15`, label `Patient P01 - FU2`). Then swap **Follow-up** to FU2 or compare FU1 → FU2.

---

### Path B — Doctor (clinical roster)

#### Step B1 — Sign in as Doctor

Land on **`/doctor`**. Search or scroll the roster.

#### Step B2 — Open a patient chart

Click any patient → **`/doctor/patients/:id`**.

- **Scans & viewer**: mock longitudinal imaging + MRI sidebar (slice viewer).
- **Longitudinal**: loads **live backend studies** (same list as radiologist); roster rows are **not** wired to backend UUIDs yet — every completed study appears in the selectors. Complete Path A first so labeled P01 studies appear.
- **Reports**: mock generate/list only.

---

### Path C — Patient portal

#### Step C1 — Sign in as Patient

Use Patient ID **`P-9001`** or **`P-1029`** plus any password.

Shows mock scans, AI summary text, recommendations, and mock reports list.

---

## How the demo maps to the codebase

| UI action | Backend / route | Code |
|-----------|-----------------|------|
| Role routing | `/radiologist`, `/doctor`, `/patient` | [`frontend/src/router.tsx`](frontend/src/router.tsx), [`frontend/src/lib/routes.ts`](frontend/src/lib/routes.ts) |
| Radiologist roster + upload | `/radiologist` | [`frontend/src/pages/RadiologistWorkspacePage.tsx`](frontend/src/pages/RadiologistWorkspacePage.tsx), [`frontend/src/components/dashboard/BackendOperatorWorkspace.tsx`](frontend/src/components/dashboard/BackendOperatorWorkspace.tsx) |
| Doctor roster | `/doctor` | [`frontend/src/pages/DoctorDashboardPage.tsx`](frontend/src/pages/DoctorDashboardPage.tsx) |
| Doctor patient chart | `/doctor/patients/:id` | [`frontend/src/pages/DoctorPatientDashboardPage.tsx`](frontend/src/pages/DoctorPatientDashboardPage.tsx) |
| Patient portal | `/patient` | [`frontend/src/pages/PatientPortalPage.tsx`](frontend/src/pages/PatientPortalPage.tsx) |
| Upload scan + mask | `POST /api/v1/jobs/nifti-segmentation` | [`backend/app/api/routes/jobs.py`](backend/app/api/routes/jobs.py), [`backend/app/modules/segmentation/nifti_pipeline.py`](backend/app/modules/segmentation/nifti_pipeline.py) |
| Status polling | `GET /api/v1/jobs/{jobId}` | same |
| Result fetch | `GET /api/v1/results/{studyId}` | [`backend/app/modules/results/service.py`](backend/app/modules/results/service.py) |
| Study list | `GET /api/v1/results/studies` | [`backend/app/modules/results/studies_listing.py`](backend/app/modules/results/studies_listing.py) |
| Run comparison | `POST /api/v1/jobs/longitudinal-comparison` | [`backend/app/modules/results/comparisons.py`](backend/app/modules/results/comparisons.py) calling `ml.inference.compare_studies` |

Demo scope: mock authentication and roster; live ingestion and comparison against SQLite + local disk. Out of scope: production auth, patient–study linkage in DB, DICOM conversion (`dcm2niix`), volumetric 3D viewer, PDF export.

---

## Presenter checklist (dry run)

1. Backend env includes **`ONCOFLOW_JOB_EXECUTION_MODE=threaded`**.
2. `./scripts/check-demo-ready.sh` or manual **`GET /api/v1/ready`** succeeds.
3. Frontend loads **`/auth`**; radiologist path completes Path A steps A3–A6 at least once.
4. Optional: doctor longitudinal tab lists P01 studies after uploads; patient **`P-9001`** loads portal.

---

## Troubleshooting

- **Status stuck at `queued`** — make sure `ONCOFLOW_JOB_EXECUTION_MODE=threaded` is set in the same shell that started `uvicorn`. The default `deferred` mode never runs jobs locally.
- **`comparison failed: ...`** — both studies need scan **and** mask. The comparison endpoint expects `nifti-source` and prefers `tumor-mask-input`.
- **`ml.inference is not available`** — install ML extras: `pip install -r ml/inference/requirements.txt`.
- **CORS errors** — default origin regex matches `localhost` / `127.0.0.1`. Otherwise set `ONCOFLOW_FRONTEND_ORIGIN_REGEX`.
- **Wrong API host from frontend** — set `VITE_API_URL` via `.env.local` (see [`frontend/.env.example`](frontend/.env.example)).
- **Source label looks concatenated after selecting a patient** — clear the **Source Label** field before typing demo labels (`Patient P01 - Baseline`, etc.).
