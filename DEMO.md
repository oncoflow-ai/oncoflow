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

In a second terminal, from the repo root:

```bash
cd frontend
npm install
npm run dev
```

Vite serves the app on `http://localhost:5173`.

### Sanity check

In a browser tab, hit `http://localhost:8000/api/v1/ready` — you should see
a JSON object with `"status": "ready"` and the configured backend
(`"local"`) and enabled models (`"nnunet"`).

---

## 2. UI walkthrough

### Step 1 — Sign in

Open `http://localhost:5173/auth`.

- Enter any email (e.g. `dr.cohen@ichilov.gov.il`) and any password.
- Click **Sign In**.

> Auth is mock-only for this demo; any non-empty values are accepted.

You will land on the dashboard.

### Step 2 — Upload the baseline scan

Scroll to the **Operator Workspace** card on the dashboard.

1. Confirm the **Scan Format** toggle is on **NIfTI (recommended)**.
   The scan and mask file pickers intentionally do **not** filter by extension,
   so `.nii.gz` files are never hidden by the browser; only `.nii` / `.nii.gz`
   names are accepted when you click **Upload And Start**.
2. Under **NIfTI Scan (.nii / .nii.gz)**, pick:
   `data/P01/BraTS/baseline/t1c.nii.gz`
3. Under **Tumor Mask (optional, .nii.gz)**, pick:
   `data/P01/tumor segmentation/P01_tumor_mask_baseline.nii.gz`
4. **Source Label**: `Patient P01 - Baseline`
5. **Acquisition Date**: `2024-01-15`
6. Click **Upload And Start**.

The right-hand "Active run status" card shows the job moving from
`queued` → `running` → `completed`. With the bundled mask this typically
finishes in under a second.

### Step 3 — Inspect the baseline result

Once the run is `completed`, scroll to the **Results and lesion packaging**
panel. You should see:

- A summary block: study ID, lesion count (1), review state (CLEAR), QC reasons.
- A **Lesion** card with:
  - `lesion-001`
  - **Volume** (mm³) and **longest diameter** (mm) computed from the mask.
  - The bounding box JSON.
  - The mask artifact path under
    `derived/studies/<studyId>/lesions/lesion-001.nii.gz`.

> Make a mental note (or copy) of the **Study ID** — you'll see it again in
> the comparison dropdown.

### Step 4 — Upload the follow-up scan

Repeat Step 2 with:

- **Scan**: `data/P01/BraTS/fu1/t1c.nii.gz`
- **Mask**: `data/P01/tumor segmentation/P01_tumor_mask_fu1.nii.gz`
- **Source Label**: `Patient P01 - FU1`
- **Acquisition Date**: `2024-04-10`

Wait until status is `completed`.

### Step 5 — Run the longitudinal comparison

Scroll one section further to the **Longitudinal Comparison** panel
("Compare two scans, see tumor change").

1. The two studies you just uploaded appear in the dropdowns
   (label · acquisition date). If not, click **Refresh studies**.
2. **Baseline Study**: pick `Patient P01 - Baseline · 2024-01-15`.
3. **Follow-up Study**: pick `Patient P01 - FU1 · 2024-04-10`.
4. Click **Run Comparison**.

The status card on the right shows a spinner while the backend runs
preprocessing → registration → metrics. With provided masks this finishes
in a few seconds.

### Step 6 — Read the growth metrics

When the comparison returns, the lower **Comparison metrics** section
populates with three rows of stat blocks:

- **Volumes**: baseline volume, follow-up volume, Δ volume (cm³,
  color-coded growth/shrinkage), and % change with an interpretation
  badge (`stable`, `progressive`, `response`, or `minor change`).
- **RECIST**: longest in-plane diameter for baseline (A) and follow-up (B),
  the RECIST ratio, and the volume growth rate (cm³/day).
- **Overlap & registration**: Dice, HD95 (mm), registration NCC, and the
  registration method (e.g. `affine`).

The **Interpretation** banner shows the RECIST-style flag
(`Progressive disease`, `Stable disease`, `Response (>=25% reduction)`,
or similar). The full backend payload is available in the collapsible
**Raw comparison payload** block.

### Step 7 — (Optional) Add another follow-up

Repeat Step 4 with `data/P01/BraTS/fu2/t1c.nii.gz` and
`P01_tumor_mask_fu2.nii.gz` (date `2024-07-15`). Then in the Comparison
panel, swap the **Follow-up Study** dropdown to FU2 and rerun. You can
demo "tracked over multiple visits" by comparing baseline → FU1, then
baseline → FU2, then FU1 → FU2.

---

## How the demo maps to the codebase

| UI action | Backend endpoint | Code |
|---|---|---|
| Upload scan + mask | `POST /api/v1/jobs/nifti-segmentation` | [`backend/app/api/routes/jobs.py`](backend/app/api/routes/jobs.py), [`backend/app/modules/segmentation/nifti_pipeline.py`](backend/app/modules/segmentation/nifti_pipeline.py) |
| Status polling | `GET /api/v1/jobs/{jobId}` | same |
| Result fetch | `GET /api/v1/results/{studyId}` | [`backend/app/modules/results/service.py`](backend/app/modules/results/service.py) |
| Study list | `GET /api/v1/results/studies` | [`backend/app/modules/results/studies_listing.py`](backend/app/modules/results/studies_listing.py) |
| Run comparison | `POST /api/v1/jobs/longitudinal-comparison` | [`backend/app/modules/results/comparisons.py`](backend/app/modules/results/comparisons.py) calling `ml.inference.compare_studies` |

Demo is intentionally narrow; out of scope: real authentication, patient
CRUD, DICOM conversion (`dcm2niix`), 3D viewer, and reports/PDF export.

---

## Troubleshooting

- **Status stuck at `queued`** — make sure
  `ONCOFLOW_JOB_EXECUTION_MODE=threaded` is set in the same shell that
  started `uvicorn`. The default `deferred` mode never runs jobs locally.
- **`comparison failed: ...`** — re-check that both studies were uploaded
  with both a scan AND a mask file. The comparison endpoint requires the
  `nifti-source` artifact on each side and prefers the `tumor-mask-input`
  artifact to bypass segmentation.
- **`ml.inference is not available`** — install the ml extras:
  `pip install -r ml/inference/requirements.txt`.
- **CORS errors in the browser** — the default origin regex matches
  `localhost`/`127.0.0.1`. If you serve the frontend from another host,
  set `ONCOFLOW_FRONTEND_ORIGIN_REGEX` accordingly.
