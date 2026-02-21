# OncoFlow – Step-by-Step Implementation Plan

> **Source:** OncoFlow Project Document (HLD)  
> **Goal:** Automated tumor-tracking web application for longitudinal MRI analysis  
> **Stack:** React/TypeScript · FastAPI · PostgreSQL · Redis/Celery · nnU-Net · MedGemma-1.5 · SAM3 · AWS

---

## Overview

OncoFlow uses a **Dual-Stream Processing Architecture**:
1. **Image Stream** – MRI ingestion → DICOM→NIfTI conversion → **Panel-of-Experts** segmentation (nnU-Net + MedGemma-1.5 + SAM3) → ensemble mask → volumetric comparison
2. **Text Stream** – Clinical notes/reports → RAG pipeline → context-enriched PDF report

The plan is divided into **7 phases** (Milestones):

| # | Phase | Focus |
|---|-------|-------|
| 1 | Foundation | Repo, CI/CD, Docker, DB schema |
| 2 | Auth & RBAC | Users, roles, JWT, 2FA |
| 3 | Patient & Scan Management | Upload, de-identification, storage |
| 4 | Image Processing Pipeline | DICOM→NIfTI, 3-model panel (nnU-Net · MedGemma-1.5 · SAM3), ensemble |
| 5 | RAG Text Pipeline | Vector DB, embedding, LLM querying |
| 6 | Report Generation | PDF, cryptographic signing |
| 7 | Frontend | Dashboard, patient view, report viewer |

Read each phase top-to-bottom. Each step has **What**, **Why**, and **How** sub-sections.

---

## Phase 1 – Foundation

### Step 1.1 – Repository & Project Structure

**What:** Set up a mono-repo (or separate repos) with clear separation between `frontend/`, `backend/`, `infra/`, and `ml/`.

**Recommended layout:**
```
oncoflow/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI routers
│   │   ├── core/         # Config, security, logging
│   │   ├── db/           # SQLAlchemy models & migrations
│   │   ├── services/     # Business logic
│   │   ├── workers/      # Celery tasks
│   │   └── main.py
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── api/          # Axios clients
│   │   └── store/        # State management (Zustand / Redux)
│   ├── Dockerfile
│   └── package.json
├── ml/
│   ├── nnunet/           # nnU-Net v2 wrapper service
│   ├── medgemma/         # MedGemma-1.5 inference service
│   ├── sam3/             # Meta SAM3 inference service
│   ├── ensemble/         # Voting / fusion logic
│   ├── registration/     # Image registration scripts
│   ├── rag/              # RAG pipeline
│   └── Dockerfile        # shared base; each model has its own
├── infra/
│   ├── docker-compose.yml
│   ├── docker-compose.prod.yml
│   └── terraform/        # AWS IaC (optional at start)
└── .github/
    └── workflows/        # CI/CD pipelines
```

**How:**
```bash
# Create folder structure
mkdir -p oncoflow/{backend/app/{api,core,db,services,workers},backend/tests,frontend/src/{components,pages,hooks,api,store},ml/{inference,registration,rag},infra,.github/workflows}
cd oncoflow && git init
```

---

### Step 1.2 – Docker Compose (Local Dev Environment)

**What:** Spin up all services locally: API, Worker, DB, Redis, ML service.

**`infra/docker-compose.yml`** should include:
- `api` – FastAPI (port 8000)
- `worker` – Celery worker
- `db` – PostgreSQL 15
- `redis` – Redis 7
- `ml-nnunet` – nnU-Net v2 inference service (port 8001)
- `ml-medgemma` – MedGemma-1.5 inference service (port 8002)
- `ml-sam3` – SAM3 inference service (port 8003)
- `ml-ensemble` – ensemble/voting service (port 8004)
- `frontend` – React dev server (port 3000)

**How:**
```yaml
# infra/docker-compose.yml (skeleton)
version: "3.9"
services:
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: oncoflow
      POSTGRES_USER: oncoflow
      POSTGRES_PASSWORD: secret
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports: ["5432:5432"]

  redis:
    image: redis:7
    ports: ["6379:6379"]

  api:
    build: ./backend
    env_file: .env
    depends_on: [db, redis]
    ports: ["8000:8000"]
    volumes:
      - ./backend:/app

  worker:
    build: ./backend
    command: celery -A app.workers.celery_app worker --loglevel=info
    env_file: .env
    depends_on: [db, redis, ml]

  ml-nnunet:
    build:
      context: ./ml/nnunet
    runtime: nvidia
    ports: ["8001:8001"]
    volumes: ["ml_models:/models"]

  ml-medgemma:
    build:
      context: ./ml/medgemma
    runtime: nvidia
    ports: ["8002:8002"]
    environment:
      HF_TOKEN: ${HF_TOKEN}   # Hugging Face access token for gated model
    volumes: ["ml_models:/models"]

  ml-sam3:
    build:
      context: ./ml/sam3
    runtime: nvidia
    ports: ["8003:8003"]
    volumes: ["ml_models:/models"]

  ml-ensemble:
    build:
      context: ./ml/ensemble
    ports: ["8004:8004"]

  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    volumes:
      - ./frontend:/app

volumes:
  pgdata:
  ml_models:
```

**Run:**
```bash
docker compose -f infra/docker-compose.yml up --build
```

---

### Step 1.3 – CI/CD Pipeline (GitHub Actions)

**What:** Automated lint, test, build, and (optionally) deploy on every PR/push.

**`/.github/workflows/ci.yml`:**
```yaml
name: CI
on: [push, pull_request]
jobs:
  backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install -r backend/requirements.txt
      - run: pytest backend/tests --cov=app
      - run: ruff check backend/

  frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: {node-version: "20"}
      - run: cd frontend && npm ci && npm run lint && npm test -- --watchAll=false
```

---

### Step 1.4 – Database Schema (PostgreSQL)

**What:** Design the core relational schema.

**Key tables:**

```sql
-- Users / Auth
CREATE TABLE users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email       TEXT UNIQUE NOT NULL,
    hashed_pw   TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('admin','doctor','radiologist','clinician')),
    two_fa_secret TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Doctor–Patient assignment (ABAC enforcement)
CREATE TABLE assignments (
    doctor_id   UUID REFERENCES users(id),
    patient_id  UUID REFERENCES patients(id),
    PRIMARY KEY (doctor_id, patient_id)
);

-- Patients (de-identified; true identity stored separately/encrypted)
CREATE TABLE patients (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pseudonym       TEXT NOT NULL,          -- internal UUID-based label
    age             INT,
    gender          TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Studies (one MRI scan session)
CREATE TABLE studies (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id  UUID REFERENCES patients(id),
    acquired_at TIMESTAMPTZ,
    modality    TEXT DEFAULT 'MRI',
    s3_path     TEXT NOT NULL,             -- pointer to DICOM/NIfTI in S3
    sha256      TEXT NOT NULL,             -- integrity hash
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Segmentation jobs
CREATE TABLE segmentation_jobs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    study_id    UUID REFERENCES studies(id),
    status      TEXT DEFAULT 'pending',     -- pending/running/done/failed
    mask_s3     TEXT,                       -- output mask path
    started_at  TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);

-- Longitudinal comparisons
CREATE TABLE comparisons (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    study_a      UUID REFERENCES studies(id),
    study_b      UUID REFERENCES studies(id),
    volume_a_cm3 FLOAT,
    volume_b_cm3 FLOAT,
    delta_cm3    FLOAT,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- Reports
CREATE TABLE reports (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id   UUID REFERENCES patients(id),
    comparison_id UUID REFERENCES comparisons(id),
    pdf_s3       TEXT,
    signature    TEXT,                      -- cryptographic PDF signature
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- Audit log (append-only)
CREATE TABLE audit_log (
    id        BIGSERIAL PRIMARY KEY,
    actor_id  UUID,
    action    TEXT,
    resource  TEXT,
    ts        TIMESTAMPTZ DEFAULT NOW()
);
```

**Migrations:** Use **Alembic** (`alembic init alembic` inside `backend/`).

---

## Phase 2 – Authentication & RBAC

### Step 2.1 – FastAPI Auth Service

**What:** Implement JWT-based login with optional 2FA (TOTP).

**Packages:**
```
python-jose[cryptography]
passlib[bcrypt]
pyotp              # TOTP 2FA
```

**Endpoints:**
```
POST /auth/register   – admin creates users
POST /auth/login      – returns access_token + refresh_token
POST /auth/refresh    – rotate tokens
POST /auth/2fa/setup  – returns TOTP QR code
POST /auth/2fa/verify – validates TOTP code
DELETE /auth/logout   – token revocation (Redis blocklist)
```

**Middleware – ABAC:**  
Every request beyond `/auth/*` must pass through `verify_resource_access()`:
```python
# backend/app/core/security.py
def verify_resource_access(current_user, patient_id: UUID, db: Session):
    """Raises 403 if current_user has no assignment for patient_id."""
    assignment = db.query(Assignment).filter_by(
        doctor_id=current_user.id, patient_id=patient_id
    ).first()
    if not assignment:
        raise HTTPException(403, "Forbidden – no assignment for this patient")
```

---

### Step 2.2 – Role Definitions

| Role | Permissions |
|------|-------------|
| `admin` | Create users, assign patients, view audit log |
| `doctor` | View/upload scans for assigned patients, trigger analysis, view reports |
| `radiologist` | Same as doctor + initiate segmentation |
| `clinician` | Read-only view of reports for assigned patients |

---

## Phase 3 – Patient & Scan Management

### Step 3.1 – Patient CRUD

**Endpoints:**
```
POST   /patients          – admin creates patient record
GET    /patients/{id}     – doctor retrieves patient (ABAC check)
PATCH  /patients/{id}     – update metadata
GET    /patients          – list assigned patients for current doctor
```

---

### Step 3.2 – DICOM/NIfTI Upload & De-identification

**What:** When a scan is uploaded, strip PHI before storage.

**Flow:**
```
Client → POST /studies/upload (multipart form) →
  1. Generate SHA-256 hash of raw DICOM bytes (integrity)
  2. Run de-identification worker:
     a. pydicom: strip VR=PN/LO/DA/DT tags
     b. Replace PatientID with internal UUID
     c. PyDeFace (optional): defacing brain MRIs
  3. Convert DICOM → NIfTI (SimpleITK or dcm2niix)
  4. Upload de-identified NIfTI + original DICOM to S3
  5. Store s3_path + sha256 in studies table
  6. Return study_id
```

**Key packages:**
```
pydicom
SimpleITK      # or nibabel + dcm2niix subprocess
boto3          # S3
```

---

### Step 3.3 – S3 Storage Structure

```
s3://oncoflow-data/
  raw/{patient_uuid}/{study_uuid}/               # original DICOM (traceability)
  processed/{patient_uuid}/{study_uuid}/input/   # de-id NIfTI fed to all models
  masks/{patient_uuid}/{study_uuid}/
    nnunet/   mask.nii.gz                        # nnU-Net output
    medgemma/ mask.nii.gz                        # MedGemma-1.5 output
    sam3/     mask.nii.gz                        # SAM3 output
    ensemble/ mask.nii.gz  metrics.json          # final fused mask + agreement scores
  reports/{patient_uuid}/{report_uuid}.pdf
```

**IAM policy:** API only has `s3:PutObject` / `s3:GetObject`; worker has additional access.

---

## Phase 4 – Image Processing Pipeline (Multi-Model Panel)

> **Design principle:** Run all three models in parallel, fuse their outputs into one ensemble mask,
> store individual masks for audit/comparison, and surface agreement scores to the clinician.

---

### Step 4.0 – Model Overview

| Model | Type | Input | Strength |
|-------|------|-------|----------|
| **nnU-Net v2** | CNN self-configuring | 3-D NIfTI | Gold-standard volumetric segmentation; BraTS-trained weights available |
| **MedGemma-1.5** | Multimodal vision-language (Google) | NIfTI slices + optional text prompt | Contextual understanding; can use clinical notes as prompt conditioning |
| **Meta SAM3** | Promptable vision foundation model | 3-D volume + optional point/box prompt | Strong generalisation across modalities; good for interactive refinement |

---

### Step 4.1 – Celery Task Architecture

```
API → Redis queue → Celery orchestrator worker
         ├──► nnU-Net service  (port 8001)  ─┐
         ├──► MedGemma service (port 8002)  ─┼──► Ensemble service (port 8004) → DB + S3
         └──► SAM3 service    (port 8003)  ─┘
```

**Task chain (parallel fan-out then merge):**
```python
# backend/app/workers/tasks.py
from celery import group, chain

def process_study(study_id: str):
    chain(
        deidentify_task.s(study_id),
        convert_to_nifti_task.s(),
        # fan-out: all three models run concurrently
        group(
            run_nnunet_task.s(),
            run_medgemma_task.s(),
            run_sam3_task.s(),
        ),
        # merge: ensemble after all three finish
        ensemble_task.s(),
        compute_volumes_task.s(),
    ).apply_async()
```

**DB – updated `segmentation_jobs` table:**
```sql
ALTER TABLE segmentation_jobs
  ADD COLUMN model       TEXT NOT NULL DEFAULT 'ensemble',
  -- model IN ('nnunet','medgemma','sam3','ensemble')
  ADD COLUMN dice_score  FLOAT,   -- vs ensemble mask (agreement metric)
  ADD COLUMN iou_score   FLOAT;
```

**Job states:** `pending → running → completed | failed`  
One row per model + one row for the ensemble.

---

### Step 4.2 – Model 1: nnU-Net v2 Service

**What:** Self-configuring 3-D CNN; use pretrained BraTS/Ichilov checkpoint.

**Setup:**
```bash
# ml/nnunet/Dockerfile
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
RUN pip install nnunetv2
COPY . /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

**Inference endpoint (`ml/nnunet/main.py`):**
```python
@app.post("/infer")
async def infer(payload: InferRequest, bg: BackgroundTasks):
    bg.add_task(_predict, payload)
    return {"status": "queued", "job_id": payload.job_id}

def _predict(p):
    # 1. Download NIfTI from S3 → /tmp/input/{uuid}_0000.nii.gz
    # 2. subprocess.run(["nnUNetv2_predict",
    #       "-i", "/tmp/input", "-o", "/tmp/out",
    #       "-d", DATASET_ID, "-c", "3d_fullres"])
    # 3. Upload /tmp/out/mask.nii.gz → S3 masks/{patient}/{study}/nnunet/
    # 4. POST callback to backend /internal/jobs/{job_id}/done
```

**NIfTI naming required by nnU-Net:**
```
/tmp/input/{study_uuid}_0000.nii.gz    # T1 (modality 0)
/tmp/input/{study_uuid}_0001.nii.gz    # T1ce (modality 1, if available)
```

---

### Step 4.3 – Model 2: MedGemma-1.5 Service

**What:** Google's medical vision-language model. Use the `google/medgemma-1.5-it` checkpoint
from HuggingFace (gated – requires approval at hf.co/google/medgemma).

**Key capability:** Can receive a text prompt alongside the image, e.g. clinical indication, which
improves segmentation precision on ambiguous cases.

**Setup:**
```bash
# ml/medgemma/Dockerfile
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
RUN pip install transformers accelerate nibabel boto3 fastapi uvicorn
COPY . /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
```

**Inference endpoint (`ml/medgemma/main.py`):**
```python
from transformers import AutoTokenizer, AutoModelForImageClassification
# (use the actual MedGemma pipeline class when released)

MODEL_ID = "google/medgemma-1.5-it"
model, processor = None, None

@app.on_event("startup")
def load_model():
    global model, processor
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )

@app.post("/infer")
async def infer(payload: InferRequest, bg: BackgroundTasks):
    bg.add_task(_predict, payload)
    return {"status": "queued", "job_id": payload.job_id}

def _predict(p):
    volume = load_nifti_from_s3(p.input_s3)   # shape (H, W, D)
    # Slice-wise inference: run on axial slices, reconstruct 3-D mask
    masks_2d = []
    prompt = p.clinical_hint or "Segment the tumor region."
    for s in range(volume.shape[2]):
        slice_img = volume[:, :, s]            # 2-D axial slice
        inputs = processor(images=to_pil(slice_img), text=prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512)
        masks_2d.append(decode_segmentation_output(out))
    mask_3d = np.stack(masks_2d, axis=-1)
    save_and_upload_mask(mask_3d, p.output_s3)
    post_callback(p.job_id)
```

> **Note:** The exact pipeline class depends on the MedGemma version released.
> Monitor [hf.co/google/medgemma](https://huggingface.co/google/medgemma) for updates.
> At minimum, slice-by-slice VQA-style inference with segmentation decoding is the expected approach.

---

### Step 4.4 – Model 3: Meta SAM3 (Segment Anything Model 3) Service

**What:** Meta's SAM3 extends SAM2 to 3-D medical volumes with improved promptability.
Use it in **automatic mode** (no user prompt) for the batch pipeline;
use **interactive/prompt mode** in the front-end for clinician refinement.

**Setup:**
```bash
# ml/sam3/Dockerfile
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
RUN pip install git+https://github.com/facebookresearch/sam3.git nibabel boto3 fastapi uvicorn
COPY . /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
```

**Inference endpoint (`ml/sam3/main.py`):**
```python
from sam3 import SAM3Predictor   # adjust import once library stabilises

predictor = SAM3Predictor.from_pretrained("facebook/sam3-hiera-large")

@app.post("/infer")
async def infer(payload: InferRequest, bg: BackgroundTasks):
    bg.add_task(_predict, payload)
    return {"status": "queued", "job_id": payload.job_id}

def _predict(p):
    volume = load_nifti_from_s3(p.input_s3)           # numpy ndarray
    predictor.set_volume(volume)
    if p.prompt_points:                                # optional: [[x,y,z], ...]
        masks, _, _ = predictor.predict(point_coords=p.prompt_points,
                                        point_labels=[1]*len(p.prompt_points),
                                        multimask_output=False)
    else:
        masks = predictor.generate_automatic()         # automatic mode
    best_mask = masks[0]                               # shape (H, W, D)
    save_and_upload_mask(best_mask.astype(np.uint8), p.output_s3)
    post_callback(p.job_id)
```

**Interactive refinement (frontend):**
Expose `POST /infer/interactive` that accepts click coordinates from the UI
so a clinician can click a point on the scan and SAM3 refines the mask in real time.

---

### Step 4.5 – Ensemble / Panel-of-Experts Fusion

**What:** After all three masks are produced, fuse them into one final mask
and compute agreement metrics that appear in the report.

**Strategy options (implement all; choose via config flag):**

| Strategy | When to use |
|----------|-------------|
| **Majority vote** (≥2/3 agree per voxel) | Default; robust baseline |
| **STAPLE** (EM-based probabilistic fusion) | When model calibration differs |
| **Weighted average** (learned or manual weights) | When one model is known to outperform on your data |
| **Confidence-weighted** (use model-reported probability maps) | Best accuracy; requires each model to output soft masks |

**`ml/ensemble/main.py`:**
```python
import nibabel as nib
import numpy as np
from scipy.ndimage import label

def majority_vote(masks: list[np.ndarray]) -> np.ndarray:
    """voxel-wise majority vote across N binary masks."""
    stack = np.stack(masks, axis=0)   # (N, H, W, D)
    return (stack.sum(axis=0) >= len(masks) / 2).astype(np.uint8)

def compute_dice(a: np.ndarray, b: np.ndarray) -> float:
    intersection = (a & b).sum()
    return 2 * intersection / (a.sum() + b.sum() + 1e-6)

@app.post("/fuse")
async def fuse(payload: FuseRequest):
    masks = {
        "nnunet":   load_mask_from_s3(payload.nnunet_s3),
        "medgemma": load_mask_from_s3(payload.medgemma_s3),
        "sam3":     load_mask_from_s3(payload.sam3_s3),
    }
    ensemble = majority_vote(list(masks.values()))

    metrics = {
        model: {"dice_vs_ensemble": compute_dice(mask, ensemble)}
        for model, mask in masks.items()
    }
    metrics["agreement_score"] = float(np.mean(
        [m["dice_vs_ensemble"] for m in metrics.values()]
    ))

    save_and_upload_mask(ensemble, payload.output_s3)
    return {"metrics": metrics, "ensemble_s3": payload.output_s3}
```

**Agreement score guide (surface in UI):**
```
≥ 0.90  ✅ High agreement – report automatically
0.75–0.89  ⚠️ Moderate – flag for radiologist review
< 0.75  🔴 Low agreement – require manual segmentation check
```

---

### Step 4.6 – Evaluation Framework (Which Model Wins?)

**What:** Log per-case model metrics so you can retrospectively analyse which model
performs best on your patient cohort.

**DB additions:**
```sql
CREATE TABLE model_evaluations (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    study_id     UUID REFERENCES studies(id),
    model        TEXT NOT NULL,           -- 'nnunet' | 'medgemma' | 'sam3'
    dice_score   FLOAT,
    iou_score    FLOAT,
    volume_cm3   FLOAT,
    runtime_s    FLOAT,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
```

**Admin dashboard page `/admin/model-stats`:** bar chart of mean Dice per model across all studies.

---

### Step 4.7 – Longitudinal Volume Comparison

**What:** After two studies are segmented (using ensemble mask), compare tumor volumes.

```python
# backend/app/services/comparison.py
import nibabel as nib
import numpy as np

def compute_volume_cm3(mask_nii: str) -> float:
    img = nib.load(mask_nii)
    voxel_vol_mm3 = float(np.prod(img.header.get_zooms()))
    n_voxels = int(np.count_nonzero(img.get_fdata()))
    return (n_voxels * voxel_vol_mm3) / 1000.0

def create_comparison(study_a_id, study_b_id, db):
    # uses ensemble mask by default; optionally per-model masks too
    vol_a = compute_volume_cm3(get_ensemble_mask(study_a_id))
    vol_b = compute_volume_cm3(get_ensemble_mask(study_b_id))
    delta = vol_b - vol_a
    pct   = (delta / vol_a * 100) if vol_a > 0 else 0
    comparison = Comparison(
        study_a=study_a_id, study_b=study_b_id,
        volume_a_cm3=vol_a, volume_b_cm3=vol_b,
        delta_cm3=delta, pct_change=pct
    )
    db.add(comparison); db.commit()
    return comparison
```

---

## Phase 5 – RAG Text Pipeline

### Step 5.1 – Document Ingestion

**What:** Clinical notes, pathology reports, physician notes → vector store.

**Flow:**
```
POST /patients/{id}/documents (multipart PDF/TXT) →
  1. Extract text (PyPDF2 / pdfplumber)
  2. Chunk text (LangChain RecursiveCharacterTextSplitter, chunk_size=500)
  3. Embed each chunk (OpenAI text-embedding-ada-002 or local HuggingFace)
  4. Store embeddings in per-patient ephemeral vector index (FAISS in-memory)
     – keys scoped to session token; flushed on session end
```

---

### Step 5.2 – RAG Query at Report Time

```python
# ml/rag/pipeline.py
def build_patient_context(patient_id: str, query: str) -> str:
    """
    Retrieve top-k relevant chunks from patient's FAISS index.
    Compose a context string for the LLM prompt.
    """
    index = load_patient_index(patient_id)
    docs  = index.similarity_search(query, k=5)
    return "\n\n".join([d.page_content for d in docs])

def generate_clinical_summary(patient_id, imaging_findings: dict) -> str:
    context = build_patient_context(patient_id, "patient history and current status")
    prompt = f"""
    You are an oncology assistant. Based on the following patient context and imaging findings,
    write a concise clinical summary.

    Patient Context:
    {context}

    Imaging Findings:
    {imaging_findings}

    Summary:
    """
    return call_llm(prompt)   # OpenAI / Mistral / local model
```

**Security:** FAISS index is created fresh per session, lives only in worker memory, and is GC'd after the session ends. No cross-patient contamination.

---

## Phase 6 – Report Generation

### Step 6.1 – PDF Generation

**What:** Combine imaging findings + RAG-generated summary into a structured PDF.

**Package:** `reportlab` or `WeasyPrint` (HTML-to-PDF).

**PDF contents:**
- Patient pseudonym & age/gender
- Scan dates (Study A vs Study B)
- Tumor volume table (vol_a, vol_b, delta, % change)
- Segmentation overlay images (axial/sagittal slices)
- Clinical summary (from RAG)
- Signature block

---

### Step 6.2 – Cryptographic Signing

**What:** Each finalized PDF is digitally signed.

```python
import hashlib
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

def sign_pdf(pdf_bytes: bytes, private_key) -> str:
    """Returns base64-encoded RSA signature."""
    signature = private_key.sign(pdf_bytes, padding.PKCS1v15(), hashes.SHA256())
    return signature.hex()

def verify_pdf(pdf_bytes: bytes, signature_hex: str, public_key) -> bool:
    sig = bytes.fromhex(signature_hex)
    try:
        public_key.verify(sig, pdf_bytes, padding.PKCS1v15(), hashes.SHA256())
        return True
    except Exception:
        return False
```

Store `signature` in `reports.signature`. Keys managed via AWS KMS.

---

## Phase 7 – Frontend

### Step 7.1 – Tech Stack Setup

```bash
cd frontend
npx create-react-app . --template typescript
# or
npm create vite@latest . -- --template react-ts
npm install axios react-router-dom @tanstack/react-query zustand
```

---

### Step 7.2 – Core Pages

| Route | Component | Description |
|-------|-----------|-------------|
| `/login` | `LoginPage` | JWT login + optional TOTP |
| `/dashboard` | `DoctorDashboard` | List of assigned patients |
| `/patients/:id` | `PatientPage` | Scan history, upload button, reports |
| `/patients/:id/studies` | `StudiesPage` | Scan timeline |
| `/reports/:id` | `ReportViewer` | PDF embed + metadata |
| `/admin` | `AdminPanel` | User management, patient assignment |

---

### Step 7.3 – Key UX Flows

**Upload Flow:**
```
PatientPage → "Upload Scan" button →
  FileInput (DICOM .dcm or NIfTI .nii.gz) →
  PUT /studies/upload →
  Progress indicator (WebSocket or polling /studies/{id}/status) →
  "Analysis complete" notification → redirect to ReportViewer
```

**Analysis Trigger:**
```
PatientPage → select Study A + Study B →
  POST /comparisons/trigger {study_a_id, study_b_id} →
  Celery task chain executes asynchronously →
  Status polling until done → auto-redirect to report
```

---

## Security Checklist

Work these into each phase above, not as an afterthought:

- [ ] **HTTPS/TLS 1.3** on all endpoints (AWS ALB or Nginx)
- [ ] **JWT short-lived** (15 min access / 7 day refresh) + Redis blocklist
- [ ] **2FA (TOTP)** – enforce for all `doctor` and above roles
- [ ] **ABAC middleware** – every patient data endpoint checks assignment
- [ ] **SHA-256 integrity** on every uploaded file; re-verify before segmentation
- [ ] **DICOM de-identification** before any processing/storage
- [ ] **PHI-free audit log** – log Actor, Action, Resource, Timestamp only
- [ ] **S3 server-side encryption** (AES-256 via AWS KMS)
- [ ] **RDS encryption at rest**
- [ ] **Private VPC** – DB and ML service not publicly routable
- [ ] **PDF cryptographic signing** before delivery
- [ ] **Container image scanning** in CI (Trivy or Snyk)

---

## Suggested Sprint Order

| Sprint | Deliverable |
|--------|-------------|
| 1 | Repo, Docker Compose (all 5 ML services), DB schema, Alembic migrations |
| 2 | Auth (login, JWT, RBAC), User & Patient CRUD |
| 3 | DICOM upload, de-identification, DICOM→NIfTI, S3 storage |
| 4a | nnU-Net service – inference endpoint, BraTS checkpoint, callback |
| 4b | MedGemma-1.5 service – HF model loading, slice-wise inference |
| 4c | SAM3 service – automatic + interactive inference endpoint |
| 4d | Ensemble service – majority vote, Dice metrics, agreement scoring |
| 4e | Celery orchestrator – parallel fan-out task group + evaluation logging |
| 5 | Volumetric comparison, longitudinal tracking (uses ensemble mask) |
| 6 | RAG pipeline (document ingestion + query) |
| 7 | PDF report generation (include per-model comparison table) + signing |
| 8 | React frontend – dashboard, patient page, upload UX, model selector |
| 9 | React frontend – report viewer, SAM3 interactive editor, admin/model-stats |
| 10 | Security hardening, CI/CD, staging deployment, QA |

---

## Key Dependencies (Python)

**Shared / Backend:**
```txt
fastapi>=0.110
uvicorn[standard]
sqlalchemy>=2.0
alembic
psycopg2-binary
celery[redis]
redis
boto3
pydicom
SimpleITK
nibabel
numpy
pyotp
python-jose[cryptography]
passlib[bcrypt]
reportlab
langchain
faiss-cpu          # or faiss-gpu
openai             # or local HuggingFace transformers
cryptography
```

**nnU-Net service:**
```txt
nnunetv2
torch>=2.2
```

**MedGemma-1.5 service:**
```txt
transformers>=4.40
accelerator
torch>=2.2
bitsandbytes       # optional: 4-bit quantisation to reduce VRAM
# HF_TOKEN env var required (gated model)
```

**SAM3 service:**
```txt
# pip install git+https://github.com/facebookresearch/sam3.git
torch>=2.2
opencv-python-headless
```

**Ensemble service:**
```txt
nibabel
numpy
scipy
simpleitk          # for STAPLE algorithm
```

## Key Dependencies (Node/React)

```
react, react-dom, react-router-dom
typescript
axios
@tanstack/react-query
zustand
@mui/material  # or antd / shadcn/ui
```

---

## Notes & Decisions to Revisit

1. **LLM choice for RAG** – OpenAI GPT-4o vs. local Mistral (data sovereignty).
2. **Image registration** – ANTsPy vs SimpleElastix for multi-scan alignment.
3. **nnU-Net checkpoint** – BraTS pretrained vs Ichilov fine-tuned; set `NNUNET_DATASET_ID` env var.
4. **MedGemma access** – Requires HuggingFace approval; apply at hf.co/google/medgemma.
   Fallback: use `LLaVA-Med` (open licence) while waiting for approval.
5. **SAM3 availability** – SAM3 package API may still be evolving; pin a commit hash in Dockerfile.
   Fallback: SAM2 with 3-D volume support works today.
6. **Ensemble strategy** – Start with majority vote (simple); upgrade to STAPLE or learned weights
   once you have ground-truth annotations from Ichilov radiologists.
7. **FAISS vs ChromaDB** – ChromaDB has persistence/multi-tenant support if needed.
8. **AWS vs on-prem** – The HLD specifies AWS; revisit if hospital data cannot leave premises.
9. **GPU budget** – Running three models in parallel requires significant VRAM; consider
   sequential inference on a single large GPU or three smaller GPU nodes.
