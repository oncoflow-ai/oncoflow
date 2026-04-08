# External Integrations

**Analysis Date:** 2026-04-08

## APIs & External Services

**Backend API (Planned):**
- FastAPI backend - Planned but not yet implemented
  - SDK/Client: axios via `frontend/src/api/client.ts`
  - Base URL: Configured via `VITE_API_URL` environment variable
  - Default: `http://localhost:8000`
  - Auth: JWT bearer token (stubbed in `apiClient` interceptor, currently commented out)
  - Current state: Mock data layer in frontend (`frontend/src/api/*.ts`)

**ML Model Providers:**
- HuggingFace Transformers - MedGemma-1.5 / LLaVA-Med inference
  - SDK/Client: `transformers` Python library (4.47+)
  - Auth: `HF_TOKEN` environment variable (optional, for gated models)
  - Usage: Medical image analysis in `ml/exploration/` notebooks
  - Location: Notebook 03 (`03_medgemma_exploration.ipynb`)

**Medical Imaging Resources:**
- Mock DICOM endpoint - Development placeholder
  - URL: `https://mock-dicom.oncoflow.internal/{scanId}.nii.gz` (in `frontend/src/api/mri.ts`)
  - Purpose: Simulated MRI scan retrieval for frontend prototyping
  - Current state: Returns mock URL structure only

## Data Storage

**Databases:**
- PostgreSQL (Planned)
  - Connection: Not yet configured
  - Client: Not yet implemented
  - Purpose: Patient records, scan metadata, physician data
  - State: Documented in `IMPLEMENTATION_PLAN.md` Phase 1

**File Storage:**
- Local filesystem only (Current)
  - ML notebooks: `data/P01/` directory for patient DICOM/NIfTI samples
  - Frontend: No file upload implementation yet
  - Planned: AWS S3 for DICOM/NIfTI storage (per implementation plan)

**Caching:**
- Redis (Planned)
  - Purpose: Celery task queue backend for async ML processing
  - State: Documented in `IMPLEMENTATION_PLAN.md`, not yet implemented

## Authentication & Identity

**Auth Provider:**
- Custom (Planned)
  - Implementation: JWT bearer tokens with FastAPI backend
  - Current state: Mock authentication in `frontend/src/store/authStore.ts`
  - Mock accepts any non-empty credentials
  - Storage: sessionStorage via Zustand persist middleware
  - Key: `oncoflow_auth`
  - Planned features: 2FA, role-based access control (RBAC) for physician/admin roles

## Monitoring & Observability

**Error Tracking:**
- None configured

**Logs:**
- Console-based logging only
  - Frontend: Browser console
  - ML: Python `logging` module in utility scripts (`ml/exploration/utils/*.py`)

## CI/CD & Deployment

**Hosting:**
- Not yet deployed
  - Planned: AWS infrastructure (per `IMPLEMENTATION_PLAN.md`)
  - Current: Local development only

**CI Pipeline:**
- None configured
  - Directory exists: `.github/` but no workflow files present

## Environment Configuration

**Required env vars:**

**Frontend (Vite):**
- `VITE_API_URL` - Backend API base URL (optional, defaults to `http://localhost:8000`)

**ML Exploration (Python):**
- `HF_TOKEN` - HuggingFace API token for MedGemma access (optional, falls back to LLaVA-Med)
- `nnUNet_raw` - nnU-Net raw data directory path
- `nnUNet_preprocessed` - nnU-Net preprocessed data directory path  
- `nnUNet_results` - nnU-Net model results directory path

**Secrets location:**
- Frontend: `frontend/.env.local` (gitignored)
- ML: Not configured (set manually in shell or notebook environment)

## Webhooks & Callbacks

**Incoming:**
- None implemented

**Outgoing:**
- None implemented

## ML Model Integrations

**Segmentation Models (Research/Exploration):**

**nnU-Net v2:**
- Type: Self-hosted inference
- Location: `ml/exploration/notebooks/02_nnunet_exploration.ipynb`
- Purpose: Medical image tumor segmentation
- Installation: `pip install nnunetv2>=2.4`
- State: Exploration phase, dataset structure planned

**SAM3 / SAM2 (Meta):**
- Type: Self-hosted inference
- Location: `ml/exploration/notebooks/04_sam3_exploration.ipynb`
- Purpose: Segment Anything model for medical imaging
- Installation: `pip install git+https://github.com/facebookresearch/sam3.git`
- Fallback: SAM2 if SAM3 unavailable
- State: Exploration phase

**MedGemma-1.5 / LLaVA-Med:**
- Type: HuggingFace Transformers model
- Location: `ml/exploration/notebooks/03_medgemma_exploration.ipynb`
- Purpose: Medical vision-language model for slice-wise analysis
- State: Exploration phase with HF token gating

## Data Format Standards

**Medical Imaging:**
- DICOM - Clinical acquisition and storage format
  - Library: `pydicom` 2.4+
  - Location: `ml/exploration/utils/dicom_utils.py`
- NIfTI - Neuroimaging analysis format
  - Libraries: `nibabel` 5.2+, `SimpleITK` 2.3+
  - Conversion: DICOM→NIfTI via SimpleITK or dcm2niix subprocess
  - Usage: All ML model inference operates on NIfTI format

## Planned Integrations (Not Yet Implemented)

**Based on `IMPLEMENTATION_PLAN.md`:**

- FastAPI backend (Python)
- PostgreSQL database
- Redis + Celery for async task processing
- AWS S3 for medical image storage
- ML inference microservices (ports 8001-8004):
  - nnU-Net inference service
  - MedGemma inference service
  - SAM3 inference service
  - Ensemble/voting service
- Vector database for RAG pipeline (clinical notes)
- PDF generation service with cryptographic signing

---

*Integration audit: 2026-04-08*
