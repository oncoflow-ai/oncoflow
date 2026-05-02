# OncoFlow Backend Inference Service

The backend exposes the application-facing API for study ingestion, asynchronous
processing, segmentation, and result retrieval. Segmentation is delegated to the
`ml.inference` package through the backend runner abstraction.

## Local Development

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e "backend[dev,ml]"
.venv/bin/python -m pip install -r ml/inference/requirements.txt
```

Run the API with the repository root on `PYTHONPATH` so both `backend/app` and
`ml.inference` are importable:

```bash
cd backend
PYTHONPATH=".:.." ../.venv/bin/uvicorn app.main:app --reload
```

Useful local settings:

```bash
export ONCOFLOW_DATABASE_URL="sqlite+pysqlite:///./var/oncoflow/dev.sqlite3"
export ONCOFLOW_STORAGE_ROOT="./var/oncoflow"
export ONCOFLOW_JOB_EXECUTION_MODE="threaded"
export OFLOW_BACKEND="local"
export OFLOW_DEVICE="auto"
export OFLOW_ENABLED_MODELS="nnunet"  # base-models only; full panel is "nnunet,medgemma,sam3"
export OFLOW_CACHE_DIR="./var/oncoflow/cache"
```

When model dependencies or weights are missing, the service remains available
but inference readiness is reported as degraded and adapter warnings are stored
with the result metadata.

## GPU Production Mode

Provision a GPU Python environment with the inference dependencies and model
weights, then configure:

```bash
export OFLOW_BACKEND="gpu-prod"
export OFLOW_DEVICE="cuda"
export OFLOW_WEIGHTS_DIR="/models/oncoflow"
export OFLOW_CACHE_DIR="/var/cache/oncoflow"
export nnUNet_results="/models/nnunet/results"
export nnUNet_raw="/models/nnunet/raw"
export nnUNet_preprocessed="/models/nnunet/preprocessed"
export HF_TOKEN="<token-for-medgemma-if-used>"
```

Enable nnU-Net first, then add SAM and MedGemma once their packages and weights
are installed:

```bash
export OFLOW_ENABLED_MODELS="nnunet"
# later: export OFLOW_ENABLED_MODELS="nnunet,sam3,medgemma"
```

## API Flow

1. `POST /api/v1/jobs/mri-ingestion` with a study archive.
2. Poll `GET /api/v1/jobs/{job_id}` until `completed` or `failed`.
3. Fetch structured results from `GET /api/v1/results/{study_id}`.
4. Check service and model readiness with `GET /api/v1/ready`.

`GET /api/v1/ready` includes the configured inference backend, enabled models,
dependency gaps, resolved device, and per-adapter availability.
