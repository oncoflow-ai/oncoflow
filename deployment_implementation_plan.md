# OncoFlow GCP Deployment Plan (MVP)

This plan details the migration and deployment architecture of the OncoFlow application to **Google Cloud Platform (GCP)**, focusing only on the core "must-have" services for now. It specifically addresses the integration of **Vertex AI** for hosting ML models and the **Google Cloud Healthcare API** for robust DICOM storage.

## Proposed GCP Architecture & Tools

### 1. Data Ingestion & Storage (Cloud Healthcare API & GCS)
Instead of manually stripping PHI from DICOM files and storing them locally, we will leverage the **Google Cloud Healthcare API**.
*   **DICOM Store:** Stores uploaded MRI scans using the DICOMweb standard.
*   **Google Cloud Storage (GCS):** Used for storing intermediate NIfTI files, derived segmentation masks, and final generated PDF reports.

### 2. Machine Learning Inference (Vertex AI)
To centralize model management and scaling, all ML models will be deployed using **Vertex AI Endpoints**:
*   **MedGemma on Vertex AI:** MedGemma (vision-language model) will be deployed as a managed endpoint from the Vertex AI Model Garden.
*   **Panel of Experts (nnU-Net v1 & SAM3):** These custom models will be packaged into custom Docker containers and deployed to Vertex AI Endpoints.

### 3. Application Backend (Cloud Run & Managed DBs)
*   **API Gateway / Backend (FastAPI):** Deployed to **Cloud Run**, Google's serverless container platform. It handles all HTTP traffic. Authentication will remain the custom JWT-based authentication currently implemented in FastAPI.
*   **Asynchronous Workers (Celery):** Deployed as Cloud Run services to handle the orchestration of the image processing pipeline.
*   **Database:** **Cloud SQL for PostgreSQL** replaces the local Postgres database for managing users, patient metadata, job states, and comparisons.
*   **Message Broker:** **Memorystore for Redis** replaces the local Redis instance for Celery queues.

---

## Implementation Phases

### Phase 1: MVP Infrastructure Setup
*   Provision GCP Project and enable required APIs (Compute, Cloud Run, Vertex AI, Healthcare, SQL, Redis).
*   Set up Cloud SQL (Postgres) and Memorystore (Redis) using default networking (no custom VPCs for now).
*   Create GCS Buckets (`oncoflow-nifti`, `oncoflow-masks`, `oncoflow-reports`).
*   Create Cloud Healthcare API Dataset and DICOM store.

### Phase 2: Vertex AI Integration
*   Deploy the MedGemma-1.5 model from the Vertex AI Model Garden to an Endpoint.
*   Package the nnU-Net v1 adapter and SAM3 into custom Vertex AI prediction containers and deploy them to Endpoints.
*   Update `ml/inference` backend logic to make asynchronous REST calls to Vertex AI Endpoints.

### Phase 3: Healthcare API & Backend Refactor
*   Refactor the upload endpoints to push DICOMs directly to the Healthcare API DICOM store.
*   Update the Celery workers to pull DICOM instances from the Healthcare API, convert them to NIfTI, and upload to GCS for inference.

### Phase 4: CI/CD & Frontend Deployment
*   Use **Cloud Build** to automate building Docker images and pushing them to **Artifact Registry**.
*   Deploy the React frontend to **Firebase Hosting** or a public-facing Cloud Run instance.
