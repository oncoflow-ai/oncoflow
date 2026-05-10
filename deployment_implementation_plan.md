# OncoFlow GCP Deployment Plan

This plan details the migration and deployment architecture of the OncoFlow application from a local/AWS-oriented design to **Google Cloud Platform (GCP)**. It specifically addresses the integration of **Vertex AI** for hosting all ML models and the **Google Cloud Healthcare API** for robust, compliant DICOM storage and de-identification.

## Proposed GCP Architecture & Tools

### 1. Data Ingestion & Storage (Cloud Healthcare API & GCS)
Instead of manually stripping PHI from DICOM files and storing them in S3, we will leverage the **Google Cloud Healthcare API**.
*   **DICOM Store:** Stores uploaded MRI scans using the DICOMweb standard.
*   **De-identification:** The Healthcare API natively integrates with Cloud Data Loss Prevention (DLP) to automatically scrub PHI (Patient Names, IDs) upon ingestion, creating a secure, de-identified DICOM dataset.
*   **Google Cloud Storage (GCS):** Used for storing intermediate NIfTI files, derived segmentation masks, and final generated PDF reports.

### 2. Machine Learning Inference (Vertex AI)
To centralize model management and scaling, all ML models will be deployed using **Vertex AI Endpoints**:
*   **MedGemma on Vertex AI:** MedGemma (vision-language model) will be deployed as a managed endpoint from the Vertex AI Model Garden. This allows dynamic scaling of GPUs (e.g., NVIDIA L4 or A100) and provides a secure REST API for the backend to query.
*   **Panel of Experts (nnU-Net v1 & SAM3):** These custom models will be packaged into custom Docker containers and deployed to Vertex AI Endpoints as custom models. This standardizes the ML architecture across the entire panel.

### 3. Application Backend (Cloud Run & Cloud SQL)
*   **API Gateway / Backend (FastAPI):** Deployed to **Cloud Run**, Google's serverless container platform. It scales automatically to zero and handles all HTTP traffic. **Authentication will remain the custom JWT-based authentication currently implemented in FastAPI.**
*   **Asynchronous Workers (Celery):** Deployed as continuous Cloud Run services or Google Kubernetes Engine (GKE) pods to handle the orchestration of the image processing pipeline.
*   **Database:** **Cloud SQL for PostgreSQL** replaces the local Postgres database for managing users, patient metadata, job states, and comparisons.
*   **Message Broker:** **Memorystore for Redis** replaces the local Redis instance for Celery queues.

### 4. Security & Compliance
*   **Cloud KMS:** Manages encryption keys for data at rest (GCS and SQL).
*   **Cloud IAM:** Enforces least-privilege access between Cloud Run services, Vertex AI, and the Healthcare API.
*   **VPC Service Controls:** Isolates the database and ML endpoints from the public internet.

---

## Implementation Phases

### Phase 1: Infrastructure Setup (Terraform)
*   Provision GCP Project, VPC, and private subnets.
*   Set up Cloud SQL (Postgres) and Memorystore (Redis).
*   Create GCS Buckets (`oncoflow-nifti`, `oncoflow-masks`, `oncoflow-reports`).
*   Create Cloud Healthcare API Dataset and DICOM store.

### Phase 2: Vertex AI Integration
*   Deploy the MedGemma-1.5 model from the Vertex AI Model Garden to an Endpoint.
*   Package the nnU-Net v1 adapter and SAM3 into custom Vertex AI prediction containers and deploy them to Endpoints.
*   Update `ml/inference` backend logic to make asynchronous REST calls to Vertex AI Endpoints instead of using local subprocesses/containers.

### Phase 3: Healthcare API & Backend Refactor
*   Refactor the upload endpoints to push DICOMs directly to the Healthcare API DICOM store.
*   Implement the automated de-identification pipeline within the Healthcare API.
*   Update the Celery workers to pull DICOM instances from the Healthcare API, convert them to NIfTI, and upload to GCS for inference.

### Phase 4: CI/CD & Frontend Deployment
*   Use **Cloud Build** to automate building Docker images and pushing them to **Artifact Registry**.
*   Deploy the React frontend to **Firebase Hosting** or a public-facing Cloud Run instance.
