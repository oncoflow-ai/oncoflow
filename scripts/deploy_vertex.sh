#!/bin/bash
set -e

PROJECT_ID="oncoflow-496517"
REGION="us-central1"
REPO_NAME="oncoflow-ml"

echo "=== OncoFlow Vertex AI Deployment Script ==="

# 1. Create Artifact Registry Repository (if not exists)
echo "[1/5] Creating Artifact Registry..."
gcloud artifacts repositories create $REPO_NAME \
    --project=$PROJECT_ID \
    --repository-format=docker \
    --location=$REGION \
    --description="Docker repository for OncoFlow ML models" \
    --quiet || echo "Repository may already exist."

# 2. Build and Push Custom Container (nnU-Net) using Cloud Build
echo "[2/4] Building Docker image natively on GCP via Cloud Build (much faster)..."
NNUNET_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/nnunet:latest"

gcloud builds submit --config cloudbuild.yaml \
    --substitutions=_NNUNET_IMAGE=$NNUNET_IMAGE \
    --project=$PROJECT_ID \
    .

# 3. Upload Models to Vertex AI
echo "[3/5] Uploading models to Vertex AI..."

# Upload nnU-Net Custom Model
gcloud ai models upload \
  --project=$PROJECT_ID \
  --region=$REGION \
  --display-name="oncoflow-nnunet" \
  --container-image-uri=$NNUNET_IMAGE \
  --container-health-route="/health" \
  --container-predict-route="/predict" \
  --container-ports=8080

# Note: For MedGemma, it is deployed directly from the Model Garden UI.
echo "NOTE: Please deploy MedGemma-1.5 directly from the Vertex AI Model Garden in the GCP Console."

# 4. Create Endpoint
echo "[4/4] Creating Vertex AI Endpoint..."
gcloud ai endpoints create --project=$PROJECT_ID --region=$REGION --display-name="endpoint-nnunet" --quiet

# 5. Instructions for Deployment
echo "[DONE] Next Steps:"
echo "1. Run 'gcloud ai models list --region=$REGION' to get your MODEL_IDs."
echo "2. Run 'gcloud ai endpoints list --region=$REGION' to get your ENDPOINT_IDs."
echo "3. Deploy the models to the endpoints using 'gcloud ai endpoints deploy-model'."
echo "   Example:"
echo "   gcloud ai endpoints deploy-model <ENDPOINT_ID> \\"
echo "     --region=$REGION \\"
echo "     --model=<MODEL_ID> \\"
echo "     --display-name=nnunet-deployment \\"
echo "     --machine-type=n1-standard-4 \\"
echo "     --accelerator=type=nvidia-tesla-t4,count=1"
echo ""
echo "4. Update ml/inference/config.py with the generated Endpoint IDs!"
