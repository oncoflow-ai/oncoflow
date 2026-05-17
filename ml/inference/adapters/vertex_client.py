import base64
import json
import logging
import os
import tempfile
from typing import Optional

import requests
import google.auth
from google.auth.transport.requests import Request

from ml.inference.adapters.base import AdapterResult, Bbox, empty_result
from ml.inference.io import Volume, load_nifti, save_nifti

logger = logging.getLogger(__name__)

class VertexClient:
    """Helper client to securely send Inference requests to Vertex AI Endpoints."""

    def __init__(self, project_id: str, region: str):
        self.project_id = project_id
        self.region = region
        try:
            self.credentials, _ = google.auth.default()
        except Exception as e:
            logger.warning(f"Failed to get Google default credentials: {e}")
            self.credentials = None

    def _get_access_token(self) -> str:
        if not self.credentials:
            raise RuntimeError("Google Cloud credentials are not initialized.")
        if not self.credentials.valid:
            self.credentials.refresh(Request())
        return self.credentials.token

    def predict(
        self, 
        endpoint_id: str, 
        vol: Volume, 
        roi: Optional[Bbox] = None,
        model_name: str = "unknown"
    ) -> AdapterResult:
        """Sends a serialized NIfTI volume to a Vertex AI Endpoint."""
        if not endpoint_id:
            return empty_result(vol.shape, error="Vertex endpoint ID not configured", model=model_name)

        # 1. Serialize NIfTI to bytes
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, "input.nii.gz")
            save_nifti(vol, temp_path)
            with open(temp_path, "rb") as f:
                nifti_bytes = f.read()

        b64_encoded = base64.b64encode(nifti_bytes).decode("utf-8")

        # 2. Build Payload
        instance_payload = {"b64": b64_encoded}
        if roi:
            instance_payload["roi"] = list(roi.as_tuple())
            
        payload = {
            "instances": [instance_payload]
        }

        # 3. Call Vertex API
        token = self._get_access_token()
        url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/endpoints/{endpoint_id}:predict"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        logger.info(f"Sending prediction request to Vertex Endpoint: {endpoint_id}")
        response = requests.post(url, headers=headers, json=payload, timeout=120)

        if response.status_code != 200:
            err_msg = f"Vertex API error {response.status_code}: {response.text[:200]}"
            logger.error(err_msg)
            return empty_result(vol.shape, error=err_msg, model=model_name)

        # 4. Deserialize Result
        try:
            resp_data = response.json()
            predictions = resp_data.get("predictions", [])
            if not predictions:
                return empty_result(vol.shape, error="No predictions returned from Vertex", model=model_name)
            
            pred_obj = predictions[0]
            out_b64 = pred_obj.get("b64")
            meta = pred_obj.get("meta", {})
            
            out_bytes = base64.b64decode(out_b64)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_out = os.path.join(tmpdir, "output.nii.gz")
                with open(temp_out, "wb") as f:
                    f.write(out_bytes)
                
                out_vol = load_nifti(temp_out)
                
                return {
                    "mask": out_vol.data,
                    "prob": None,
                    "runtime_s": 0.0,  # Could measure this across the API call
                    "meta": meta
                }
                
        except Exception as e:
            logger.exception("Failed to parse Vertex API response")
            return empty_result(vol.shape, error=f"Failed to parse response: {e}", model=model_name)
