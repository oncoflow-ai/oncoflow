import base64
import os
import tempfile
import logging
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from ml.inference.config import InferenceConfig
from ml.inference.io import load_nifti, save_nifti, Volume
from ml.inference.adapters.base import build_adapter, Bbox

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OncoFlow Vertex AI Inference Server")

# Initialize the adapter globally so weights are loaded on startup
model_name = os.environ.get("OFLOW_MODEL_NAME", "nnunet").lower()
cfg = InferenceConfig(backend="local", device="cuda")
adapter = build_adapter(model_name, cfg)

@app.on_event("startup")
def load_model():
    logger.info(f"Loading model adapter for: {model_name}")
    try:
        adapter.load()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

@app.get("/health")
def health() -> Dict[str, str]:
    if adapter.is_available():
        return {"status": "healthy", "model": model_name}
    else:
        raise HTTPException(status_code=503, detail="Model not available")

@app.post("/predict")
async def predict(request: Request) -> JSONResponse:
    """
    Vertex AI Custom Prediction Routine endpoint.
    Expected payload format:
    {
      "instances": [
        {
          "b64": "<base64_encoded_nifti_bytes>",
          "roi": [xmin, ymin, zmin, xmax, ymax, zmax]  # optional
        }
      ]
    }
    """
    body = await request.json()
    if "instances" not in body or not body["instances"]:
        raise HTTPException(status_code=400, detail="Missing 'instances' in payload")
    
    instance = body["instances"][0]
    if "b64" not in instance:
        raise HTTPException(status_code=400, detail="Missing 'b64' field in instance")
    
    try:
        nifti_bytes = base64.b64decode(instance["b64"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {e}")

    roi = None
    if "roi" in instance and instance["roi"]:
        r = instance["roi"]
        roi = Bbox(r[0], r[1], r[2], r[3], r[4], r[5])

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.nii.gz")
        output_path = os.path.join(tmpdir, "output.nii.gz")

        with open(input_path, "wb") as f:
            f.write(nifti_bytes)
        
        # Load NIfTI and run inference
        vol = load_nifti(input_path)
        logger.info(f"Running inference on volume shape {vol.shape}")
        
        result = adapter.predict(vol, roi=roi)
        mask_vol = vol.copy_with(data=result["mask"], meta=result["meta"])
        
        # Save output mask and encode it
        save_nifti(mask_vol, output_path)
        with open(output_path, "rb") as f:
            out_bytes = f.read()
            
    out_b64 = base64.b64encode(out_bytes).decode("utf-8")
    
    return JSONResponse(content={
        "predictions": [
            {
                "b64": out_b64,
                "meta": result.get("meta", {})
            }
        ]
    })

if __name__ == "__main__":
    port = int(os.environ.get("AIP_HTTP_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
