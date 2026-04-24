"""
FastAPI backend for Audio Deepfake Detector.

Endpoints:
  GET  /              → health check
  GET  /health        → health check JSON
  POST /predict       → accepts audio file, returns prediction
  GET  /model/info    → model metadata

Run:
  uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import tempfile
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model.predict import predict

app = FastAPI(
    title="Audio Deepfake Detector",
    description="Detects AI-cloned or synthetic speech in audio files.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_FILE_SIZE_MB = 20


# ─── Response schemas ─────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    fake_score: float
    real_score: float
    features: int
    processing_time_ms: float
    filename: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health():
    model_ready = os.path.exists(
        os.path.join(os.path.dirname(__file__), "../model/detector.joblib")
    )
    return HealthResponse(status="ok", model_loaded=model_ready)


@app.post("/predict", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...)):
    # Validate extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}"
        )

    # Read + size check
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB"
        )

    # Write to temp file and run inference
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        start = time.time()
        result = predict(tmp_path)
        elapsed_ms = (time.time() - start) * 1000
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    finally:
        os.unlink(tmp_path)

    return PredictionResponse(
        **result,
        processing_time_ms=round(elapsed_ms, 1),
        filename=file.filename or "unknown",
    )


@app.get("/model/info")
async def model_info():
    return {
        "model": "SVM + GradientBoosting Ensemble",
        "features": "MFCC (40 coeffs × delta × delta2) + spectral + chroma + prosody",
        "input": "Audio files up to 20MB (.wav, .mp3, .flac, .ogg, .m4a)",
        "output": "label (REAL/FAKE), confidence, fake_score, real_score",
        "dataset": "ASVspoof 2019 LA / synthetic",
        "framework": "scikit-learn 1.x",
    }