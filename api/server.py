"""
FastAPI backend v2 — includes explainability fields in response.
"""

import os, sys, tempfile, time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model.predict import predict

app = FastAPI(title="VoiceGuard API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

ALLOWED = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


class FeatureContrib(BaseModel):
    name:   str
    key:    str
    value:  float
    weight: float


class PredictionResponse(BaseModel):
    label:              str
    confidence:         float
    fake_score:         float
    real_score:         float
    features:           int
    processing_time_ms: float
    filename:           str
    top_features:       List[FeatureContrib] = []
    verdict_reason:     str = ""


def _verdict_reason(label, fake_score, top_features):
    if not top_features:
        return ""
    top = top_features[0]["name"] if top_features else "audio features"
    if label == "FAKE":
        return (f"Flagged as synthetic. Key signal: {top}. "
                f"Confidence {round(fake_score*100)}% — characteristic patterns "
                f"match AI-generated speech from the training corpus.")
    return (f"Likely authentic human speech. Key signal: {top}. "
            f"Confidence {round((1-fake_score)*100)}% — natural prosody and "
            f"spectral variation consistent with real voice.")


@app.get("/health")
async def health():
    return {"status": "ok",
            "model_loaded": os.path.exists("model/detector.joblib")}


@app.post("/predict", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Unsupported type '{ext}'")

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 20 MB)")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content); tmp_path = tmp.name

    try:
        t0     = time.time()
        result = predict(tmp_path)
        ms     = (time.time() - t0) * 1000
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")
    finally:
        os.unlink(tmp_path)

    return PredictionResponse(
        **{k: v for k, v in result.items() if k != "top_features"},
        top_features     = result.get("top_features", []),
        processing_time_ms = round(ms, 1),
        filename           = file.filename or "unknown",
        verdict_reason     = _verdict_reason(
            result["label"], result["fake_score"],
            result.get("top_features", [])
        ),
    )


@app.get("/model/info")
async def model_info():
    return {
        "model":    "SVM + GradientBoosting + XGBoost (soft-voting ensemble, v2)",
        "features": "270-dim: MFCC×delta×delta2 + spectral + chroma + prosody",
        "dataset":  "ASVspoof 2019 LA",
        "explainability": "Top-5 feature contributions returned per prediction",
    }