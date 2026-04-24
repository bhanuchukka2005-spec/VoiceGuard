"""
FastAPI backend v2.1 — VoiceGuard
New in this version:
  - POST /predict/batch   → analyse multiple files in one request
  - GET  /stats           → session stats (total analysed, fake/real counts)
  - Improved health check (checks actual model file exists)
  - Confidence calibration note in response
"""

import os, sys, tempfile, time
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model.predict import predict

app = FastAPI(title="VoiceGuard API", version="2.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"]
)

ALLOWED = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_MB  = 20

# In-memory session stats (resets on server restart)
_stats = {"total": 0, "fake": 0, "real": 0, "errors": 0}


# ─── Schemas ──────────────────────────────────────────────────────────────────

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
    model_version:      str = "2.1.0"


class BatchItem(BaseModel):
    filename:   str
    label:      str
    confidence: float
    fake_score: float
    real_score: float
    error:      str = ""


class BatchResponse(BaseModel):
    results:            List[BatchItem]
    total_files:        int
    fake_count:         int
    real_count:         int
    error_count:        int
    total_time_ms:      float


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _verdict(label, fake_score, top_features):
    top = top_features[0]["name"] if top_features else "audio features"
    if label == "FAKE":
        return (f"Flagged as synthetic. Key signal: {top}. "
                f"Confidence {round(fake_score * 100)}% — patterns match AI-generated speech.")
    return (f"Likely authentic human speech. Key signal: {top}. "
            f"Confidence {round((1 - fake_score) * 100)}% — natural prosody consistent with real voice.")


async def _run_predict(tmp_path: str):
    """Run predict() in executor so it doesn't block the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, predict, tmp_path)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    model_path = os.path.join(os.path.dirname(__file__), "../model/detector.joblib")
    model_ok   = os.path.exists(model_path)
    return {
        "status":       "ok" if model_ok else "degraded",
        "model_loaded": model_ok,
        "version":      "2.1.0",
        "stats":        _stats,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Unsupported type '{ext}'. Allowed: {ALLOWED}")

    content  = await file.read()
    size_mb  = len(content) / (1024 * 1024)
    if size_mb > MAX_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB). Max: {MAX_MB} MB")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        t0     = time.time()
        result = await _run_predict(tmp_path)
        ms     = (time.time() - t0) * 1000
        _stats["total"] += 1
        _stats[result["label"].lower()] += 1
    except FileNotFoundError as e:
        _stats["errors"] += 1
        raise HTTPException(503, str(e))
    except Exception as e:
        _stats["errors"] += 1
        raise HTTPException(500, f"Inference error: {e}")
    finally:
        try: os.unlink(tmp_path)
        except: pass

    return PredictionResponse(
        **{k: v for k, v in result.items() if k != "top_features"},
        top_features       = result.get("top_features", []),
        processing_time_ms = round(ms, 1),
        filename           = file.filename or "unknown",
        verdict_reason     = _verdict(result["label"], result["fake_score"],
                                      result.get("top_features", [])),
    )


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    """Analyse up to 10 audio files in one request."""
    if len(files) > 10:
        raise HTTPException(400, "Max 10 files per batch request.")

    results = []
    fake_count = real_count = error_count = 0
    t_start = time.time()

    for file in files:
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in ALLOWED:
            results.append(BatchItem(
                filename=file.filename or "unknown", label="ERROR",
                confidence=0, fake_score=0, real_score=0,
                error=f"Unsupported file type '{ext}'"
            ))
            error_count += 1
            continue

        content = await file.read()
        if len(content) > MAX_MB * 1024 * 1024:
            results.append(BatchItem(
                filename=file.filename or "unknown", label="ERROR",
                confidence=0, fake_score=0, real_score=0,
                error="File too large"
            ))
            error_count += 1
            continue

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            r = await _run_predict(tmp_path)
            results.append(BatchItem(
                filename   = file.filename or "unknown",
                label      = r["label"],
                confidence = r["confidence"],
                fake_score = r["fake_score"],
                real_score = r["real_score"],
            ))
            if r["label"] == "FAKE": fake_count += 1
            else: real_count += 1
            _stats["total"] += 1
            _stats[r["label"].lower()] += 1
        except Exception as e:
            results.append(BatchItem(
                filename=file.filename or "unknown", label="ERROR",
                confidence=0, fake_score=0, real_score=0, error=str(e)
            ))
            error_count += 1
            _stats["errors"] += 1
        finally:
            try: os.unlink(tmp_path)
            except: pass

    return BatchResponse(
        results       = results,
        total_files   = len(files),
        fake_count    = fake_count,
        real_count    = real_count,
        error_count   = error_count,
        total_time_ms = round((time.time() - t_start) * 1000, 1),
    )


@app.get("/stats")
async def session_stats():
    """Return session-level analysis statistics."""
    total = _stats["total"] or 1  # avoid div by zero
    return {
        **_stats,
        "fake_rate": round(_stats["fake"] / total, 4),
        "real_rate": round(_stats["real"] / total, 4),
    }


@app.get("/model/info")
async def model_info():
    return {
        "model":          "SVM + GradientBoosting + XGBoost (soft-voting, v2)",
        "features":       "270-dim: MFCC×delta×delta2 + spectral + chroma + prosody",
        "input_formats":  list(ALLOWED),
        "max_file_mb":    MAX_MB,
        "max_batch_size": 10,
        "explainability": "Top-5 feature contributions per prediction",
        "dataset":        "ASVspoof 2019 LA / synthetic",
        "version":        "2.1.0",
    }