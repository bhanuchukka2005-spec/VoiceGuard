"""
FastAPI backend v2.3 — VoiceGuard
"""

import os
import sys
import tempfile
import time
import mimetypes
import asyncio
import logging
from pathlib import Path
from typing import List
from urllib.request import urlretrieve
from urllib.error import URLError

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Fix import path so predict.py is always found
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "model"))

from model.predict import predict, predict_segments  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voiceguard")

# ── Rate limiter ───────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="VoiceGuard API", version="2.3.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ───────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000,http://localhost:5500",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Audio type detection ───────────────────────────────────────────────────────
_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus",
    ".wma", ".aiff", ".aif", ".au", ".ra", ".amr", ".webm",
    ".mp4", ".3gp", ".caf", ".gsm",
}
MAX_MB = 50


def _is_audio(filename: str, content: bytes) -> bool:
    ext = Path(filename).suffix.lower()
    if ext in _AUDIO_EXTENSIONS:
        return True
    mime, _ = mimetypes.guess_type(filename)
    if mime and mime.startswith("audio/"):
        return True
    sigs = {
        b"RIFF": True, b"fLaC": True,
        b"\xff\xfb": True, b"\xff\xf3": True, b"\xff\xf2": True,
        b"ID3": True, b"OggS": True, b"\x1aE\xdf\xa3": True,
    }
    header = content[:4]
    return any(header.startswith(sig) for sig in sigs)


# ── Thread-safe in-memory session stats ───────────────────────────────────────
_stats: dict = {"total": 0, "fake": 0, "real": 0, "errors": 0}
_stats_lock = asyncio.Lock()


async def _inc_label(label: str):
    async with _stats_lock:
        key = label.lower()
        if key in _stats:
            _stats[key] += 1
        _stats["total"] += 1


async def _inc_errors():
    async with _stats_lock:
        _stats["errors"] += 1


# ── Schemas ────────────────────────────────────────────────────────────────────

class FeatureContrib(BaseModel):
    name: str
    key: str
    value: float
    weight: float


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    fake_score: float
    real_score: float
    features: int
    processing_time_ms: float
    filename: str
    top_features: List[FeatureContrib] = []
    verdict_reason: str = ""
    model_version: str = "2.3.0"


class BatchItem(BaseModel):
    filename: str
    label: str
    confidence: float
    fake_score: float
    real_score: float
    error: str = ""


class BatchResponse(BaseModel):
    results: List[BatchItem]
    total_files: int
    fake_count: int
    real_count: int
    error_count: int
    total_time_ms: float


# ── Helpers ────────────────────────────────────────────────────────────────────

def _verdict(label: str, fake_score: float, top_features: list) -> str:
    top = top_features[0]["name"] if top_features else "audio features"
    if label == "FAKE":
        return (
            f"Flagged as synthetic. Key signal: {top}. "
            f"Confidence {round(fake_score * 100)}% — patterns match AI-generated speech."
        )
    return (
        f"Likely authentic human speech. Key signal: {top}. "
        f"Confidence {round((1 - fake_score) * 100)}% — natural prosody consistent with real voice."
    )


async def _run_predict(tmp_path: str) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, predict, tmp_path)


async def _save_upload(file: UploadFile) -> tuple:
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB). Max: {MAX_MB} MB")
    ext = Path(file.filename or "audio.wav").suffix.lower() or ".wav"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        return content, tmp.name


def _compute_stress(audio_path: str) -> dict:
    """Derive 5 biometric stress indicators from audio features."""
    import numpy as np
    from model.predict import _load
    from model.features import extract_with_names

    _load()

    vec, named = extract_with_names(audio_path)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    f0_std = float(named.get("f0_std", 0.0))
    pitch_stability = float(np.clip(f0_std / 40.0, 0.0, 1.0))

    dmfcc_std_avg = float(np.mean([
        named.get(f"dmfcc_{i}_std", 0.0) for i in range(5)
    ]))
    rhythm_naturalness = float(np.clip(dmfcc_std_avg / 8.0, 0.0, 1.0))

    voiced_ratio = float(named.get("voiced_ratio", 1.0))
    breath_patterns = float(np.clip(1.0 - abs(voiced_ratio - 0.7) / 0.7, 0.0, 1.0))

    mfcc_std_avg = float(np.mean([
        named.get(f"mfcc_{i}_std", 0.0) for i in range(10)
    ]))
    micro_variations = float(np.clip(mfcc_std_avg / 20.0, 0.0, 1.0))

    sc = float(named.get("spectral_centroid", 0.0))
    sb = float(named.get("spectral_bandwidth", 0.0))
    formant_naturalness = float(np.clip(sb / max(sc, 1.0), 0.0, 1.0))

    return {
        "pitch_stability": round(pitch_stability, 3),
        "rhythm_naturalness": round(rhythm_naturalness, 3),
        "breath_patterns": round(breath_patterns, 3),
        "micro_variations": round(micro_variations, 3),
        "formant_stability": round(formant_naturalness, 3),
        "is_demo": False,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    model_path = BASE_DIR / "model" / "detector.joblib"
    model_ok = model_path.exists()
    return JSONResponse(status_code=200, content={
        "status": "ok" if model_ok else "degraded",
        "model_loaded": model_ok,
        "model_path": str(model_path),
        "version": "2.3.0",
        "stats": _stats,
    })


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("30/minute")
async def predict_single(request: Request, file: UploadFile = File(...)):
    content, tmp_path = await _save_upload(file)

    if not _is_audio(file.filename or "", content):
        os.unlink(tmp_path)
        raise HTTPException(
            400,
            "File does not appear to be audio. "
            "Supported: wav, mp3, flac, ogg, m4a, aac, opus, wma, aiff, webm, mp4, 3gp, and more.",
        )

    try:
        t0 = time.perf_counter()
        result = await _run_predict(tmp_path)
        ms = (time.perf_counter() - t0) * 1000
        await _inc_label(result["label"])
    except FileNotFoundError as exc:
        await _inc_errors()
        raise HTTPException(503, str(exc))
    except ValueError as exc:
        await _inc_errors()
        raise HTTPException(422, str(exc))
    except Exception as exc:
        await _inc_errors()
        logger.exception("Inference error on %s", file.filename)
        raise HTTPException(500, f"Inference error: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    top_feats = result.get("top_features", [])
    return PredictionResponse(
        label=result["label"],
        confidence=result["confidence"],
        fake_score=result["fake_score"],
        real_score=result["real_score"],
        features=result["features"],
        processing_time_ms=round(ms, 1),
        filename=file.filename or "unknown",
        top_features=[FeatureContrib(**f) for f in top_feats],
        verdict_reason=_verdict(result["label"], result["fake_score"], top_feats),
    )


@app.post("/predict/batch", response_model=BatchResponse)
@limiter.limit("10/minute")
async def predict_batch(request: Request, files: List[UploadFile] = File(...)):
    if len(files) > 10:
        raise HTTPException(400, "Max 10 files per batch request.")

    results: List[BatchItem] = []
    fake_count = real_count = error_count = 0
    t_start = time.perf_counter()

    for file in files:
        try:
            content, tmp_path = await _save_upload(file)
        except HTTPException as exc:
            results.append(BatchItem(
                filename=file.filename or "unknown",
                label="ERROR",
                confidence=0,
                fake_score=0,
                real_score=0,
                error=exc.detail,
            ))
            error_count += 1
            await _inc_errors()
            continue

        if not _is_audio(file.filename or "", content):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            results.append(BatchItem(
                filename=file.filename or "unknown",
                label="ERROR",
                confidence=0,
                fake_score=0,
                real_score=0,
                error="Not a recognised audio file",
            ))
            error_count += 1
            await _inc_errors()
            continue

        try:
            r = await _run_predict(tmp_path)
            results.append(BatchItem(
                filename=file.filename or "unknown",
                label=r["label"],
                confidence=r["confidence"],
                fake_score=r["fake_score"],
                real_score=r["real_score"],
            ))
            if r["label"] == "FAKE":
                fake_count += 1
            else:
                real_count += 1
            await _inc_label(r["label"])
        except Exception as exc:
            logger.warning("Batch item %s failed: %s", file.filename, exc)
            results.append(BatchItem(
                filename=file.filename or "unknown",
                label="ERROR",
                confidence=0,
                fake_score=0,
                real_score=0,
                error=str(exc),
            ))
            error_count += 1
            await _inc_errors()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return BatchResponse(
        results=results,
        total_files=len(files),
        fake_count=fake_count,
        real_count=real_count,
        error_count=error_count,
        total_time_ms=round((time.perf_counter() - t_start) * 1000, 1),
    )


@app.get("/stats")
async def session_stats():
    async with _stats_lock:
        snapshot = dict(_stats)
    total = snapshot["total"] or 1
    return {
        **snapshot,
        "fake_rate": round(snapshot["fake"] / total, 4),
        "real_rate": round(snapshot["real"] / total, 4),
    }


@app.get("/model/info")
async def model_info():
    return {
        "model": "SVM + GradientBoosting + XGBoost (soft-voting, v2)",
        "features": "270-dim: MFCC×delta×delta2 + spectral + chroma + prosody",
        "input_formats": "wav, mp3, flac, ogg, m4a, aac, opus, wma, aiff, webm, mp4, 3gp, caf …",
        "max_file_mb": MAX_MB,
        "max_batch_size": 10,
        "explainability": "Top-5 feature contributions per prediction",
        "dataset": "ASVspoof 2019 LA / synthetic",
        "version": "2.3.0",
    }


@app.post("/predict/segments")
@limiter.limit("20/minute")
async def predict_temporal(request: Request, file: UploadFile = File(...)):
    content, tmp_path = await _save_upload(file)

    if not _is_audio(file.filename or "", content):
        os.unlink(tmp_path)
        raise HTTPException(400, "File does not appear to be audio.")

    try:
        t0 = time.perf_counter()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, predict_segments, tmp_path)
        ms = (time.perf_counter() - t0) * 1000
        result["processing_time_ms"] = round(ms, 1)
        result["filename"] = file.filename or "unknown"
        await _inc_label(result["overall"]["label"])
    except Exception as exc:
        await _inc_errors()
        logger.exception("Segment analysis error on %s", file.filename)
        raise HTTPException(500, f"Segment analysis error: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return result


@app.post("/predict/compare")
@limiter.limit("15/minute")
async def predict_compare(
    request: Request,
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
):
    results = {}
    for label, file in [("a", file_a), ("b", file_b)]:
        content, tmp_path = await _save_upload(file)
        if not _is_audio(file.filename or "", content):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise HTTPException(400, f"File {label.upper()} does not appear to be audio.")

        try:
            t0 = time.perf_counter()
            r = await _run_predict(tmp_path)
            ms = (time.perf_counter() - t0) * 1000
            top_feats = r.get("top_features", [])
            results[label] = {
                "label": r["label"],
                "confidence": r["confidence"],
                "fake_score": r["fake_score"],
                "real_score": r["real_score"],
                "features": r["features"],
                "processing_time_ms": round(ms, 1),
                "filename": file.filename or "unknown",
                "top_features": top_feats,
                "verdict_reason": _verdict(r["label"], r["fake_score"], top_feats),
            }
            await _inc_label(r["label"])
        except Exception as exc:
            await _inc_errors()
            logger.exception("Compare error on file %s", label.upper())
            raise HTTPException(500, f"Compare error on file {label.upper()}: {exc}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return results


@app.post("/predict/stress")
@limiter.limit("20/minute")
async def predict_stress(request: Request, file: UploadFile = File(...)):
    content, tmp_path = await _save_upload(file)

    if not _is_audio(file.filename or "", content):
        os.unlink(tmp_path)
        raise HTTPException(400, "File does not appear to be audio.")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _compute_stress, tmp_path)
    except Exception as exc:
        await _inc_errors()
        logger.exception("Stress analysis error on %s", file.filename)
        raise HTTPException(500, f"Stress analysis error: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return result


@app.post("/predict/url")
@limiter.limit("10/minute")
async def predict_from_url(
    request: Request,
    url: str = Query(..., description="Direct URL to an audio file"),
):
    """Fetch audio from a URL and run deepfake detection."""
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "URL must start with http:// or https://")

    ext = Path(url.split("?")[0]).suffix.lower() or ".wav"
    if ext not in _AUDIO_EXTENSIONS:
        ext = ".wav"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, urlretrieve, url, tmp_path)
        except (URLError, ValueError) as exc:
            raise HTTPException(400, f"Could not fetch URL: {exc}")

        with open(tmp_path, "rb") as f:
            content = f.read(8)
        if not _is_audio(ext, content):
            raise HTTPException(400, "URL does not appear to point to an audio file.")

        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        if size_mb > MAX_MB:
            raise HTTPException(413, f"Remote file too large ({size_mb:.1f} MB). Max: {MAX_MB} MB")

        t0 = time.perf_counter()
        result = await _run_predict(tmp_path)
        ms = (time.perf_counter() - t0) * 1000
        await _inc_label(result["label"])

        top_feats = result.get("top_features", [])
        return PredictionResponse(
            label=result["label"],
            confidence=result["confidence"],
            fake_score=result["fake_score"],
            real_score=result["real_score"],
            features=result["features"],
            processing_time_ms=round(ms, 1),
            filename=url.split("/")[-1].split("?")[0] or "remote_audio",
            top_features=[FeatureContrib(**f) for f in top_feats],
            verdict_reason=_verdict(result["label"], result["fake_score"], top_feats),
        )
    except HTTPException:
        raise
    except Exception as exc:
        await _inc_errors()
        logger.exception("URL predict error for %s", url)
        raise HTTPException(500, f"Inference error: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
