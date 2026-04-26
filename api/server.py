"""
FastAPI backend v2.2 — VoiceGuard
Fixes in this version:
  - ACCEPTS ANY audio file type (whitelist replaced with mime+extension check)
  - Fixed MODEL_PATH resolution (was looking in wrong relative dir)
  - Fixed _stats["fake"] / _stats["real"] KeyError when label is lowercase
  - Fixed predict() import path — now works regardless of CWD
  - batch endpoint: errors no longer silently corrupt _stats
  - /health now returns 200 even when model missing (was crashing callers)
  - Added /predict/url endpoint for URL-based audio (bonus)
  - Removed duplicate `response_model` crash when top_features keys mismatch
"""

import os, sys, tempfile, time, mimetypes
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio

# ── Fix import path so predict.py is always found ──────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent   # project root
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "model"))

from model.predict import predict, predict_segments  # noqa: E402

app = FastAPI(title="VoiceGuard API", version="2.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Accept ANY audio type ──────────────────────────────────────────────────────
# Extension-based allow-list replaced with a broad audio MIME check.
# Any file whose MIME type starts with "audio/" is accepted, plus common
# extensions whose MIME detection can fail (e.g. .ogg on some systems).
_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus",
    ".wma", ".aiff", ".aif", ".au", ".ra", ".amr", ".webm",
    ".mp4",   # often audio-only (voice memos)
    ".3gp",   # phone recordings
    ".caf",   # Apple Core Audio
    ".gsm",   # telephony
}
MAX_MB = 50          # raised from 20 MB to 50 MB

def _is_audio(filename: str, content: bytes) -> bool:
    """Accept file if extension is known audio OR mime-type says audio."""
    ext = Path(filename).suffix.lower()
    if ext in _AUDIO_EXTENSIONS:
        return True
    mime, _ = mimetypes.guess_type(filename)
    if mime and mime.startswith("audio/"):
        return True
    # Fallback: sniff magic bytes for common formats
    sigs = {
        b"RIFF": True,           # WAV
        b"fLaC": True,           # FLAC
        b"\xff\xfb": True,       # MP3
        b"\xff\xf3": True,       # MP3
        b"\xff\xf2": True,       # MP3
        b"ID3":  True,           # MP3 with ID3 tag
        b"OggS": True,           # OGG
        b"\x1aE\xdf\xa3": True,  # WebM / MKV
    }
    header = content[:4]
    return any(header.startswith(sig) for sig in sigs)


# ── In-memory session stats ────────────────────────────────────────────────────
# BUG FIX: original code did _stats[result["label"].lower()] which would KeyError
# on any label other than "fake"/"real" (e.g. "error"). Now using .get() with default.
_stats: dict = {"total": 0, "fake": 0, "real": 0, "errors": 0}

def _inc_label(label: str):
    """Safely increment fake/real counter."""
    key = label.lower()
    if key in _stats:
        _stats[key] += 1


# ── Schemas ────────────────────────────────────────────────────────────────────

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
    model_version:      str = "2.2.0"


class BatchItem(BaseModel):
    filename:   str
    label:      str
    confidence: float
    fake_score: float
    real_score: float
    error:      str = ""


class BatchResponse(BaseModel):
    results:       List[BatchItem]
    total_files:   int
    fake_count:    int
    real_count:    int
    error_count:   int
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
    """Run predict() in thread-pool so it doesn't block the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, predict, tmp_path)


async def _save_upload(file: UploadFile) -> tuple[bytes, str]:
    """Read upload, validate size, return (content, tmp_path)."""
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB). Max: {MAX_MB} MB")
    # BUG FIX: always preserve original extension so librosa gets the right decoder
    ext = Path(file.filename or "audio.wav").suffix.lower() or ".wav"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        return content, tmp.name


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    # BUG FIX: original used relative path "../model/detector.joblib" which
    # breaks depending on CWD. Now resolved from this file's location.
    model_path = BASE_DIR / "model" / "detector.joblib"
    model_ok   = model_path.exists()
    # Always return 200 so the frontend health-check dot works correctly.
    return JSONResponse(status_code=200, content={
        "status":       "ok" if model_ok else "degraded",
        "model_loaded": model_ok,
        "version":      "2.2.0",
        "stats":        _stats,
    })


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(file: UploadFile = File(...)):
    content, tmp_path = await _save_upload(file)

    # BUG FIX: was raising 400 for any unknown extension. Now accepts all audio.
    if not _is_audio(file.filename or "", content):
        os.unlink(tmp_path)
        raise HTTPException(
            400,
            "File does not appear to be audio. "
            "Supported: wav, mp3, flac, ogg, m4a, aac, opus, wma, aiff, webm, mp4, 3gp, and more."
        )

    try:
        t0     = time.perf_counter()
        result = await _run_predict(tmp_path)
        ms     = (time.perf_counter() - t0) * 1000
        _stats["total"] += 1
        _inc_label(result["label"])
    except FileNotFoundError as exc:
        _stats["errors"] += 1
        raise HTTPException(503, str(exc))
    except Exception as exc:
        _stats["errors"] += 1
        raise HTTPException(500, f"Inference error: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    top_feats = result.get("top_features", [])
    return PredictionResponse(
        label              = result["label"],
        confidence         = result["confidence"],
        fake_score         = result["fake_score"],
        real_score         = result["real_score"],
        features           = result["features"],
        processing_time_ms = round(ms, 1),
        filename           = file.filename or "unknown",
        top_features       = [FeatureContrib(**f) for f in top_feats],
        verdict_reason     = _verdict(result["label"], result["fake_score"], top_feats),
    )


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    """Analyse up to 10 audio files in one request."""
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
                filename=file.filename or "unknown", label="ERROR",
                confidence=0, fake_score=0, real_score=0,
                error=exc.detail,
            ))
            error_count += 1
            _stats["errors"] += 1
            continue

        if not _is_audio(file.filename or "", content):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            results.append(BatchItem(
                filename=file.filename or "unknown", label="ERROR",
                confidence=0, fake_score=0, real_score=0,
                error="Not a recognised audio file",
            ))
            error_count += 1
            _stats["errors"] += 1
            continue

        try:
            r = await _run_predict(tmp_path)
            results.append(BatchItem(
                filename   = file.filename or "unknown",
                label      = r["label"],
                confidence = r["confidence"],
                fake_score = r["fake_score"],
                real_score = r["real_score"],
            ))
            if r["label"] == "FAKE":
                fake_count += 1
            else:
                real_count += 1
            _stats["total"] += 1
            _inc_label(r["label"])
        except Exception as exc:
            results.append(BatchItem(
                filename=file.filename or "unknown", label="ERROR",
                confidence=0, fake_score=0, real_score=0, error=str(exc),
            ))
            error_count += 1
            _stats["errors"] += 1
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return BatchResponse(
        results       = results,
        total_files   = len(files),
        fake_count    = fake_count,
        real_count    = real_count,
        error_count   = error_count,
        total_time_ms = round((time.perf_counter() - t_start) * 1000, 1),
    )


@app.get("/stats")
async def session_stats():
    """Return session-level analysis statistics."""
    total = _stats["total"] or 1   # BUG FIX: avoid ZeroDivisionError
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
        "input_formats":  "any audio file (wav, mp3, flac, ogg, m4a, aac, opus, wma, aiff, webm, mp4, 3gp, caf …)",
        "max_file_mb":    MAX_MB,
        "max_batch_size": 10,
        "explainability": "Top-5 feature contributions per prediction",
        "dataset":        "ASVspoof 2019 LA / synthetic",
        "version":        "2.2.0",
    }


@app.post("/predict/segments")
async def predict_temporal(file: UploadFile = File(...)):
    """Analyse audio in temporal segments for confidence timeline."""
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
        _stats["total"] += 1
        _inc_label(result["overall"]["label"])
    except Exception as exc:
        _stats["errors"] += 1
        raise HTTPException(500, f"Segment analysis error: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return result


@app.post("/predict/compare")
async def predict_compare(file_a: UploadFile = File(...), file_b: UploadFile = File(...)):
    """Compare two audio files side-by-side."""
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
            _stats["total"] += 1
            _inc_label(r["label"])
        except Exception as exc:
            _stats["errors"] += 1
            raise HTTPException(500, f"Compare error on file {label.upper()}: {exc}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return results


@app.post("/predict/stress")
async def predict_stress(file: UploadFile = File(...)):
    """
    Compute voice biometric stress indicators.
    Returns 5 scores measuring human-likeness of the voice:
      pitch_stability, rhythm_naturalness, breath_patterns,
      micro_variations, formant_stability.
    All scores in [0, 1]. Higher = more human-like EXCEPT
    pitch_stability and formant_stability (lower = more human).
    """
    content, tmp_path = await _save_upload(file)

    if not _is_audio(file.filename or "", content):
        os.unlink(tmp_path)
        raise HTTPException(400, "File does not appear to be audio.")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _compute_stress, tmp_path)
        _stats["total"] += 1
    except Exception as exc:
        _stats["errors"] += 1
        raise HTTPException(500, f"Stress analysis error: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return result


def _compute_stress(audio_path: str) -> dict:
    """
    Derive 5 biometric stress indicators from the audio features.
    Uses the already-extracted feature vector — no extra ML inference needed.
    """
    import numpy as np
    from model.predict import _load, _scaler, _model
    from model.features import extract_with_names, load_audio, SAMPLE_RATE
    import librosa

    _load()

    vec, named = extract_with_names(audio_path)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Pitch Stability ──────────────────────────────────────────────────────
    # High f0_std = natural variation = more human-like
    # We invert so that 1.0 = max human-likeness
    f0_std   = float(named.get("f0_std", 0.0))
    # Typical human f0_std: 15–40 Hz. AI: 0–5 Hz.
    pitch_stability = float(np.clip(f0_std / 40.0, 0.0, 1.0))

    # ── Rhythm Naturalness ───────────────────────────────────────────────────
    # ZCR variation across time indicates natural rhythm changes
    # Use mfcc delta std as proxy for temporal dynamics
    dmfcc_std_avg = float(np.mean([
        named.get(f"dmfcc_{i}_std", 0.0) for i in range(5)
    ]))
    rhythm_naturalness = float(np.clip(dmfcc_std_avg / 8.0, 0.0, 1.0))

    # ── Breath Patterns ──────────────────────────────────────────────────────
    # voiced_ratio: humans pause to breathe → ratio < 1.0
    # AI voices: voiced_ratio close to 1.0 (no breath pauses)
    voiced_ratio = float(named.get("voiced_ratio", 1.0))
    # More human-like if voiced_ratio is moderate (0.5–0.85)
    breath_patterns = float(np.clip(1.0 - abs(voiced_ratio - 0.7) / 0.7, 0.0, 1.0))

    # ── Micro Variations ─────────────────────────────────────────────────────
    # Human voices have natural variation in MFCC std values
    mfcc_std_avg = float(np.mean([
        named.get(f"mfcc_{i}_std", 0.0) for i in range(10)
    ]))
    # Typical human range: 8–20. AI: 1–5.
    micro_variations = float(np.clip(mfcc_std_avg / 20.0, 0.0, 1.0))

    # ── Formant Stability ────────────────────────────────────────────────────
    # Use spectral centroid variation as formant proxy
    # High variation = natural human formant movement
    sc = float(named.get("spectral_centroid", 0.0))
    sb = float(named.get("spectral_bandwidth", 0.0))
    # Humans: wide bandwidth relative to centroid
    formant_naturalness = float(np.clip(sb / max(sc, 1.0), 0.0, 1.0))

    return {
        "pitch_stability":    round(pitch_stability,    3),
        "rhythm_naturalness": round(rhythm_naturalness, 3),
        "breath_patterns":    round(breath_patterns,    3),
        "micro_variations":   round(micro_variations,   3),
        "formant_stability":  round(formant_naturalness,3),
        "is_demo": False,
    }