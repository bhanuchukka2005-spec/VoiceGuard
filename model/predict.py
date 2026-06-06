"""
Inference module v2.3 — VoiceGuard
Changes from v2.2:
  - predict_segments: chunks now padded to segment_samples (1s), not DURATION (3s)
    — the 3s pad was filling 2/3 of every chunk with silence, biasing voiced_ratio
    and prosody features toward artificial values.
  - ValueError (silent / too-short clips) now propagates cleanly to the API layer
    which returns HTTP 422 instead of 500.
  - Segment fallback logs a warning instead of silently swallowing the exception.
  - All v2.2 fixes retained.
"""

import os
import sys
import json
import threading
import logging
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger("voiceguard.predict")

# ── Path resolution ────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from features import extract_all, extract_with_names, FEATURE_NAMES  # noqa: E402

MODEL_PATH      = os.path.join(_HERE, "detector.joblib")
SCALER_PATH     = os.path.join(_HERE, "scaler.joblib")
IMPORTANCE_PATH = os.path.join(_HERE, "feature_importance.json")

# ── Lazy-load with thread safety ───────────────────────────────────────────────
_lock       = threading.Lock()
_model      = None
_scaler     = None
_importance = None


def _load():
    global _model, _scaler, _importance
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run `python model/train.py` first."
            )
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(
                f"Scaler not found at {SCALER_PATH}. "
                "Run `python model/train.py` first."
            )
        _model  = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        if os.path.exists(IMPORTANCE_PATH):
            with open(IMPORTANCE_PATH) as f:
                _importance = json.load(f)
        else:
            _importance = {}


# ── Human-readable feature labels ─────────────────────────────────────────────
_READABLE = {
    "mfcc_1_mean":       "MFCC-1 mean (vocal tract shape)",
    "mfcc_2_mean":       "MFCC-2 mean (spectral envelope)",
    "mfcc_2_std":        "MFCC-2 variation (temporal consistency)",
    "spectral_flatness": "Spectral flatness (synthesis artifact)",
    "spectral_centroid": "Spectral centroid (frequency balance)",
    "voiced_ratio":      "Voiced ratio (natural speech rhythm)",
    "f0_std":            "Pitch variation (vibrato / naturalness)",
    "f0_mean":           "Average pitch",
    "zcr":               "Zero-crossing rate (noise texture)",
    "rms":               "Energy level",
    "spectral_rolloff":  "Spectral rolloff (brightness)",
    "d2mfcc_7_std":      "Δ²-MFCC-7 std (articulation speed)",
    "dmfcc_5_std":       "Δ-MFCC-5 std (temporal dynamics)",
    "mfcc_26_std":       "MFCC-26 std (high-freq variation)",
    "mfcc_10_std":       "MFCC-10 std (spectral texture)",
    "mfcc_20_std":       "MFCC-20 std (spectral detail)",
}


def predict(audio_path: str) -> dict:
    """
    Run inference on an audio file.

    Returns dict with label, confidence, fake_score, real_score,
    features (count), and top_features list.

    Raises:
        FileNotFoundError: model or scaler not trained yet.
        ValueError: clip is silent or too short (from features.load_audio).
    """
    _load()

    vec, named = extract_with_names(audio_path)

    if not np.isfinite(vec).all():
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    vec_s  = _scaler.transform(vec.reshape(1, -1))
    proba  = _model.predict_proba(vec_s)[0]

    classes    = list(_model.classes_)
    fake_idx   = classes.index(1) if 1 in classes else 1
    real_idx   = classes.index(0) if 0 in classes else 0
    fake_prob  = float(proba[fake_idx])
    real_prob  = float(proba[real_idx])

    label      = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = max(real_prob, fake_prob)

    top_features = []
    if _importance:
        for fname, imp in list(_importance.items())[:5]:
            top_features.append({
                "name":   _READABLE.get(fname, fname),
                "key":    fname,
                "value":  round(float(named.get(fname, 0.0)), 4),
                "weight": round(float(imp) * 100, 1),
            })

    return {
        "label":        label,
        "confidence":   round(confidence, 4),
        "fake_score":   round(fake_prob,  4),
        "real_score":   round(real_prob,  4),
        "features":     int(vec.shape[0]),
        "top_features": top_features,
    }


def predict_segments(audio_path: str, segment_duration: float = 1.0) -> dict:
    """
    Run inference on overlapping 1-second segments of the audio file.
    Returns per-segment confidence scores for temporal analysis.

    FIX (v2.3): chunks are now padded to `segment_samples` (1s), not
    `target_len` (3s). The previous 3-second padding added 2 full seconds
    of silence to every segment window, artificially depressing voiced_ratio
    and prosody features — making every segment look slightly more "fake"
    than it really is.
    """
    _load()

    import librosa
    from features import (
        SAMPLE_RATE, extract_mfcc, extract_spectral,
        extract_chroma, extract_prosody,
    )

    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception:
        try:
            from features import _load_via_pydub
            y = _load_via_pydub(audio_path)
            sr = SAMPLE_RATE
        except Exception:
            return {"segments": [], "overall": predict(audio_path)}

    if len(y) == 0:
        return {"segments": [], "overall": predict(audio_path)}

    segment_samples = int(segment_duration * sr)
    hop_samples     = segment_samples // 2   # 50% overlap

    min_viable = 2048

    classes  = list(_model.classes_)
    fake_idx = classes.index(1) if 1 in classes else 1

    segments = []
    pos      = 0
    prev_fake = None

    while pos < len(y):
        end   = min(pos + segment_samples, len(y))
        chunk = y[pos:end].copy()

        if len(chunk) < min_viable or np.abs(chunk).max() < 1e-6:
            pos += hop_samples
            continue

        # ── FIX: pad to 1 second only (segment_samples), not 3 seconds ───────
        chunk_padded = np.zeros(segment_samples, dtype=np.float32)
        chunk_padded[:len(chunk)] = chunk

        try:
            vec = np.concatenate([
                extract_mfcc(chunk_padded),
                extract_spectral(chunk_padded),
                extract_chroma(chunk_padded),
                extract_prosody(chunk_padded),
            ]).astype(np.float32)

            if not np.isfinite(vec).all():
                vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

            vec_s     = _scaler.transform(vec.reshape(1, -1))
            proba     = _model.predict_proba(vec_s)[0]
            fake_prob = float(proba[fake_idx])

            # Exponential smoothing to reduce jitter between segments
            if prev_fake is not None:
                fake_prob = 0.65 * fake_prob + 0.35 * prev_fake
            prev_fake = fake_prob

        except Exception as exc:
            logger.warning("Segment %s–%s failed: %s", pos, end, exc)
            fake_prob = prev_fake if prev_fake is not None else 0.5

        segments.append({
            "start_sec": round(pos / sr, 2),
            "end_sec":   round(end / sr, 2),
            "fake_score": round(float(np.clip(fake_prob, 0.0, 1.0)), 4),
            "real_score": round(float(np.clip(1.0 - fake_prob, 0.0, 1.0)), 4),
        })

        pos += hop_samples

    overall = predict(audio_path)
    return {"segments": segments, "overall": overall}


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file>")
        sys.exit(1)
    print(json.dumps(predict(sys.argv[1]), indent=2))
