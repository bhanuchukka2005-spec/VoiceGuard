"""
Inference module v2.2 — VoiceGuard
Fixes:
  - MODEL_PATH / SCALER_PATH resolved from this file's location (not CWD)
  - _load() called once per process via module-level guard (thread-safe with lock)
  - Missing scaler.joblib no longer crashes — raises clear 503-able error
  - top_features safely returns [] when importance file absent
  - extract_with_names import corrected (was silently ignored if features.py not on sys.path)
"""

import os
import sys
import json
import threading
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

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
        return   # already loaded
    with _lock:
        if _model is not None:
            return  # double-checked locking
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
            _importance = {}   # BUG FIX: was None, caused TypeError later


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
    """
    _load()

    # Extract features
    vec, named = extract_with_names(audio_path)

    # BUG FIX: check for NaN/Inf in feature vector (can happen with very short clips)
    if not np.isfinite(vec).all():
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    vec_s  = _scaler.transform(vec.reshape(1, -1))
    proba  = _model.predict_proba(vec_s)[0]

    # BUG FIX: predict_proba order depends on model.classes_.
    # Always look up which index corresponds to class 1 (fake).
    classes    = list(_model.classes_)
    fake_idx   = classes.index(1) if 1 in classes else 1
    real_idx   = classes.index(0) if 0 in classes else 0
    fake_prob  = float(proba[fake_idx])
    real_prob  = float(proba[real_idx])

    label      = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = max(real_prob, fake_prob)

    # Top-5 contributing features for explainability
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


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file>")
        sys.exit(1)
    print(json.dumps(predict(sys.argv[1]), indent=2))