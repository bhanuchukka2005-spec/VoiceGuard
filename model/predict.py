"""
Inference module v2 — returns prediction + top feature contributions.
"""

import os, sys, json
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from features import extract_all, extract_with_names, FEATURE_NAMES

MODEL_PATH      = os.path.join(os.path.dirname(__file__), "detector.joblib")
SCALER_PATH     = os.path.join(os.path.dirname(__file__), "scaler.joblib")
IMPORTANCE_PATH = os.path.join(os.path.dirname(__file__), "feature_importance.json")

_model = _scaler = _importance = None


def _load():
    global _model, _scaler, _importance
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model not found. Run `python model/train.py` first.")
        _model    = joblib.load(MODEL_PATH)
        _scaler   = joblib.load(SCALER_PATH)
        if os.path.exists(IMPORTANCE_PATH):
            _importance = json.load(open(IMPORTANCE_PATH))


# Human-readable labels for top features shown in the UI
_READABLE = {
    "spectral_flatness": "Spectral flatness (synthesis artifact)",
    "spectral_centroid": "Spectral centroid (frequency balance)",
    "voiced_ratio":      "Voiced ratio (natural speech rhythm)",
    "f0_std":            "Pitch variation (vibrato / naturalness)",
    "f0_mean":           "Average pitch",
    "zcr":               "Zero-crossing rate (noise texture)",
    "rms":               "Energy level",
    "spectral_rolloff":  "Spectral rolloff (brightness)",
}


def predict(audio_path: str) -> dict:
    _load()
    vec, named = extract_with_names(audio_path)
    vec_s      = _scaler.transform(vec.reshape(1, -1))
    proba      = _model.predict_proba(vec_s)[0]
    real_prob, fake_prob = float(proba[0]), float(proba[1])
    label      = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = max(real_prob, fake_prob)

    # Build top-5 contributing features for explainability
    top_features = []
    if _importance:
        for fname, imp in list(_importance.items())[:5]:
            top_features.append({
                "name":     _READABLE.get(fname, fname),
                "key":      fname,
                "value":    round(named.get(fname, 0.0), 4),
                "weight":   round(imp * 100, 1),   # as % of total importance
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
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file>")
        sys.exit(1)
    import json
    print(json.dumps(predict(sys.argv[1]), indent=2))