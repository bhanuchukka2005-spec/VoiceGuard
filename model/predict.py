"""
Inference module — loads trained model and runs prediction on a single audio file.
Used by the FastAPI server.
"""

import os
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.dirname(__file__))
from features import extract_all

MODEL_PATH = os.path.join(os.path.dirname(__file__), "detector.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.joblib")

_model = None
_scaler = None


def _load():
    global _model, _scaler
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run `python model/train.py` first."
            )
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)


def predict(audio_path: str) -> dict:
    """
    Run deepfake detection on a single audio file.

    Returns:
        {
            "label":      "REAL" | "FAKE",
            "confidence": float (0-1, confidence in the predicted label),
            "fake_score": float (0-1, probability of being fake),
            "real_score": float (0-1, probability of being real),
            "features":   int   (number of features extracted)
        }
    """
    _load()

    features = extract_all(audio_path)
    features_scaled = _scaler.transform(features.reshape(1, -1))

    proba = _model.predict_proba(features_scaled)[0]
    real_prob, fake_prob = float(proba[0]), float(proba[1])

    label = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = max(real_prob, fake_prob)

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "fake_score": round(fake_prob, 4),
        "real_score": round(real_prob, 4),
        "features": int(features.shape[0]),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file>")
        sys.exit(1)
    result = predict(sys.argv[1])
    print(result)