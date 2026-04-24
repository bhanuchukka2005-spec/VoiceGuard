"""
Train the deepfake audio classifier.

For the hackathon demo we support two modes:
  1. REAL data  — point --data_dir at an ASVspoof-style folder
  2. SYNTHETIC  — generates a labelled dataset from sine/noise synthesis
                  so the pipeline runs end-to-end without downloading 3 GB

Usage:
  python train.py                        # synthetic demo mode
  python train.py --data_dir ./data/LA   # real ASVspoof data
"""

import os
import sys
import argparse
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from features import extract_all, SAMPLE_RATE

MODEL_PATH = os.path.join(os.path.dirname(__file__), "detector.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.joblib")


# ─── Synthetic data generator ────────────────────────────────────────────────

def make_real_voice(n_samples: int, rng: np.random.Generator) -> list:
    """
    Simulate real speech: voiced (F0 vibrato) + small noise.
    Returns list of temp .wav paths.
    """
    import tempfile, soundfile as sf
    paths = []
    sr = SAMPLE_RATE
    dur = 3.0
    t = np.linspace(0, dur, int(sr * dur))
    for _ in range(n_samples):
        f0 = rng.uniform(90, 300)
        vibrato = np.sin(2 * np.pi * 5.5 * t) * rng.uniform(3, 8)
        wave = np.sin(2 * np.pi * (f0 + vibrato) * t)
        # Add harmonics — natural voices are harmonically rich
        for h in [2, 3, 4]:
            wave += rng.uniform(0.1, 0.4) * np.sin(2 * np.pi * h * f0 * t)
        wave += rng.normal(0, 0.02, len(t))
        wave = wave / np.abs(wave).max() * 0.85
        tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tf.name, wave, sr)
        paths.append(tf.name)
    return paths


def make_fake_voice(n_samples: int, rng: np.random.Generator) -> list:
    """
    Simulate TTS/cloned speech: very steady pitch, flat prosody, phase artifacts.
    """
    import tempfile, soundfile as sf
    paths = []
    sr = SAMPLE_RATE
    dur = 3.0
    t = np.linspace(0, dur, int(sr * dur))
    for _ in range(n_samples):
        f0 = rng.uniform(100, 280)
        # Almost no vibrato — flat pitch = TTS artifact
        wave = np.sin(2 * np.pi * f0 * t)
        # Weaker harmonics, unusual ratios
        wave += rng.uniform(0.05, 0.15) * np.sin(2 * np.pi * 2.7 * f0 * t)
        # Quantization-like noise (synthesis artifact)
        wave += rng.uniform(0.0, 0.05) * np.sign(rng.normal(0, 1, len(t)))
        wave = wave / np.abs(wave).max() * 0.85
        tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tf.name, wave, sr)
        paths.append(tf.name)
    return paths


def build_synthetic_dataset(n_per_class: int = 150):
    print(f"[INFO] Generating synthetic dataset ({n_per_class} real + {n_per_class} fake)...")
    rng = np.random.default_rng(42)
    real_paths = make_real_voice(n_per_class, rng)
    fake_paths = make_fake_voice(n_per_class, rng)
    return real_paths, fake_paths


# ─── Feature extraction ───────────────────────────────────────────────────────

def build_features(real_paths, fake_paths):
    print("[INFO] Extracting features...")
    X, y = [], []
    for i, p in enumerate(real_paths):
        try:
            X.append(extract_all(p))
            y.append(0)  # 0 = real
        except Exception as e:
            print(f"  [WARN] Skip {p}: {e}")
        if (i + 1) % 20 == 0:
            print(f"  Real: {i+1}/{len(real_paths)}")

    for i, p in enumerate(fake_paths):
        try:
            X.append(extract_all(p))
            y.append(1)  # 1 = fake
        except Exception as e:
            print(f"  [WARN] Skip {p}: {e}")
        if (i + 1) % 20 == 0:
            print(f"  Fake: {i+1}/{len(fake_paths)}")

    return np.array(X), np.array(y)


# ─── Load real ASVspoof data ──────────────────────────────────────────────────

def load_asvspoof(data_dir: str):
    """
    Expects:
      data_dir/
        real/  ← genuine speech .flac / .wav
        fake/  ← spoofed speech
    Or ASVspoof LA structure (reads protocol file).
    """
    real_paths, fake_paths = [], []
    for label, folder in [(0, "real"), (1, "fake")]:
        d = os.path.join(data_dir, folder)
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.endswith((".wav", ".flac", ".mp3")):
                    p = os.path.join(d, fn)
                    if label == 0:
                        real_paths.append(p)
                    else:
                        fake_paths.append(p)
    print(f"[INFO] Loaded {len(real_paths)} real, {len(fake_paths)} fake from {data_dir}")
    return real_paths, fake_paths


# ─── Model ───────────────────────────────────────────────────────────────────

def build_model():
    """Voting ensemble: SVM + GradientBoosting for robustness."""
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    gbc = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                      learning_rate=0.05, random_state=42)
    ensemble = VotingClassifier(
        estimators=[("svm", svm), ("gbc", gbc)],
        voting="soft"
    )
    return ensemble


# ─── Train ───────────────────────────────────────────────────────────────────

def train(data_dir=None):
    if data_dir:
        real_paths, fake_paths = load_asvspoof(data_dir)
    else:
        real_paths, fake_paths = build_synthetic_dataset(150)

    X, y = build_features(real_paths, fake_paths)
    print(f"[INFO] Dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"real={sum(y==0)}, fake={sum(y==1)}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train
    print("[INFO] Training model...")
    model = build_model()
    model.fit(X_train_s, y_train)

    # Evaluate
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print("\n─── Evaluation ───────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
    print(f"ROC-AUC: {auc:.4f}")
    print("──────────────────────────────────────────────\n")

    # Save
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[INFO] Model saved → {MODEL_PATH}")
    print(f"[INFO] Scaler saved → {SCALER_PATH}")

    # Cleanup synthetic files
    if not data_dir:
        import os
        for p in real_paths + fake_paths:
            try:
                os.unlink(p)
            except Exception:
                pass

    return model, scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None,
                        help="Path to real/fake audio folders (optional)")
    args = parser.parse_args()
    train(args.data_dir)