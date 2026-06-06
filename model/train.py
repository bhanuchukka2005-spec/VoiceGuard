"""
Train the deepfake audio classifier — v2.3

3-model stacked ensemble: SVM + GradientBoosting + XGBoost
with feature importance tracking for explainability.

IMPORTANT — synthetic mode warning
====================================
When run without --data_dir, this script generates idealized sine-wave
signals as training data. The "real" voices are multi-harmonic sine waves
and the "fake" voices are clean sine waves with minimal noise. These are
trivially separable and bear little resemblance to real TTS / neural
vocoder output.

Accuracy metrics in synthetic mode are NOT indicative of real-world
deepfake detection performance. Always use --data_dir with the ASVspoof
2019 LA partition for any production or research use.

Usage:
  python train.py                        # synthetic demo (NOT for production)
  python train.py --data_dir ./data/LA   # ASVspoof 2019 LA (recommended)
"""

import os
import sys
import argparse
import tempfile
import numpy as np
import joblib
import json
import warnings

warnings.filterwarnings("ignore")

# sklearn / xgboost imports must come after warnings.filterwarnings
from sklearn.svm import SVC  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from features import extract_all, SAMPLE_RATE, FEATURE_NAMES  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "detector.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.joblib")
IMPORTANCE_PATH = os.path.join(os.path.dirname(__file__), "feature_importance.json")

_SYNTHETIC_WARNING = """
╔══════════════════════════════════════════════════════════════════╗
║  WARNING — SYNTHETIC TRAINING MODE                               ║
║                                                                  ║
║  Training on idealized sine-wave signals, NOT real TTS/          ║
║  vocoder output. Reported accuracy is artificially high          ║
║  and will NOT reflect real-world deepfake detection.             ║
║                                                                  ║
║  For a real model, use:                                          ║
║    python train.py --data_dir ./data/LA                          ║
║  with the ASVspoof 2019 LA partition.                            ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ── Synthetic data ─────────────────────────────────────────────────────────────

def make_real_voice(n_samples, rng):
    import soundfile as sf
    paths = []
    sr = SAMPLE_RATE
    t = np.linspace(0, 3.0, int(sr * 3.0))
    for _ in range(n_samples):
        f0 = rng.uniform(90, 300)
        vibrato = np.sin(2 * np.pi * 5.5 * t) * rng.uniform(3, 8)
        wave = np.sin(2 * np.pi * (f0 + vibrato) * t)
        for h in [2, 3, 4]:
            wave += rng.uniform(0.1, 0.4) * np.sin(2 * np.pi * h * f0 * t)
        wave += rng.normal(0, 0.02, len(t))
        wave = wave / np.abs(wave).max() * 0.85
        tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tf.name, wave, sr)
        paths.append(tf.name)
    return paths


def make_fake_voice(n_samples, rng):
    import soundfile as sf
    paths = []
    sr = SAMPLE_RATE
    t = np.linspace(0, 3.0, int(sr * 3.0))
    for _ in range(n_samples):
        f0 = rng.uniform(100, 280)
        wave = np.sin(2 * np.pi * f0 * t)
        wave += rng.uniform(0.05, 0.15) * np.sin(2 * np.pi * 2.7 * f0 * t)
        wave += rng.uniform(0.0, 0.05) * np.sign(rng.normal(0, 1, len(t)))
        wave = wave / np.abs(wave).max() * 0.85
        tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tf.name, wave, sr)
        paths.append(tf.name)
    return paths


def build_synthetic_dataset(n_per_class=200):
    print(_SYNTHETIC_WARNING)
    print(f"[INFO] Generating synthetic dataset ({n_per_class} per class)...")
    rng = np.random.default_rng(42)
    return make_real_voice(n_per_class, rng), make_fake_voice(n_per_class, rng)


# ── Real data loader ───────────────────────────────────────────────────────────

def load_asvspoof(data_dir):
    real_paths, fake_paths = [], []
    for folder, paths in [("real", real_paths), ("fake", fake_paths)]:
        d = os.path.join(data_dir, folder)
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.endswith((".wav", ".flac", ".mp3")):
                    paths.append(os.path.join(d, fn))
    print(f"[INFO] Loaded {len(real_paths)} real, {len(fake_paths)} fake from {data_dir}")
    if not real_paths or not fake_paths:
        raise ValueError(
            f"No audio files found in {data_dir}/real or {data_dir}/fake. "
            "Expected .wav / .flac / .mp3 files."
        )
    return real_paths, fake_paths


# ── Feature extraction ─────────────────────────────────────────────────────────

def build_features(real_paths, fake_paths):
    print("[INFO] Extracting features...")
    X, y = [], []
    for label, paths in [(0, real_paths), (1, fake_paths)]:
        name = "Real" if label == 0 else "Fake"
        for i, p in enumerate(paths):
            try:
                X.append(extract_all(p))
                y.append(label)
            except ValueError as e:
                print(f"  [SKIP] {p}: {e}")
            except Exception as e:
                print(f"  [WARN] {p}: {e}")
            if (i + 1) % 50 == 0:
                print(f"  {name}: {i+1}/{len(paths)}")
    if not X:
        raise RuntimeError("No features could be extracted. Check your audio files.")
    return np.array(X, dtype=np.float32), np.array(y)


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model():
    """3-model soft-voting ensemble: SVM + GBC + XGBoost"""
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    gbc = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42, verbosity=0,
    )
    return VotingClassifier(
        estimators=[("svm", svm), ("gbc", gbc), ("xgb", xgb)],
        voting="soft",
        weights=[1, 2, 2],
    )


def compute_feature_importance(model, feature_names):
    """Extract feature importance from GBC and XGB sub-models."""
    importances = {}
    for _name, clf in model.named_estimators_.items():
        if hasattr(clf, "feature_importances_"):
            for fname, imp in zip(feature_names, clf.feature_importances_):
                importances[fname] = importances.get(fname, 0) + float(imp)
    total = sum(importances.values()) or 1
    importances = {k: round(v / total, 6) for k, v in importances.items()}
    top10 = dict(sorted(importances.items(), key=lambda x: -x[1])[:10])
    return top10


# ── Train ──────────────────────────────────────────────────────────────────────

def train(data_dir=None):
    if data_dir:
        real_paths, fake_paths = load_asvspoof(data_dir)
    else:
        real_paths, fake_paths = build_synthetic_dataset(200)

    X, y = build_features(real_paths, fake_paths)
    print(
        f"[INFO] Dataset: {X.shape[0]} samples, {X.shape[1]} features | "
        f"real={sum(y == 0)}, fake={sum(y == 1)}"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("[INFO] Training 3-model ensemble (SVM + GBC + XGBoost)...")
    model = build_model()
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n─── Evaluation ───────────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
    print(f"ROC-AUC  : {auc:.4f}")
    print(f"Confusion:\n{cm}")
    if not data_dir:
        print("\n[WARNING] These metrics are from synthetic data and do NOT reflect")
        print("          real-world deepfake detection accuracy.")
    print("──────────────────────────────────────────────────────\n")

    fi = compute_feature_importance(model, FEATURE_NAMES)
    print("[INFO] Top 10 most important features:")
    for k, v in fi.items():
        print(f"  {k:<35} {v:.4f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(IMPORTANCE_PATH, "w") as f:
        json.dump(fi, f, indent=2)

    print(f"\n[INFO] Saved model      → {MODEL_PATH}")
    print(f"[INFO] Saved scaler     → {SCALER_PATH}")
    print(f"[INFO] Saved importance → {IMPORTANCE_PATH}")

    if not data_dir:
        for p in real_paths + fake_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    return model, scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train VoiceGuard deepfake detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                        # synthetic demo (NOT for production)
  python train.py --data_dir ./data/LA   # ASVspoof 2019 LA (recommended)
        """,
    )
    parser.add_argument(
        "--data_dir", default=None,
        help="Path to directory with real/ and fake/ subfolders of audio files",
    )
    args = parser.parse_args()
    train(args.data_dir)
