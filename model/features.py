"""
Feature extraction for audio deepfake detection — v2.
Exports FEATURE_NAMES so the API can return per-feature explanations.
"""

import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000
DURATION    = 3.0
N_MFCC      = 40
HOP_LENGTH  = 512
N_FFT       = 2048

# Build feature name list (must stay in sync with extract_all output order)
_MFCC_NAMES = (
    [f"mfcc_{i}_mean"  for i in range(N_MFCC)] +
    [f"mfcc_{i}_std"   for i in range(N_MFCC)] +
    [f"dmfcc_{i}_mean" for i in range(N_MFCC)] +
    [f"dmfcc_{i}_std"  for i in range(N_MFCC)] +
    [f"d2mfcc_{i}_mean"for i in range(N_MFCC)] +
    [f"d2mfcc_{i}_std" for i in range(N_MFCC)]
)
_SPEC_NAMES    = ["spectral_centroid","spectral_bandwidth",
                  "spectral_rolloff","spectral_flatness","zcr","rms"]
_CHROMA_NAMES  = [f"chroma_{i}" for i in range(12)] + [f"contrast_{i}" for i in range(7)]
_PROSODY_NAMES = ["f0_mean","f0_std","f0_p10","f0_p90","voiced_ratio"]

FEATURE_NAMES = _MFCC_NAMES + _SPEC_NAMES + _CHROMA_NAMES + _PROSODY_NAMES


def load_audio(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    target = int(SAMPLE_RATE * DURATION)
    return np.pad(y, (0, max(0, target - len(y))))[:target]


def extract_mfcc(y):
    mfcc  = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
                                   hop_length=HOP_LENGTH, n_fft=N_FFT)
    d1    = librosa.feature.delta(mfcc)
    d2    = librosa.feature.delta(mfcc, order=2)
    feats = []
    for mat in [mfcc, d1, d2]:
        feats.extend([mat.mean(axis=1), mat.std(axis=1)])
    return np.concatenate(feats)


def extract_spectral(y):
    return np.array([
        librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE).mean(),
        librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE).mean(),
        librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE).mean(),
        librosa.feature.spectral_flatness(y=y).mean(),
        librosa.feature.zero_crossing_rate(y).mean(),
        librosa.feature.rms(y=y).mean(),
    ])


def extract_chroma(y):
    chroma   = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    contrast = librosa.feature.spectral_contrast(y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    return np.concatenate([chroma.mean(axis=1), contrast.mean(axis=1)])


def extract_prosody(y):
    f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    f0c = f0[~np.isnan(f0)] if f0 is not None else np.array([0.0])
    if len(f0c) == 0: f0c = np.array([0.0])
    return np.array([
        f0c.mean(), f0c.std(),
        float(np.percentile(f0c, 10)),
        float(np.percentile(f0c, 90)),
        float(voiced.mean()) if voiced is not None else 0.0,
    ])


def extract_all(path):
    """Returns 270-dim feature vector."""
    y = load_audio(path)
    return np.concatenate([
        extract_mfcc(y),
        extract_spectral(y),
        extract_chroma(y),
        extract_prosody(y),
    ]).astype(np.float32)


def extract_with_names(path):
    """Returns (feature_vector, {name: value}) dict for explainability."""
    vec = extract_all(path)
    return vec, {name: round(float(val), 5)
                 for name, val in zip(FEATURE_NAMES, vec)}