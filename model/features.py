"""
Feature extraction for audio deepfake detection.
Extracts MFCC, spectral, and prosody features from audio files.
"""

import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000
DURATION = 3.0          # seconds — fixed window per clip
N_MFCC = 40
HOP_LENGTH = 512
N_FFT = 2048
N_MELS = 128


def load_audio(path: str) -> np.ndarray:
    """Load and normalize audio to fixed length."""
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


def extract_mfcc(y: np.ndarray) -> np.ndarray:
    """Extract MFCC + delta + delta-delta features."""
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
                                  hop_length=HOP_LENGTH, n_fft=N_FFT)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    # Aggregate: mean + std across time axis
    features = []
    for mat in [mfcc, delta, delta2]:
        features.append(mat.mean(axis=1))
        features.append(mat.std(axis=1))
    return np.concatenate(features)  # 240-dim


def extract_spectral(y: np.ndarray) -> np.ndarray:
    """Spectral features that expose synthesis artifacts."""
    centroid = librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE).mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE).mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE).mean()
    flatness = librosa.feature.spectral_flatness(y=y).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    rms = librosa.feature.rms(y=y).mean()
    return np.array([centroid, bandwidth, rolloff, flatness, zcr, rms])


def extract_chroma(y: np.ndarray) -> np.ndarray:
    """Chroma features + contrast — capture harmonic patterns."""
    chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE,
                                          hop_length=HOP_LENGTH)
    contrast = librosa.feature.spectral_contrast(y=y, sr=SAMPLE_RATE,
                                                  hop_length=HOP_LENGTH)
    return np.concatenate([chroma.mean(axis=1), contrast.mean(axis=1)])  # 19-dim


def extract_prosody(y: np.ndarray) -> np.ndarray:
    """Pitch + energy envelope — deepfakes often have unnatural prosody."""
    f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500,
                                       sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([0.0])
    if len(f0_clean) == 0:
        f0_clean = np.array([0.0])
    voiced_ratio = voiced_flag.mean() if voiced_flag is not None else 0.0
    return np.array([
        f0_clean.mean(),
        f0_clean.std(),
        float(np.percentile(f0_clean, 10)),
        float(np.percentile(f0_clean, 90)),
        voiced_ratio,
    ])


def extract_all(path: str) -> np.ndarray:
    """Full feature vector for one audio file. ~270-dim."""
    y = load_audio(path)
    mfcc_feats = extract_mfcc(y)
    spec_feats = extract_spectral(y)
    chroma_feats = extract_chroma(y)
    prosody_feats = extract_prosody(y)
    vec = np.concatenate([mfcc_feats, spec_feats, chroma_feats, prosody_feats])
    return vec.astype(np.float32)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        vec = extract_all(sys.argv[1])
        print(f"Feature vector shape: {vec.shape}")
        print(f"Sample values: {vec[:10]}")
    else:
        print("Usage: python features.py <audio_file>")