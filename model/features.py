"""
Feature extraction for audio deepfake detection — v2.3
Changes from v2.2:
  - load_audio() raises ValueError for silent clips (< 1e-4 peak amplitude)
  - load_audio() raises ValueError for clips shorter than MIN_DURATION_SEC (0.5s)
  - predict_segments pads chunks to segment_samples (1s) not DURATION (3s)
  - Removed unused imports: os, tempfile
"""

import io
import numpy as np
import librosa
import warnings

warnings.filterwarnings("ignore")

SAMPLE_RATE = 16_000
DURATION = 3.0
N_MFCC = 40
HOP_LENGTH = 512
N_FFT = 2048
MIN_SAMPLES = HOP_LENGTH * 2
MIN_DURATION_SEC = 0.5

# ── Feature name list (must stay in sync with extract_all output order) ────────
_MFCC_NAMES = (
    [f"mfcc_{i}_mean" for i in range(N_MFCC)]
    + [f"mfcc_{i}_std" for i in range(N_MFCC)]
    + [f"dmfcc_{i}_mean" for i in range(N_MFCC)]
    + [f"dmfcc_{i}_std" for i in range(N_MFCC)]
    + [f"d2mfcc_{i}_mean" for i in range(N_MFCC)]
    + [f"d2mfcc_{i}_std" for i in range(N_MFCC)]
)
_SPEC_NAMES = [
    "spectral_centroid", "spectral_bandwidth",
    "spectral_rolloff", "spectral_flatness", "zcr", "rms",
]
_CHROMA_NAMES = (
    [f"chroma_{i}" for i in range(12)]
    + [f"contrast_{i}" for i in range(7)]
)
_PROSODY_NAMES = ["f0_mean", "f0_std", "f0_p10", "f0_p90", "voiced_ratio"]

FEATURE_NAMES = _MFCC_NAMES + _SPEC_NAMES + _CHROMA_NAMES + _PROSODY_NAMES
assert len(FEATURE_NAMES) == 270, f"Feature count mismatch: {len(FEATURE_NAMES)}"


# ── Audio loading ──────────────────────────────────────────────────────────────

def _load_via_pydub(path: str) -> np.ndarray:
    """
    Fallback loader for formats librosa can't handle natively.
    Converts to WAV in memory via pydub/ffmpeg, then loads with librosa.
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise RuntimeError(
            "pydub is not installed. Install it with: pip install pydub"
        )
    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    y, _ = librosa.load(wav_io, sr=SAMPLE_RATE, mono=True)
    return y


def load_audio(path: str) -> np.ndarray:
    """
    Load any audio file to a mono 16 kHz float32 array of exactly
    DURATION seconds (zero-padded or truncated).

    Raises:
        ValueError: if the clip is too short or silent.
    """
    try:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception:
        y = _load_via_pydub(path)

    actual_duration = len(y) / SAMPLE_RATE
    if actual_duration < MIN_DURATION_SEC:
        raise ValueError(
            f"Audio too short ({actual_duration:.2f}s). "
            f"Minimum required: {MIN_DURATION_SEC}s"
        )

    if np.abs(y).max() < 1e-4:
        raise ValueError(
            "Audio file appears to be silent or nearly silent. "
            "Please upload a clip that contains speech."
        )

    target = int(SAMPLE_RATE * DURATION)
    if len(y) == 0:
        y = np.zeros(target, dtype=np.float32)
    y = np.pad(y, (0, max(0, target - len(y))))[:target]
    return y.astype(np.float32)


# ── Feature extractors ─────────────────────────────────────────────────────────

def extract_mfcc(y: np.ndarray) -> np.ndarray:
    if len(y) < MIN_SAMPLES:
        y = np.pad(y, (0, MIN_SAMPLES - len(y)))

    mfcc = librosa.feature.mfcc(
        y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH, n_fft=N_FFT,
    )
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    feats = []
    for mat in [mfcc, d1, d2]:
        feats.extend([mat.mean(axis=1), mat.std(axis=1)])
    return np.concatenate(feats)


def extract_spectral(y: np.ndarray) -> np.ndarray:
    if len(y) < MIN_SAMPLES:
        y = np.pad(y, (0, MIN_SAMPLES - len(y)))
    return np.array([
        librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE).mean(),
        librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE).mean(),
        librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE).mean(),
        librosa.feature.spectral_flatness(y=y).mean(),
        librosa.feature.zero_crossing_rate(y).mean(),
        librosa.feature.rms(y=y).mean(),
    ], dtype=np.float32)


def extract_chroma(y: np.ndarray) -> np.ndarray:
    if len(y) < MIN_SAMPLES:
        y = np.pad(y, (0, MIN_SAMPLES - len(y)))
    chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    contrast = librosa.feature.spectral_contrast(y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    return np.concatenate([chroma.mean(axis=1), contrast.mean(axis=1)]).astype(np.float32)


def extract_prosody(y: np.ndarray) -> np.ndarray:
    if len(y) < MIN_SAMPLES:
        return np.zeros(5, dtype=np.float32)

    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=50, fmax=500, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
        )
    except Exception:
        return np.zeros(5, dtype=np.float32)

    voiced_ratio = float(voiced_flag.mean()) if voiced_flag is not None else 0.0

    if f0 is None:
        return np.array([0.0, 0.0, 0.0, 0.0, voiced_ratio], dtype=np.float32)

    f0c = f0[~np.isnan(f0)]
    if len(f0c) == 0:
        return np.array([0.0, 0.0, 0.0, 0.0, voiced_ratio], dtype=np.float32)

    return np.array([
        float(f0c.mean()),
        float(f0c.std()),
        float(np.percentile(f0c, 10)),
        float(np.percentile(f0c, 90)),
        voiced_ratio,
    ], dtype=np.float32)


def extract_all(path: str) -> np.ndarray:
    """Returns 270-dim feature vector for any audio file."""
    y = load_audio(path)
    vec = np.concatenate([
        extract_mfcc(y),
        extract_spectral(y),
        extract_chroma(y),
        extract_prosody(y),
    ]).astype(np.float32)

    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec


def extract_with_names(path: str) -> tuple:
    """Returns (feature_vector, {name: value}) for explainability."""
    vec = extract_all(path)
    named = {name: round(float(val), 5) for name, val in zip(FEATURE_NAMES, vec)}
    return vec, named
