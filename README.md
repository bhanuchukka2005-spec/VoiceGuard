# VoiceGuard 🛡️ — Audio Deepfake Detector

> **AI-powered audio forensics platform** that detects synthetic and cloned voices using a 270-feature ML pipeline with temporal analysis, biometric stress scoring, and explainable AI.

[![VoiceGuard CI](https://github.com/bhanuchukka2005-spec/VoiceGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/bhanuchukka2005-spec/VoiceGuard/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange?style=flat-square)](https://xgboost.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)

---

## 📌 What Is VoiceGuard?

VoiceGuard is a web application that takes any audio file and tells you — **is this voice REAL or AI-generated?** — with a confidence score, temporal analysis, and a full biometric breakdown.

Built in 24 hours at **AIML Hackathon 2026** by a team of 4.

---

## 🚨 The Problem

AI can now clone anyone's voice using just **3–5 seconds** of audio. These cloned voices are being used for:

- 📞 **Phone fraud** — scammers impersonating family members
- 🗳️ **Misinformation** — fake political speeches
- 🏦 **Identity theft** — bypassing bank voice authentication
- ⚖️ **Evidence tampering** — fake audio in legal cases

Human ears can no longer tell the difference. **VoiceGuard can.**

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **REAL / FAKE Detection** | Classifies any audio with confidence score |
| 📈 **Temporal Timeline** | Per-second fake confidence graph across full audio |
| 🧬 **Voice Biometric Analysis** | 5-point human likeness score (pitch, rhythm, breath, micro-variations, formants) |
| 🔬 **Explainable AI** | Top-5 features that influenced the decision |
| 🌊 **Spectral Fingerprint** | Visual frequency signature of the audio |
| 🧿 **Audio DNA Fingerprint** | Unique circular identity visual for every voice |
| 📦 **Batch Processing** | Analyze up to 10 files in one request |
| 🎤 **Live Mic Recording** | Record directly from browser and analyze |
| 📄 **Export Report** | Download full analysis report |
| 🌙 **Dark / Light Theme** | Toggle between themes, preference saved |
| 🕐 **Session History** | All analyses saved, persists across page refreshes |
| 🐳 **Docker Ready** | One command deployment |
| 🚦 **Rate Limiting** | 30 req/min per IP on `/predict` |

---

## 🚀 Quick Start

### Option 1 — Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-team/voiceguard.git
cd voiceguard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (~60 seconds on synthetic data)
#    ⚠ Synthetic mode only — see Dataset section for real data
python model/train.py

# 4. Start the backend server
uvicorn api.server:app --reload --port 8000

# 5. Open the frontend
# Simply open frontend/index.html in your browser
```

### Option 2 — Docker (Recommended)

```bash
# Build and run everything in one command
docker build -t voiceguard .
docker run -p 8000:8000 voiceguard

# Open http://localhost:8000 in your browser
```

### Option 3 — Real ASVspoof Data

```bash
# Download ASVspoof 2019 LA partition
# Organise your data as:
#   data/LA/real/   ← real voice files (.wav / .flac / .mp3)
#   data/LA/fake/   ← fake voice files (.wav / .flac / .mp3)

python model/train.py --data_dir ./data/LA
```

---

## 📁 Project Structure

```
voiceguard/
│
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI pipeline
│
├── api/
│   └── server.py               # FastAPI backend — all endpoints
│
├── model/
│   ├── train.py                # Model training script
│   ├── predict.py              # Inference + segment analysis
│   ├── features.py             # 270-feature extraction pipeline
│   ├── detector.joblib         # Trained ensemble model (git-ignored)
│   ├── scaler.joblib           # StandardScaler (git-ignored)
│   └── feature_importance.json # Top feature weights
│
├── frontend/
│   ├── index.html              # Main UI
│   ├── styles.css              # Dark + Light theme styles
│   ├── app-core.js             # Core logic — upload, analyze, history
│   └── app-features.js         # Visualizations — timeline, radar, DNA, stress
│
├── tests/
│   └── test_api.py             # pytest API test suite
│
├── Dockerfile                  # Container definition
├── pytest.ini                  # pytest + asyncio config
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🧠 How It Works

```
User uploads audio file
         ↓
FastAPI backend receives & validates file
  (rejects: silent, < 0.5s, > 50 MB, non-audio)
         ↓
features.py extracts 270 voice features:
  ├─ MFCC (240)      → vocal tract shape
  ├─ Spectral (6)    → frequency patterns
  ├─ Chroma (19)     → pitch and tonal content
  └─ Prosody (5)     → rhythm, pitch variation, breath
         ↓
StandardScaler normalizes all 270 features
         ↓
3-model ensemble votes:
  ├─ SVM (weight 1)               → non-linear boundaries
  ├─ Gradient Boosting (weight 2) → sequential error correction
  └─ XGBoost (weight 2)           → fast, regularized boosting
         ↓
/predict/segments → per-second temporal analysis
  (1s windows, 50% overlap, padded to 1s — not 3s)
         ↓
Result: REAL ✅ or FAKE ❌
  + confidence score
  + temporal confidence timeline
  + top-5 feature explanations
  + biometric stress analysis
```

---

## 🔬 Feature Pipeline — 270 Dimensions

| Group | Features | Count | What It Detects |
|-------|----------|-------|----------------|
| **MFCC** × mean/std × delta × delta2 | 40 coefficients × 6 stats | **240** | Vocal tract shape — AI voices have unnatural shapes |
| **Spectral** | centroid, bandwidth, rolloff, flatness, ZCR, RMS | **6** | Frequency patterns — AI is too clean/smooth |
| **Chroma + Contrast** | 12 pitch classes + 7 contrast bands | **19** | Tonal structure — AI lacks natural note variation |
| **Prosody** | F0 mean, std, P10, P90, voiced ratio | **5** | Rhythm + breath — AI is too metronomic |
| **Total** | | **270** | |

---

## 🤖 Model Architecture

### Soft-Voting Ensemble

```
Input: 270-dim feature vector (StandardScaler normalized)
           │
    ┌──────┼──────┐
    │      │      │
   SVM    GBC   XGBoost
 (w=1)  (w=2)   (w=2)
    │      │      │
    └──────┼──────┘
           │
    Weighted soft vote
           │
    Probability → Label (REAL / FAKE)
```

| Model | Config | Weight | Strength |
|-------|--------|--------|----------|
| SVM | RBF kernel, C=10 | 1 | Non-linear decision boundaries |
| Gradient Boosting | 300 trees, depth=4, lr=0.05 | 2 | Sequential tabular learning |
| XGBoost | 300 trees, depth=5, lr=0.05 | 2 | Fast, regularized, robust |

**Why ensemble?** Each model has different failure modes. Voting together reduces individual errors.

---

## 🌐 API Reference

**Base URL:** `http://localhost:8000`

| Method | Endpoint | Rate limit | Description |
|--------|----------|-----------|-------------|
| `GET` | `/health` | — | Server + model health check |
| `GET` | `/model/info` | — | Model metadata |
| `GET` | `/stats` | — | Session statistics |
| `POST` | `/predict` | 30/min | Single file deepfake analysis |
| `POST` | `/predict/batch` | 10/min | Up to 10 files at once |
| `POST` | `/predict/segments` | 20/min | Per-second temporal analysis |
| `POST` | `/predict/compare` | 15/min | Side-by-side comparison of 2 files |
| `POST` | `/predict/stress` | 20/min | Biometric stress indicators |
| `POST` | `/predict/url` | 10/min | Analyze audio from URL |

### Example — Single File

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@sample.wav"
```

```json
{
  "label": "FAKE",
  "confidence": 0.91,
  "fake_score": 0.91,
  "real_score": 0.09,
  "features": 270,
  "processing_time_ms": 145.2,
  "filename": "sample.wav",
  "model_version": "2.3.0",
  "verdict_reason": "Flagged as synthetic. Key signal: MFCC-1 mean. Confidence 91%.",
  "top_features": [
    { "name": "MFCC-1 mean (vocal tract shape)", "key": "mfcc_1_mean", "value": -14.2, "weight": 24.9 }
  ]
}
```

### Example — From URL

```bash
curl -X POST "http://localhost:8000/predict/url?url=https://example.com/voice.wav"
```

---

## 🧪 Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Train model first (required)
python model/train.py

# Run all tests
pytest tests/ -v

# Run with coverage
pip install pytest-cov
pytest tests/ -v --cov=api --cov=model --cov-report=term-missing
```

---

## 📊 Dataset

| Mode | Command | Use case |
|------|---------|----------|
| Synthetic (default) | `python model/train.py` | Demo / development only |
| ASVspoof 2019 LA | `python model/train.py --data_dir ./data/LA` | Research / production |

The [ASVspoof 2019 LA](https://datashare.ed.ac.uk/handle/10283/3336) (logical access) partition contains genuine speech and 19 types of spoofed speech from TTS and voice conversion systems.

> **Note:** Synthetic mode trains on idealized sine-wave signals. Accuracy figures from synthetic mode are NOT representative of real-world deepfake detection performance.

---

## ⚠️ Known Limitations

| Limitation | Detail |
|---|---|
| Synthetic training | Default mode trains on idealized sine waves, not real TTS output |
| Short clips | Clips under 0.5s are rejected; clips 0.5–1s may have lower accuracy |
| Compression artifacts | Heavy MP3/AAC encoding can distort spectral features |
| Language-agnostic | Model is not tuned per language — accuracy may vary |
| No adversarial hardening | Pitch-shifting or noise injection can reduce detection accuracy |
| In-memory stats | Session stats reset on server restart (no persistence) |

---

## 🎵 Supported Audio Formats

`WAV` · `MP3` · `FLAC` · `OGG` · `M4A` · `AAC` · `OPUS` · `WMA` · `AIFF` · `WebM` · `MP4` · `3GP` · `CAF` · `AMR` · `GSM` · and more

**Max file size:** 50 MB · **Max batch:** 10 files · **Min duration:** 0.5s

---

## 🧬 Voice Biometric Stress Analysis

| Indicator | Human Voice | AI Voice |
|-----------|------------|----------|
| Pitch Stability | Low — natural wobble | High — unnaturally perfect |
| Rhythm Naturalness | High — irregular, organic | Low — robotic, metronomic |
| Breath Patterns | High — pauses to breathe | Low — no breath pauses |
| Micro Variations | High — tiny imperfections | Low — too clean |
| Formant Stability | Low — natural shifts | High — artificially smooth |

---

## 👥 Team

| Member | Role | What They Built |
|--------|------|----------------|
| **Ch. Bhanu Prakash** | ML Model Training | `train.py`, `detector.joblib`, ensemble design |
| **K. Vishal Varma** | Frontend Development | `index.html`, `styles.css`, `app-core.js`, `app-features.js` |
| **R. Sai Mahesh** | Data Collection | Audio datasets, synthetic data pipeline |
| **G. Krishna Mahesh** | Backend Development | `server.py`, REST API, Docker setup |

---

## 🔮 Future Roadmap

**Short term**
- Real-time microphone stream detection
- Mobile app (Android + iOS)
- Browser extension for live call verification
- SHAP-based per-prediction explainability (replacing static feature importance)

**Medium term**
- Clone Source Identification — detect which AI tool generated the voice
- Voice Twin Detection — verify if two clips are from the same person
- Deep learning model (CNN / Transformer on raw waveform)
- PostgreSQL for permanent analysis history

**Long term**
- Blockchain audio certificates — tamper-proof authenticity verification
- WhatsApp / Telegram bot integration
- Business API for banks, call centers, legal firms
- Adversarial robustness against compression, pitch-shift and noise attacks

---

## 📦 Requirements

```
Python >= 3.10
fastapi>=0.110.0      uvicorn>=0.29.0       slowapi>=0.1.9
librosa>=0.10.0       soundfile>=0.12.1     numpy>=1.24.0
pydub>=0.25.1         scikit-learn>=1.4.0   xgboost>=2.0.0
joblib>=1.3.0         pydantic>=2.0.0
ffmpeg (system — included in Docker)
```

---

## 📄 License

MIT License — free to use, modify and distribute.

---

<div align="center">

**VoiceGuard v2.3** &nbsp;·&nbsp; MFCC · Spectral · Prosody · Ensemble ML

*Protecting people from AI voice fraud — one audio file at a time.*

Built at **AIML Hackathon 2026** in 24 hours.

</div>
