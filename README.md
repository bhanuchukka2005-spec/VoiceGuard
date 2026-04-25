# VoiceGuard 🛡️ — Audio Deepfake Detector

> **AI-powered audio forensics platform** that detects synthetic and cloned voices using a 270-feature ML pipeline with temporal analysis, biometric stress scoring, and explainable AI.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange?style=flat-square)](https://xgboost.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)

---

## 📌 What Is VoiceGuard?

VoiceGuard is a web application that takes any audio file and tells you — **is this voice REAL or AI-generated?** — with a confidence score, temporal analysis, and a full biometric breakdown.

Built in 24 hours at **AIML Hackathon 2025** by a team of 4.

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
├── api/
│   └── server.py               # FastAPI backend — all endpoints
│
├── model/
│   ├── train.py                # Model training script
│   ├── predict.py              # Inference + segment analysis
│   ├── features.py             # 270-feature extraction pipeline
│   ├── detector.joblib         # Trained ensemble model
│   ├── scaler.joblib           # StandardScaler for features
│   └── feature_importance.json # Top feature weights
│
├── frontend/
│   ├── index.html              # Main UI
│   ├── styles.css              # Dark + Light theme styles
│   ├── app-core.js             # Core logic — upload, analyze, history
│   └── app-features.js         # Visualizations — timeline, radar, DNA, stress
│
├── Dockerfile                  # Container definition
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🧠 How It Works

```
User uploads audio file
         ↓
FastAPI backend receives & validates file
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
predict/segments → per-second temporal analysis
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

### Top Features by Importance

| Feature | Importance | What It Measures |
|---------|-----------|-----------------|
| mfcc_1_mean | 24.9% | Vocal tract shape |
| mfcc_2_mean | 20.6% | Spectral envelope |
| mfcc_2_std | 9.9% | Temporal consistency |
| voiced_ratio | 7.3% | Natural speech rhythm |
| mfcc_26_std | 4.8% | High-frequency variation |

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

**Why ensemble?** Each model has different failure modes. Voting together reduces individual errors — consistently outperforms any single model by 3–8% AUC.

---

## 🌐 API Reference

**Base URL:** `http://localhost:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server + model health check |
| `POST` | `/predict` | Single file deepfake analysis |
| `POST` | `/predict/batch` | Up to 10 files at once |
| `POST` | `/predict/segments` | Per-second temporal analysis |
| `POST` | `/predict/compare` | Side-by-side comparison of 2 files |
| `GET` | `/stats` | Session statistics |
| `GET` | `/model/info` | Model metadata |

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
  "model_version": "2.2.0",
  "verdict_reason": "Flagged as synthetic. Key signal: MFCC-1 mean. Confidence 91%.",
  "top_features": [
    { "name": "MFCC-1 mean (vocal tract shape)", "key": "mfcc_1_mean", "value": -14.2, "weight": 24.9 },
    { "name": "MFCC-2 mean (spectral envelope)",  "key": "mfcc_2_mean", "value": 8.1,  "weight": 20.6 }
  ]
}
```

### Example — Temporal Segments

```bash
curl -X POST http://localhost:8000/predict/segments \
  -F "file=@sample.wav"
```

```json
{
  "segments": [
    { "start_sec": 0.0, "end_sec": 1.0, "fake_score": 0.87, "real_score": 0.13 },
    { "start_sec": 0.5, "end_sec": 1.5, "fake_score": 0.91, "real_score": 0.09 },
    { "start_sec": 1.0, "end_sec": 2.0, "fake_score": 0.45, "real_score": 0.55 }
  ],
  "overall": { "label": "FAKE", "confidence": 0.91 }
}
```

Enables **partial deepfake detection** — identifies exactly which seconds of an audio clip are suspicious.

---

## 🎵 Supported Audio Formats

VoiceGuard accepts **any audio format** via dual-loader strategy — Librosa for standard formats, Pydub + FFmpeg as fallback for everything else.

`WAV` · `MP3` · `FLAC` · `OGG` · `M4A` · `AAC` · `OPUS` · `WMA` · `AIFF` · `WebM` · `MP4` · `3GP` · `CAF` · `AMR` · `GSM` · and more

**Max file size:** 50 MB &nbsp;·&nbsp; **Max batch:** 10 files

---

## 🧬 Voice Biometric Stress Analysis

A 5-point breakdown comparing the voice against known human speech patterns:

| Indicator | Human Voice | AI Voice |
|-----------|------------|----------|
| Pitch Stability | Low — natural wobble | High — unnaturally perfect |
| Rhythm Naturalness | High — irregular, organic | Low — robotic, metronomic |
| Breath Patterns | High — pauses to breathe | Low — no breath pauses |
| Micro Variations | High — tiny imperfections | Low — too clean |
| Formant Stability | Low — natural shifts | High — artificially smooth |

Combined into a single **Human Likeness Score (0–100%)**.

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

**Medium term**
- Clone Source Identification — detect which AI tool generated the voice
- Voice Twin Detection — verify if two clips are from the same person
- Deepfake Heatmap on spectrogram — red overlay on suspicious regions
- Deep learning model (CNN / Transformer on raw waveform)

**Long term**
- Blockchain audio certificates — tamper-proof authenticity verification
- WhatsApp / Telegram bot integration
- Business API for banks, call centers, legal firms
- Forensic PDF report for legal evidence
- Adversarial robustness against compression, pitch-shift and noise attacks
- PostgreSQL for permanent analysis history

---

## 📦 Requirements

```
Python >= 3.10
fastapi >= 0.110.0        uvicorn >= 0.29.0
librosa >= 0.10.0         soundfile >= 0.12.1
numpy >= 1.24.0           pydub >= 0.25.1
scikit-learn >= 1.4.0     xgboost >= 2.0.0
joblib >= 1.3.0           pydantic >= 2.0.0
ffmpeg (system — included in Docker)
```

---

## 📄 License

MIT License — free to use, modify and distribute.

---

<div align="center">

**VoiceGuard v2.2** &nbsp;·&nbsp; MFCC · Spectral · Prosody · Ensemble ML

*Protecting people from AI voice fraud — one audio file at a time.*

Built at **AIML Hackathon 2025** in 24 hours.

</div>