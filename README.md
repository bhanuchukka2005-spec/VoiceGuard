# VoiceGuard 🎙️ — Audio Deepfake Detector

> Detects AI-generated or cloned voices in any audio file using a 270-feature ML pipeline.  
> Built at AIML Hackathon 2025 in 24 hours.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (uses synthetic data, takes ~60 seconds)
python model/train.py

# 3. Start the backend server
uvicorn api.server:app --reload --port 8000

# 4. Open frontend in your browser
open frontend/index.html
```

---

## 🐳 Docker (Recommended)

```bash
# Build and run everything in one command
docker build -t voiceguard .
docker run -p 8000:8000 voiceguard
```

Then open `http://localhost:8000` in your browser.

---

## 📁 Project Structure

```
audio-deepfake-detection/
├── api/
│   └── server.py          # FastAPI backend
├── model/
│   ├── train.py           # Model training
│   ├── predict.py         # Inference logic
│   ├── features.py        # 270-feature extraction
│   ├── detector.joblib    # Trained model
│   ├── scaler.joblib      # Feature scaler
│   └── feature_importance.json
├── frontend/
│   └── index.html         # Web UI
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🧠 How It Works

```
User uploads audio
       ↓
FastAPI backend receives file
       ↓
features.py extracts 270 voice features
  - MFCC (vocal tract shape)
  - Spectral (frequency patterns)
  - Chroma (pitch and notes)
  - Prosody (rhythm and flow)
       ↓
StandardScaler normalizes features
       ↓
3-model ensemble votes:
  SVM + Gradient Boosting + XGBoost
       ↓
Result: REAL ✅ or FAKE ❌ + confidence score
```

---

## 🔬 Feature Pipeline

| Group | Features | Count |
|-------|----------|-------|
| MFCC (40 coefficients × mean/std × delta × delta2) | Vocal tract shape | 240 |
| Spectral (centroid, bandwidth, rolloff, flatness, ZCR, RMS) | Frequency content | 6 |
| Chroma + Spectral Contrast | Pitch and notes | 19 |
| Prosody (F0 mean, std, percentiles, voiced ratio) | Rhythm and flow | 5 |
| **Total** | | **270** |

---

## 🤖 Model

**Voting Ensemble (soft voting):**

| Model | Weight | Strength |
|-------|--------|----------|
| SVM (RBF kernel) | 1 | Handles non-linear boundaries |
| Gradient Boosting | 2 | Strong on tabular features |
| XGBoost | 2 | Fast, accurate, robust |

---

## 🌐 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server health check |
| POST | `/predict` | Upload audio, get REAL/FAKE result |
| POST | `/predict/batch` | Upload up to 10 files at once |
| GET | `/stats` | Session statistics |
| GET | `/model/info` | Model metadata |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@sample.wav"
```

### Example Response

```json
{
  "label": "FAKE",
  "confidence": 0.91,
  "fake_score": 0.91,
  "real_score": 0.09,
  "features": 270,
  "processing_time_ms": 145.2,
  "filename": "sample.wav",
  "verdict_reason": "Flagged as synthetic. Key signal: MFCC-1 mean. Confidence 91%."
}
```

---

## 🎵 Supported Audio Formats

WAV, MP3, FLAC, OGG, M4A, AAC, OPUS, WMA, AIFF, WebM, MP4, 3GP, CAF and more.  
Maximum file size: **50 MB**

---

## 📊 With Real ASVspoof Data

```bash
# Download ASVspoof 2019 LA partition and organise as:
# data/LA/real/   ← real voice files
# data/LA/fake/   ← fake voice files

python model/train.py --data_dir ./data/LA
```

---

## 👥 Team

| Member | Role |
|--------|------|
| Ch. Bhanu Prakash | ML Model Training |
| K. Vishal Varma | Frontend Development |
| R. Sai Mahesh | Data Collection |
| G. Krishna Mahesh | Backend Development |

---

## 🔮 Future Improvements

- Real-time microphone detection
- Mobile app (Android & iOS)
- Browser extension for live call verification
- Deep learning model (CNN / Transformer)
- Blockchain-based audio verification certificates
- WhatsApp / Telegram bot integration
- Partial deepfake detection with confidence timeline graph

---

## 📄 License

MIT License — free to use, modify and distribute.