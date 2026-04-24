# VoiceGuard — Audio Deepfake Detector

Detects AI-cloned or synthetic speech using MFCC, spectral, and prosody feature analysis.

## Quick start

```bash
pip install -r requirements.txt
python model/train.py          # trains on synthetic data (~60s)
uvicorn api.server:app --reload --port 8000
# open frontend/index.html in browser
```

## With real ASVspoof data

```bash
# Download ASVspoof 2019 LA partition, unzip to ./data/LA
# Organise as:   data/LA/real/   data/LA/fake/
python model/train.py --data_dir ./data/LA
```

## Docker

```bash
docker build -t voiceguard .
docker run -p 8000:8000 voiceguard
```

## API

| Method | Endpoint     | Description              |
|--------|-------------|--------------------------|
| GET    | /health     | Health check             |
| POST   | /predict    | Upload audio, get result |
| GET    | /model/info | Model metadata           |

### POST /predict response

```json
{
  "label": "FAKE",
  "confidence": 0.91,
  "fake_score": 0.91,
  "real_score": 0.09,
  "features": 270,
  "processing_time_ms": 145.2,
  "filename": "sample.wav"
}
```

## Feature pipeline

- **MFCC** (40 coefficients × delta × delta2) — 240 features
- **Spectral** (centroid, bandwidth, rolloff, flatness, ZCR, RMS) — 6 features
- **Chroma + contrast** — 19 features
- **Prosody** (F0 mean/std/percentiles, voiced ratio) — 5 features

Total: **270-dimensional feature vector**

## Model

Voting ensemble: SVM (RBF kernel) + Gradient Boosting Classifier  
Trained on ASVspoof 2019 LA or synthetic data  
Evaluation metric: ROC-AUC

## Team

Built at AIML Hackathon 2025 in 24 hours.