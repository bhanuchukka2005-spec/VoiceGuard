FROM python:3.11-slim

WORKDIR /app

# System deps:
#   libsndfile1 — soundfile / librosa WAV+FLAC support
#   ffmpeg      — pydub fallback for wma, opus, 3gp, amr, caf, etc.
#   curl        — used by Docker HEALTHCHECK and CI smoke test
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root user for container security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Docker health check — container reports "healthy" once /health returns 200
HEALTHCHECK --interval=30s --timeout=10s --start-period=25s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Train model on first run if artefacts are missing, then start API
CMD ["sh", "-c", "[ -f model/detector.joblib ] || python model/train.py && uvicorn api.server:app --host 0.0.0.0 --port 8000"]

EXPOSE 8000
