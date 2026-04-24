FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train model on container start if not already trained
CMD ["sh", "-c", "[ -f model/detector.joblib ] || python model/train.py && uvicorn api.server:app --host 0.0.0.0 --port 8000"]

EXPOSE 8000