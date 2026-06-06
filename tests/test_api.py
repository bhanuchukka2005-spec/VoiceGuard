"""
VoiceGuard API tests — v2.3
Run with: pytest tests/ -v

Requires the model to be trained first:
  python model/train.py
"""

import io
import os
import sys
import pytest
import numpy as np
import soundfile as sf

# Make sure the project root is on sys.path so server.py imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from httpx import AsyncClient, ASGITransport
from api.server import app


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_wav(duration: float = 2.0, sr: int = 16000, freq: float = 150.0) -> bytes:
    """Generate a minimal valid mono WAV file in memory."""
    t    = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    wave = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)
    # Add a second harmonic and tiny noise so the clip isn't pure sine
    wave += 0.2 * np.sin(2 * np.pi * 2 * freq * t)
    wave += np.random.default_rng(42).normal(0, 0.01, len(wave)).astype(np.float32)
    wave /= np.abs(wave).max()
    wave *= 0.85
    buf = io.BytesIO()
    sf.write(buf, wave, sr, format="WAV")
    return buf.getvalue()


def _make_silent_wav(duration: float = 2.0, sr: int = 16000) -> bytes:
    """WAV file containing only silence."""
    wave = np.zeros(int(sr * duration), dtype=np.float32)
    buf  = io.BytesIO()
    sf.write(buf, wave, sr, format="WAV")
    return buf.getvalue()


def _make_short_wav(duration: float = 0.1, sr: int = 16000) -> bytes:
    """WAV file shorter than MIN_DURATION_SEC (0.5s)."""
    t    = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    wave = (np.sin(2 * np.pi * 200 * t) * 0.5).astype(np.float32)
    buf  = io.BytesIO()
    sf.write(buf, wave, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture
def wav_bytes():
    return _make_wav()


@pytest.fixture
def silent_wav_bytes():
    return _make_silent_wav()


@pytest.fixture
def short_wav_bytes():
    return _make_short_wav()


# ── Health & info ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_returns_200():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_health_schema():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    body = r.json()
    assert "status"       in body
    assert "model_loaded" in body
    assert "version"      in body
    assert "stats"        in body


@pytest.mark.asyncio
async def test_model_info():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["features"].startswith("270")


@pytest.mark.asyncio
async def test_stats_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    for key in ("total", "fake", "real", "errors", "fake_rate", "real_rate"):
        assert key in body, f"Missing key: {key}"


# ── /predict ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_predict_valid_wav(wav_bytes):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/predict",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in ("REAL", "FAKE")
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["fake_score"] <= 1.0
    assert 0.0 <= body["real_score"] <= 1.0
    assert abs(body["fake_score"] + body["real_score"] - 1.0) < 0.01
    assert body["features"] == 270
    assert body["processing_time_ms"] > 0
    assert body["filename"] == "test.wav"
    assert isinstance(body["top_features"], list)
    assert isinstance(body["verdict_reason"], str)
    assert len(body["verdict_reason"]) > 0


@pytest.mark.asyncio
async def test_predict_rejects_non_audio():
    """Uploading a text/binary file should return 400."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/predict",
            files={"file": ("evil.exe", b"MZ\x90\x00not audio", "application/octet-stream")},
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_predict_rejects_oversized_file():
    """File > MAX_MB should return 413."""
    big = b"\x00" * (51 * 1024 * 1024)   # 51 MB of zeros
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/predict",
            files={"file": ("big.wav", big, "audio/wav")},
        )
    assert r.status_code == 413


@pytest.mark.asyncio
async def test_predict_rejects_silent_audio(silent_wav_bytes):
    """Silent audio should return 422 (Unprocessable Entity)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/predict",
            files={"file": ("silent.wav", silent_wav_bytes, "audio/wav")},
        )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_predict_rejects_short_audio(short_wav_bytes):
    """Audio shorter than 0.5s should return 422."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/predict",
            files={"file": ("short.wav", short_wav_bytes, "audio/wav")},
        )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_predict_accepts_mp3_extension(wav_bytes):
    """A WAV file uploaded with .mp3 extension should be accepted (magic-byte sniff)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/predict",
            files={"file": ("test.mp3", wav_bytes, "audio/mpeg")},
        )
    # Should succeed (200) or fail due to model not trained (503), never 400
    assert r.status_code in (200, 503)


# ── /predict/segments ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_segments_returns_list(wav_bytes):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/predict/segments",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
    assert r.status_code == 200
    body = r.json()
    assert "segments" in body
    assert "overall"  in body
    assert isinstance(body["segments"], list)
    if body["segments"]:
        seg = body["segments"][0]
        assert "start_sec"  in seg
        assert "end_sec"    in seg
        assert "fake_score" in seg
        assert "real_score" in seg
        assert 0.0 <= seg["fake_score"] <= 1.0


@pytest.mark.asyncio
async def test_segments_fake_score_range(wav_bytes):
    """All segment fake_scores must be in [0, 1]."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/predict/segments",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
    body = r.json()
    for seg in body.get("segments", []):
        assert 0.0 <= seg["fake_score"] <= 1.0, f"Out-of-range fake_score: {seg}"
        assert 0.0 <= seg["real_score"] <= 1.0, f"Out-of-range real_score: {seg}"


# ── /predict/stress ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stress_returns_five_metrics(wav_bytes):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/predict/stress",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
    assert r.status_code == 200
    body = r.json()
    for key in ("pitch_stability", "rhythm_naturalness", "breath_patterns",
                "micro_variations", "formant_stability"):
        assert key in body, f"Missing stress metric: {key}"
        assert 0.0 <= body[key] <= 1.0, f"Out-of-range stress metric {key}: {body[key]}"


# ── /predict/batch ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_batch_two_files(wav_bytes):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/predict/batch",
            files=[
                ("files", ("a.wav", wav_bytes, "audio/wav")),
                ("files", ("b.wav", wav_bytes, "audio/wav")),
            ],
        )
    assert r.status_code == 200
    body = r.json()
    assert body["total_files"] == 2
    assert len(body["results"]) == 2
    assert body["fake_count"] + body["real_count"] + body["error_count"] == 2


@pytest.mark.asyncio
async def test_batch_rejects_more_than_ten(wav_bytes):
    files = [("files", (f"file{i}.wav", wav_bytes, "audio/wav")) for i in range(11)]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/predict/batch", files=files)
    assert r.status_code == 400


# ── /predict/compare ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_compare_two_files(wav_bytes):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/predict/compare",
            files=[
                ("file_a", ("a.wav", wav_bytes, "audio/wav")),
                ("file_b", ("b.wav", wav_bytes, "audio/wav")),
            ],
        )
    assert r.status_code == 200
    body = r.json()
    assert "a" in body and "b" in body
    assert body["a"]["label"] in ("REAL", "FAKE")
    assert body["b"]["label"] in ("REAL", "FAKE")


# ── Stats are updated by predictions ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_stats_increment_after_predict(wav_bytes):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        before = (await client.get("/stats")).json()["total"]
        await client.post("/predict", files={"file": ("t.wav", wav_bytes, "audio/wav")})
        after  = (await client.get("/stats")).json()["total"]
    assert after == before + 1
