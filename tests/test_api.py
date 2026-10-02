import io

import numpy as np
import pytest
from scipy.io import wavfile

from index import MAX_UPLOAD_BYTES, app


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def wav_bytes(x, sr, dtype=np.int16):
    buf = io.BytesIO()
    if dtype == np.int16:
        data = np.clip(x * 32767, -32768, 32767).astype(np.int16)
    else:
        data = x.astype(np.float32)
    wavfile.write(buf, sr, data)
    return buf.getvalue()


def sung(f0=220.0, sr=44100, dur=1.0):
    t = np.arange(int(sr * dur)) / sr
    return 0.3 * sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))


def test_health(client):
    r = client.get("/api/health")
    assert r.get_json() == {"status": "ok"}
    assert r.headers["Cache-Control"] == "no-store"


def test_upload_limit_is_json(client):
    r = client.post("/api/analyze", data=b"\0" * (MAX_UPLOAD_BYTES + 1))
    assert r.status_code == 413
    assert "error" in r.get_json()


def test_forty_seconds_at_48k_fits_the_limit():
    assert 44 + 40 * 48000 * 2 < MAX_UPLOAD_BYTES < 4.5e6


def test_analyze_raw_wav(client):
    r = client.post("/api/analyze", data=wav_bytes(sung(), 44100), content_type="audio/wav")
    assert r.status_code == 200
    body = r.get_json()
    assert body["segments"][0]["note"]["label"] == "A3"
    assert body["sample_rate"] == 44100


def test_analyze_multipart_stereo_float(client):
    x = sung(f0=261.63, sr=48000)
    stereo = np.stack([x, x], axis=1)
    data = {"audio": (io.BytesIO(wav_bytes(stereo, 48000, np.float32)), "take.wav")}
    r = client.post("/api/analyze?a4=440", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    assert r.get_json()["segments"][0]["note"]["label"] == "C4"


def test_custom_a4(client):
    r = client.post("/api/analyze?a4=432", data=wav_bytes(sung(432.0), 44100))
    assert r.get_json()["segments"][0]["note"]["label"] == "A4"


@pytest.mark.parametrize("query", ["a4=abc", "a4=1000", "gate=5"])
def test_bad_parameters(client, query):
    r = client.post(f"/api/analyze?{query}", data=wav_bytes(sung(), 44100))
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_bad_audio(client):
    for payload in (b"", b"not a wav file"):
        r = client.post("/api/analyze", data=payload)
        assert r.status_code == 400
        assert "error" in r.get_json()


def test_wrong_method_is_json(client):
    r = client.get("/api/analyze")
    assert r.status_code == 405
    assert "error" in r.get_json()
