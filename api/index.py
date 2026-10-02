"""Voice Harmonics Analyzer — Vercel serverless function (Flask, WSGI).

Vercel serves the page from ``public/`` through its CDN, and ``vercel.json``
rewrites every ``/api/*`` request to this function.  The function is
stateless: it analyses one WAV file per request and never stores audio.
"""

from __future__ import annotations

import math
import os
import sys

# ``harmonics`` lives at the project root (bundled through ``includeFiles``).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flask import Flask, jsonify, request  # noqa: E402
from werkzeug.exceptions import HTTPException  # noqa: E402

from harmonics import AnalysisError, AudioDecodeError, analyze, read_wav_bytes  # noqa: E402

# Vercel rejects function request bodies above 4.5 MB before they reach Flask.
# 40 s of 48 kHz 16-bit mono WAV is about 3.84 MB.
MAX_UPLOAD_BYTES = 4_400_000

app = Flask(__name__, static_folder=None)  # static files are served by Vercel's CDN
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def _float_arg(name: str, default: float, lo: float, hi: float) -> float:
    raw = request.args.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        raise AnalysisError(f"Parameter '{name}' must be a number.")
    if not math.isfinite(value) or not lo <= value <= hi:
        raise AnalysisError(f"Parameter '{name}' must be between {lo} and {hi}.")
    return value


@app.after_request
def _no_store(resp):
    resp.headers.setdefault("Cache-Control", "no-store")
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    return resp


@app.get("/api/health")
def health():
    return jsonify(status="ok")


@app.post("/api/analyze")
def api_analyze():
    try:
        a4 = _float_arg("a4", 440.0, 400.0, 480.0)
        gate_db = _float_arg("gate", -50.0, -90.0, -10.0)
        upload = request.files.get("audio")
        data = upload.read() if upload else request.get_data(cache=False)
        x, sr = read_wav_bytes(data)
        result = analyze(x, sr, a4=a4, gate_db=gate_db)
    except (AnalysisError, AudioDecodeError) as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(result)


@app.errorhandler(413)
def too_large(_exc):
    return jsonify(error=f"Upload too large (limit {MAX_UPLOAD_BYTES / 1e6:.1f} MB, about 40 s of audio)."), 413


@app.errorhandler(HTTPException)
def http_error(exc: HTTPException):
    return jsonify(error=exc.description), exc.code


@app.errorhandler(Exception)
def unexpected(_exc: Exception):
    app.logger.exception("Unhandled error")
    return jsonify(error="Internal error while analysing the recording."), 500
