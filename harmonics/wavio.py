"""Decode WAV bytes into a mono float signal."""

from __future__ import annotations

import io

import numpy as np
from scipy.io import wavfile


class AudioDecodeError(ValueError):
    pass


def read_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Return (mono float64 signal in [-1, 1], sample rate)."""
    if not data:
        raise AudioDecodeError("Empty audio payload.")
    try:
        sr, x = wavfile.read(io.BytesIO(data))
    except Exception as exc:  # scipy raises ValueError / struct errors on bad input
        raise AudioDecodeError(f"Not a readable WAV file ({exc}).") from exc

    if x.dtype == np.uint8:
        y = (x.astype(np.float64) - 128.0) / 128.0
    elif x.dtype == np.int16:
        y = x.astype(np.float64) / 32768.0
    elif x.dtype == np.int32:  # 32-bit, and 24-bit left-justified by scipy
        y = x.astype(np.float64) / 2147483648.0
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float64)
    else:
        raise AudioDecodeError(f"Unsupported WAV sample format: {x.dtype}.")

    if y.ndim == 2:
        y = y.mean(axis=1)
    if y.ndim != 1 or y.size == 0:
        raise AudioDecodeError("WAV file contains no samples.")
    if not np.all(np.isfinite(y)):
        raise AudioDecodeError("WAV file contains non-finite samples.")
    return y, int(sr)
