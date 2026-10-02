"""Pitch tracking (YIN) and harmonic decomposition of a monophonic voice.

Pipeline
    1. DC removal + 50 Hz high-pass (mains hum / rumble).
    2. Frame-wise YIN f0 estimate with a noise gate and voicing decision.
    3. Median smoothing and segmentation into stable notes.
    4. Per note: zero-padded FFT, parabolic peak interpolation, and
       measurement of every harmonic n*f0 (frequency, level, deviation in cents).

A single voice has one f0 at a time; every other peak is a harmonic.  f0 is
taken from the periodicity of the waveform, never from the loudest spectral
peak, so a strong 2nd harmonic or a missing fundamental does not fool it.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft as sp_fft, signal
from scipy.ndimage import median_filter

from .notes import A4_DEFAULT, cents_between, note_info

# --- Analysis parameters -------------------------------------------------
FMIN_HZ = 60.0                # lowest f0 searched (bass)
FMAX_HZ = 1100.0              # highest f0 searched (soprano)
HOP_SEC = 0.010               # pitch-track resolution
HIGHPASS_HZ = 50.0
YIN_THRESHOLD = 0.15          # first CMNDF dip below this is the period
VOICING_LIMIT = 0.30          # frame is unvoiced if best CMNDF dip is above this
OCTAVE_CHECK_RATIO = 0.5      # prefer 2*tau if its dip is this much deeper
GATE_DB_DEFAULT = -50.0       # absolute RMS gate (dBFS)
GATE_RELATIVE_DB = 45.0       # frames this far below the loudest frame are unvoiced
SMOOTH_FRAMES = 5             # median filter length on the f0 track
SEGMENT_TOLERANCE_CENTS = 70  # a frame further than this from the note centre starts a new note
SEGMENT_MAX_GAP_SEC = 0.05    # unvoiced gap tolerated inside a note
SEGMENT_CONTOUR_SEC = 0.25    # contour smoothing used for note-change detection (> one vibrato cycle)
MIN_NOTE_SEC = 0.12
HARM_WINDOW_SEC = 0.085       # FFT window for harmonic measurement
HARM_ZERO_PAD = 4
HARM_SEARCH_CENTS = 40        # search window around n*f0
HARM_MAX_FREQ_HZ = 8000.0
HARM_MAX_COUNT = 30
HARM_PRESENT_SNR_DB = 10.0    # peak must stand this far above the local floor
HARM_MIN_REL_DB = 60.0        # ... and be at most this far below the strongest harmonic
HARM_MAX_FRAMES = 24          # frames sampled per note for harmonic averaging
SPECTRUM_POINTS = 480
VIBRATO_MIN_SEC = 0.6
VIBRATO_BAND_HZ = (3.0, 9.0)
MAX_DURATION_SEC = 180.0
MAX_TRACK_POINTS = 4000


class AnalysisError(ValueError):
    pass


# --- YIN -------------------------------------------------------------------

def _next_pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def _cmndf(frames: np.ndarray, w: int, tau_max: int) -> np.ndarray:
    """Cumulative mean normalised difference function for each frame row.

    ``frames`` has shape (n, w + tau_max); the result has shape (n, tau_max + 1).
    """
    n, length = frames.shape
    nfft = _next_pow2(length + w)
    spec_full = sp_fft.rfft(frames, nfft, axis=1, workers=-1)
    spec_head = sp_fft.rfft(frames[:, :w], nfft, axis=1, workers=-1)
    # r[tau] = sum_{j<w} x[j] * x[j + tau]
    r = sp_fft.irfft(spec_full * np.conj(spec_head), nfft, axis=1, workers=-1)[:, : tau_max + 1]

    cs = np.zeros((n, length + 1))
    np.cumsum(frames * frames, axis=1, out=cs[:, 1:])
    taus = np.arange(tau_max + 1)
    energy_0 = cs[:, w][:, None]
    energy_tau = cs[:, taus + w] - cs[:, taus]
    d = np.maximum(energy_0 + energy_tau - 2.0 * r, 0.0)
    d[:, 0] = 0.0

    out = np.ones_like(d)
    cum = np.cumsum(d[:, 1:], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[:, 1:] = np.where(cum > 0, d[:, 1:] * taus[1:] / cum, 1.0)
    return out


def _parabolic(y: np.ndarray, k: int) -> tuple[float, float]:
    """Vertex (position, value) of the parabola through y[k-1..k+1]."""
    if k <= 0 or k >= len(y) - 1:
        return float(k), float(y[k])
    a, b, c = y[k - 1], y[k], y[k + 1]
    denom = a - 2.0 * b + c
    if denom == 0:
        return float(k), float(b)
    delta = 0.5 * (a - c) / denom
    delta = max(-1.0, min(1.0, delta))
    return k + delta, float(b - 0.25 * (a - c) * delta)


def _local_min(row: np.ndarray, k: int) -> int:
    while k + 1 < len(row) and row[k + 1] < row[k]:
        k += 1
    return k


def _pick_period(row: np.ndarray, tau_min: int, tau_max: int) -> tuple[float, float]:
    """Return (tau, cmndf value) for one frame, or (nan, best value) if unvoiced."""
    seg = row[tau_min : tau_max + 1]
    below = np.flatnonzero(seg < YIN_THRESHOLD)
    if below.size:
        k = _local_min(row, tau_min + int(below[0]))
    else:
        k = tau_min + int(np.argmin(seg))
    if row[k] > VOICING_LIMIT:
        return math.nan, float(row[k])

    # Octave-up guard: when one harmonic dominates (strong H2, or H4 in
    # overtone singing), a dip at tau/m can pass the threshold.  If a dip at an
    # integer multiple m*tau is much deeper, the true period is m*tau.
    if row[k] > 0.02:
        best_k, best_v = k, row[k]
        for m in range(2, 7):
            m_lo, m_hi = int(m * k * 0.97), int(math.ceil(m * k * 1.03))
            if m_hi > tau_max:
                break
            km = m_lo + int(np.argmin(row[m_lo : m_hi + 1]))
            if row[km] < best_v:
                best_k, best_v = km, row[km]
        if best_v < OCTAVE_CHECK_RATIO * row[k]:
            k = best_k

    tau, val = _parabolic(row, k)
    return tau, max(val, 0.0)


def _yin_track(x: np.ndarray, sr: int, hop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    tau_min = max(2, int(sr / FMAX_HZ))
    tau_max = int(math.ceil(sr / FMIN_HZ))
    w = tau_max
    length = w + tau_max
    if x.size < length:
        raise AnalysisError("Recording is too short to analyse (needs at least ~40 ms).")

    windows = sliding_window_view(x, length)[::hop]
    n = windows.shape[0]
    f0 = np.full(n, np.nan)
    aperiodicity = np.ones(n)
    chunk = 512
    for start in range(0, n, chunk):
        block = np.ascontiguousarray(windows[start : start + chunk])
        cm = _cmndf(block, w, tau_max)
        for i, row in enumerate(cm):
            tau, val = _pick_period(row, tau_min, tau_max)
            aperiodicity[start + i] = val
            if math.isfinite(tau) and tau > 0:
                f0[start + i] = sr / tau

    rms = np.sqrt(np.mean(windows * windows, axis=1))
    rms_db = 20.0 * np.log10(rms + 1e-12)
    return f0, aperiodicity, rms_db, length


def _median_smooth(f0: np.ndarray, size: int) -> np.ndarray:
    out = f0.copy()
    half = size // 2
    voiced = np.isfinite(f0)
    logf = np.where(voiced, np.log2(np.where(voiced, f0, 1.0)), np.nan)
    for i in np.flatnonzero(voiced):
        win = logf[max(0, i - half) : i + half + 1]
        out[i] = 2.0 ** np.nanmedian(win)
    return out


# --- Segmentation ----------------------------------------------------------

def _segments(f0: np.ndarray, hop_sec: float) -> list[tuple[int, int]]:
    """Group voiced frames into notes; returns inclusive (first, last) frame indices.

    Phrases are split at unvoiced gaps.  Inside a phrase, note changes are
    detected on a median-smoothed contour longer than a vibrato cycle, so
    vibrato (up to about +-50 cents) stays one note while steps and glides split.
    """
    max_gap = max(1, int(round(SEGMENT_MAX_GAP_SEC / hop_sec)))
    contour_len = int(round(SEGMENT_CONTOUR_SEC / hop_sec)) | 1
    min_frames = int(round(MIN_NOTE_SEC / hop_sec))
    voiced_idx = np.flatnonzero(np.isfinite(f0))
    if voiced_idx.size == 0:
        return []

    breaks = np.flatnonzero(np.diff(voiced_idx) > max_gap + 1)
    phrases = np.split(voiced_idx, breaks + 1)

    out: list[tuple[int, int]] = []
    for ph in phrases:
        first, last = int(ph[0]), int(ph[-1])
        span = np.arange(first, last + 1)
        good = np.isfinite(f0[span])
        cents = 1200.0 * np.interp(span, span[good], np.log2(f0[span][good]))
        contour = median_filter(cents, size=contour_len, mode="nearest")
        start, total, count = first, 0.0, 0
        for j, c in enumerate(contour):
            if count and abs(c - total / count) > SEGMENT_TOLERANCE_CENTS:
                out.append((start, first + j - 1))
                start, total, count = first + j, 0.0, 0
            total += c
            count += 1
        out.append((start, last))
    return [(a, b) for a, b in out if b - a + 1 >= min_frames]


# --- Harmonics ---------------------------------------------------------------

def _measure_harmonics(x: np.ndarray, sr: int, centres: np.ndarray, f0s: np.ndarray) -> tuple[list[dict], dict]:
    nwin = int(round(HARM_WINDOW_SEC * sr))
    nfft = _next_pow2(nwin) * HARM_ZERO_PAD
    window = signal.get_window("hann", nwin, fftbins=True)
    gain = window.sum() / 2.0
    df = sr / nfft
    freqs = np.arange(nfft // 2 + 1) * df
    fmax = min(HARM_MAX_FREQ_HZ, 0.45 * sr)

    per_n: dict[int, list[tuple[float, float, float]]] = {}
    power_sum = np.zeros(nfft // 2 + 1)
    used = 0
    half = nwin // 2
    for c, f0 in zip(centres, f0s):
        seg = np.zeros(nwin)
        lo, hi = c - half, c - half + nwin
        src_lo, src_hi = max(lo, 0), min(hi, x.size)
        if src_hi - src_lo < nwin // 2:
            continue
        seg[src_lo - lo : src_hi - lo] = x[src_lo:src_hi]
        amp = np.abs(np.fft.rfft(seg * window, nfft)) / gain
        power_sum += amp * amp
        used += 1
        db = 20.0 * np.log10(amp + 1e-12)

        count = min(HARM_MAX_COUNT, int(fmax // f0))
        for n in range(1, count + 1):
            target = n * f0
            span = max(target * (2 ** (HARM_SEARCH_CENTS / 1200.0) - 1.0), 1.5 * df)
            k_lo, k_hi = int((target - span) / df), int(math.ceil((target + span) / df))
            k_lo, k_hi = max(k_lo, 1), min(k_hi, len(db) - 2)
            if k_hi <= k_lo:
                continue
            k = k_lo + int(np.argmax(db[k_lo : k_hi + 1]))
            # A maximum on the window edge is the skirt of a neighbour, not a peak.
            is_peak = k_lo < k < k_hi and db[k] >= db[k - 1] and db[k] >= db[k + 1]
            pos, peak_db = _parabolic(db, k)
            peak_freq = pos * df
            band_lo = max(1, int((target - 0.5 * f0) / df))
            band_hi = min(len(db) - 1, int((target + 0.5 * f0) / df))
            # Low percentile: for low f0 the window's main lobe covers much of the band.
            band = db[band_lo : band_hi + 1]
            q = int(0.2 * (band.size - 1))
            floor_db = float(np.partition(band, q)[q])  # 20th percentile, without np.percentile overhead
            snr = peak_db - floor_db if is_peak else -math.inf
            per_n.setdefault(n, []).append((cents_between(peak_freq, target), peak_db, snr))

    if used == 0:
        return [], {"freq": [], "db": []}

    levels = {n: float(10.0 * np.log10(np.mean(10.0 ** (np.array(v)[:, 1] / 10.0)))) for n, v in per_n.items()}
    top_db = max(levels.values(), default=0.0)
    harmonics = []
    for n in sorted(per_n):
        vals = np.array(per_n[n])
        present_ratio = float(np.mean(vals[:, 2] >= HARM_PRESENT_SNR_DB))
        harmonics.append({
            "n": n,
            "cents_dev": float(np.median(vals[:, 0])),
            "level_dbfs": levels[n],
            "snr_db": float(np.median(np.maximum(vals[:, 2], -99.0))),
            # Both above the local noise floor and within range of the strongest
            # harmonic (window leakage of a clean tone has a huge local SNR).
            "present": present_ratio >= 0.5 and levels[n] >= top_db - HARM_MIN_REL_DB,
        })

    spectrum = _log_spectrum(freqs, 10.0 * np.log10(power_sum / used + 1e-24), fmax)
    return harmonics, spectrum


def _log_spectrum(freqs: np.ndarray, db: np.ndarray, fmax: float) -> dict:
    """Max-pool the spectrum onto a log-spaced grid for plotting."""
    grid = np.geomspace(40.0, fmax, SPECTRUM_POINTS + 1)
    idx = np.searchsorted(freqs, grid)
    out_f, out_db = [], []
    for i in range(SPECTRUM_POINTS):
        a, b = idx[i], max(idx[i + 1], idx[i] + 1)
        if a >= len(db):
            break
        out_f.append(round(float(math.sqrt(grid[i] * grid[i + 1])), 2))
        out_db.append(round(float(np.max(db[a:b])), 1))
    return {"freq": out_f, "db": out_db}


def _vibrato(f0_seg: np.ndarray, hop_sec: float) -> dict | None:
    if f0_seg.size * hop_sec < VIBRATO_MIN_SEC:
        return None
    t = np.arange(f0_seg.size)
    good = np.isfinite(f0_seg)
    if good.sum() < 0.8 * f0_seg.size:
        return None
    f = np.interp(t, t[good], f0_seg[good])
    cents = 1200.0 * np.log2(f / np.median(f))
    cents = signal.detrend(cents)
    fps = 1.0 / hop_sec
    nfft = max(4096, _next_pow2(cents.size) * 4)
    spec = np.abs(np.fft.rfft(cents * np.hanning(cents.size), nfft)) ** 2
    fr = np.fft.rfftfreq(nfft, hop_sec)
    band = (fr >= VIBRATO_BAND_HZ[0]) & (fr <= VIBRATO_BAND_HZ[1])
    total = spec[(fr > 0.5) & (fr < fps / 2)].sum()
    if total <= 0 or not band.any():
        return None
    k = np.flatnonzero(band)[np.argmax(spec[band])]
    pos, _ = _parabolic(spec, int(k))
    rate = pos * fr[1]
    near = (fr >= rate - 1.0) & (fr <= rate + 1.0)
    extent = math.sqrt(2.0) * float(np.std(cents))
    if spec[near].sum() / total < 0.5 or extent < 10.0:
        return None
    return {"rate_hz": round(rate, 2), "extent_cents": round(extent, 1)}


# --- Public API ------------------------------------------------------------

def _round(v: float | None, nd: int = 2):
    return None if v is None or not math.isfinite(v) else round(float(v), nd)


def analyze(x: np.ndarray, sr: int, a4: float = A4_DEFAULT, gate_db: float = GATE_DB_DEFAULT) -> dict:
    """Analyse a mono signal and return a JSON-serialisable result."""
    if sr < 8000:
        raise AnalysisError(f"Sample rate {sr} Hz is too low (minimum 8000 Hz).")
    duration = x.size / sr
    if duration > MAX_DURATION_SEC:
        raise AnalysisError(f"Recording is {duration:.0f} s long; the limit is {MAX_DURATION_SEC:.0f} s.")

    warnings: list[str] = []
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak >= 0.999:
        warnings.append("The recording clips (peak at 0 dBFS). Lower the input level or move away from the microphone.")

    x = x - np.mean(x)
    sos = signal.butter(4, HIGHPASS_HZ, btype="highpass", fs=sr, output="sos")
    if x.size > 3 * 27:  # sosfiltfilt padding requirement
        x = signal.sosfiltfilt(sos, x)

    hop = max(1, int(round(HOP_SEC * sr)))
    hop_sec = hop / sr
    f0_raw, aper, rms_db, frame_len = _yin_track(x, sr, hop)
    times = (np.arange(f0_raw.size) * hop + frame_len / 2) / sr

    loudest = float(np.max(rms_db))
    gated = (rms_db < gate_db) | (rms_db < loudest - GATE_RELATIVE_DB)
    f0_raw[gated] = np.nan
    if loudest < gate_db:
        warnings.append(f"The recording is very quiet (loudest frame {loudest:.0f} dBFS). "
                        "Check the microphone selection, its level and the OS privacy settings.")

    f0 = _median_smooth(f0_raw, SMOOTH_FRAMES)
    voiced = np.isfinite(f0)
    voiced_ratio = float(voiced.mean()) if voiced.size else 0.0

    segments = []
    for first, last in _segments(f0, hop_sec):
        idx = np.arange(first, last + 1)
        idx_v = idx[np.isfinite(f0[idx])]
        seg_f0 = float(2.0 ** np.median(np.log2(f0[idx_v])))
        # Keep the FFT windows inside the note so neighbouring notes do not leak in.
        margin = HARM_WINDOW_SEC / 2
        core = idx_v[(times[idx_v] - times[first] >= margin) & (times[last] - times[idx_v] >= margin)]
        if core.size >= 3:
            idx_v = core
        pick = idx_v if idx_v.size <= HARM_MAX_FRAMES else idx_v[np.linspace(0, idx_v.size - 1, HARM_MAX_FRAMES).astype(int)]
        centres = np.round(times[pick] * sr).astype(int)
        harmonics, spectrum = _measure_harmonics(x, sr, centres, f0[pick])

        present = [h for h in harmonics if h["present"]]
        ref_db = max((h["level_dbfs"] for h in present), default=None)
        for h in harmonics:
            freq = h["n"] * seg_f0 * 2 ** (h["cents_dev"] / 1200.0)
            note = note_info(freq, a4)
            h.update({
                "freq": _round(freq),
                "ideal_freq": _round(h["n"] * seg_f0),
                "note": note["label"] if note else None,
                "note_fr": note["label_fr"] if note else None,
                "note_cents": note["cents"] if note else None,
                "level_rel_db": _round(h["level_dbfs"] - ref_db, 1) if ref_db is not None else None,
                "cents_dev": _round(h["cents_dev"], 1),
                "level_dbfs": _round(h["level_dbfs"], 1),
                "snr_db": _round(h["snr_db"], 1),
            })
        strongest = max(present, key=lambda h: h["level_dbfs"], default=None)

        note = note_info(seg_f0, a4)
        segments.append({
            "start": _round(times[first] - hop_sec / 2, 3),
            "end": _round(times[last] + hop_sec / 2, 3),
            "duration": _round((last - first + 1) * hop_sec, 3),
            "f0": _round(seg_f0),
            "note": note,
            "confidence": _round(float(np.mean(1.0 - aper[idx_v])), 3),
            "vibrato": _vibrato(f0[first : last + 1], hop_sec),
            "harmonics": harmonics,
            "harmonics_present": len(present),
            "strongest_harmonic": strongest["n"] if strongest else None,
            "spectrum": spectrum,
        })

    if voiced_ratio < 0.05 and not warnings:
        warnings.append("No clear pitch was found. Sing a sustained vowel closer to the microphone.")

    stride = max(1, int(math.ceil(f0.size / MAX_TRACK_POINTS)))
    median_f0 = float(2.0 ** np.median(np.log2(f0[voiced]))) if voiced.any() else None
    by_pitch = sorted(segments, key=lambda s: s["f0"])

    return {
        "sample_rate": sr,
        "duration": _round(duration, 3),
        "a4": a4,
        "summary": {
            "voiced_ratio": _round(voiced_ratio, 3),
            "median_f0": _round(median_f0),
            "median_note": note_info(median_f0, a4) if median_f0 else None,
            "lowest": by_pitch[0]["note"] if by_pitch else None,
            "highest": by_pitch[-1]["note"] if by_pitch else None,
            "segment_count": len(segments),
            "peak_dbfs": _round(20.0 * math.log10(peak + 1e-12), 1),
            "loudest_frame_dbfs": _round(loudest, 1),
        },
        "track": {
            "t": [_round(v, 3) for v in times[::stride]],
            "f0": [_round(v) for v in f0[::stride]],
        },
        "segments": segments,
        "warnings": warnings,
        "params": {
            "fmin": FMIN_HZ, "fmax": FMAX_HZ, "hop_sec": hop_sec, "gate_db": gate_db,
            "harm_window_sec": HARM_WINDOW_SEC, "harm_search_cents": HARM_SEARCH_CENTS,
        },
    }
