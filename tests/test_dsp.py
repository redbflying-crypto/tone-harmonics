"""Synthetic-signal tests for the analysis pipeline (no audio hardware needed)."""

import numpy as np
import pytest

from harmonics import AnalysisError, analyze

SR = 48000


def tone(f0, amps, dur=1.0, sr=SR, vibrato=None, level=0.3):
    """Harmonic tone; amps[k] is the linear amplitude of harmonic k+1."""
    t = np.arange(int(dur * sr)) / sr
    if vibrato:
        rate, extent_cents = vibrato
        inst = f0 * 2 ** (extent_cents * np.sin(2 * np.pi * rate * t) / 1200)
        phase = 2 * np.pi * np.cumsum(inst) / sr
    else:
        phase = 2 * np.pi * f0 * t
    x = sum(a * np.sin((k + 1) * phase) for k, a in enumerate(amps) if a)
    x = x / np.max(np.abs(x)) * level
    fade = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.02)
    return x * fade


def main_segment(result):
    assert result["segments"], result["warnings"]
    return max(result["segments"], key=lambda s: s["duration"])


def harmonic(seg, n):
    return next(h for h in seg["harmonics"] if h["n"] == n)


def voice_amps(count=12):
    return [1.0 / k for k in range(1, count + 1)]


def test_pure_sine():
    r = analyze(tone(220.0, [1.0]), SR)
    seg = main_segment(r)
    assert seg["f0"] == pytest.approx(220.0, abs=0.5)
    assert seg["note"]["label"] == "A3"
    assert [h["n"] for h in seg["harmonics"] if h["present"]] == [1]
    assert len(r["segments"]) == 1


def test_piano_case_strong_second_harmonic():
    # H2 is 6 dB louder than H1: the old analyzer reported C5 (as "D#4").
    amps = [1.0, 2.0, 0.6, 0.4, 0.25, 0.15]
    seg = main_segment(analyze(tone(261.63, amps), SR))
    assert seg["note"]["label"] == "C4"
    assert seg["f0"] == pytest.approx(261.63, abs=1.0)
    assert seg["strongest_harmonic"] == 2
    assert abs(harmonic(seg, 2)["cents_dev"]) < 5


def test_missing_fundamental():
    amps = [0.0] + [1.0 / k for k in range(2, 9)]
    seg = main_segment(analyze(tone(110.0, amps), SR))
    assert seg["note"]["label"] == "A2"
    assert seg["f0"] == pytest.approx(110.0, abs=0.5)
    assert harmonic(seg, 1)["present"] is False
    assert all(harmonic(seg, n)["present"] for n in range(2, 9))


def test_overtone_singing():
    amps = voice_amps(10)
    amps[3] = 4.0  # H4 12 dB above the fundamental
    seg = main_segment(analyze(tone(150.0, amps), SR))
    assert seg["f0"] == pytest.approx(150.0, abs=0.5)
    assert seg["strongest_harmonic"] == 4
    assert harmonic(seg, 4)["level_rel_db"] == 0.0


def test_vibrato_is_one_note():
    r = analyze(tone(220.0, voice_amps(), dur=2.0, vibrato=(5.5, 50)), SR)
    assert len(r["segments"]) == 1
    seg = r["segments"][0]
    assert abs(1200 * np.log2(seg["f0"] / 220.0)) < 10
    assert seg["vibrato"]["rate_hz"] == pytest.approx(5.5, abs=0.3)
    assert seg["vibrato"]["extent_cents"] == pytest.approx(50, abs=12)


def test_noise_and_silence_have_no_pitch():
    rng = np.random.default_rng(0)
    r = analyze(0.1 * rng.standard_normal(SR), SR)
    assert r["segments"] == []
    r = analyze(np.zeros(SR), SR)
    assert r["segments"] == []
    assert r["warnings"]


def test_mains_hum_does_not_corrupt_pitch():
    t = np.arange(SR) / SR
    voice = tone(300.0, voice_amps(), level=0.3)
    hum = 0.03 * sum(np.sin(2 * np.pi * 50 * k * t) / k for k in range(1, 6))
    seg = main_segment(analyze(voice + hum, SR))
    assert seg["f0"] == pytest.approx(300.0, abs=1.0)
    assert seg["note"]["label"] == "D4"


@pytest.mark.parametrize("sr", [16000, 44100, 48000])
def test_sample_rate_independence(sr):
    seg = main_segment(analyze(tone(196.0, voice_amps(), sr=sr), sr))
    assert seg["note"]["label"] == "G3"
    assert seg["f0"] == pytest.approx(196.0, abs=0.5)


@pytest.mark.parametrize("f0, label", [(82.41, "E2"), (1046.5, "C6")])
def test_vocal_range_limits(f0, label):
    seg = main_segment(analyze(tone(f0, voice_amps(6)), SR))
    assert seg["note"]["label"] == label


def test_two_sung_notes_are_two_segments():
    x = np.concatenate([tone(220.0, voice_amps()), np.zeros(SR // 5), tone(246.94, voice_amps())])
    r = analyze(x, SR)
    assert [s["note"]["label"] for s in r["segments"]] == ["A3", "B3"]


def test_legato_semitone_step_splits():
    x = np.concatenate([tone(220.0, voice_amps(), level=0.3), tone(233.08, voice_amps(), level=0.3)])
    segs = analyze(x, SR)["segments"]
    assert [s["note"]["label"] for s in segs] == ["A3", "A#3/Bb3"]
    # The neighbouring note must not leak extra harmonics into either note.
    assert [s["harmonics_present"] for s in segs] == [12, 12]


def test_harmonic_frequencies_are_precise():
    seg = main_segment(analyze(tone(233.08, voice_amps()), SR))
    for n in range(1, 9):
        h = harmonic(seg, n)
        assert h["present"]
        assert abs(h["cents_dev"]) < 3, (n, h)


def test_no_phantom_harmonics_with_vibrato():
    # 12 harmonics only: H13+ must not be reported even when vibrato smears the spectrum.
    seg = main_segment(analyze(tone(220.0, voice_amps(12), dur=1.5, vibrato=(5.5, 30)), SR))
    present = [h["n"] for h in seg["harmonics"] if h["present"]]
    assert present == list(range(1, 13))


def test_too_short_and_bad_rate():
    with pytest.raises(AnalysisError):
        analyze(np.zeros(100), SR)
    with pytest.raises(AnalysisError):
        analyze(np.zeros(4000), 4000)
