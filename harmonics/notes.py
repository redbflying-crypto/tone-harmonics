"""Frequency <-> MIDI <-> note name conversion.

Note names are derived from the rounded MIDI number, whose pitch class 0 is C.
(The previous version indexed a C-based name list with an A-based semitone
count, which labelled A4 as "C4" and C4 as "D#4".)
"""

from __future__ import annotations

import math

A4_DEFAULT = 440.0

NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
NAMES_SHARP_FR = ["Do", "Do#", "Ré", "Ré#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]
NAMES_FLAT_FR = ["Do", "Réb", "Ré", "Mib", "Mi", "Fa", "Solb", "Sol", "Lab", "La", "Sib", "Si"]


def freq_to_midi(freq: float, a4: float = A4_DEFAULT) -> float:
    """Fractional MIDI number (A4 = 69)."""
    return 69.0 + 12.0 * math.log2(freq / a4)


def midi_to_freq(midi: float, a4: float = A4_DEFAULT) -> float:
    return a4 * 2.0 ** ((midi - 69.0) / 12.0)


def _label(sharp_names: list[str], flat_names: list[str], pc: int, octave: int) -> str:
    sharp = f"{sharp_names[pc]}{octave}"
    flat = f"{flat_names[pc]}{octave}"
    return sharp if sharp == flat else f"{sharp}/{flat}"


def note_info(freq: float | None, a4: float = A4_DEFAULT) -> dict | None:
    """Describe the nearest equal-tempered note to ``freq``.

    Returns None for missing, non-finite or non-positive frequencies.
    """
    if freq is None or not math.isfinite(freq) or freq <= 0:
        return None
    midi = freq_to_midi(freq, a4)
    m = int(round(midi))
    pc = m % 12
    octave = m // 12 - 1
    return {
        "midi": m,
        "cents": round(100.0 * (midi - m), 1),
        "name": f"{NAMES_SHARP[pc]}{octave}",
        "flat": f"{NAMES_FLAT[pc]}{octave}",
        "label": _label(NAMES_SHARP, NAMES_FLAT, pc, octave),
        "label_fr": _label(NAMES_SHARP_FR, NAMES_FLAT_FR, pc, octave),
        "ref_freq": round(midi_to_freq(m, a4), 3),
    }


def cents_between(freq: float, ref: float) -> float:
    return 1200.0 * math.log2(freq / ref)
