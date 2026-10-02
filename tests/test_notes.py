import pytest

from harmonics.notes import note_info


@pytest.mark.parametrize(
    "freq, label, cents",
    [
        (440.0, "A4", 0.0),
        (261.63, "C4", 0.0),
        (523.25, "C5", 0.0),
        (259.5, "C4", -14.1),
        (277.18, "C#4/Db4", 0.0),
        (27.5, "A0", 0.0),
        (4186.01, "C8", 0.0),
        (65.41, "C2", 0.0),
        (263.78, "C4", 14.2),   # was reported as "D#4" by the previous version
        (522.18, "C5", -3.5),   # was reported as "D#4" by the previous version
    ],
)
def test_note_names(freq, label, cents):
    info = note_info(freq)
    assert info["label"] == label
    assert info["cents"] == pytest.approx(cents, abs=0.15)


def test_french_names_and_a4():
    assert note_info(440.0)["label_fr"] == "La4"
    assert note_info(311.13)["label_fr"] == "Ré#4/Mib4"
    assert note_info(432.0, a4=432.0)["label"] == "A4"


@pytest.mark.parametrize("bad", [None, 0.0, -5.0, float("nan"), float("inf")])
def test_invalid_frequencies(bad):
    assert note_info(bad) is None
