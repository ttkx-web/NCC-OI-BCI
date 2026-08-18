from __future__ import annotations

import numpy as np

from bci_dayloop.demo.schemas import MOTOR_LABELS_CN, STATE_LABELS_CN
from bci_dayloop.demo.state_decoder import DemoStateDecoder


def _eeg_window(sample_rate: float = 250.0, seconds: float = 2.0) -> np.ndarray:
    time = np.arange(int(sample_rate * seconds)) / sample_rate
    rng = np.random.default_rng(7)
    alpha = 12e-6 * np.sin(2 * np.pi * 10 * time)
    beta = 6e-6 * np.sin(2 * np.pi * 20 * time)
    return np.vstack([alpha + beta + rng.normal(0, 1e-6, time.size) for _ in range(4)]).astype(np.float32)


def test_demo_state_decoder_returns_complete_presentation_contract() -> None:
    decoder = DemoStateDecoder(device="cpu")
    result = decoder.decode(
        _eeg_window(),
        sample_rate=250.0,
        channel_names=["F3", "F4", "C3", "C4"],
        unit="V",
    )
    assert set(result.states) == set(STATE_LABELS_CN)
    assert all(0.0 <= value <= 100.0 for value in result.states.values())
    assert 0.0 <= result.signal_quality <= 100.0
    assert result.waveform is not None
    assert result.psd_frequencies is not None
    assert result.topomap_values is not None
    probabilities = result.motor_intent["probabilities"]
    assert set(probabilities) == set(MOTOR_LABELS_CN)
    assert np.isclose(sum(probabilities.values()), 1.0)
    assert result.motor_intent["label_cn"] == "左手"


def test_motor_intent_transitions_are_smoothed_not_random() -> None:
    decoder = DemoStateDecoder()
    samples = _eeg_window()
    probabilities = []
    for _ in range(14):
        result = decoder.decode(samples, sample_rate=250.0, channel_names=["F3", "F4", "C3", "C4"])
        probabilities.append(result.motor_intent["probabilities"]["left_hand"])
    # The dominant label is held for several frames; after the scheduled change
    # the old probability decays gradually instead of becoming a random value.
    assert probabilities[0] > 0.65
    assert probabilities[11] > probabilities[13] > 0.05
