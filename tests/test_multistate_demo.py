from __future__ import annotations

import numpy as np

from bci_dayloop.demo.schemas import DemoEEGWindow, MOTOR_LABELS_CN, STATE_LABELS_CN
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
    assert result.cortical_activity is not None
    assert result.cortical_activity.left_rgba.shape[-1] == 4
    assert result.cortical_activity.right_rgba.shape[-1] == 4
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


class CountingMotorDecoder:
    display_name = "Counting"

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, **_: object) -> dict[str, object]:
        self.calls += 1
        return {
            "label": "left_hand",
            "label_cn": "左手",
            "confidence": 1.0,
            "probabilities": {name: 0.25 for name in MOTOR_LABELS_CN},
        }


def test_new_state_indices_are_bounded_and_emotion_is_structured() -> None:
    decoder = DemoStateDecoder(compute_motor_intent=False)
    results = [
        decoder.decode(_eeg_window(), sample_rate=250.0, channel_names=["F3", "F4", "C3", "C4"])
        for _ in range(6)
    ]

    assert list(results[-1].states) == list(STATE_LABELS_CN)
    assert len(results[-1].states) == 16
    assert all(0.0 <= value <= 100.0 for result in results for value in result.states.values())
    assert results[-1].emotion is not None
    assert results[-1].emotion.label_cn in {"放松", "愉悦", "平稳", "专注", "紧张"}
    assert results[-1].emotion.emoji
    assert results[-1].motor_intent["decoder_type"] == "disabled"


def test_hidden_motor_intent_skips_motor_decoder_inference() -> None:
    motor = CountingMotorDecoder()
    decoder = DemoStateDecoder(motor_decoder=motor, compute_motor_intent=False)
    decoder.decode(_eeg_window(), sample_rate=250.0, channel_names=["F3", "F4", "C3", "C4"])

    assert motor.calls == 0


def test_decode_window_accepts_32_channel_device_contract_and_excludes_invalid_channels() -> None:
    sample_rate = 250.0
    seconds = 4.0
    time = np.arange(int(sample_rate * seconds)) / sample_rate
    amplitudes = np.linspace(5.0, 20.0, 32)
    samples_uv = np.vstack([amplitude * np.sin(2 * np.pi * 10.0 * time) for amplitude in amplitudes]).astype(np.float32)
    valid = np.ones(32, dtype=bool)
    valid[[3, 19]] = False
    samples_uv[3] = 1e6  # Must not alter valid-channel feature/cortical statistics.
    decoder = DemoStateDecoder(compute_motor_intent=False)
    result = decoder.decode_window(
        DemoEEGWindow(
            samples=samples_uv,
            sample_rate=sample_rate,
            channel_names=[f"CH{index:02d}" for index in range(1, 33)],
            unit="uV",
            timestamp=12.5,
            channel_valid_mask=valid,
            device_name="BCIGo",
            montage_name="bcigo_32_placeholder",
        )
    )

    assert result.timestamp == 12.5
    assert result.cortical_activity is not None
    assert result.cortical_activity.available
    assert result.cortical_activity.mapped_channel_count == 30
    assert all(np.isfinite(value) and 0.0 <= value <= 100.0 for value in result.states.values())
