from __future__ import annotations

import numpy as np

from bci_dayloop.data.preprocessing import EEGPreprocessor, PreprocessingConfig
from bci_dayloop.inference.realtime import SlidingWindowDecoder
from bci_dayloop.models.base import BaseModelAdapter


class FakeModel(BaseModelAdapter):
    model_name = "fake"

    def fit(self, X, y, **kwargs):
        return {}

    def predict_proba(self, X):
        return np.tile(np.array([[0.2, 0.3, 0.4, 0.1]], dtype=np.float32), (len(X), 1))

    def save(self, path, **kwargs):
        return path

    def load(self, path):
        return self

    def update(self, X, y, **kwargs):
        return {}


def test_decoder_prediction_latency_and_low_confidence_stop():
    decoder = SlidingWindowDecoder(
        FakeModel(),
        EEGPreprocessor(PreprocessingConfig()),
        ["left_hand", "right_hand", "feet", "tongue"],
        sample_rate=200,
        input_unit="uV",
        window_sec=1.0,
        step_sec=1.0,
        confidence_threshold=0.55,
    )
    samples = np.random.default_rng(4).normal(size=(2, 200)).astype(np.float32)
    result = decoder.push(samples, trial_id=7, expected_class_id=2)
    assert result is not None
    assert result.prediction == "feet"
    assert result.command == "STOP"
    assert result.latency_ms >= 0
    assert result.trial_id == 7


def test_decoder_command_mapping_above_threshold():
    class Confident(FakeModel):
        def predict_proba(self, X):
            return np.tile(np.array([[0.05, 0.05, 0.85, 0.05]], dtype=np.float32), (len(X), 1))

    decoder = SlidingWindowDecoder(
        Confident(),
        EEGPreprocessor(PreprocessingConfig()),
        ["left_hand", "right_hand", "feet", "tongue"],
        sample_rate=200,
        input_unit="uV",
        window_sec=1.0,
        step_sec=1.0,
        confidence_threshold=0.55,
    )
    result = decoder.push(np.random.default_rng(5).normal(size=(2, 200)).astype(np.float32))
    assert result is not None and result.command == "FORWARD"

