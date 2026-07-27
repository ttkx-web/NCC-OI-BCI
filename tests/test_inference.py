from __future__ import annotations

import numpy as np
import pytest

from bci_dayloop.data.preprocessing import EEGPreprocessor, PreprocessingConfig
from bci_dayloop.inference.realtime import SlidingWindowDecoder
from bci_dayloop.models.base import BaseModelAdapter, add_batch_dimension


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


def test_add_batch_dimension_for_ndarray_does_not_change_input():
    values = np.arange(6, dtype=np.float32).reshape(2, 3)

    batched = add_batch_dimension(values)

    assert isinstance(batched, np.ndarray)
    assert values.shape == (2, 3)
    assert batched.shape == (1, 2, 3)
    np.testing.assert_array_equal(batched[0], values)


def test_add_batch_dimension_for_dict_does_not_change_input():
    values = {"signal": np.ones((2, 3), dtype=np.float32), "mask": np.array([1, 0], dtype=np.int64)}

    batched = add_batch_dimension(values)

    assert isinstance(batched, dict)
    assert batched is not values
    assert values["signal"].shape == (2, 3)
    assert values["mask"].shape == (2,)
    assert batched["signal"].shape == (1, 2, 3)
    assert batched["mask"].shape == (1, 2)


@pytest.mark.parametrize(
    ("value", "exception"),
    [({}, ValueError), ("not-an-input", TypeError), ({"signal": [1, 2]}, TypeError)],
)
def test_add_batch_dimension_rejects_invalid_inputs(value, exception):
    with pytest.raises(exception):
        add_batch_dimension(value)


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


def test_decoder_accepts_dict_model_input_from_preprocessor():
    class DictPreprocessor:
        def transform(self, samples, sample_rate, input_unit, *, reshape=True):
            assert sample_rate == 200
            assert input_unit == "uV"
            assert reshape is True
            return {"signal": samples.copy(), "scale": np.array([1.0], dtype=np.float32)}

    class DictModel(FakeModel):
        def predict_proba(self, X):
            assert isinstance(X, dict)
            assert X["signal"].shape == (1, 2, 200)
            assert X["scale"].shape == (1, 1)
            return np.array([[0.05, 0.05, 0.85, 0.05]], dtype=np.float32)

    decoder = SlidingWindowDecoder(
        DictModel(),
        DictPreprocessor(),
        ["left_hand", "right_hand", "feet", "tongue"],
        sample_rate=200,
        input_unit="uV",
        window_sec=1.0,
        step_sec=1.0,
        confidence_threshold=0.55,
    )

    result = decoder.push(np.random.default_rng(6).normal(size=(2, 200)).astype(np.float32))

    assert result is not None
    assert result.prediction == "feet"
    assert result.confidence == pytest.approx(0.85)

