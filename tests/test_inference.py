from __future__ import annotations

import numpy as np
import pytest

from bci_dayloop.inference.realtime import (
    SlidingWindowDecoder,
)
from bci_dayloop.models.base import (
    add_batch_dimension,
)
from runtime_fakes import (
    build_fixed_runtime,
)
import torch

from bci_dayloop.inference.predictor import (
    PreparedPredictor,
)
from bci_dayloop.runtime.types import (
    ModelOutput,
    PreparedModelInput,
)


class RecordingPreparedPredictor:
    """
    用来验证 Decoder 是否调用了注入的 predictor，
    而不是继续调用 RuntimeModel 的 backend。
    """

    def __init__(
        self,
        probabilities: tuple[
            float,
            ...,
        ],
    ) -> None:
        self.probabilities = torch.tensor(
            [probabilities],
            dtype=torch.float32,
        )

        self.call_count = 0

        self.last_prepared: (
            PreparedModelInput | None
        ) = None

        self.last_return_features: (
            bool | None
        ) = None

        self.model_revision = (
            "test-revision-3"
        )

        self.update_step = 3

    def predict_prepared(
        self,
        prepared: PreparedModelInput,
        *,
        return_features: bool = False,
    ) -> ModelOutput:
        self.call_count += 1
        self.last_prepared = prepared
        self.last_return_features = (
            return_features
        )

        probabilities = (
            self.probabilities.clone()
        )

        logits = torch.log(
            probabilities.clamp_min(
                1e-8
            )
        )

        confidence, prediction = (
            probabilities.max(
                dim=-1
            )
        )

        return ModelOutput(
            logits=logits,
            probabilities=probabilities,
            predicted_class=int(
                prediction[0].item()
            ),
            confidence=float(
                confidence[0].item()
            ),
            features=None,
            diagnostics={
                "online_strategy": (
                    "test-predictor"
                )
            },
        )

def test_decoder_uses_runtime_model_as_default_predictor():
    runtime_model = build_fixed_runtime(
        channel_names=(
            "C3",
            "C4",
        ),
        sample_rate=200.0,
        window_sec=1.0,
        probabilities=(
            0.05,
            0.05,
            0.85,
            0.05,
        ),
    )

    decoder = SlidingWindowDecoder(
        runtime_model=runtime_model,
        class_names=(
            "left_hand",
            "right_hand",
            "feet",
            "tongue",
        ),
        channel_names=(
            "C3",
            "C4",
        ),
        sample_rate=200.0,
        input_unit="uV",
        window_sec=1.0,
        step_sec=1.0,
    )

    assert (
        decoder.predictor
        is runtime_model
    )

    assert isinstance(
        decoder.predictor,
        PreparedPredictor,
    )

def test_decoder_uses_injected_prepared_predictor():
    runtime_model = build_fixed_runtime(
        channel_names=(
            "C3",
            "C4",
        ),
        sample_rate=200.0,
        window_sec=1.0,
        probabilities=(
            0.25,
            0.25,
            0.25,
            0.25,
        ),
        error_message=(
            "Static Runtime backend "
            "must not be called."
        ),
    )

    predictor = (
        RecordingPreparedPredictor(
            probabilities=(
                0.05,
                0.80,
                0.10,
                0.05,
            )
        )
    )

    assert isinstance(
        predictor,
        PreparedPredictor,
    )

    decoder = SlidingWindowDecoder(
        runtime_model=runtime_model,
        predictor=predictor,
        class_names=(
            "left_hand",
            "right_hand",
            "feet",
            "tongue",
        ),
        channel_names=(
            "C3",
            "C4",
        ),
        sample_rate=200.0,
        input_unit="uV",
        window_sec=1.0,
        step_sec=1.0,
        confidence_threshold=0.55,
        command_map={
            "left_hand": "LEFT",
            "right_hand": "RIGHT",
            "feet": "FORWARD",
            "tongue": "STOP",
        },
    )

    samples = (
        np.random.default_rng(123)
        .normal(
            size=(2, 200)
        )
        .astype(np.float32)
    )

    result = decoder.push(
        samples,
        trial_id=9,
        expected_class_id=1,
    )

    assert result is not None

    assert predictor.call_count == 1

    assert (
        predictor.last_prepared
        is not None
    )

    assert (
        predictor.last_return_features
        is False
    )

    # predictor 给出的最大概率类别是 1。
    assert result.class_id == 1
    assert result.prediction == "right_hand"
    assert result.confidence == pytest.approx(
        0.80
    )

    # right_hand 经过 command_map 后是 RIGHT。
    assert result.command == "RIGHT"

    assert (
        result.model_revision
        == "test-revision-3"
    )

    assert (
        result.online_update_step
        == 3
    )

    assert (
        result.online_update_applied
        is False
    )

    assert (
        result.model_diagnostics[
            "online_strategy"
        ]
        == "test-predictor"
    )

    assert (
        result.model_diagnostics[
            "predictor"
        ]
        == (
            "RecordingPreparedPredictor"
        )
    )
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


def test_decoder_rejects_invalid_predictor():
    runtime_model = build_fixed_runtime(
        channel_names=(
            "C3",
            "C4",
        ),
        sample_rate=200.0,
        window_sec=1.0,
    )

    with pytest.raises(
        TypeError,
        match=(
            "must implement "
            "PreparedPredictor"
        ),
    ):
        SlidingWindowDecoder(
            runtime_model=runtime_model,
            predictor=object(),
            class_names=(
                "left_hand",
                "right_hand",
                "feet",
                "tongue",
            ),
            channel_names=(
                "C3",
                "C4",
            ),
            sample_rate=200.0,
            input_unit="uV",
            window_sec=1.0,
            step_sec=1.0,
        )

def test_decoder_prediction_latency_and_low_confidence_stop():
    runtime_model = build_fixed_runtime(
        channel_names=(
            "C3",
            "C4",
        ),
        sample_rate=200.0,
        window_sec=1.0,
        probabilities=(
            0.2,
            0.3,
            0.4,
            0.1,
        ),
    )

    # 不传 predictor，验证原来的普通静态模式。
    decoder = SlidingWindowDecoder(
        runtime_model=runtime_model,
        class_names=(
            "left_hand",
            "right_hand",
            "feet",
            "tongue",
        ),
        channel_names=(
            "C3",
            "C4",
        ),
        sample_rate=200.0,
        input_unit="uV",
        window_sec=1.0,
        step_sec=1.0,
        confidence_threshold=0.55,
    )

    samples = (
        np.random.default_rng(4)
        .normal(
            size=(2, 200)
        )
        .astype(np.float32)
    )

    # 必须先调用 push()，才能得到 result。
    result = decoder.push(
        samples,
        trial_id=7,
        expected_class_id=2,
    )

    assert result is not None
    assert result.prediction == "feet"

    # 最大置信度只有 0.4，小于阈值 0.55，
    # 所以无论预测类别是什么，命令都应该是 STOP。
    assert result.command == "STOP"

    assert result.latency_ms >= 0
    assert result.trial_id == 7

    samples = (
        np.random.default_rng(4)
        .normal(size=(2, 200))
        .astype(np.float32)
    )

    result = decoder.push(
        samples,
        trial_id=7,
        expected_class_id=2,
    )

    assert result is not None
    assert result.prediction == "feet"
    assert result.command == "STOP"
    assert result.latency_ms >= 0
    assert result.trial_id == 7


def test_decoder_command_mapping_above_threshold():
    runtime_model = build_fixed_runtime(
        channel_names=("C3", "C4"),
        sample_rate=200.0,
        window_sec=1.0,
        probabilities=(
            0.05,
            0.05,
            0.85,
            0.05,
        ),
        include_scale=True,
        expect_scale=True,
    )

    decoder = SlidingWindowDecoder(
        runtime_model=runtime_model,
        class_names=(
            "left_hand",
            "right_hand",
            "feet",
            "tongue",
        ),
        channel_names=(
            "C3",
            "C4",
        ),
        sample_rate=200.0,
        input_unit="uV",
        window_sec=1.0,
        step_sec=1.0,
        confidence_threshold=0.55,
    )

    samples = (
        np.random.default_rng(5)
        .normal(size=(2, 200))
        .astype(np.float32)
    )

    result = decoder.push(samples)

    assert result is not None
    assert result.command == "FORWARD"


def test_decoder_accepts_dict_model_input_from_transform():
    runtime_model = build_fixed_runtime(
        channel_names=("C3", "C4"),
        sample_rate=200.0,
        window_sec=1.0,
        probabilities=(
            0.05,
            0.05,
            0.85,
            0.05,
        ),
        include_scale=True,
        expect_scale=True,
    )

    decoder = SlidingWindowDecoder(
        runtime_model=runtime_model,
        class_names=(
            "left_hand",
            "right_hand",
            "feet",
            "tongue",
        ),
        channel_names=(
            "C3",
            "C4",
        ),
        sample_rate=200.0,
        input_unit="uV",
        window_sec=1.0,
        step_sec=1.0,
        confidence_threshold=0.55,
    )

    samples = (
        np.random.default_rng(6)
        .normal(size=(2, 200))
        .astype(np.float32)
    )

    result = decoder.push(samples)

    assert result is not None
    assert result.prediction == "feet"
    assert result.confidence == pytest.approx(
        0.85
    )

