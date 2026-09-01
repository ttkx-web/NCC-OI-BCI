from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bci_dayloop.data.trial_reader import open_trial_reader
from bci_dayloop.inference.multi_head import (
    HeadPrediction,
    MultiHeadPrediction,
    MultiHeadPredictor,
)
from bci_dayloop.inference.realtime import (
    DecodeResult,
    MultiHeadDecodeResult,
    SlidingWindowDecoder,
)
from bci_dayloop.inference.observability import JsonlWindowLogger
from bci_dayloop.runtime.types import RawEEGWindow
from runtime_fakes import build_fixed_runtime


ROOT = Path(__file__).resolve().parents[1]
BACKBONE = ROOT / "checkpoints/backbones/50m/model_deploy.pt"
WORKLOAD = ROOT / (
    "checkpoints/heads/stage1/bnci2014_001/subject_01/Workload/"
    "subject_01/population/2s_flatten/head.pt"
)
ATTENTION = ROOT / (
    "checkpoints/heads/stage1/bnci2014_001/subject_01/MEMA/"
    "subject_01/population/2s_flatten/head.pt"
)
EMOTION = ROOT / (
    "checkpoints/heads/stage1/bnci2014_001/subject_01/SEED/"
    "subject_01/population/2s_flatten/head.pt"
)
YAXIN = ROOT / "data/processed/yaxin/smr_control_yaxin_0819_combined.h5"


class RecordingRawMultiHeadPredictor:
    window_seconds = 2.0

    def __init__(self) -> None:
        self.calls = 0
        self.last_window: RawEEGWindow | None = None
        self.prediction = MultiHeadPrediction(
            workload=HeadPrediction(1, "diff", 0.8, (0.2, 0.8)),
            attention=HeadPrediction(2, "concentrating", 0.7, (0.1, 0.2, 0.7)),
            emotion=HeadPrediction(0, "negative", 0.6, (0.6, 0.3, 0.1)),
        )
        self.last_diagnostics = SimpleNamespace(
            preprocessing_calls=1,
            backbone_forwards=1,
            head_forwards={"workload": 1, "attention": 1, "emotion": 1},
            shared_feature_shape=(1, 65_536),
            preprocessing_latency_ms=1.0,
            backbone_latency_ms=2.0,
            heads_latency_ms=0.1,
        )

    def predict(self, window: RawEEGWindow) -> MultiHeadPrediction:
        self.calls += 1
        self.last_window = window
        return self.prediction


def _assert_prediction_equal(
    direct: MultiHeadPrediction,
    decoded: MultiHeadPrediction,
) -> None:
    for task in ("workload", "attention", "emotion"):
        before = getattr(direct, task)
        after = getattr(decoded, task)
        assert after.label_id == before.label_id
        assert after.label == before.label
        np.testing.assert_allclose(after.probabilities, before.probabilities)
        assert after.confidence == pytest.approx(before.confidence)


def test_single_task_decoder_remains_a_decode_result() -> None:
    runtime_model = build_fixed_runtime(
        channel_names=("C3", "C4"),
        sample_rate=10.0,
        window_sec=2.0,
        probabilities=(0.1, 0.8, 0.05, 0.05),
    )
    decoder = SlidingWindowDecoder(
        runtime_model=runtime_model,
        class_names=("left", "right", "both", "rest"),
        channel_names=("C3", "C4"),
        sample_rate=10.0,
        input_unit="uV",
        window_sec=2.0,
        step_sec=2.0,
    )
    result = decoder.push(np.ones((2, 20), dtype=np.float32))
    assert isinstance(result, DecodeResult)
    assert result.prediction == "right"
    assert result.probabilities == pytest.approx([0.1, 0.8, 0.05, 0.05])


def test_decoder_accepts_raw_multi_head_predictor_once_and_preserves_result() -> None:
    predictor = RecordingRawMultiHeadPredictor()
    decoder = SlidingWindowDecoder(
        predictor=predictor,
        channel_names=("C3", "C4"),
        sample_rate=10.0,
        input_unit="uV",
        window_sec=2.0,
        step_sec=2.0,
    )
    result = decoder.push(np.ones((2, 20), dtype=np.float32), trial_id=9)

    assert isinstance(result, MultiHeadDecodeResult)
    assert predictor.calls == 1
    assert predictor.last_window is not None
    assert predictor.last_window.data.shape == (2, 20)
    assert result.trial_id == 9
    _assert_prediction_equal(predictor.prediction, result.prediction)
    assert result.model_diagnostics["preprocessing_calls"] == 1
    assert result.model_diagnostics["backbone_forwards"] == 1
    assert result.model_diagnostics["head_forwards"] == {
        "workload": 1,
        "attention": 1,
        "emotion": 1,
    }
    for task in ("workload", "attention", "emotion"):
        probabilities = getattr(result.prediction, task).probabilities
        assert np.isfinite(probabilities).all()
        assert np.isclose(sum(probabilities), 1.0)


def test_decoder_rejects_raw_predictor_window_contract_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="decoder=4.0, predictor=2.0",
    ):
        SlidingWindowDecoder(
            predictor=RecordingRawMultiHeadPredictor(),
            channel_names=("C3", "C4"),
            sample_rate=10.0,
            input_unit="uV",
            window_sec=4.0,
            step_sec=1.0,
        )


def test_multi_head_decode_is_jsonl_loggable(tmp_path: Path) -> None:
    log_path = tmp_path / "multi_head.jsonl"
    decoder = SlidingWindowDecoder(
        predictor=RecordingRawMultiHeadPredictor(),
        channel_names=("C3", "C4"),
        sample_rate=10.0,
        input_unit="uV",
        window_sec=2.0,
        step_sec=2.0,
        jsonl_logger=JsonlWindowLogger(log_path),
    )
    result = decoder.push(np.ones((2, 20), dtype=np.float32))
    assert isinstance(result, MultiHeadDecodeResult)
    record = json.loads(log_path.read_text().strip())
    assert record["prediction_type"] == "multi_head"
    assert record["prediction"]["workload"]["label"] == "diff"
    assert record["prediction"]["emotion"]["probabilities"] == pytest.approx(
        [0.6, 0.3, 0.1]
    )


@pytest.mark.skipif(
    not all(path.is_file() for path in (BACKBONE, WORKLOAD, ATTENTION, EMOTION, YAXIN)),
    reason="requires local three-head checkpoints and canonical yaxin H5",
)
def test_real_yaxin_direct_and_decoder_multi_head_predictions_are_equivalent() -> None:
    reader = open_trial_reader(data_reader="eeg", path=YAXIN, canonical_subject_id=1)
    source = reader.load(session="S6")
    raw = np.asarray(source["data"][0], dtype=np.float32)
    predictor = MultiHeadPredictor.from_checkpoints(
        backbone_checkpoint=BACKBONE,
        workload_head=WORKLOAD,
        attention_head=ATTENTION,
        emotion_head=EMOTION,
        device="cpu",
    )
    direct = predictor.predict(
        RawEEGWindow(
            data=raw,
            channel_names=list(reader.metadata.channel_names),
            sample_rate=float(reader.metadata.sample_rate),
            unit=str(reader.metadata.unit),
            trial_id=str(source["trial_ids"][0]),
        )
    )
    decoder = SlidingWindowDecoder(
        predictor=predictor,
        channel_names=reader.metadata.channel_names,
        sample_rate=float(reader.metadata.sample_rate),
        input_unit=str(reader.metadata.unit),
        window_sec=2.0,
        step_sec=2.0,
    )
    decoded = decoder.push(raw, trial_id=int(source["trial_ids"][0]))

    assert isinstance(decoded, MultiHeadDecodeResult)
    _assert_prediction_equal(direct, decoded.prediction)
    diagnostics = predictor.last_diagnostics
    assert diagnostics is not None
    assert diagnostics.preprocessing_calls == 1
    assert diagnostics.backbone_forwards == 1
    assert diagnostics.head_forwards == {
        "workload": 1,
        "attention": 1,
        "emotion": 1,
    }
