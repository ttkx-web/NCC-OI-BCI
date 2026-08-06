from __future__ import annotations

import json

import numpy as np
import pytest

from bci_dayloop.inference.observability import (
    JsonlWindowLogger,
    LatencyBreakdown,
    PipelineRunStats,
    calculate_expected_windows,
)
from bci_dayloop.inference.realtime import (
    SlidingWindowDecoder,
)
from bci_dayloop.runtime.types import (
    CanonicalEEGWindow,
    PreparedModelInput,
)
from tests.runtime_fakes import (
    IdentityInputTransform,
    build_fixed_runtime,
)

class FailingInputTransform(
    IdentityInputTransform
):
    def transform(
        self,
        window: CanonicalEEGWindow,
    ) -> PreparedModelInput:
        raise RuntimeError(
            "preprocessing failed"
        )



def make_decoder(
    *,
    stats: PipelineRunStats | None = None,
    logger: JsonlWindowLogger | None = None,
) -> SlidingWindowDecoder:
    runtime_model = build_fixed_runtime(
        channel_names=("C3", "C4"),
        sample_rate=20.0,
        window_sec=1.0,
        probabilities=(
            0.05,
            0.05,
            0.85,
            0.05,
        ),
    )

    return SlidingWindowDecoder(
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
        sample_rate=20.0,
        input_unit="uV",
        window_sec=1.0,
        step_sec=0.5,
        confidence_threshold=0.55,
        run_stats=stats,
        jsonl_logger=logger,
    )


def test_calculate_expected_windows_and_invalid_arguments():
    assert calculate_expected_windows(100, 20, 10) == 9
    assert calculate_expected_windows(19, 20, 10) == 0
    with pytest.raises(ValueError, match="window_samples must be positive"):
        calculate_expected_windows(100, 0, 10)
    with pytest.raises(ValueError, match="step_samples must be positive"):
        calculate_expected_windows(100, 20, 0)


def test_pipeline_stats_averages_p95_and_reset():
    stats = PipelineRunStats()
    stats.set_expected_windows(3)
    stats.record_chunk()
    stats.record_success(LatencyBreakdown(1.0, 2.0, 3.0))
    stats.record_success(LatencyBreakdown(2.0, 3.0, 5.0))
    stats.record_failure()

    snapshot = stats.snapshot()
    assert snapshot.expected_windows == 3
    assert snapshot.chunks_received == 1
    assert snapshot.emitted_windows == 3
    assert snapshot.successful_windows == 2
    assert snapshot.failed_windows == 1
    assert snapshot.current_latency_ms == 5.0
    assert snapshot.average_latency_ms == 4.0
    assert snapshot.p95_latency_ms == pytest.approx(4.9)
    assert snapshot.preprocessing_average_ms == 1.5
    assert snapshot.model_average_ms == 2.5

    stats.reset()
    reset = stats.snapshot()
    assert reset.chunks_received == reset.emitted_windows == 0
    assert reset.successful_windows == reset.failed_windows == 0
    assert reset.current_latency_ms is None
    assert reset.average_latency_ms is None
    assert reset.p95_latency_ms is None


def test_decoder_records_latency_and_success():
    stats = PipelineRunStats()
    decoder = make_decoder(stats=stats)

    result = decoder.push(np.ones((2, 20), dtype=np.float32), trial_id=4, expected_class_id=2)

    assert result is not None
    assert result.preprocessing_latency_ms >= 0
    assert result.model_latency_ms >= 0
    assert result.total_latency_ms >= result.preprocessing_latency_ms
    assert result.total_latency_ms >= result.model_latency_ms
    assert result.latency_ms == result.total_latency_ms
    snapshot = stats.snapshot()
    assert snapshot.chunks_received == 1
    assert snapshot.emitted_windows == 1
    assert snapshot.successful_windows == 1
    assert snapshot.failed_windows == 0


def test_decoder_records_failure_reraises_and_writes_jsonl(
    tmp_path,
):
    log_path = (
        tmp_path
        / "logs"
        / "windows.jsonl"
    )

    stats = PipelineRunStats()
    logger = JsonlWindowLogger(log_path)

    decoder = make_decoder(
        stats=stats,
        logger=logger,
    )

    success = decoder.push(
        np.ones(
            (2, 20),
            dtype=np.float32,
        ),
        trial_id=1,
        expected_class_id=2,
    )

    assert success is not None

    decoder.runtime_model.input_transform = (
        FailingInputTransform(
            channel_names=("C3", "C4"),
            sample_rate=20.0,
            window_sec=1.0,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="preprocessing failed",
    ):
        decoder.push(
            np.ones(
                (2, 10),
                dtype=np.float32,
            )
        )

    snapshot = stats.snapshot()

    assert snapshot.chunks_received == 2
    assert snapshot.emitted_windows == 2
    assert snapshot.successful_windows == 1
    assert snapshot.failed_windows == 1

    records = [
        json.loads(line)
        for line in log_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert [
        record["status"]
        for record in records
    ] == [
        "success",
        "error",
    ]

    success_record, error_record = records

    assert {
        "timestamp",
        "window_id",
        "trial_id",
        "expected_class_id",
        "prediction",
        "confidence",
        "probabilities",
        "command",
        "preprocessing_latency_ms",
        "model_latency_ms",
        "total_latency_ms",
    } <= success_record.keys()

    assert {
        "timestamp",
        "window_id",
        "error_type",
        "error_message",
    } <= error_record.keys()
