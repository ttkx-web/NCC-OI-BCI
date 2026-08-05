import json

import numpy as np
import pytest

from bci_dayloop.realtime.buffer import TimestampedRingBuffer
from bci_dayloop.realtime.contracts import EEGChunk, EventMarker
from bci_dayloop.realtime.logging import RealtimeRunLogger
from bci_dayloop.realtime.metrics import RealtimeMetrics
from bci_dayloop.realtime.sync import map_event_to_nearest_sample
from bci_dayloop.realtime.windowing import FixedSlidingWindowGenerator


def _chunk(sequence_id: int, start_sample: int, *, timestamp_start: float | None = None) -> EEGChunk:
    sampling_rate = 10.0
    timestamps = (
        start_sample / sampling_rate
        if timestamp_start is None
        else timestamp_start
    ) + np.arange(25, dtype=np.float64) / sampling_rate
    return EEGChunk(
        samples=np.tile(np.arange(start_sample, start_sample + 25, dtype=np.float32), (2, 1)),
        channel_names=("C3", "C4"),
        sampling_rate=sampling_rate,
        unit="uV",
        timestamps=timestamps,
        sequence_id=sequence_id,
        device_id="device-for-hash-only",
        received_at=float(timestamps[-1]),
    )


def _buffer(*, gap: bool = False) -> TimestampedRingBuffer:
    buffer = TimestampedRingBuffer(capacity_samples=100)
    buffer.append(_chunk(1, 0))
    buffer.append(_chunk(2, 25, timestamp_start=4.0 if gap else None))
    return buffer


def test_four_second_windows_use_real_samples_and_half_second_step_without_duplicates() -> None:
    generator = FixedSlidingWindowGenerator(_buffer())
    generator.add_events((EventMarker(1.2, "imagery", code=1),))

    results = generator.generate()
    emitted = [result.window for result in results if result.window is not None]

    assert [window.start_sample_index for window in emitted] == [0, 5, 10]
    assert [window.end_sample_index for window in emitted] == [40, 45, 50]
    assert all(window.samples.shape == (2, 40) for window in emitted)
    assert all(window.source_sequence_start == 1 for window in emitted)
    assert all(window.source_sequence_end == 2 for window in emitted)
    assert len(emitted[0].markers) == 1
    assert generator.generate() == ()


def test_windows_crossing_timestamp_gaps_are_rejected_without_padding() -> None:
    results = FixedSlidingWindowGenerator(_buffer(gap=True)).generate()

    assert len(results) == 3
    assert all(result.status == "failed" for result in results)
    assert all("timestamp gap" in (result.reason or "") for result in results)


def test_event_sync_logging_and_summary_are_parseable(tmp_path) -> None:
    buffer = _buffer()
    event = EventMarker(1.23, "imagery", code=1)
    alignment = map_event_to_nearest_sample(event, buffer)
    assert alignment.sample_index == 12
    assert alignment.error_seconds == pytest.approx(0.03)

    generator = FixedSlidingWindowGenerator(buffer)
    result = generator.generate()[0]
    metrics = RealtimeMetrics()
    metrics.record_chunk(receive_seconds=0.01, buffer_seconds=0.02)
    metrics.record_event_alignment(alignment)
    metrics.record_window(result, elapsed_seconds=0.03)
    logger = RealtimeRunLogger(tmp_path)
    logger.log_chunk(buffer.snapshot()[0].chunk)
    logger.log_event(event, alignment)
    logger.log_window(result)
    logger.write_summary(metrics.summary(buffer.stats))

    for filename in ("chunks.jsonl", "events.jsonl", "windows.jsonl"):
        lines = (tmp_path / filename).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])
    summary = json.loads((tmp_path / "runtime_summary.json").read_text(encoding="utf-8"))
    assert summary["expected_windows"] == 1
    assert summary["emitted_windows"] == 1
    assert summary["failed_windows"] == 0
    assert summary["event_to_eeg_error_seconds"]["count"] == 1
