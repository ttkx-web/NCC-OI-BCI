import numpy as np
import pytest

from bci_dayloop.realtime.buffer import BufferOverflowError, TimestampedRingBuffer
from bci_dayloop.realtime.contracts import EEGChunk, EventMarker
from bci_dayloop.realtime.source import ReplayRealtimeEEGSource, ReplayRealtimeEventSource


def _chunk(sequence_id: int, *, start: float, sample_count: int = 4) -> EEGChunk:
    return EEGChunk(
        samples=np.full((2, sample_count), sequence_id, dtype=np.float32),
        channel_names=("C3", "C4"),
        sampling_rate=10.0,
        unit="uV",
        timestamps=start + np.arange(sample_count, dtype=np.float64) / 10,
        sequence_id=sequence_id,
        device_id=None,
        received_at=start,
    )


def test_replay_sources_preserve_supplied_order() -> None:
    chunks = (_chunk(1, start=0.0), _chunk(2, start=0.4))
    eeg_source = ReplayRealtimeEEGSource(chunks)
    event_source = ReplayRealtimeEventSource((EventMarker(0.1, "cue", code=1),))
    eeg_source.connect()
    event_source.connect()

    assert eeg_source.read_chunk() is chunks[0]
    assert eeg_source.read_chunk() is chunks[1]
    assert eeg_source.read_chunk() is None
    assert event_source.read_event() == EventMarker(0.1, "cue", code=1)
    assert event_source.read_event() is None


def test_buffer_detects_missing_duplicate_and_out_of_order_sequences() -> None:
    buffer = TimestampedRingBuffer(capacity_samples=20)
    assert buffer.append(_chunk(1, start=0.0)) is True
    assert buffer.append(_chunk(3, start=0.8)) is True
    assert buffer.append(_chunk(3, start=1.2)) is False
    assert buffer.append(_chunk(2, start=0.4)) is False

    stats = buffer.stats
    assert stats.inferred_missing_samples == 4
    assert stats.duplicate_chunks == 1
    assert stats.out_of_order_chunks == 1
    assert [item.chunk.sequence_id for item in buffer.snapshot()] == [1, 3]


def test_buffer_overflow_is_explicit_and_does_not_discard_data() -> None:
    buffer = TimestampedRingBuffer(capacity_samples=6)
    first = _chunk(1, start=0.0, sample_count=4)
    assert buffer.append(first) is True
    with pytest.raises(BufferOverflowError, match="overwrite"):
        buffer.append(_chunk(2, start=0.4, sample_count=3))

    assert buffer.stats.buffer_overflows == 1
    assert buffer.backlog_samples == 4
    assert buffer.snapshot()[0].chunk is first
