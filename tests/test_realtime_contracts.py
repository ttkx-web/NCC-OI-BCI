import numpy as np
import pytest

from bci_dayloop.realtime.contracts import EEGChunk, EventMarker, RealtimeWindow, WindowResult


def _chunk(*, timestamps: np.ndarray | None = None) -> EEGChunk:
    samples = np.arange(12, dtype=np.float32).reshape(2, 6)
    return EEGChunk(
        samples=samples,
        channel_names=("C3", "C4"),
        sampling_rate=10.0,
        unit="uV",
        timestamps=np.arange(6, dtype=np.float64) / 10 if timestamps is None else timestamps,
        sequence_id=1,
        device_id="test-device",
        received_at=10.0,
        metadata={"transport": "replay"},
    )


def test_valid_chunk_is_immutable_and_unit_explicit() -> None:
    chunk = _chunk()

    assert chunk.samples.dtype == np.float32
    assert chunk.samples.flags.writeable is False
    assert chunk.timestamps.flags.writeable is False
    assert chunk.unit == "uV"
    with pytest.raises(TypeError):
        chunk.metadata["transport"] = "other"  # type: ignore[index]


def test_chunk_rejects_timestamp_length_and_timestamp_regression() -> None:
    with pytest.raises(ValueError, match="length N"):
        _chunk(timestamps=np.arange(5))
    with pytest.raises(ValueError, match="non-decreasing"):
        _chunk(timestamps=np.array([0.0, 0.1, 0.2, 0.15, 0.4, 0.5]))


def test_chunk_rejects_channel_count_mismatch() -> None:
    with pytest.raises(ValueError, match="channel count"):
        EEGChunk(
            samples=np.zeros((2, 3)),
            channel_names=("C3",),
            sampling_rate=10.0,
            unit="uV",
            timestamps=np.array([0.0, 0.1, 0.2]),
            sequence_id=1,
            device_id=None,
            received_at=0.0,
        )


def test_window_and_result_validate_real_sample_bounds() -> None:
    chunk = _chunk()
    marker = EventMarker(0.2, "cue", code=1)
    window = RealtimeWindow(
        window_id=3,
        samples=chunk.samples,
        channel_names=chunk.channel_names,
        sampling_rate=10.0,
        unit="uV",
        timestamps=chunk.timestamps,
        start_sample_index=0,
        end_sample_index=6,
        source_sequence_start=1,
        source_sequence_end=1,
        markers=(marker,),
    )
    assert WindowResult(3, "emitted", window=window).window is window
    assert WindowResult(4, "failed", reason="gap").reason == "gap"
