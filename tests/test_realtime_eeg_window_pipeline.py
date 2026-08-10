import numpy as np
import pytest

from bci_dayloop.realtime.buffer import BufferOverflowError
from bci_dayloop.realtime.channel_units import EEG_UNIT, VENDOR_CONFIRMED
from bci_dayloop.realtime.contracts import EEGChunk, EventMarker
from bci_dayloop.realtime.pipeline import RealtimeEEGWindowPipeline, RealtimePipelineError


def _chunk(
    sequence_id: int,
    start: int,
    sample_count: int,
    *,
    model_safe: bool = True,
    unit: str = "uV",
    metadata_overrides: dict[str, object] | None = None,
) -> EEGChunk:
    metadata = {
        "channel_types": ("EEG", "eeg"),
        "channel_units": (EEG_UNIT, EEG_UNIT),
        "unit_evidence_level": VENDOR_CONFIRMED,
        "model_safe": model_safe,
        "raw_start_timestamp": start,
    }
    if metadata_overrides is not None:
        metadata.update(metadata_overrides)
    return EEGChunk(
        samples=np.vstack(
            (
                np.arange(start, start + sample_count, dtype=np.float32),
                np.arange(start + 10000, start + 10000 + sample_count, dtype=np.float32),
            )
        ),
        channel_names=("C3", "C4"),
        sampling_rate=1000.0,
        unit=unit,
        timestamps=(start + np.arange(sample_count, dtype=np.float64)) / 1000.0,
        sequence_id=sequence_id,
        device_id=None,
        received_at=float(start) / 1000.0,
        metadata=metadata,
    )


def _process_in_chunks(sizes: tuple[int, ...]) -> tuple[RealtimeEEGWindowPipeline, list[object]]:
    pipeline = RealtimeEEGWindowPipeline()
    results: list[object] = []
    start = 0
    for sequence_id, size in enumerate(sizes):
        results.extend(pipeline.process(_chunk(sequence_id, start, size)))
        start += size
    return pipeline, results


def test_window_counts_for_under_exact_and_four_point_five_seconds() -> None:
    under, under_results = _process_in_chunks((3999,))
    exact, exact_results = _process_in_chunks((4000,))
    extended, extended_results = _process_in_chunks((4500,))

    assert under_results == []
    assert under.expected_windows == 0
    assert len(exact_results) == 1
    assert exact.expected_windows == exact.emitted_windows == 1
    assert len(extended_results) == 2
    assert extended.expected_windows == extended.emitted_windows == 2


def test_packet_partitioning_preserves_fixed_window_and_step_contract() -> None:
    whole, whole_results = _process_in_chunks((4500,))
    partitioned, partitioned_results = _process_in_chunks((1000, 750, 1250, 1500))

    for results in (whole_results, partitioned_results):
        windows = [result.window for result in results if result.window is not None]
        assert [window.samples.shape[1] for window in windows] == [4000, 4000]
        assert [window.start_sample_index for window in windows] == [0, 500]
        assert [window.end_sample_index for window in windows] == [4000, 4500]
        assert np.all(np.diff(windows[0].timestamps) > 0)
        np.testing.assert_array_equal(windows[0].samples[0], np.arange(4000, dtype=np.float32))
        assert windows[0].metadata["channel_types"] == ("EEG", "eeg")
        assert windows[0].metadata["channel_units"] == (EEG_UNIT, EEG_UNIT)
        assert windows[0].metadata["unit_evidence_level"] == VENDOR_CONFIRMED
        assert windows[0].metadata["model_safe"] is True
        assert windows[0].metadata["continuous_segment_id"] == 1
    assert whole.emitted_windows == partitioned.emitted_windows == 2


def test_gap_starts_a_new_segment_without_crossing_samples() -> None:
    pipeline = RealtimeEEGWindowPipeline()
    first = pipeline.process(_chunk(0, 0, 4000))
    second = pipeline.process(_chunk(1, 5000, 4000))

    assert len(first) == len(second) == 1
    assert pipeline.contiguous_segment_count == 2
    assert pipeline.timestamp_gap_count == 1
    assert pipeline.expected_windows == pipeline.emitted_windows == 2
    assert second[0].window is not None
    assert second[0].window.timestamps[0] == pytest.approx(5.0)


def test_pipeline_rejects_mixed_unknown_or_non_model_safe_inputs() -> None:
    pipeline = RealtimeEEGWindowPipeline()
    with pytest.raises(RealtimePipelineError, match="uV"):
        pipeline.process(_chunk(0, 0, 10, unit="mixed"))
    with pytest.raises(RealtimePipelineError, match="model-safe"):
        pipeline.process(_chunk(0, 0, 10, model_safe=False))
    unknown = _chunk(0, 0, 10)
    unknown_metadata = dict(unknown.metadata)
    unknown_metadata["channel_units"] = ("unknown", EEG_UNIT)
    unknown = EEGChunk(
        samples=unknown.samples,
        channel_names=unknown.channel_names,
        sampling_rate=unknown.sampling_rate,
        unit=unknown.unit,
        timestamps=unknown.timestamps,
        sequence_id=unknown.sequence_id,
        device_id=None,
        received_at=unknown.received_at,
        metadata=unknown_metadata,
    )
    with pytest.raises(RealtimePipelineError, match="non-uV"):
        pipeline.process(unknown)


def test_duplicate_timestamps_and_overflow_are_explicit() -> None:
    pipeline = RealtimeEEGWindowPipeline()
    pipeline.process(_chunk(0, 0, 10))
    with pytest.raises(RealtimePipelineError, match="duplicate or out-of-order"):
        pipeline.process(_chunk(1, 0, 10))

    overflowing = RealtimeEEGWindowPipeline(capacity_samples=4000)
    overflowing.process(_chunk(0, 0, 3500))
    with pytest.raises(BufferOverflowError):
        overflowing.process(_chunk(1, 3500, 1000))
    assert overflowing.buffer_overflow_count == 1


@pytest.mark.parametrize(
    ("metadata_overrides", "expected_reason"),
    [
        ({"channel_types": ("eeg", "EEG")}, "provenance changed"),
        ({"channel_units": (EEG_UNIT, "unknown")}, "non-uV"),
        ({"unit_evidence_level": "unknown"}, "vendor-confirmed"),
    ],
)
def test_pipeline_rejects_changed_provenance_before_buffering(metadata_overrides, expected_reason) -> None:
    pipeline = RealtimeEEGWindowPipeline()
    pipeline.process(_chunk(0, 0, 10))

    with pytest.raises(RealtimePipelineError, match=expected_reason):
        pipeline.process(_chunk(1, 10, 10, metadata_overrides=metadata_overrides))


def test_marker_boundaries_order_and_overlap_follow_half_open_windows() -> None:
    pipeline = RealtimeEEGWindowPipeline()
    first = pipeline.process(
        _chunk(0, 0, 4000),
        (
            EventMarker(0.0, "trigger", code=1),
            EventMarker(0.5, "trigger", code=2),
            EventMarker(4.0, "trigger", code=3),
        ),
    )
    second = pipeline.process(_chunk(1, 4000, 500))

    assert first[0].window is not None and second[0].window is not None
    assert [event.code for event in first[0].window.markers] == [1, 2]
    assert [event.code for event in second[0].window.markers] == [2, 3]
    assert pipeline.unique_trigger_count == 3
    assert pipeline.marker_window_association_count == 4


def test_reset_drops_previous_run_state() -> None:
    pipeline, results = _process_in_chunks((4000,))
    assert results
    pipeline.reset()

    assert pipeline.accepted_eeg_sample_count == 0
    assert pipeline.expected_windows == pipeline.emitted_windows == 0
    assert pipeline.contiguous_segment_count == 0
    assert pipeline.process(_chunk(0, 0, 4000))[0].window_id == 0
