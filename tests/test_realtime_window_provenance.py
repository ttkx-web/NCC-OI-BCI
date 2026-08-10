import numpy as np

from bci_dayloop.realtime.buffer import TimestampedRingBuffer
from bci_dayloop.realtime.contracts import EEGChunk
from bci_dayloop.realtime.windowing import FixedSlidingWindowGenerator


def _chunk(sequence_id: int, start: int, *, metadata: dict[str, object] | None = None) -> EEGChunk:
    provenance = {
        "channel_types": ("eeg", "eeg"),
        "channel_units": ("uV", "uV"),
        "unit_evidence_level": "vendor_confirmed",
        "model_safe": True,
    }
    if metadata is not None:
        provenance.update(metadata)
    return EEGChunk(
        samples=np.arange(start, start + 25, dtype=np.float32).repeat(2).reshape(2, 25),
        channel_names=("C3", "C4"),
        sampling_rate=10.0,
        unit="uV",
        timestamps=(start + np.arange(25, dtype=np.float64)) / 10.0,
        sequence_id=sequence_id,
        device_id=None,
        received_at=float(start),
        metadata=provenance,
    )


def _results(*, second_metadata: dict[str, object] | None = None):
    buffer = TimestampedRingBuffer(capacity_samples=100)
    first = _chunk(1, 0)
    second = _chunk(2, 25, metadata=second_metadata)
    buffer.append(first)
    buffer.append(second)
    generator = FixedSlidingWindowGenerator(buffer)
    generator.set_continuous_segment_id(3)
    return first, generator.generate()


def test_window_propagates_whitelisted_provenance_without_changing_data() -> None:
    first, results = _results()
    window = results[0].window

    assert window is not None
    assert window.metadata == {
        "channel_types": ("eeg", "eeg"),
        "channel_units": ("uV", "uV"),
        "unit_evidence_level": "vendor_confirmed",
        "model_safe": True,
        "source_sampling_rate": 10.0,
        "source_unit": "uV",
        "continuous_segment_id": 3,
    }
    np.testing.assert_array_equal(window.samples[:, :25], first.samples)
    np.testing.assert_array_equal(window.timestamps[:25], first.timestamps)
    assert window.channel_names == first.channel_names


def test_conflicting_channel_types_units_or_evidence_fail_closed() -> None:
    for conflict in (
        {"channel_types": ("eeg", "eog")},
        {"channel_units": ("uV", "unknown")},
        {"unit_evidence_level": "unknown"},
    ):
        _, results = _results(second_metadata=conflict)
        assert all(result.status == "failed" for result in results)
        assert all("provenance changed" in (result.reason or "") for result in results)


def test_unsafe_or_length_mismatched_provenance_fails_closed() -> None:
    _, unsafe = _results(second_metadata={"model_safe": False})
    _, mismatched = _results(second_metadata={"channel_units": ("uV",)})

    assert all(result.status == "failed" for result in unsafe)
    assert "model_safe" in (unsafe[0].reason or "")
    assert all(result.status == "failed" for result in mismatched)
    assert "channel_units length" in (mismatched[0].reason or "")
