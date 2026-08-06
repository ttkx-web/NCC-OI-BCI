"""Unit-gated Source-to-window orchestration built from the existing realtime primitives."""

from __future__ import annotations

import math
from collections.abc import Iterable

from .buffer import BufferOverflowError, TimestampedRingBuffer
from .channel_units import EEG_UNIT, VENDOR_CONFIRMED, normalize_channel_type
from .contracts import EEGChunk, EventMarker, WindowResult
from .windowing import FixedSlidingWindowGenerator, WindowProvenance, chunk_window_provenance


class RealtimePipelineError(ValueError):
    """Raised when input violates the EEG-only realtime window contract."""


class RealtimeEEGWindowPipeline:
    """Validate EEG-only chunks, split timestamp gaps, and generate fixed windows."""

    def __init__(
        self,
        *,
        sampling_rate: float = 1000.0,
        window_seconds: float = 4.0,
        step_seconds: float = 0.5,
        capacity_samples: int = 8000,
    ) -> None:
        if not math.isfinite(sampling_rate) or sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive and finite")
        self.sampling_rate = sampling_rate
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.capacity_samples = capacity_samples
        self._new_segment()
        self.accepted_eeg_sample_count = 0
        self.contiguous_segment_count = 0
        self.timestamp_gap_count = 0
        self.buffer_overflow_count = 0
        self.buffer_peak_samples = 0
        self._last_timestamp: float | None = None
        self._segment_provenance: WindowProvenance | None = None
        self._marker_keys: set[tuple[object, ...]] = set()
        self.unique_trigger_count = 0
        self.marker_window_association_count = 0
        self.last_failure_reason: str | None = None

    @property
    def buffer(self) -> TimestampedRingBuffer:
        return self._buffer

    @property
    def windowing(self) -> FixedSlidingWindowGenerator:
        return self._windowing

    @property
    def expected_windows(self) -> int:
        return self._windowing.expected_windows

    @property
    def emitted_windows(self) -> int:
        return self._windowing.emitted_windows

    @property
    def failed_windows(self) -> int:
        return self._windowing.failed_windows

    @property
    def window_samples(self) -> int:
        return round(self.window_seconds * self.sampling_rate)

    @property
    def step_samples(self) -> int:
        return round(self.step_seconds * self.sampling_rate)

    def process(self, chunk: EEGChunk, markers: Iterable[EventMarker] = ()) -> tuple[WindowResult, ...]:
        """Accept one verified EEG chunk and any same-device-clock markers."""
        try:
            _validate_eeg_chunk(chunk, expected_sampling_rate=self.sampling_rate)
            provenance = chunk_window_provenance(chunk)
        except RealtimePipelineError as exc:
            self.last_failure_reason = str(exc)
            raise
        except ValueError as exc:
            self.last_failure_reason = str(exc)
            raise RealtimePipelineError(str(exc)) from exc
        if self._last_timestamp is not None:
            expected = self._last_timestamp + (1.0 / self.sampling_rate)
            first = float(chunk.timestamps[0])
            if first <= self._last_timestamp:
                self.last_failure_reason = "duplicate or out-of-order EEG timestamps are rejected"
                raise RealtimePipelineError(self.last_failure_reason)
            if first - expected > (0.5 / self.sampling_rate):
                self.timestamp_gap_count += 1
                self._new_segment()
                self.contiguous_segment_count += 1
        if self.contiguous_segment_count == 0:
            self.contiguous_segment_count = 1
        if self._segment_provenance is None:
            self._segment_provenance = provenance
        elif self._segment_provenance != provenance:
            self.last_failure_reason = "chunk provenance changed within one continuous segment"
            raise RealtimePipelineError(self.last_failure_reason)
        self._windowing.set_continuous_segment_id(self.contiguous_segment_count)
        self._add_markers(markers)
        try:
            accepted = self._buffer.append(chunk)
        except BufferOverflowError:
            self.buffer_overflow_count += 1
            self.last_failure_reason = "TimestampBuffer capacity overflow"
            raise
        if not accepted:
            self.last_failure_reason = "duplicate or out-of-order EEG chunk is rejected"
            raise RealtimePipelineError(self.last_failure_reason)
        self.accepted_eeg_sample_count += chunk.samples.shape[1]
        self._last_timestamp = float(chunk.timestamps[-1])
        self.buffer_peak_samples = max(self.buffer_peak_samples, self._buffer.backlog_samples)
        results = self._windowing.generate()
        self.marker_window_association_count += sum(
            len(result.window.markers) for result in results if result.window is not None
        )
        next_start = self._windowing.next_start_sample
        if next_start is not None:
            self._buffer.discard_before(next_start)
        return results

    def reset(self) -> None:
        """Drop all buffered data and counters before a new run."""
        self._buffer = TimestampedRingBuffer(self.capacity_samples)
        self._windowing = FixedSlidingWindowGenerator(
            self._buffer,
            window_seconds=self.window_seconds,
            step_seconds=self.step_seconds,
        )
        self.accepted_eeg_sample_count = 0
        self.contiguous_segment_count = 0
        self.timestamp_gap_count = 0
        self.buffer_overflow_count = 0
        self.buffer_peak_samples = 0
        self._last_timestamp = None
        self._segment_provenance = None
        self._marker_keys.clear()
        self.unique_trigger_count = 0
        self.marker_window_association_count = 0
        self.last_failure_reason = None

    def close(self) -> None:
        """Release buffered chunks after a run; the instance may later be reused via reset()."""
        self.reset()

    def _new_segment(self) -> None:
        self._buffer = TimestampedRingBuffer(self.capacity_samples)
        self._segment_provenance = None
        if hasattr(self, "_windowing"):
            self._windowing.reset_for_buffer(self._buffer)
        else:
            self._windowing = FixedSlidingWindowGenerator(
                self._buffer,
                window_seconds=self.window_seconds,
                step_seconds=self.step_seconds,
            )

    def _add_markers(self, markers: Iterable[EventMarker]) -> None:
        fresh: list[EventMarker] = []
        for marker in markers:
            key = (
                marker.timestamp,
                marker.event_type,
                marker.code,
                marker.sequence_id,
                marker.metadata.get("raw_device_timestamp"),
            )
            if key not in self._marker_keys:
                self._marker_keys.add(key)
                self.unique_trigger_count += 1
                fresh.append(marker)
        self._windowing.add_events(fresh)


def _validate_eeg_chunk(chunk: EEGChunk, *, expected_sampling_rate: float) -> None:
    metadata = chunk.metadata
    channel_types = _metadata_sequence(metadata, "channel_types")
    channel_units = _metadata_sequence(metadata, "channel_units")
    channel_count = chunk.samples.shape[0]
    if chunk.sampling_rate != expected_sampling_rate:
        raise RealtimePipelineError("EEG sampling_rate does not match the realtime contract")
    if chunk.unit != EEG_UNIT:
        raise RealtimePipelineError("pipeline only accepts EEG-only uV chunks")
    if metadata.get("model_safe") is not True:
        raise RealtimePipelineError("EEG chunk is not model-safe")
    if metadata.get("unit_evidence_level") != VENDOR_CONFIRMED:
        raise RealtimePipelineError("EEG chunk lacks vendor-confirmed unit evidence")
    if len(chunk.channel_names) != channel_count or len(channel_types) != channel_count or len(channel_units) != channel_count:
        raise RealtimePipelineError("EEG channel metadata lengths must match samples")
    if any(normalize_channel_type(channel_type) != "eeg" for channel_type in channel_types):
        raise RealtimePipelineError("pipeline rejects non-EEG channels")
    if any(channel_unit != EEG_UNIT for channel_unit in channel_units):
        raise RealtimePipelineError("pipeline rejects non-uV EEG channels")


def _metadata_sequence(metadata: object, name: str) -> tuple[object, ...]:
    if not isinstance(metadata, dict) and not hasattr(metadata, "get"):
        raise RealtimePipelineError("EEG metadata must be mapping-like")
    value = metadata.get(name)  # type: ignore[union-attr]
    if isinstance(value, (str, bytes)) or value is None:
        raise RealtimePipelineError(f"EEG metadata missing sequence field: {name}")
    try:
        return tuple(value)
    except TypeError as exc:
        raise RealtimePipelineError(f"EEG metadata field must be a sequence: {name}") from exc
