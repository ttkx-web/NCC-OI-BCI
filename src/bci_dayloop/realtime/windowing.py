"""Generate fixed sliding windows from contiguous, timestamped realtime EEG data."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
import time

import numpy as np

from .buffer import TimestampedRingBuffer
from .contracts import EEGChunk, EventMarker, RealtimeWindow, WindowResult


@dataclass(frozen=True, slots=True)
class WindowProvenance:
    """Whitelisted, per-channel source provenance required by an EEG window."""

    channel_names: tuple[str, ...]
    channel_types: tuple[object, ...]
    channel_units: tuple[object, ...]
    sampling_rate: float
    unit: str
    unit_evidence_level: object
    model_safe: bool

    def to_metadata(self, *, continuous_segment_id: int | None) -> dict[str, object]:
        return {
            "channel_types": self.channel_types,
            "channel_units": self.channel_units,
            "unit_evidence_level": self.unit_evidence_level,
            "model_safe": self.model_safe,
            "source_sampling_rate": self.sampling_rate,
            "source_unit": self.unit,
            "continuous_segment_id": continuous_segment_id,
        }


def chunk_window_provenance(chunk: EEGChunk) -> WindowProvenance:
    """Validate the provenance that must survive from an EEG chunk to a window."""
    channel_types = _metadata_sequence(chunk, "channel_types")
    channel_units = _metadata_sequence(chunk, "channel_units")
    channel_count = chunk.samples.shape[0]
    if len(chunk.channel_names) != channel_count:
        raise ValueError("chunk channel_names length must match samples")
    if len(channel_types) != channel_count:
        raise ValueError("chunk channel_types length must match samples")
    if len(channel_units) != channel_count:
        raise ValueError("chunk channel_units length must match samples")
    if chunk.metadata.get("unit_evidence_level") is None:
        raise ValueError("chunk provenance is missing unit_evidence_level")
    if chunk.metadata.get("model_safe") is not True:
        raise ValueError("chunk provenance requires model_safe=true")
    return WindowProvenance(
        channel_names=chunk.channel_names,
        channel_types=channel_types,
        channel_units=channel_units,
        sampling_rate=chunk.sampling_rate,
        unit=chunk.unit,
        unit_evidence_level=chunk.metadata["unit_evidence_level"],
        model_safe=True,
    )


def _metadata_sequence(chunk: EEGChunk, name: str) -> tuple[object, ...]:
    value = chunk.metadata.get(name)
    if value is None or isinstance(value, (str, bytes)):
        raise ValueError(f"chunk provenance is missing sequence field: {name}")
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"chunk provenance field is not a sequence: {name}") from exc


class FixedSlidingWindowGenerator:
    """Emit only real contiguous samples; gaps become failed attempts, never padding."""

    def __init__(
        self,
        buffer: TimestampedRingBuffer,
        *,
        window_seconds: float = 4.0,
        step_seconds: float = 0.5,
    ) -> None:
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be positive and finite")
        if not math.isfinite(step_seconds) or step_seconds <= 0:
            raise ValueError("step_seconds must be positive and finite")
        self.buffer = buffer
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self._next_start_sample: int | None = None
        self._next_window_id = 0
        self._markers: list[EventMarker] = []
        self._continuous_segment_id: int | None = None
        self.expected_windows = 0
        self.emitted_windows = 0
        self.failed_windows = 0

    def add_events(self, markers: Iterable[EventMarker]) -> None:
        for marker in markers:
            if marker not in self._markers:
                self._markers.append(marker)
        self._markers.sort(key=lambda marker: marker.timestamp)

    @property
    def next_start_sample(self) -> int | None:
        return self._next_start_sample

    def reset_for_buffer(self, buffer: TimestampedRingBuffer) -> None:
        """Start a new contiguous segment without reusing old samples or markers."""
        self.buffer = buffer
        self._next_start_sample = None
        self._markers.clear()

    def set_continuous_segment_id(self, segment_id: int) -> None:
        if segment_id <= 0:
            raise ValueError("continuous_segment_id must be positive")
        self._continuous_segment_id = segment_id

    def generate(self) -> tuple[WindowResult, ...]:
        """Generate all currently decidable window attempts in stable sample order."""
        items = self.buffer.snapshot()
        if not items:
            return ()
        sampling_rate = items[0].chunk.sampling_rate
        window_samples = round(self.window_seconds * sampling_rate)
        step_samples = round(self.step_seconds * sampling_rate)
        if window_samples <= 0 or step_samples <= 0:
            raise ValueError("window and step must resolve to at least one sample")
        if self._next_start_sample is None:
            self._next_start_sample = items[0].start_sample_index

        results: list[WindowResult] = []
        latest_sample = items[-1].end_sample_index
        while self._next_start_sample + window_samples <= latest_sample:
            start = self._next_start_sample
            end = start + window_samples
            window_id = self._next_window_id
            self._next_window_id += 1
            self._next_start_sample += step_samples
            self.expected_windows += 1
            try:
                samples, timestamps, source_chunks = self.buffer.samples_between(start, end)
                if self._has_timestamp_gap(timestamps, sampling_rate):
                    raise ValueError("window crosses an EEG timestamp gap")
                provenance = _window_provenance(source_chunks)
                window = RealtimeWindow(
                    window_id=window_id,
                    samples=samples,
                    channel_names=provenance.channel_names,
                    sampling_rate=sampling_rate,
                    unit=source_chunks[0].chunk.unit,
                    timestamps=timestamps,
                    start_sample_index=start,
                    end_sample_index=end,
                    source_sequence_start=source_chunks[0].chunk.sequence_id,
                    source_sequence_end=source_chunks[-1].chunk.sequence_id,
                    markers=tuple(
                        marker
                        for marker in self._markers
                        if timestamps[0] <= marker.timestamp < timestamps[0] + (window_samples / sampling_rate)
                    ),
                    metadata=provenance.to_metadata(
                        continuous_segment_id=self._continuous_segment_id
                    ),
                )
            except ValueError as exc:
                self.failed_windows += 1
                results.append(WindowResult(window_id, "failed", reason=str(exc)))
            else:
                self.emitted_windows += 1
                results.append(
                    WindowResult(window_id, "emitted", window=window, emitted_at=time.monotonic())
                )
        return tuple(results)

    @staticmethod
    def _has_timestamp_gap(timestamps: np.ndarray, sampling_rate: float) -> bool:
        if timestamps.shape[0] < 2:
            return False
        allowed_delta = 1.5 / sampling_rate
        return bool(np.any(np.diff(timestamps) > allowed_delta))


def _window_provenance(source_chunks: tuple[object, ...]) -> WindowProvenance:
    if not source_chunks:
        raise ValueError("window has no source chunks")
    reference = chunk_window_provenance(source_chunks[0].chunk)  # type: ignore[union-attr]
    for item in source_chunks[1:]:
        candidate = chunk_window_provenance(item.chunk)  # type: ignore[union-attr]
        if candidate != reference:
            raise ValueError("source chunk provenance changed within one window")
    return reference
