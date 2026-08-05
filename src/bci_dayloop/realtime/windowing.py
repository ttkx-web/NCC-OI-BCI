"""Generate fixed sliding windows from contiguous, timestamped realtime EEG data."""

from __future__ import annotations

from collections.abc import Iterable
import math
import time

import numpy as np

from .buffer import TimestampedRingBuffer
from .contracts import EventMarker, RealtimeWindow, WindowResult


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
        self.expected_windows = 0
        self.emitted_windows = 0
        self.failed_windows = 0

    def add_events(self, markers: Iterable[EventMarker]) -> None:
        self._markers.extend(markers)
        self._markers.sort(key=lambda marker: marker.timestamp)

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
                window = RealtimeWindow(
                    window_id=window_id,
                    samples=samples,
                    channel_names=source_chunks[0].chunk.channel_names,
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
                        if timestamps[0] <= marker.timestamp <= timestamps[-1]
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
