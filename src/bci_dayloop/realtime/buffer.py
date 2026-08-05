"""Timestamp-aware ring buffer that refuses to overwrite unconsumed EEG samples."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math

import numpy as np

from .contracts import EEGChunk


class BufferOverflowError(RuntimeError):
    """Raised when accepting a chunk would overwrite buffered, unconsumed samples."""


@dataclass(frozen=True)
class BufferedChunk:
    """An accepted chunk placed at an absolute received-sample range."""

    chunk: EEGChunk
    start_sample_index: int
    end_sample_index: int


@dataclass(frozen=True)
class BufferStats:
    received_chunks: int
    received_samples: int
    inferred_missing_samples: int
    out_of_order_chunks: int
    duplicate_chunks: int
    buffer_overflows: int
    maximum_backlog_samples: int


class TimestampedRingBuffer:
    """Bounded EEG storage with explicit backpressure and timestamp gap accounting."""

    def __init__(self, capacity_samples: int) -> None:
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be positive")
        self.capacity_samples = capacity_samples
        self._chunks: deque[BufferedChunk] = deque()
        self._next_sample_index = 0
        self._received_chunks = 0
        self._received_samples = 0
        self._inferred_missing_samples = 0
        self._out_of_order_chunks = 0
        self._duplicate_chunks = 0
        self._buffer_overflows = 0
        self._maximum_backlog_samples = 0
        self._last_sequence_id: int | None = None
        self._last_timestamp: float | None = None
        self._channel_names: tuple[str, ...] | None = None
        self._sampling_rate: float | None = None
        self._unit: str | None = None

    @property
    def backlog_samples(self) -> int:
        return sum(item.end_sample_index - item.start_sample_index for item in self._chunks)

    @property
    def next_sample_index(self) -> int:
        return self._next_sample_index

    @property
    def oldest_sample_index(self) -> int | None:
        return self._chunks[0].start_sample_index if self._chunks else None

    @property
    def stats(self) -> BufferStats:
        return BufferStats(
            received_chunks=self._received_chunks,
            received_samples=self._received_samples,
            inferred_missing_samples=self._inferred_missing_samples,
            out_of_order_chunks=self._out_of_order_chunks,
            duplicate_chunks=self._duplicate_chunks,
            buffer_overflows=self._buffer_overflows,
            maximum_backlog_samples=self._maximum_backlog_samples,
        )

    def append(self, chunk: EEGChunk) -> bool:
        """Accept a chunk, or explicitly reject duplicate/out-of-order/overflow input."""
        self._received_chunks += 1
        self._received_samples += chunk.samples.shape[1]
        self._validate_stream_contract(chunk)
        if self._last_sequence_id is not None:
            if chunk.sequence_id == self._last_sequence_id:
                self._duplicate_chunks += 1
                return False
            if chunk.sequence_id < self._last_sequence_id:
                self._out_of_order_chunks += 1
                return False
            self._infer_timestamp_gap(chunk)
        if self.backlog_samples + chunk.samples.shape[1] > self.capacity_samples:
            self._buffer_overflows += 1
            raise BufferOverflowError(
                "Buffer capacity would overwrite unconsumed samples; consume explicitly first"
            )

        start = self._next_sample_index
        end = start + chunk.samples.shape[1]
        self._chunks.append(BufferedChunk(chunk, start, end))
        self._next_sample_index = end
        self._last_sequence_id = chunk.sequence_id
        self._last_timestamp = float(chunk.timestamps[-1])
        self._maximum_backlog_samples = max(self._maximum_backlog_samples, self.backlog_samples)
        return True

    def snapshot(self) -> tuple[BufferedChunk, ...]:
        """Return the accepted chunks without exposing mutable buffer internals."""
        return tuple(self._chunks)

    def discard_before(self, sample_index: int) -> None:
        """Explicitly discard consumed data; this is the only way to free capacity."""
        while self._chunks and self._chunks[0].end_sample_index <= sample_index:
            self._chunks.popleft()
        if not self._chunks or sample_index <= self._chunks[0].start_sample_index:
            return
        first = self._chunks.popleft()
        offset = sample_index - first.start_sample_index
        trimmed = replace(
            first.chunk,
            samples=first.chunk.samples[:, offset:],
            timestamps=first.chunk.timestamps[offset:],
        )
        self._chunks.appendleft(
            BufferedChunk(trimmed, sample_index, first.end_sample_index)
        )

    def samples_between(self, start_sample_index: int, end_sample_index: int) -> tuple[np.ndarray, np.ndarray, tuple[BufferedChunk, ...]]:
        """Return actual received samples for an absolute range, never padding a gap."""
        if start_sample_index < 0 or end_sample_index <= start_sample_index:
            raise ValueError("sample range must be positive and ordered")
        selected: list[BufferedChunk] = []
        parts: list[np.ndarray] = []
        timestamp_parts: list[np.ndarray] = []
        covered_until = start_sample_index
        for item in self._chunks:
            if item.end_sample_index <= start_sample_index:
                continue
            if item.start_sample_index >= end_sample_index:
                break
            if item.start_sample_index > covered_until:
                raise ValueError("requested sample range is no longer fully buffered")
            local_start = max(start_sample_index, item.start_sample_index) - item.start_sample_index
            local_end = min(end_sample_index, item.end_sample_index) - item.start_sample_index
            if local_end > local_start:
                selected.append(item)
                parts.append(item.chunk.samples[:, local_start:local_end])
                timestamp_parts.append(item.chunk.timestamps[local_start:local_end])
                covered_until = item.start_sample_index + local_end
            if covered_until == end_sample_index:
                break
        if covered_until != end_sample_index:
            raise ValueError("requested sample range is not yet fully buffered")
        return np.concatenate(parts, axis=1), np.concatenate(timestamp_parts), tuple(selected)

    def _validate_stream_contract(self, chunk: EEGChunk) -> None:
        if self._channel_names is None:
            self._channel_names = chunk.channel_names
            self._sampling_rate = chunk.sampling_rate
            self._unit = chunk.unit
            return
        if chunk.channel_names != self._channel_names:
            raise ValueError("chunk channel_names changed within one realtime stream")
        if chunk.sampling_rate != self._sampling_rate:
            raise ValueError("chunk sampling_rate changed within one realtime stream")
        if chunk.unit != self._unit:
            raise ValueError("chunk unit changed within one realtime stream")

    def _infer_timestamp_gap(self, chunk: EEGChunk) -> None:
        assert self._last_timestamp is not None
        assert self._sampling_rate is not None
        expected_next = self._last_timestamp + (1.0 / self._sampling_rate)
        elapsed = float(chunk.timestamps[0]) - expected_next
        if elapsed > 0:
            self._inferred_missing_samples += max(0, math.floor(elapsed * self._sampling_rate + 0.5))
