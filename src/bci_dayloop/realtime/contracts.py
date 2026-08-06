"""Immutable, unit-explicit contracts shared by realtime ingestion components."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np


def _immutable_metadata(metadata: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(metadata))


def _readonly_float_array(value: object, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).copy()
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must not contain NaN or Inf")
    array.setflags(write=False)
    return array


def _readonly_timestamps(value: object, *, sample_count: int) -> np.ndarray:
    timestamps = np.asarray(value, dtype=np.float64).copy()
    if timestamps.ndim != 1 or timestamps.shape[0] != sample_count:
        raise ValueError("timestamps must be one-dimensional with length N")
    if not np.isfinite(timestamps).all():
        raise ValueError("timestamps must contain only finite values")
    if timestamps.size > 1 and not np.all(np.diff(timestamps) >= 0):
        raise ValueError("timestamps must be non-decreasing")
    timestamps.setflags(write=False)
    return timestamps


@dataclass(frozen=True)
class EEGChunk:
    """One received, unit-declared EEG chunk with shape ``[channels, samples]``."""

    samples: np.ndarray
    channel_names: tuple[str, ...]
    sampling_rate: float
    unit: str
    timestamps: np.ndarray
    sequence_id: int
    device_id: str | None
    received_at: float
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        samples = _readonly_float_array(self.samples, name="samples", ndim=2)
        channel_names = tuple(self.channel_names)
        if samples.shape[0] != len(channel_names):
            raise ValueError("samples channel count must match channel_names")
        if samples.shape[1] == 0:
            raise ValueError("samples must contain at least one sample")
        if not math.isfinite(self.sampling_rate) or self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive and finite")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("unit must be explicitly declared and non-empty")
        if isinstance(self.sequence_id, bool) or not isinstance(self.sequence_id, int):
            raise ValueError("sequence_id must be an integer")
        if not math.isfinite(float(self.received_at)):
            raise ValueError("received_at must be finite")
        timestamps = _readonly_timestamps(self.timestamps, sample_count=samples.shape[1])
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "channel_names", channel_names)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "metadata", _immutable_metadata(self.metadata))


@dataclass(frozen=True)
class EventMarker:
    """A timestamped device or paradigm event, kept independent of any protocol."""

    timestamp: float
    event_type: str
    code: int | str | None = None
    sequence_id: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.timestamp)):
            raise ValueError("event timestamp must be finite")
        if not isinstance(self.event_type, str) or not self.event_type.strip():
            raise ValueError("event_type must be non-empty")
        object.__setattr__(self, "metadata", _immutable_metadata(self.metadata))


@dataclass(frozen=True)
class RealtimeWindow:
    """A non-padded, contiguous realtime EEG window derived from received chunks."""

    window_id: int
    samples: np.ndarray
    channel_names: tuple[str, ...]
    sampling_rate: float
    unit: str
    timestamps: np.ndarray
    start_sample_index: int
    end_sample_index: int
    source_sequence_start: int
    source_sequence_end: int
    markers: tuple[EventMarker, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        samples = _readonly_float_array(self.samples, name="samples", ndim=2)
        channel_names = tuple(self.channel_names)
        if samples.shape[0] != len(channel_names):
            raise ValueError("samples channel count must match channel_names")
        if not math.isfinite(self.sampling_rate) or self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive and finite")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("unit must be explicitly declared and non-empty")
        timestamps = _readonly_timestamps(self.timestamps, sample_count=samples.shape[1])
        if self.start_sample_index < 0 or self.end_sample_index <= self.start_sample_index:
            raise ValueError("window sample bounds must be positive and ordered")
        if self.end_sample_index - self.start_sample_index != samples.shape[1]:
            raise ValueError("window sample bounds must match samples")
        if self.source_sequence_end < self.source_sequence_start:
            raise ValueError("source sequence range must be ordered")
        if any(not isinstance(marker, EventMarker) for marker in self.markers):
            raise ValueError("markers must contain EventMarker instances")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "channel_names", channel_names)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "markers", tuple(self.markers))
        object.__setattr__(self, "metadata", _immutable_metadata(self.metadata))


@dataclass(frozen=True)
class WindowResult:
    """An emitted or rejected window attempt; no model output is implied."""

    window_id: int
    status: str
    window: RealtimeWindow | None = None
    reason: str | None = None
    emitted_at: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"emitted", "failed"}:
            raise ValueError("status must be 'emitted' or 'failed'")
        if self.status == "emitted" and self.window is None:
            raise ValueError("emitted WindowResult requires a window")
        if self.status == "failed" and not self.reason:
            raise ValueError("failed WindowResult requires a reason")
        if self.emitted_at is not None and not math.isfinite(float(self.emitted_at)):
            raise ValueError("emitted_at must be finite when present")
        object.__setattr__(self, "metadata", _immutable_metadata(self.metadata))
