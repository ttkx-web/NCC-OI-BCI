"""Timestamp-only event-to-EEG alignment helpers for realtime observability."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .buffer import TimestampedRingBuffer
from .contracts import EventMarker


@dataclass(frozen=True)
class EventSampleAlignment:
    """Nearest received EEG sample for one event; no clock offset is assumed."""

    event: EventMarker
    sample_index: int
    sequence_id: int
    eeg_timestamp: float
    error_seconds: float


def map_event_to_nearest_sample(
    event: EventMarker, buffer: TimestampedRingBuffer
) -> EventSampleAlignment:
    """Map an event to the nearest currently buffered EEG timestamp."""
    candidates = buffer.snapshot()
    if not candidates:
        raise ValueError("cannot align an event without buffered EEG samples")
    timestamps = np.concatenate([item.chunk.timestamps for item in candidates])
    distances = np.abs(timestamps - event.timestamp)
    offset = int(np.argmin(distances))
    cumulative = 0
    for item in candidates:
        size = item.chunk.timestamps.shape[0]
        if offset < cumulative + size:
            local_index = offset - cumulative
            eeg_timestamp = float(item.chunk.timestamps[local_index])
            return EventSampleAlignment(
                event=event,
                sample_index=item.start_sample_index + local_index,
                sequence_id=item.chunk.sequence_id,
                eeg_timestamp=eeg_timestamp,
                error_seconds=abs(eeg_timestamp - event.timestamp),
            )
        cumulative += size
    raise RuntimeError("event alignment index was not found")
