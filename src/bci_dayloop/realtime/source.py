"""Protocol-neutral realtime source interfaces and deterministic replay sources."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from .contracts import EEGChunk, EventMarker


@runtime_checkable
class RealtimeEEGSource(Protocol):
    """A realtime EEG source; concrete protocols are deliberately unspecified."""

    def connect(self) -> None: ...

    def read_chunk(self) -> EEGChunk | None: ...

    def disconnect(self) -> None: ...

    def reconnect(self) -> None: ...

    def health(self) -> Mapping[str, object]: ...


@runtime_checkable
class RealtimeEventSource(Protocol):
    """A realtime event source that is independent from the EEG transport."""

    def connect(self) -> None: ...

    def read_event(self) -> EventMarker | None: ...

    def disconnect(self) -> None: ...

    def reconnect(self) -> None: ...

    def health(self) -> Mapping[str, object]: ...


class ReplayRealtimeEEGSource:
    """A deterministic test source that returns supplied chunks in their given order."""

    def __init__(self, chunks: tuple[EEGChunk, ...]) -> None:
        self._chunks = tuple(chunks)
        self._index = 0
        self._connected = False
        self._reconnects = 0

    def connect(self) -> None:
        self._connected = True

    def read_chunk(self) -> EEGChunk | None:
        if not self._connected:
            raise RuntimeError("source is not connected")
        if self._index >= len(self._chunks):
            return None
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk

    def disconnect(self) -> None:
        self._connected = False

    def reconnect(self) -> None:
        self._reconnects += 1
        self.connect()

    def health(self) -> Mapping[str, object]:
        return {
            "connected": self._connected,
            "remaining_chunks": len(self._chunks) - self._index,
            "reconnect_count": self._reconnects,
        }


class ReplayRealtimeEventSource:
    """A deterministic test source that returns supplied events in their given order."""

    def __init__(self, events: tuple[EventMarker, ...]) -> None:
        self._events = tuple(events)
        self._index = 0
        self._connected = False
        self._reconnects = 0

    def connect(self) -> None:
        self._connected = True

    def read_event(self) -> EventMarker | None:
        if not self._connected:
            raise RuntimeError("source is not connected")
        if self._index >= len(self._events):
            return None
        event = self._events[self._index]
        self._index += 1
        return event

    def disconnect(self) -> None:
        self._connected = False

    def reconnect(self) -> None:
        self._reconnects += 1
        self.connect()

    def health(self) -> Mapping[str, object]:
        return {
            "connected": self._connected,
            "remaining_events": len(self._events) - self._index,
            "reconnect_count": self._reconnects,
        }
