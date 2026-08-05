"""Low-cost, model-free metrics for realtime ingestion health and latency."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import statistics
import time
import tracemalloc

from .buffer import BufferStats
from .sync import EventSampleAlignment
from .contracts import WindowResult


@dataclass
class RealtimeMetrics:
    """Accumulate transparent runtime metrics without storing EEG samples."""

    started_at: float = field(default_factory=time.monotonic)
    chunk_exceptions: int = 0
    window_exceptions: int = 0
    disconnect_count: int = 0
    reconnect_count: int = 0
    expected_windows: int = 0
    emitted_windows: int = 0
    failed_windows: int = 0
    _receive_seconds: list[float] = field(default_factory=list, repr=False)
    _buffer_seconds: list[float] = field(default_factory=list, repr=False)
    _window_seconds: list[float] = field(default_factory=list, repr=False)
    _event_errors: list[float] = field(default_factory=list, repr=False)
    _memory_start_bytes: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        self._memory_start_bytes = tracemalloc.get_traced_memory()[0]

    def record_chunk(self, *, receive_seconds: float, buffer_seconds: float) -> None:
        self._receive_seconds.append(receive_seconds)
        self._buffer_seconds.append(buffer_seconds)

    def record_window(self, result: WindowResult, *, elapsed_seconds: float) -> None:
        self.expected_windows += 1
        self._window_seconds.append(elapsed_seconds)
        if result.status == "emitted":
            self.emitted_windows += 1
        else:
            self.failed_windows += 1
            self.window_exceptions += 1

    def record_event_alignment(self, alignment: EventSampleAlignment) -> None:
        self._event_errors.append(alignment.error_seconds)

    def record_chunk_exception(self) -> None:
        self.chunk_exceptions += 1

    def record_disconnect(self) -> None:
        self.disconnect_count += 1

    def record_reconnect(self) -> None:
        self.reconnect_count += 1

    def summary(self, buffer_stats: BufferStats) -> dict[str, object]:
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        return {
            "runtime_seconds": time.monotonic() - self.started_at,
            "memory_change_bytes": current_memory - self._memory_start_bytes,
            "peak_traced_memory_bytes": peak_memory,
            "received_chunks": buffer_stats.received_chunks,
            "received_samples": buffer_stats.received_samples,
            "inferred_missing_samples": buffer_stats.inferred_missing_samples,
            "out_of_order_chunks": buffer_stats.out_of_order_chunks,
            "duplicate_chunks": buffer_stats.duplicate_chunks,
            "buffer_overflows": buffer_stats.buffer_overflows,
            "maximum_backlog_samples": buffer_stats.maximum_backlog_samples,
            "expected_windows": self.expected_windows,
            "emitted_windows": self.emitted_windows,
            "failed_windows": self.failed_windows,
            "chunk_exceptions": self.chunk_exceptions,
            "window_exceptions": self.window_exceptions,
            "disconnect_count": self.disconnect_count,
            "reconnect_count": self.reconnect_count,
            "event_to_eeg_error_seconds": _distribution(self._event_errors),
            "receive_seconds": _distribution(self._receive_seconds),
            "buffer_seconds": _distribution(self._buffer_seconds),
            "window_seconds": _distribution(self._window_seconds),
        }


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "max": max(values) if values else None,
    }
