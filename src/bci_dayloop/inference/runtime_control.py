from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from bci_dayloop.acquisition.base import AbstractAcquirer
from bci_dayloop.inference.observability import PipelineRunStats
from bci_dayloop.inference.realtime import DecodeResult, SlidingWindowDecoder


class PipelineState(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class PipelineControllerSnapshot:
    state: PipelineState
    run_id: int
    started_at: float | None
    stopped_at: float | None
    results_emitted: int
    thread_alive: bool
    last_error_type: str | None
    last_error_message: str | None


AcquirerFactory = Callable[[], AbstractAcquirer]
ResultCallback = Callable[[DecodeResult], None]
StateCallback = Callable[[PipelineControllerSnapshot], None]


class PipelineController:
    """Thread-safe lifecycle manager for a decoder and freshly created acquirers."""

    def __init__(
        self,
        decoder: SlidingWindowDecoder,
        acquirer_factory: AcquirerFactory,
        *,
        max_windows: int | None = None,
        on_result: ResultCallback | None = None,
        on_state_change: StateCallback | None = None,
    ) -> None:
        self.decoder = decoder
        self.acquirer_factory = acquirer_factory
        self.max_windows = max_windows
        self.on_result = on_result
        self.on_state_change = on_state_change
        self.stats = decoder.run_stats or PipelineRunStats()
        self.decoder.run_stats = self.stats
        self._lock = threading.RLock()
        self._state = PipelineState.IDLE
        self._run_id = 0
        self._started_at: float | None = None
        self._stopped_at: float | None = None
        self._results_emitted = 0
        self._last_error: Exception | None = None
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None

    def _snapshot_locked(self) -> PipelineControllerSnapshot:
        return PipelineControllerSnapshot(
            state=self._state,
            run_id=self._run_id,
            started_at=self._started_at,
            stopped_at=self._stopped_at,
            results_emitted=self._results_emitted,
            thread_alive=self._thread is not None and self._thread.is_alive(),
            last_error_type=type(self._last_error).__name__ if self._last_error else None,
            last_error_message=str(self._last_error) if self._last_error else None,
        )

    def _notify_state_change(self, snapshot: PipelineControllerSnapshot) -> None:
        if self.on_state_change is None:
            return
        try:
            self.on_state_change(snapshot)
        except Exception as error:
            self._mark_failed(error)

    def _mark_failed(self, error: Exception) -> None:
        with self._lock:
            self._last_error = error
            self._state = PipelineState.FAILED
            self._stopped_at = time.perf_counter()
            if self._stop_event is not None:
                self._stop_event.set()

    def _set_terminal_state(self, state: PipelineState, run_id: int) -> None:
        with self._lock:
            if run_id != self._run_id:
                return
            if self._state != PipelineState.FAILED:
                self._state = state
            self._stopped_at = time.perf_counter()
            self._thread = None
            snapshot = self._snapshot_locked()
        self._notify_state_change(snapshot)

    def _worker(self, run_id: int, acquirer: AbstractAcquirer, stop_event: threading.Event) -> None:
        try:
            for result in self.decoder.run(
                acquirer,
                max_windows=self.max_windows,
                stop_event=stop_event,
            ):
                with self._lock:
                    if run_id != self._run_id:
                        return
                    self._results_emitted += 1
                if self.on_result is not None:
                    self.on_result(result)
        except Exception as error:
            self._mark_failed(error)
            with self._lock:
                run_matches = run_id == self._run_id
                if run_matches:
                    self._thread = None
                    snapshot = self._snapshot_locked()
                else:
                    snapshot = None
            if snapshot is not None:
                self._notify_state_change(snapshot)
            return

        terminal_state = PipelineState.STOPPED if stop_event.is_set() else PipelineState.COMPLETED
        self._set_terminal_state(terminal_state, run_id)

    def start(self) -> int:
        with self._lock:
            if self._state in {PipelineState.RUNNING, PipelineState.STOPPING}:
                raise RuntimeError(f"Cannot start PipelineController while state is {self._state.value}")
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Cannot start PipelineController while a worker thread is alive")

            self._run_id += 1
            run_id = self._run_id
            self.decoder.reset()
            self.stats.reset()
            self.stats.start()
            self._results_emitted = 0
            self._last_error = None
            self._started_at = time.perf_counter()
            self._stopped_at = None
            self._stop_event = threading.Event()
            acquirer = self.acquirer_factory()
            thread = threading.Thread(
                target=self._worker,
                args=(run_id, acquirer, self._stop_event),
                name=f"pipeline-controller-{run_id}",
                daemon=True,
            )
            self._thread = thread
            self._state = PipelineState.RUNNING
            thread.start()
            snapshot = self._snapshot_locked()
        self._notify_state_change(snapshot)
        return run_id

    def stop(self, *, wait: bool = True, timeout: float | None = None) -> bool:
        """Request a normal stop.

        Returns ``False`` without starting work when already inactive. When
        ``wait`` is true, a worker that does not finish before ``timeout``
        raises ``TimeoutError``.
        """
        with self._lock:
            if self._state in {PipelineState.IDLE, PipelineState.STOPPED, PipelineState.COMPLETED, PipelineState.FAILED}:
                return False
            if self._stop_event is not None:
                self._stop_event.set()
            self._state = PipelineState.STOPPING
            thread = self._thread
            snapshot = self._snapshot_locked()
        self._notify_state_change(snapshot)
        if wait and thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                raise TimeoutError("Timed out waiting for the PipelineController worker to stop")
        return True

    def restart(self, *, timeout: float | None = None) -> int:
        self.stop(wait=True, timeout=timeout)
        return self.start()

    def wait(self, timeout: float | None = None) -> bool:
        with self._lock:
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def snapshot(self) -> PipelineControllerSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def raise_if_failed(self) -> None:
        with self._lock:
            error = self._last_error
            failed = self._state == PipelineState.FAILED
        if failed:
            if error is not None:
                raise error
            raise RuntimeError("PipelineController failed without a recorded exception")
