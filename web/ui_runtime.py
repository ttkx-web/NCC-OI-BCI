"""Thread-safe, Streamlit-free state helpers for the replay UI."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Literal, MutableMapping

import numpy as np

from bci_dayloop.inference.runtime_control import PipelineControllerSnapshot, PipelineState


UiEventKind = Literal["result", "state", "error"]


@dataclass(frozen=True, slots=True)
class UiEvent:
    """A plain event passed from a pipeline worker to the UI thread."""

    kind: UiEventKind
    payload: dict[str, Any]


class UiEventQueue:
    """Queue-only bridge: worker callbacks never need Streamlit state."""

    def __init__(self) -> None:
        self._queue: Queue[UiEvent] = Queue()

    def publish_result(self, result: Any, samples: np.ndarray) -> None:
        waveform = np.asarray(samples).copy()
        record = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        self._queue.put(UiEvent("result", {"result": dict(record), "waveform": waveform}))

    def publish_state(self, snapshot: PipelineControllerSnapshot) -> None:
        self._queue.put(UiEvent("state", {"snapshot": snapshot}))
        if snapshot.state == PipelineState.FAILED:
            self.publish_error(snapshot.last_error_type, snapshot.last_error_message)

    def publish_error(self, error_type: str | None, error_message: str | None) -> None:
        self._queue.put(
            UiEvent(
                "error",
                {
                    "error_type": error_type or "RuntimeError",
                    "error_message": error_message or "Pipeline worker failed",
                },
            )
        )

    def drain(self) -> list[UiEvent]:
        events: list[UiEvent] = []
        while True:
            try:
                events.append(self._queue.get_nowait())
            except Empty:
                return events


@dataclass(frozen=True, slots=True)
class UiControlAvailability:
    start_enabled: bool
    stop_enabled: bool
    restart_enabled: bool
    configuration_enabled: bool


def control_availability(
    state: PipelineState | None,
    *,
    thread_alive: bool = False,
) -> UiControlAvailability:
    """Return the UI button and configuration availability for a controller state."""

    active = thread_alive or state in {PipelineState.RUNNING, PipelineState.STOPPING}
    terminal = state in {PipelineState.STOPPED, PipelineState.COMPLETED, PipelineState.FAILED}
    return UiControlAvailability(
        start_enabled=not active,
        stop_enabled=state == PipelineState.RUNNING,
        restart_enabled=state == PipelineState.RUNNING or (terminal and not thread_alive),
        configuration_enabled=not active,
    )


def append_bounded(history: list[dict[str, Any]], record: dict[str, Any], *, limit: int = 500) -> None:
    if limit <= 0:
        raise ValueError("limit must be positive")
    history.append(dict(record))
    overflow = len(history) - limit
    if overflow > 0:
        del history[:overflow]


def apply_events(state: MutableMapping[str, Any], events: list[UiEvent], *, history_limit: int = 500) -> None:
    """Apply drained events in the UI thread to a session-state-like mapping."""

    for event in events:
        if event.kind == "result":
            record = dict(event.payload["result"])
            state["last_result"] = record
            append_bounded(state.setdefault("history", []), record, limit=history_limit)
            state["waveform"] = np.asarray(event.payload["waveform"]).copy()
        elif event.kind == "state":
            snapshot = event.payload["snapshot"]
            state["controller_snapshot"] = snapshot
        elif event.kind == "error":
            state["runtime_error"] = {
                "error_type": event.payload["error_type"],
                "error_message": event.payload["error_message"],
            }


def target_window_count(expected_windows: int, maximum_windows: int | None) -> int:
    if expected_windows < 0:
        raise ValueError("expected_windows must be non-negative")
    if maximum_windows is not None and maximum_windows <= 0:
        raise ValueError("maximum_windows must be positive or None")
    return expected_windows if maximum_windows is None else min(expected_windows, maximum_windows)
