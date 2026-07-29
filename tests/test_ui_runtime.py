from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bci_dayloop.inference.runtime_control import PipelineControllerSnapshot, PipelineState
from web.ui_runtime import (
    UiEventQueue,
    apply_events,
    control_availability,
    target_window_count,
)


@dataclass
class FakeResult:
    prediction: str = "feet"
    confidence: float = 0.85
    command: str = "FORWARD"

    def to_dict(self) -> dict[str, object]:
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "command": self.command,
            "preprocessing_latency_ms": 1.0,
            "model_latency_ms": 2.0,
            "total_latency_ms": 3.0,
            "trial_id": 4,
            "expected_class_id": 2,
        }


def make_snapshot(state: PipelineState, *, error: Exception | None = None) -> PipelineControllerSnapshot:
    return PipelineControllerSnapshot(
        state=state,
        run_id=3,
        started_at=1.0,
        stopped_at=None,
        results_emitted=0,
        thread_alive=state in {PipelineState.RUNNING, PipelineState.STOPPING},
        last_error_type=type(error).__name__ if error else None,
        last_error_message=str(error) if error else None,
    )


def test_result_events_are_ordered_and_waveform_is_copied():
    queue = UiEventQueue()
    first = np.ones((2, 3), dtype=np.float32)
    second = np.full((2, 3), 2.0, dtype=np.float32)
    queue.publish_result(FakeResult("left_hand"), first)
    queue.publish_result(FakeResult("right_hand"), second)
    first[:, :] = 99.0

    events = queue.drain()

    assert [event.payload["result"]["prediction"] for event in events] == ["left_hand", "right_hand"]
    assert len(events) == 2
    np.testing.assert_array_equal(events[0].payload["waveform"], np.ones((2, 3), dtype=np.float32))


def test_apply_events_bounds_history_updates_state_and_preserves_error():
    queue = UiEventQueue()
    state: dict[str, object] = {"history": []}
    for index in range(502):
        queue.publish_result(FakeResult(f"class-{index}"), np.full((1, 2), index, dtype=np.float32))
    failure = RuntimeError("decoder failed")
    queue.publish_state(make_snapshot(PipelineState.FAILED, error=failure))

    apply_events(state, queue.drain())

    history = state["history"]
    assert len(history) == 500
    assert history[0]["prediction"] == "class-2"
    assert state["controller_snapshot"].state == PipelineState.FAILED
    assert state["runtime_error"] == {"error_type": "RuntimeError", "error_message": "decoder failed"}


def test_control_availability_matches_controller_lifecycle():
    running = control_availability(PipelineState.RUNNING, thread_alive=True)
    assert not running.start_enabled
    assert running.stop_enabled
    assert running.restart_enabled
    assert not running.configuration_enabled

    for state in (PipelineState.STOPPED, PipelineState.COMPLETED, PipelineState.FAILED):
        terminal = control_availability(state)
        assert terminal.start_enabled
        assert not terminal.stop_enabled
        assert terminal.restart_enabled
        assert terminal.configuration_enabled


def test_target_window_count_applies_maximum_windows():
    assert target_window_count(9, None) == 9
    assert target_window_count(9, 20) == 9
    assert target_window_count(9, 4) == 4
