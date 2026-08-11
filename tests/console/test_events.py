from __future__ import annotations

import json

from app.schemas.events import (
    error_event,
    input_contract_event,
    latency_event,
    prediction_event,
    runtime_health_event,
    state_event,
)


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for item in value.values():
            keys.update(_all_keys(item))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in value:
            keys.update(_all_keys(item))
        return keys
    return set()


def test_all_websocket_events_are_json_serializable_and_data_minimized() -> None:
    events = [
        state_event("run_test", "running"),
        prediction_event(
            "run_test",
            window_id=2,
            trial_id=1,
            predicted_class=0,
            predicted_name="left_hand",
            command="LEFT",
            confidence=0.82,
            probabilities=[0.82, 0.1, 0.05, 0.03],
            expected_class_id=0,
            expected_class_name="left_hand",
        ),
        latency_event(
            "run_test", prepare_ms=18.2, inference_ms=31.1, total_ms=52.4, p50_ms=51.8, p95_ms=67.1
        ),
        runtime_health_event("run_test", successful_windows=2, failed_windows=0, expected_windows=100),
        input_contract_event(
            "run_test",
            safe=True,
            source_channels=59,
            target_channels=64,
            valid_channels=57,
            window_sec=4.0,
            target_sample_rate=100,
        ),
        error_event("run_test", code="MODEL_INPUT_UNSAFE", message="Unexpected mapping", fatal=True),
    ]
    assert {event["type"] for event in events} == {
        "state",
        "prediction",
        "latency",
        "runtime_health",
        "input_contract",
        "error",
    }
    json.dumps(events)
    prohibited = {"samples", "raw_eeg", "waveform", "waveform_preview"}
    assert _all_keys(events).isdisjoint(prohibited)


def test_fatal_error_serialization_preserves_fail_closed_flag() -> None:
    event = error_event("run_test", code="MODEL_INPUT_UNSAFE", message="blocked", fatal=True)
    assert event["payload"]["fatal"] is True

