from __future__ import annotations

import time
from typing import Any, Literal

from app.schemas.common import ConsoleModel


EventType = Literal["state", "prediction", "latency", "runtime_health", "input_contract", "error"]


class RunEvent(ConsoleModel):
    type: EventType
    run_id: str
    timestamp: float
    payload: dict[str, Any]


def run_event(event_type: EventType, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return RunEvent(type=event_type, run_id=run_id, timestamp=time.time(), payload=payload).model_dump()


def state_event(run_id: str, state: str) -> dict[str, Any]:
    return run_event("state", run_id, {"state": state})


def prediction_event(
    run_id: str,
    *,
    window_id: int,
    trial_id: int | None,
    predicted_class: int,
    predicted_name: str,
    command: str,
    confidence: float,
    probabilities: list[float],
    expected_class_id: int | None,
    expected_class_name: str | None,
) -> dict[str, Any]:
    return run_event(
        "prediction",
        run_id,
        {
            "window_id": window_id,
            "trial_id": trial_id,
            "predicted_class": predicted_class,
            "predicted_name": predicted_name,
            "command": command,
            "confidence": confidence,
            "probabilities": probabilities,
            "expected_class_id": expected_class_id,
            "expected_class_name": expected_class_name,
        },
    )


def latency_event(
    run_id: str,
    *,
    prepare_ms: float,
    inference_ms: float,
    total_ms: float,
    p50_ms: float | None,
    p95_ms: float | None,
) -> dict[str, Any]:
    return run_event(
        "latency",
        run_id,
        {
            "prepare_ms": prepare_ms,
            "inference_ms": inference_ms,
            "total_ms": total_ms,
            "p50_ms": p50_ms,
            "p95_ms": p95_ms,
        },
    )


def runtime_health_event(
    run_id: str, *, successful_windows: int, failed_windows: int, expected_windows: int | None
) -> dict[str, Any]:
    return run_event(
        "runtime_health",
        run_id,
        {
            "successful_windows": successful_windows,
            "failed_windows": failed_windows,
            "expected_windows": expected_windows,
        },
    )


def input_contract_event(
    run_id: str,
    *,
    safe: bool,
    source_channels: int,
    target_channels: int,
    valid_channels: int,
    window_sec: float,
    target_sample_rate: float,
) -> dict[str, Any]:
    return run_event(
        "input_contract",
        run_id,
        {
            "safe": safe,
            "source_channels": source_channels,
            "target_channels": target_channels,
            "valid_channels": valid_channels,
            "window_sec": window_sec,
            "target_sample_rate": target_sample_rate,
        },
    )


def error_event(run_id: str, *, code: str, message: str, fatal: bool) -> dict[str, Any]:
    return run_event("error", run_id, {"code": code, "message": message, "fatal": fatal})
