from __future__ import annotations

from dataclasses import dataclass
import json
import math
import uuid
from typing import Any

import numpy as np

from bci_dayloop.serving.profiles import PROTOCOL_VERSION, DeviceProfile, match_device_profile


def _describe_window_shape(channels: int, sample_rate: float, samples: int) -> str:
    duration = samples / sample_rate if sample_rate else 0.0
    return f"{channels}ch @ {sample_rate:g} Hz × {samples} samples (~{duration:g}s)"


def _describe_profile(profile: DeviceProfile) -> str:
    return (
        f"{profile.id} ({profile.channels}ch @ {profile.sample_rate:g} Hz × "
        f"{profile.samples} samples, {profile.window_sec:g}s)"
    )


class ProtocolError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        request_id: str | None = None,
        observation_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.request_id = request_id
        self.observation_id = observation_id


@dataclass(frozen=True, slots=True)
class ParsedWindow:
    request_id: str
    window_id: int
    segment_id: str
    sample_rate: float
    channel_names: tuple[str, ...]
    channels: int
    samples: int
    start_time_sec: float | None
    end_time_sec: float | None
    data: np.ndarray
    profile: DeviceProfile | None


@dataclass(frozen=True, slots=True)
class ParsedFeedback:
    feedback_id: str
    observation_id: str
    label: int | None
    reward: float | None
    timestamp_sec: float | None
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PredictionResult:
    request_id: str
    observation_id: str
    window_id: int
    segment_id: str
    class_id: int
    class_name: str
    class_names: tuple[str, ...]
    probabilities: tuple[float, ...]
    confidence: float
    model_revision: str
    online_update_step: int = 0
    online_update_applied: bool = False
    prepare_latency_ms: float = 0.0
    inference_latency_ms: float = 0.0
    task: str = "motor_imagery"
    output_semantics: str | None = None


def dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def error_message(
    error: ProtocolError | Exception,
    *,
    request_id: str | None = None,
    observation_id: str | None = None,
) -> dict[str, Any]:
    if isinstance(error, ProtocolError):
        payload: dict[str, Any] = {
            "type": "error",
            "schema_version": PROTOCOL_VERSION,
            "code": error.code,
            "message": str(error),
        }
        if error.request_id:
            payload["request_id"] = error.request_id
        if error.observation_id:
            payload["observation_id"] = error.observation_id
        return payload
    payload = {
        "type": "error",
        "schema_version": PROTOCOL_VERSION,
        "code": "internal_error",
        "message": f"{type(error).__name__}: {error}",
    }
    if request_id:
        payload["request_id"] = request_id
    if observation_id:
        payload["observation_id"] = observation_id
    return payload


def hello_message(
    *,
    service: str,
    model_name: str,
    model_type: str,
    task: str,
    class_names: tuple[str, ...] | list[str],
    model_revision: str,
    strategy: str,
    window_sec: float | None = None,
    step_sec: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "hello",
        "schema_version": PROTOCOL_VERSION,
        "service": service,
        "model": {
            "name": model_name,
            "type": model_type,
            "task": task,
            "class_names": list(class_names),
        },
        "online": {
            "model_revision": model_revision,
            "strategy": strategy,
        },
    }
    if window_sec is not None:
        payload["input"] = {
            "window_sec": float(window_sec),
            "step_sec": float(step_sec if step_sec is not None else 0.5),
            "unit": "uV",
            "layout": "CT",
        }
    return payload


def prediction_message(result: PredictionResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "prediction",
        "schema_version": PROTOCOL_VERSION,
        "request_id": result.request_id,
        "observation_id": result.observation_id,
        "window_id": result.window_id,
        "segment_id": result.segment_id,
        "class_id": result.class_id,
        "class_name": result.class_name,
        "class_names": list(result.class_names),
        "probabilities": [float(value) for value in result.probabilities],
        "confidence": float(result.confidence),
        "model_revision": result.model_revision,
        "online_update_step": int(result.online_update_step),
        "online_update_applied": bool(result.online_update_applied),
        "prepare_latency_ms": float(result.prepare_latency_ms),
        "inference_latency_ms": float(result.inference_latency_ms),
        "task": result.task,
    }
    if result.output_semantics:
        payload["output_semantics"] = result.output_semantics
    return payload


def feedback_ack_message(
    feedback: ParsedFeedback,
    *,
    accepted: bool = True,
    duplicate: bool = False,
    reason: str | None = None,
    model_revision: str = "base",
    online_update_step: int = 0,
    online_update_applied: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "feedback_ack",
        "schema_version": PROTOCOL_VERSION,
        "feedback_id": feedback.feedback_id,
        "observation_id": feedback.observation_id,
        "accepted": accepted,
        "duplicate": duplicate,
        "model_revision": model_revision,
        "online_update_step": online_update_step,
        "online_update_applied": online_update_applied,
    }
    if reason:
        payload["reason"] = reason
    return payload


def parse_json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ProtocolError("Client JSON is not valid", code="invalid_json") from error
    if not isinstance(value, dict):
        raise ProtocolError("Client JSON root must be an object", code="invalid_json")
    return value


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def coerce_window_id(value: object) -> int:
    if isinstance(value, bool):
        raise ProtocolError("window_id must be a non-negative integer", code="invalid_window")
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
        if parsed >= 0:
            return parsed
    raise ProtocolError("window_id must be a non-negative integer", code="invalid_window")


def parse_window_header(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("type") != "window":
        raise ProtocolError("Expected type=window", code="invalid_window")
    schema = payload.get("schema_version", PROTOCOL_VERSION)
    if int(schema) != PROTOCOL_VERSION:
        raise ProtocolError(
            f"Unsupported schema_version {schema}",
            code="invalid_window",
            request_id=payload.get("request_id") if isinstance(payload.get("request_id"), str) else None,
        )
    request_id = payload.get("request_id")
    segment_id = payload.get("segment_id")
    if not isinstance(request_id, str) or not request_id:
        raise ProtocolError("window request_id is required", code="invalid_window")
    if not isinstance(segment_id, str) or not segment_id:
        raise ProtocolError("window segment_id is required", code="invalid_window", request_id=request_id)
    if payload.get("unit") != "uV":
        raise ProtocolError(
            "window unit must be uV",
            code="invalid_window",
            request_id=request_id,
        )
    if payload.get("layout") != "CT":
        raise ProtocolError(
            "window layout must be CT",
            code="invalid_window",
            request_id=request_id,
        )
    channel_names_raw = payload.get("channel_names")
    if not isinstance(channel_names_raw, list) or not channel_names_raw or not all(
        isinstance(name, str) and name for name in channel_names_raw
    ):
        raise ProtocolError(
            "window channel_names must be a non-empty string list",
            code="invalid_window",
            request_id=request_id,
        )
    channels = payload.get("channels")
    samples = payload.get("samples", payload.get("sample_count"))
    sample_rate = _finite_number(payload.get("sample_rate"))
    if not isinstance(channels, int) or channels <= 0:
        raise ProtocolError("window channels must be a positive integer", code="invalid_window", request_id=request_id)
    if not isinstance(samples, int) or samples <= 0:
        raise ProtocolError("window samples must be a positive integer", code="invalid_window", request_id=request_id)
    if sample_rate is None or sample_rate <= 0:
        raise ProtocolError("window sample_rate must be positive", code="invalid_window", request_id=request_id)
    if channels != len(channel_names_raw):
        raise ProtocolError(
            "window channels does not match channel_names",
            code="invalid_window",
            request_id=request_id,
        )
    return {
        "request_id": request_id,
        "window_id": coerce_window_id(payload.get("window_id")),
        "segment_id": segment_id,
        "sample_rate": sample_rate,
        "channel_names": tuple(str(name) for name in channel_names_raw),
        "channels": channels,
        "samples": samples,
        "start_time_sec": _finite_number(payload.get("start_time_sec")),
        "end_time_sec": _finite_number(payload.get("end_time_sec")),
    }


def complete_window(
    header: dict[str, Any],
    payload: bytes | bytearray | memoryview,
    *,
    required_profile: DeviceProfile | None = None,
) -> ParsedWindow:
    request_id = str(header["request_id"])
    expected_bytes = int(header["channels"]) * int(header["samples"]) * 4
    if len(payload) != expected_bytes:
        raise ProtocolError(
            f"window payload size mismatch: expected {expected_bytes} bytes, got {len(payload)}",
            code="invalid_window",
            request_id=request_id,
        )
    data = np.frombuffer(payload, dtype="<f4").reshape(int(header["channels"]), int(header["samples"])).copy()
    if not np.isfinite(data).all():
        raise ProtocolError(
            "window payload contains NaN or Inf",
            code="invalid_window",
            request_id=request_id,
        )
    channel_names = tuple(header["channel_names"])
    sample_rate = float(header["sample_rate"])
    samples = int(header["samples"])
    profile = match_device_profile(
        channel_names=channel_names,
        sample_rate=sample_rate,
        samples=samples,
        require_samples=False,
    )
    if required_profile is not None:
        identity = profile
        actual = _describe_window_shape(len(channel_names), sample_rate, samples)
        expected = _describe_profile(required_profile)
        if identity is None or identity.id != required_profile.id:
            raise ProtocolError(
                f"window does not match required profile {expected}; got {actual}",
                code="invalid_window",
                request_id=request_id,
            )
        if samples != required_profile.samples:
            raise ProtocolError(
                f"window duration does not match required profile {expected}; got {actual}",
                code="invalid_window",
                request_id=request_id,
            )
        profile = required_profile
    return ParsedWindow(
        request_id=request_id,
        window_id=int(header["window_id"]),
        segment_id=str(header["segment_id"]),
        sample_rate=float(header["sample_rate"]),
        channel_names=channel_names,
        channels=int(header["channels"]),
        samples=int(header["samples"]),
        start_time_sec=header["start_time_sec"],
        end_time_sec=header["end_time_sec"],
        data=data,
        profile=profile,
    )


def parse_feedback(payload: dict[str, Any]) -> ParsedFeedback:
    feedback_id = payload.get("feedback_id")
    observation_id = payload.get("observation_id")
    if not isinstance(feedback_id, str) or not feedback_id:
        raise ProtocolError("feedback_id is required", code="invalid_feedback")
    if not isinstance(observation_id, str) or not observation_id:
        raise ProtocolError(
            "observation_id is required",
            code="invalid_feedback",
            observation_id=None,
        )
    label_raw = payload.get("label")
    reward_raw = payload.get("reward")
    label = int(label_raw) if isinstance(label_raw, int) and not isinstance(label_raw, bool) else None
    reward = _finite_number(reward_raw)
    if label is None and reward is None:
        raise ProtocolError(
            "feedback requires label or reward",
            code="invalid_feedback",
            observation_id=observation_id,
        )
    metadata = payload.get("metadata")
    return ParsedFeedback(
        feedback_id=feedback_id,
        observation_id=observation_id,
        label=label,
        reward=reward,
        timestamp_sec=_finite_number(payload.get("timestamp_sec")),
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )


def new_observation_id() -> str:
    return f"obs-{uuid.uuid4()}"
