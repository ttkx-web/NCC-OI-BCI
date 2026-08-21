"""Small, dependency-free schema for the localhost EEG inference contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "1.0"


class InferenceSchemaError(ValueError):
    """Raised when an inference-contract v1 payload is invalid."""


def _required(payload: Mapping[str, Any], field: str) -> Any:
    if field not in payload:
        raise InferenceSchemaError(f"Missing required field: {field}.")
    return payload[field]


def _integer(payload: Mapping[str, Any], field: str) -> int:
    value = _required(payload, field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InferenceSchemaError(f"{field} must be an integer.")
    return int(value)


@dataclass(frozen=True, slots=True)
class EEGInferenceRequest:
    """Validated v1 request. ``eeg`` is always a finite float32 ``[C, T]`` array."""

    schema_version: str
    sample_rate_hz: float
    unit: str
    channel_names: tuple[str, ...]
    sequence_start: int
    sequence_end: int
    eeg: np.ndarray

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EEGInferenceRequest":
        if not isinstance(payload, Mapping):
            raise InferenceSchemaError("Request JSON must be an object.")

        schema_version = _required(payload, "schema_version")
        if schema_version != SCHEMA_VERSION:
            raise InferenceSchemaError(
                f"schema_version must be {SCHEMA_VERSION!r}, got {schema_version!r}."
            )
        unit = _required(payload, "unit")
        if unit != "uV":
            raise InferenceSchemaError("unit must be exactly 'uV'.")

        sample_rate_hz = _required(payload, "sample_rate_hz")
        if isinstance(sample_rate_hz, bool) or not isinstance(sample_rate_hz, (int, float)):
            raise InferenceSchemaError("sample_rate_hz must be a number.")
        sample_rate_hz = float(sample_rate_hz)
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise InferenceSchemaError("sample_rate_hz must be finite and greater than zero.")

        raw_names = _required(payload, "channel_names")
        if not isinstance(raw_names, list) or not raw_names:
            raise InferenceSchemaError("channel_names must be a non-empty array.")
        if any(not isinstance(name, str) or not name.strip() for name in raw_names):
            raise InferenceSchemaError("channel_names must contain non-empty strings.")
        channel_names = tuple(raw_names)

        sequence_start = _integer(payload, "sequence_start")
        sequence_end = _integer(payload, "sequence_end")
        if sequence_end < sequence_start:
            raise InferenceSchemaError("sequence_end must be greater than or equal to sequence_start.")

        raw_eeg = _required(payload, "eeg")
        if not isinstance(raw_eeg, list):
            raise InferenceSchemaError("eeg must be a two-dimensional JSON array in [C, T] layout.")
        try:
            eeg = np.asarray(raw_eeg, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise InferenceSchemaError("eeg must contain numeric values only.") from error
        if eeg.ndim != 2:
            raise InferenceSchemaError(f"eeg must have shape [C, T], got {eeg.shape}.")
        if eeg.shape[0] != len(channel_names):
            raise InferenceSchemaError(
                "eeg channel count must equal len(channel_names): "
                f"{eeg.shape[0]} != {len(channel_names)}."
            )
        if eeg.shape[1] <= 0:
            raise InferenceSchemaError("eeg must contain at least one sample per channel.")
        if sequence_end - sequence_start + 1 != eeg.shape[1]:
            raise InferenceSchemaError(
                "sequence range length must equal eeg.shape[1]: "
                f"{sequence_end - sequence_start + 1} != {eeg.shape[1]}."
            )
        if not np.isfinite(eeg).all():
            raise InferenceSchemaError("eeg must not contain NaN or Inf.")

        return cls(
            schema_version=schema_version,
            sample_rate_hz=sample_rate_hz,
            unit=unit,
            channel_names=channel_names,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            eeg=np.ascontiguousarray(eeg),
        )


@dataclass(frozen=True, slots=True)
class Prediction:
    task_id: str
    class_id: int
    label: str
    confidence: float
    probabilities: tuple[float, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "class_id": self.class_id,
            "label": self.label,
            "confidence": self.confidence,
            "probabilities": list(self.probabilities),
        }


@dataclass(frozen=True, slots=True)
class EEGInferenceResponse:
    schema_version: str
    sequence_start: int
    sequence_end: int
    predictions: tuple[Prediction, ...]
    latency_ms: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sequence_start": self.sequence_start,
            "sequence_end": self.sequence_end,
            "predictions": [prediction.to_payload() for prediction in self.predictions],
            "latency_ms": self.latency_ms,
        }
