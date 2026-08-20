from __future__ import annotations

import json
import threading
import time
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class LatencyBreakdown:
    preprocessing_ms: float
    model_ms: float
    total_ms: float

    def __post_init__(self) -> None:
        for field_name in ("preprocessing_ms", "model_ms", "total_ms"):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be a finite non-negative float, got {value}")
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True, slots=True)
class PipelineStatsSnapshot:
    started_at: float | None
    runtime_sec: float
    chunks_received: int
    expected_windows: int | None
    emitted_windows: int
    successful_windows: int
    failed_windows: int
    current_latency_ms: float | None
    average_latency_ms: float | None
    p95_latency_ms: float | None
    preprocessing_average_ms: float | None
    model_average_ms: float | None


class PipelineRunStats:
    """In-memory counters and latency statistics for one decoder run."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.reset()

    def start(self) -> None:
        with self._lock:
            if self._started_at is None:
                self._started_at = time.perf_counter()

    def reset(self) -> None:
        with self._lock:
            self._started_at: float | None = None
            self._chunks_received = 0
            self._expected_windows: int | None = None
            self._emitted_windows = 0
            self._successful_windows = 0
            self._failed_windows = 0
            self._latencies: list[LatencyBreakdown] = []

    def record_chunk(self) -> None:
        with self._lock:
            self.start()
            self._chunks_received += 1

    def record_success(self, latency: LatencyBreakdown) -> None:
        with self._lock:
            self.start()
            self._emitted_windows += 1
            self._successful_windows += 1
            self._latencies.append(latency)

    def record_failure(self) -> None:
        with self._lock:
            self.start()
            self._emitted_windows += 1
            self._failed_windows += 1

    def set_expected_windows(self, value: int | None) -> None:
        if value is not None and value < 0:
            raise ValueError(f"expected_windows must be non-negative or None, got {value}")
        with self._lock:
            self._expected_windows = value

    def snapshot(self) -> PipelineStatsSnapshot:
        with self._lock:
            runtime_sec = 0.0 if self._started_at is None else time.perf_counter() - self._started_at
            if not self._latencies:
                current_latency_ms = average_latency_ms = p95_latency_ms = None
                preprocessing_average_ms = model_average_ms = None
            else:
                totals = np.asarray([item.total_ms for item in self._latencies], dtype=float)
                preprocessings = np.asarray([item.preprocessing_ms for item in self._latencies], dtype=float)
                models = np.asarray([item.model_ms for item in self._latencies], dtype=float)
                current_latency_ms = float(totals[-1])
                average_latency_ms = float(totals.mean())
                p95_latency_ms = float(np.percentile(totals, 95))
                preprocessing_average_ms = float(preprocessings.mean())
                model_average_ms = float(models.mean())
            return PipelineStatsSnapshot(
                started_at=self._started_at,
                runtime_sec=float(runtime_sec),
                chunks_received=self._chunks_received,
                expected_windows=self._expected_windows,
                emitted_windows=self._emitted_windows,
                successful_windows=self._successful_windows,
                failed_windows=self._failed_windows,
                current_latency_ms=current_latency_ms,
                average_latency_ms=average_latency_ms,
                p95_latency_ms=p95_latency_ms,
                preprocessing_average_ms=preprocessing_average_ms,
                model_average_ms=model_average_ms,
            )


def calculate_expected_windows(total_samples: int, window_samples: int, step_samples: int) -> int:
    if window_samples <= 0:
        raise ValueError(f"window_samples must be positive, got {window_samples}")
    if step_samples <= 0:
        raise ValueError(f"step_samples must be positive, got {step_samples}")
    if total_samples < 0:
        raise ValueError(f"total_samples must be non-negative, got {total_samples}")
    if total_samples < window_samples:
        return 0
    return 1 + (total_samples - window_samples) // step_samples


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class JsonlWindowLogger:
    def __init__(
        self,
        path: str | Path,
        *,
        trace_mode: str = "on_change",
    ) -> None:
        self.path = Path(path)

        if trace_mode not in {
            "never",
            "first",
            "always",
            "on_change",
        }:
            raise ValueError(
                f"Unsupported trace_mode: "
                f"{trace_mode!r}."
            )

        self.trace_mode = trace_mode
        self._last_trace_id: str | None = None
        self._trace_written = False

    def _build_trace_payload(
            self,
            result: Any,
    ) -> dict[str, Any]:
        trace_content = {
            "steps": list(
                result.preprocessing_trace
            ),
            "diagnostics": dict(
                result.preprocessing_diagnostics
            ),
        }

        serialized = json.dumps(
            _json_value(trace_content),
            ensure_ascii=False,
            sort_keys=True,
        )

        trace_id = hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()[:16]

        include_full = False

        if self.trace_mode == "always":
            include_full = True
        elif self.trace_mode == "first":
            include_full = not self._trace_written
        elif self.trace_mode == "on_change":
            include_full = (
                    trace_id != self._last_trace_id
            )

        self._last_trace_id = trace_id

        if include_full:
            self._trace_written = True

        payload: dict[str, Any] = {
            "preprocessing_trace_id": trace_id,
        }

        if include_full:
            payload[
                "preprocessing_trace"
            ] = trace_content["steps"]

            payload[
                "preprocessing_diagnostics"
            ] = trace_content["diagnostics"]

        return payload

    def _write(self, record: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(_json_value(dict(record)), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(payload + "\n")
            handle.flush()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def log_success(
            self,
            *,
            window_id: int,
            result: Any,
    ) -> None:
        record: dict[str, Any] = {
            "timestamp": self._timestamp(),
            "status": "success",
            "window_id": window_id,
            "trial_id": result.trial_id,
            "expected_class_id": (
                result.expected_class_id
            ),
            "preprocessing_latency_ms": (
                result.preprocessing_latency_ms
            ),
            "model_latency_ms": (
                result.model_latency_ms
            ),
            "total_latency_ms": (
                result.total_latency_ms
            ),
            "model_diagnostics": (
                result.model_diagnostics
            ),
            "model_revision": (
                result.model_revision
            ),
            "online_update_step": (
                result.online_update_step
            ),
            "online_update_applied": (
                result.online_update_applied
            ),
        }

        if hasattr(result.prediction, "workload"):
            record.update(
                {
                    "prediction_type": "multi_head",
                    "prediction": {
                        task: {
                            "label_id": value.label_id,
                            "label": value.label,
                            "confidence": value.confidence,
                            "probabilities": list(value.probabilities),
                        }
                        for task, value in (
                            ("workload", result.prediction.workload),
                            ("attention", result.prediction.attention),
                            ("emotion", result.prediction.emotion),
                        )
                    },
                }
            )
        else:
            record.update(
                {
                    "prediction": result.prediction,
                    "confidence": result.confidence,
                    "probabilities": result.probabilities,
                    "command": result.command,
                }
            )

        record.update(
            self._build_trace_payload(result)
        )

        self._write(record)

    def log_error(self, *, window_id: int, error: Exception) -> None:
        self._write(
            {
                "timestamp": self._timestamp(),
                "status": "error",
                "window_id": window_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
