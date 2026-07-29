"""Machine-readable final reports for pseudo-realtime pipeline runs."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _native_json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_native_json_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _native_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class PipelineRunReport:
    generated_at: str
    controller_state: str
    run_id: int
    model_package: str
    model_name: str | None
    device: str
    data_path: str
    session: str
    sample_rate: float | None
    input_unit: str | None
    window_sec: float
    step_sec: float
    replay_speed: float
    maximum_windows: int | None
    expected_windows: int
    target_windows: int
    emitted_windows: int
    successful_windows: int
    failed_windows: int
    chunks_received: int
    runtime_sec: float
    current_latency_ms: float | None
    average_latency_ms: float | None
    p95_latency_ms: float | None
    preprocessing_average_ms: float | None
    model_average_ms: float | None
    jsonl_log_path: str | None
    last_error_type: str | None
    last_error_message: str | None
    is_test_head: bool = False
    model_warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _native_json_value(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def save_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(self.to_json())
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return target
