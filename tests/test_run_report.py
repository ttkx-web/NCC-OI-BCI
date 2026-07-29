from __future__ import annotations

import json

import numpy as np

from bci_dayloop.inference.run_report import PipelineRunReport


def make_report(**overrides) -> PipelineRunReport:
    values = {
        "generated_at": "2026-07-29T00:00:00+00:00",
        "controller_state": "COMPLETED",
        "run_id": np.int64(3),
        "model_package": "runs/demo/model_package",
        "model_name": "labram-linear",
        "device": "cpu",
        "data_path": "data/demo.h5",
        "session": "1test",
        "sample_rate": np.float64(200.0),
        "input_unit": "uV",
        "window_sec": np.float32(4.0),
        "step_sec": np.float32(0.5),
        "replay_speed": np.float64(10.0),
        "maximum_windows": np.int64(7),
        "expected_windows": np.int64(9),
        "target_windows": np.int64(7),
        "emitted_windows": np.int64(7),
        "successful_windows": np.int64(7),
        "failed_windows": np.int64(0),
        "chunks_received": np.int64(10),
        "runtime_sec": np.float64(1.25),
        "current_latency_ms": np.float64(3.0),
        "average_latency_ms": np.float64(2.5),
        "p95_latency_ms": np.float64(2.9),
        "preprocessing_average_ms": np.float64(1.0),
        "model_average_ms": np.float64(1.5),
        "jsonl_log_path": None,
        "last_error_type": None,
        "last_error_message": None,
    }
    values.update(overrides)
    return PipelineRunReport(**values)


def test_report_json_is_native_and_summary_creates_parent_directory(tmp_path):
    report = make_report()

    payload = json.loads(report.to_json())
    target = tmp_path / "nested" / "summary.json"
    report.save_json(target)

    assert isinstance(payload["run_id"], int)
    assert isinstance(payload["sample_rate"], float)
    assert json.loads(target.read_text(encoding="utf-8"))["target_windows"] == 7
    assert not list(target.parent.glob("*.tmp"))
