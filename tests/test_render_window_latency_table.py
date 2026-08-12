from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _renderer_module():
    path = Path("scripts/render_window_latency_table.py")
    spec = importlib.util.spec_from_file_location("latency_renderer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stats() -> dict[str, float]:
    return {"count": 200.0, "mean": 3.0, "min": 1.0, "p50": 2.0, "p95": 4.0, "max": 6.0}


def test_renderer_includes_live_deadline_integrity_and_max(tmp_path: Path) -> None:
    renderer = _renderer_module()
    candidate = {
        "candidate": {
            "candidate_id": "model_50m_4s",
            "model_name": "50M",
            "model_type": "model_50m",
            "package_path": "model_packages/safe",
            "package_sha256": "abc",
            "window_sec": 4.0,
            "step_sec": 0.5,
            "device": "cuda",
            "source_mode": "device",
            "warmup_windows": 20,
            "measured_windows": 200,
        },
        "num_records": 200,
        "preprocessing_ms": _stats(),
        "inference_ms": _stats(),
        "output_materialization_ms": _stats(),
        "compute_total_ms": _stats(),
        "window_ready_to_prediction_ms": _stats(),
        "last_sample_received_to_prediction_ms": _stats(),
        "deadline_ms": 500.0,
        "deadline_miss_count": 1,
        "deadline_miss_rate": 0.005,
        "expected_windows": 200,
        "completed_windows": 200,
        "failed_windows": 0,
        "source_integrity": {
            "missing_packets": 0,
            "duplicate_packets": 0,
            "out_of_order_packets": 0,
            "gap_count": 0,
            "malformed_packets": 0,
        },
        "status": "PASS",
        "created_at_utc": "2026-01-01T00:00:00+00:00",
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"schema_version": 1, "candidates": [candidate]}), encoding="utf-8")

    rows = renderer.read_summary_rows(path)
    long_table = renderer.build_long_markdown(rows, summary_path=path)
    wide_table = renderer.build_wide_markdown(rows, metric_name="last_sample_received_to_prediction_ms", summary_path=path)
    wide_csv = tmp_path / "wide.csv"
    renderer.write_wide_csv(
        rows,
        metric_name="last_sample_received_to_prediction_ms",
        path=wide_csv,
    )

    assert "Deadline misses" in long_table
    assert "200/200" in long_table
    assert "PASS" in long_table
    assert "Max (ms)" in wide_table
    assert "model_50m_p95_ms" in wide_csv.read_text(encoding="utf-8")
