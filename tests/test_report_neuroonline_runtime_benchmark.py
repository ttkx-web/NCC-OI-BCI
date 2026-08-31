from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import report_neuroonline_runtime_benchmark as report_script


def _summary(*, mode: str, accuracy: float, balanced_accuracy: float, macro_f1: float, updates: int, latency: dict[str, float | None]) -> dict[str, object]:
    return {
        mode: {
            "metrics": {
                "overall": {
                    "accuracy": accuracy,
                    "balanced_accuracy": balanced_accuracy,
                    "macro_f1": macro_f1,
                }
            },
            "updates": {
                "num_updates": updates,
                "latency": latency,
            },
        }
    }


def test_runtime_benchmark_report_compares_static_and_neuroonline(tmp_path: Path) -> None:
    static = _summary(
        mode="static",
        accuracy=0.60,
        balanced_accuracy=0.55,
        macro_f1=0.50,
        updates=0,
        latency={"mean_ms": None, "p50_ms": None, "p95_ms": None},
    )
    neuroonline = _summary(
        mode="neuroonline",
        accuracy=0.70,
        balanced_accuracy=0.65,
        macro_f1=0.62,
        updates=80,
        latency={"mean_ms": 3.0, "p50_ms": 2.5, "p95_ms": 4.5},
    )
    static_path = tmp_path / "static_none" / "summary.json"
    online_path = tmp_path / "neuroonline_80" / "summary.json"
    static_path.parent.mkdir()
    online_path.parent.mkdir()
    static_path.write_text(json.dumps(static), encoding="utf-8")
    online_path.write_text(json.dumps(neuroonline), encoding="utf-8")

    output_dir = tmp_path / "report"
    assert report_script.main([
        "--static-summary", str(static_path),
        "--neuroonline-summary", str(online_path),
        "--output-dir", str(output_dir),
    ]) == 0

    result = json.loads((output_dir / "runtime_benchmark_report.json").read_text(encoding="utf-8"))
    assert result["static_none"]["accuracy"] == 0.60
    assert result["neuroonline_80"]["update_count"] == 80
    assert result["neuroonline_80"]["latency_ms"] == {"mean_ms": 3.0, "p50_ms": 2.5, "p95_ms": 4.5}
    assert result["gain"] == pytest.approx(
        {
            "accuracy": 0.10,
            "balanced_accuracy": 0.10,
            "macro_f1": 0.12,
        }
    )

    markdown = (output_dir / "runtime_benchmark_report.md").read_text(encoding="utf-8")
    assert "static_none" in markdown
    assert "neuroonline_80" in markdown
    assert "Latency P95" in markdown
