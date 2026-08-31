from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import report_neuroonline_benchmark as report_script


def _summary(static: float, online: float) -> dict[str, object]:
    gain = online - static
    metric_names = ("accuracy", "balanced_accuracy", "macro_f1")
    return {
        "static": {
            "metrics": {
                "overall": {name: static for name in metric_names}
            }
        },
        "neuroonline": {
            "metrics": {
                "overall": {name: online for name in metric_names}
            },
            "updates": {
                "num_updates": 2,
                "latency": {
                    "mean_ms": 1.0,
                    "p50_ms": 0.9,
                    "p95_ms": 1.5,
                },
            },
        },
        "gains": {
            "overall": {f"{name}_gain": gain for name in metric_names}
        },
    }


def _write_matrix(root: Path) -> None:
    for subject in (1, 2):
        for model in ("model_a", "model_b"):
            path = root / f"s{subject}" / model / "summary.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(_summary(0.5, 0.6)), encoding="utf-8"
            )


def test_unified_entry_is_dataset_and_model_agnostic(tmp_path: Path) -> None:
    input_root = tmp_path / "evaluations"
    output_dir = tmp_path / "report"
    _write_matrix(input_root)

    assert report_script.main(
        [
            "--input-root",
            str(input_root),
            "--output-dir",
            str(output_dir),
            "--dataset-name",
            "synthetic_dataset",
            "--subjects",
            "1",
            "2",
            "--models",
            "model_a",
            "model_b",
            "--summary-template",
            "s{subject}/{model}/summary.json",
            "--filename-prefix",
            "seed_emotion_benchmark",
            "--markdown-filename",
            "seed_emotion_benchmark.md",
        ]
    ) == 0

    payload = json.loads(
        (output_dir / "seed_emotion_benchmark_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["dataset"] == "synthetic_dataset"
    assert payload["subjects"] == [1, 2]
    assert payload["models"] == ["model_a", "model_b"]
    assert len(payload["subject_results"]) == 4
    assert payload["model_summaries"]["model_a"]["metrics"][
        "balanced_accuracy"
    ]["gain"]["mean"] == pytest.approx(0.1)
    markdown = (output_dir / "seed_emotion_benchmark.md").read_text(
        encoding="utf-8"
    )
    assert markdown.strip()
    assert "# synthetic_dataset NeuroOnline Benchmark Results" in markdown
    assert not (output_dir / "seed_emotion_benchmark_report.md").exists()


def test_unified_entry_fails_closed_for_missing_model_result(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "evaluations"
    _write_matrix(input_root)
    (input_root / "s2" / "model_b" / "summary.json").unlink()

    with pytest.raises(ValueError, match="Required evaluation summary"):
        report_script.load_benchmark_results(
            input_root=input_root,
            subjects=(1, 2),
            models=("model_a", "model_b"),
            summary_template="s{subject}/{model}/summary.json",
        )
