from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import report_seed_loso_results as report_script


def _evaluation_summary(
    *,
    static: float,
    online: float,
    update_count: int,
) -> dict[str, object]:
    static_metrics = {
        "accuracy": static,
        "balanced_accuracy": static,
        "macro_f1": static,
    }
    online_metrics = {
        "accuracy": online,
        "balanced_accuracy": online,
        "macro_f1": online,
    }
    gain = online - static
    return {
        "static": {
            "metrics": {"overall": static_metrics},
            "updates": {
                "num_updates": 0,
                "latency": {
                    "mean_ms": None,
                    "p50_ms": None,
                    "p95_ms": None,
                },
            },
        },
        "neuroonline": {
            "metrics": {"overall": online_metrics},
            "updates": {
                "num_updates": update_count,
                "latency": {
                    "mean_ms": 3.0,
                    "p50_ms": 2.5,
                    "p95_ms": 4.5,
                },
            },
        },
        "gains": {
            "overall": {
                "accuracy_gain": gain,
                "balanced_accuracy_gain": gain,
                "macro_f1_gain": gain,
            }
        },
    }


def _write_complete_fixture(root: Path) -> None:
    for subject in report_script.SUBJECTS:
        for model in report_script.MODELS:
            if model == "labram":
                gain = 0.1 if subject <= 5 else (0.0 if subject <= 10 else -0.05)
            else:
                gain = 0.02
            path = root / f"subject_{subject:02d}" / model / "evaluation"
            path.mkdir(parents=True)
            (path / "summary.json").write_text(
                json.dumps(
                    _evaluation_summary(
                        static=0.5,
                        online=0.5 + gain,
                        update_count=subject,
                    )
                ),
                encoding="utf-8",
            )


def test_complete_fifteen_subject_report_and_gain_statistics(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "seed_loso_full"
    output_dir = tmp_path / "report"
    _write_complete_fixture(input_root)

    assert report_script.main(
        ["--input-root", str(input_root), "--output-dir", str(output_dir)]
    ) == 0

    with (output_dir / "seed_loso_subject_results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 30
    assert rows[0]["subject"] == "1"
    assert rows[0]["model"] == "labram"
    assert float(rows[0]["latency_p95_ms"]) == 4.5

    report = json.loads(
        (output_dir / "seed_loso_summary.json").read_text(encoding="utf-8")
    )
    labram = report["model_summaries"]["labram"]["metrics"]
    balanced_gain = labram["balanced_accuracy"]["gain"]
    assert balanced_gain["mean"] == pytest.approx(1.0 / 60.0)
    assert balanced_gain["median"] == pytest.approx(0.0)
    assert balanced_gain["subject_counts"] == {
        "positive": 5,
        "zero": 5,
        "negative": 5,
    }
    cbramod_counts = report["model_summaries"]["cbramod"]["metrics"][
        "balanced_accuracy"
    ]["gain"]["subject_counts"]
    assert cbramod_counts == {"positive": 15, "zero": 0, "negative": 0}

    markdown = (output_dir / "seed_loso_report.md").read_text(encoding="utf-8")
    assert "Primary comparison metric: **balanced accuracy**" in markdown
    assert "LaBraM" not in markdown
    assert "labram" in markdown
    assert "cbramod" in markdown


def test_missing_subject_fails_closed(tmp_path: Path) -> None:
    input_root = tmp_path / "seed_loso_full"
    _write_complete_fixture(input_root)
    missing = input_root / "subject_15"
    for path in sorted(missing.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    missing.rmdir()

    with pytest.raises(ValueError, match="subject_15"):
        report_script.load_subject_results(input_root)
