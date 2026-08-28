from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts import run_seed_order_control as control


def _mode(accuracy: float, balanced: float, macro_f1: float) -> dict:
    return {
        "metrics": {
            "overall": {
                "accuracy": accuracy,
                "balanced_accuracy": balanced,
                "macro_f1": macro_f1,
                "confusion_matrix": [[3, 1], [2, 4]],
            }
        }
    }


def _summary(data_path: Path, package_path: Path) -> dict:
    return {
        "data": {"path": str(data_path)},
        "runtime_package": {"path": str(package_path)},
        "neuroonline_config": {
            "warmup_feedback": 32,
            "update_interval": 16,
        },
        "static": _mode(0.7, 0.71, 0.69),
        "neuroonline": _mode(0.75, 0.76, 0.74),
    }


def _write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_static_invariance_accepts_reordered_aggregate_metrics():
    original = {
        "static": _mode(0.7, 0.71, 0.69),
    }
    shuffled = {
        "static": _mode(0.7, 0.71, 0.69),
    }
    result = control.check_static_invariance(original, shuffled)
    assert result["passed"] is True
    assert result["confusion_matrix_equal"] is True


def test_static_invariance_fails_closed_on_metric_or_confusion_change():
    original = {"static": _mode(0.7, 0.71, 0.69)}
    changed_metric = {"static": _mode(0.7, 0.70, 0.69)}
    with pytest.raises(RuntimeError, match="balanced_accuracy"):
        control.check_static_invariance(original, changed_metric)

    changed_matrix = {"static": _mode(0.7, 0.71, 0.69)}
    changed_matrix["static"]["metrics"]["overall"]["confusion_matrix"] = [
        [4, 0],
        [2, 4],
    ]
    with pytest.raises(RuntimeError, match="confusion_matrix"):
        control.check_static_invariance(original, changed_matrix)


def test_dry_run_plans_mini_matrix_without_subprocess(tmp_path, monkeypatch, capsys):
    input_root = tmp_path / "seed_loso_full"
    output_root = tmp_path / "seed_order_control" / "seed_20260826"
    for subject in (1, 2, 3):
        data_path = tmp_path / "data" / f"subject_{subject:02d}.h5"
        for model in ("labram", "cbramod"):
            summary = (
                input_root
                / f"subject_{subject:02d}"
                / model
                / "evaluation"
                / "summary.json"
            )
            _write_summary(
                summary,
                _summary(
                    data_path,
                    input_root / f"subject_{subject:02d}" / model / "package",
                ),
            )

    calls: list[object] = []
    monkeypatch.setattr(
        control.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    result = control.main(
        [
            "--subjects",
            "1",
            "2",
            "3",
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            "--models",
            "labram",
            "cbramod",
            "--session",
            "S3",
            "--device",
            "cuda",
            "--order-seed",
            "20260826",
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out

    assert result == 0
    assert calls == []
    assert output.count("evaluate_neuroonline_sequential.py") == 6
    assert "--evaluation-order random_permutation" in output
    assert "--online-strategy both" in output
    assert "subject_01" in output
    assert "subject_02" in output
    assert "subject_03" in output
    assert not output_root.exists()


def test_output_root_cannot_overwrite_or_nest_under_frozen_results(tmp_path):
    input_root = tmp_path / "seed_loso_full"
    with pytest.raises(ValueError, match="must not be seed_loso_full"):
        control._safe_output_root(input_root, input_root)
    with pytest.raises(ValueError, match="must not be seed_loso_full"):
        control._safe_output_root(input_root, input_root / "diagnostic")


def test_collect_result_computes_order_dependency_after_strong_gate(tmp_path):
    input_root = tmp_path / "seed_loso_full"
    output_root = tmp_path / "seed_order_control"
    original_path = (
        input_root / "subject_01" / "labram" / "evaluation" / "summary.json"
    )
    original = _summary(tmp_path / "data.h5", tmp_path / "package")
    _write_summary(original_path, original)
    plan = control.build_plan(
        subject=1,
        model="labram",
        input_root=input_root,
        output_root=output_root,
        session="S3",
        device="cpu",
        order_seed=7,
        config=Path("config.yaml"),
        require_artifacts=False,
    )
    shuffled = _summary(tmp_path / "data.h5", tmp_path / "package")
    shuffled["neuroonline"] = _mode(0.8, 0.81, 0.79)
    shuffled["order_control"] = {
        "evaluation_order": "random_permutation",
        "order_seed": 7,
    }
    _write_summary(plan.output_dir / "summary.json", shuffled)

    result = control.collect_result(plan)
    metrics = result["balanced_accuracy"]
    assert metrics["gain_original"] == pytest.approx(0.05)
    assert metrics["gain_shuffled"] == pytest.approx(0.10)
    assert metrics["order_dependency"] == pytest.approx(-0.05)
