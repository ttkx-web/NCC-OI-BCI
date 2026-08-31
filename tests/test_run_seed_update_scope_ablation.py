from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from bci_dayloop.inference.neuroonline_strategy import NeuroOnlineConfig
from scripts import run_seed_update_scope_ablation as ablation


def _mode(balanced_accuracy: float, *, updates: int = 0) -> dict:
    update_rows = [
        {
            "trial_ordinal": 48,
            "update_step_after_prediction": 1,
            "model_revision_after_prediction": "neuroonline-1",
            "samples_used": 32,
            "latency_ms": 2.0,
            "reason": None,
            "metrics": {"last_gradient_norm": 1.5},
        }
        for _ in range(updates)
    ]
    return {
        "metrics": {
            "overall": {
                "accuracy": balanced_accuracy - 0.01,
                "balanced_accuracy": balanced_accuracy,
                "macro_f1": balanced_accuracy - 0.02,
            }
        },
        "updates": {
            "num_updates": updates,
            "latency": {"mean_ms": 2.0, "p50_ms": 2.0, "p95_ms": 2.0},
            "updates": update_rows,
        },
    }


def _frozen_summary(data: Path, package: Path) -> dict:
    config = asdict(NeuroOnlineConfig())
    config.pop("update_scope")  # Frozen Stage3 summaries predate the scope field.
    return {
        "data": {"path": str(data)},
        "runtime_package": {"path": str(package)},
        "neuroonline_config": config,
        "static": _mode(0.48),
        "neuroonline": _mode(0.40, updates=2),
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _scope_summary(scope: str, balanced_accuracy: float) -> dict:
    config = asdict(NeuroOnlineConfig(update_scope=scope))
    mode = _mode(balanced_accuracy, updates=1)
    mode["parameter_audit"] = {
        "update_scope": scope,
        "backbone_trainable_param_count": 0,
        "generator_trainable_param_count": 10 if scope == "generator_only" else 0,
        "head_trainable_param_count": 5 if scope == "head_only" else 0,
        "optimizer_param_count": 10 if scope == "generator_only" else 5,
    }
    return {
        "neuroonline_config": config,
        "neuroonline": mode,
        "identity_initialization_check": {
            "prediction_agreement_rate": 1.0,
            "maximum_probability_absolute_difference": 0.0,
            "equivalent": True,
            "warning": None,
        },
    }


def test_dry_run_plans_two_scopes_without_subprocess_or_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_root = tmp_path / "seed_loso_full"
    output_root = tmp_path / "seed_update_scope_ablation"
    for subject in (1, 2, 3):
        for model in ("labram", "cbramod"):
            _write(
                input_root
                / f"subject_{subject:02d}"
                / model
                / "evaluation"
                / "summary.json",
                _frozen_summary(
                    tmp_path / "data" / f"subject_{subject:02d}.h5",
                    input_root / f"subject_{subject:02d}" / model / "package",
                ),
            )
    calls: list[object] = []
    monkeypatch.setattr(
        ablation.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = ablation.main(
        [
            "--subjects", "1", "2", "3",
            "--input-root", str(input_root),
            "--output-root", str(output_root),
            "--models", "labram", "cbramod",
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out
    assert result == 0
    assert calls == []
    assert output.count("evaluate_neuroonline_sequential.py") == 12
    assert output.count("--update-scope generator_only") == 6
    assert output.count("--update-scope head_only") == 6
    assert "--evaluation-order persisted" in output
    assert "--online-strategy both" in output
    assert not output_root.exists()


def test_frozen_a1_config_is_checked_and_legacy_scope_is_current_default() -> None:
    summary = _frozen_summary(Path("data.h5"), Path("package"))
    validated = ablation.validate_frozen_current_config(summary)
    assert validated == asdict(NeuroOnlineConfig())

    summary["neuroonline_config"]["learning_rate"] = 2e-4
    with pytest.raises(RuntimeError, match="Frozen A1"):
        ablation.validate_frozen_current_config(summary)


def test_output_root_cannot_overwrite_frozen_or_order_control(tmp_path: Path) -> None:
    input_root = tmp_path / "seed_loso_full"
    with pytest.raises(ValueError, match="seed_loso_full"):
        ablation.validate_output_root(input_root, input_root / "diagnostic")

    order_control = ROOT / "experiments" / "seed_order_control"
    with pytest.raises(ValueError, match="seed_order_control"):
        ablation.validate_output_root(input_root, order_control / "bad")


def test_collect_result_reports_scope_gains_and_gradient_clipping(tmp_path: Path) -> None:
    input_root = tmp_path / "seed_loso_full"
    output_root = tmp_path / "seed_update_scope_ablation"
    frozen_path = input_root / "subject_01" / "labram" / "evaluation" / "summary.json"
    _write(frozen_path, _frozen_summary(tmp_path / "data.h5", tmp_path / "package"))
    plans = [
        ablation.build_plan(
            subject=1,
            model="labram",
            update_scope=scope,
            input_root=input_root,
            output_root=output_root,
            session="S3",
            device="cpu",
            config=Path("config.yaml"),
            require_artifacts=False,
        )
        for scope in ablation.ABLATION_SCOPES
    ]
    _write(plans[0].output_dir / "summary.json", _scope_summary("generator_only", 0.46))
    _write(plans[1].output_dir / "summary.json", _scope_summary("head_only", 0.30))

    result = ablation.collect_subject_model(plans)
    comparison = result["balanced_accuracy_comparison"]
    assert comparison["current_gain"] == pytest.approx(-0.08)
    assert comparison["generator_only_gain"] == pytest.approx(-0.02)
    assert comparison["head_only_gain"] == pytest.approx(-0.18)
    assert comparison["generator_only_vs_current"] == pytest.approx(0.06)
    assert result["generator_only"]["gradient_norm"]["clipping_count"] == 1
    assert result["generator_only"]["gradient_norm"]["clipping_rate"] == 1.0
