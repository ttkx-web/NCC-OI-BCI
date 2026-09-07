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
from scripts import run_seed_buffer_ablation as ablation


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def frozen_summary(data: Path, package: Path) -> dict:
    config = asdict(NeuroOnlineConfig())
    for key in (
        "update_scope",
        "update_trigger",
        "min_feedback_per_class",
        "memory_strategy",
        "balanced_memory_per_class",
    ):
        config.pop(key)
    return {
        "data": {"path": str(data)},
        "runtime_package": {"path": str(package)},
        "neuroonline_config": config,
    }


def test_arm_configs_change_only_trigger_or_memory() -> None:
    baseline = asdict(NeuroOnlineConfig())
    assert asdict(ablation.arm_config("current")) == baseline
    coverage = ablation.arm_config("coverage_only")
    assert coverage.update_trigger == "class_coverage"
    assert coverage.min_feedback_per_class == 8
    assert coverage.memory_strategy == "recent_fifo"
    balanced = ablation.arm_config("balanced_only")
    assert balanced.update_trigger == "feedback_count"
    assert balanced.memory_strategy == "class_balanced_history"
    assert balanced.balanced_memory_per_class == 32
    combined = ablation.arm_config("combined")
    assert combined.update_trigger == "class_coverage"
    assert combined.memory_strategy == "class_balanced_history"
    for config in (coverage, balanced, combined):
        assert config.update_scope == "generator_and_head"
        assert config.learning_rate == baseline["learning_rate"]
        assert config.update_interval == baseline["update_interval"]


def test_dry_run_is_s2_only_and_does_not_execute_or_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_root = tmp_path / "seed_loso_full"
    output_root = tmp_path / "seed_buffer_ablation_s2"
    for subject in (1, 2, 3):
        for model in ("labram", "cbramod"):
            summary_path = (
                input_root
                / f"subject_{subject:02d}"
                / model
                / "evaluation"
                / "summary.json"
            )
            write_json(
                summary_path,
                frozen_summary(
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
            "--session", "S2",
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out
    assert result == 0
    assert calls == []
    assert output.count("evaluate_neuroonline_sequential.py") == 24
    assert output.count("--session S2") == 24
    assert "--session S3" not in output
    assert output.count("--update-scope generator_and_head") == 24
    assert output.count("--memory-strategy class_balanced_history") == 12
    assert output.count("--update-trigger class_coverage") == 12
    assert not output_root.exists()


def test_s3_is_rejected_by_parser_and_plan(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        ablation.build_parser().parse_args(["--session", "S3"])
    with pytest.raises(ValueError, match="requires S2"):
        ablation.build_plan(
            subject=1,
            model="labram",
            arm="current",
            input_root=tmp_path,
            output_root=tmp_path / "out",
            session="S3",
            device="cpu",
            evaluator_config=Path("config.yaml"),
            require_artifacts=False,
        )


def test_output_root_cannot_overwrite_any_frozen_diagnostic(tmp_path: Path) -> None:
    input_root = tmp_path / "seed_loso_full"
    with pytest.raises(ValueError, match="seed_loso_full"):
        ablation.validate_output_root(input_root, input_root / "bad")
    for name in ("seed_order_control", "seed_update_scope_ablation"):
        protected = ROOT / "experiments" / name
        with pytest.raises(ValueError, match=name):
            ablation.validate_output_root(input_root, protected / "bad")


def test_collect_arm_reads_first_update_and_buffer_telemetry(tmp_path: Path) -> None:
    config = ablation.arm_config("combined")
    output_dir = tmp_path / "result"
    plan = ablation.BufferAblationPlan(
        subject=1,
        model="labram",
        arm="combined",
        frozen_summary=tmp_path / "frozen.json",
        output_dir=output_dir,
        config=config,
        argv=(),
    )
    update_metrics = {
        "last_gradient_norm": 1.5,
        "gradient_clipping_count": 1,
        "batches": 2,
        "memory_label_histogram": {"0": 8, "1": 8, "2": 8},
        "first_update_evaluation_ordinal": 200,
        "first_update_label_histogram": {"0": 8, "1": 8, "2": 8},
    }
    write_json(
        output_dir / "summary.json",
        {
            "online_strategy": "both",
            "protocol": {"evaluation_order": "persisted"},
            "data": {"session": "S2"},
            "neuroonline_config": asdict(config),
            "identity_initialization_check": {"equivalent": True},
            "static": {
                "metrics": {
                    "overall": {
                        "accuracy": 0.5,
                        "balanced_accuracy": 0.5,
                        "macro_f1": 0.5,
                    }
                }
            },
            "neuroonline": {
                "metrics": {
                    "overall": {
                        "accuracy": 0.6,
                        "balanced_accuracy": 0.62,
                        "macro_f1": 0.58,
                    }
                },
                "updates": {
                    "num_updates": 1,
                    "latency": {"mean_ms": 3.0, "p50_ms": 3.0, "p95_ms": 3.0},
                    "updates": [{"metrics": update_metrics}],
                },
            },
        },
    )
    result = ablation.collect_arm(plan)
    assert result["balanced_accuracy_gain"] == pytest.approx(0.12)
    assert result["first_update_ordinal"] == 200
    assert result["first_update_label_histogram"] == {"0": 8, "1": 8, "2": 8}
    assert result["mean_buffer_class_imbalance"] == 0.0
    assert result["clipping_count"] == 1
    assert result["clipping_rate"] == 0.5


def test_comparisons_to_current_are_named_and_static_must_match() -> None:
    rows = [
        {
            "subject": 1,
            "model": "labram",
            "arm": arm,
            "static_balanced_accuracy": 0.5,
            "online_balanced_accuracy": score,
        }
        for arm, score in (
            ("current", 0.4),
            ("coverage_only", 0.45),
            ("balanced_only", 0.47),
            ("combined", 0.52),
        )
    ]
    comparisons = ablation.add_current_comparisons(rows)
    assert comparisons == [
        {
            "subject": 1,
            "model": "labram",
            "coverage_vs_current": pytest.approx(0.05),
            "balanced_vs_current": pytest.approx(0.07),
            "combined_vs_current": pytest.approx(0.12),
        }
    ]
    rows[-1]["static_balanced_accuracy"] = 0.51
    with pytest.raises(RuntimeError, match="Static balanced accuracy"):
        ablation.add_current_comparisons(rows)
