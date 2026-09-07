"""Run SEED S2 NeuroOnline trigger/memory ablations without retraining."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from _bootstrap import ROOT

from bci_dayloop.inference.neuroonline_strategy import NeuroOnlineConfig


DEFAULT_SUBJECTS = (1, 2, 3)
VALID_MODELS = ("labram", "cbramod")
ARM_CONFIGS = {
    "current": {},
    "coverage_only": {
        "update_trigger": "class_coverage",
        "min_feedback_per_class": 8,
    },
    "balanced_only": {
        "memory_strategy": "class_balanced_history",
        "balanced_memory_per_class": 32,
    },
    "combined": {
        "update_trigger": "class_coverage",
        "min_feedback_per_class": 8,
        "memory_strategy": "class_balanced_history",
        "balanced_memory_per_class": 32,
    },
}
PROTECTED_EXPERIMENTS = (
    "seed_loso_full",
    "seed_order_control",
    "seed_update_scope_ablation",
)


@dataclass(frozen=True, slots=True)
class BufferAblationPlan:
    subject: int
    model: str
    arm: str
    frozen_summary: Path
    output_dir: Path
    config: NeuroOnlineConfig
    argv: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run persisted-order SEED S2 trigger/memory diagnostics."
    )
    parser.add_argument("--subjects", nargs="+", type=int, default=DEFAULT_SUBJECTS)
    parser.add_argument(
        "--input-root", type=Path, default=Path("experiments/seed_loso_full")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/seed_buffer_ablation_s2"),
    )
    parser.add_argument("--session", choices=("S2",), default="S2")
    parser.add_argument(
        "--models", nargs="+", choices=VALID_MODELS, default=VALID_MODELS
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/stage0/day1_bnci_s01.yaml")
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def normalize_subjects(subjects: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(sorted(set(int(subject) for subject in subjects)))
    if not normalized or any(subject <= 0 for subject in normalized):
        raise ValueError("subjects must contain positive integers.")
    return normalized


def _is_same_or_child(path: Path, parent: Path) -> bool:
    resolved = path.resolve()
    protected = parent.resolve()
    return resolved == protected or protected in resolved.parents


def validate_output_root(input_root: Path, output_root: Path) -> None:
    if _is_same_or_child(output_root, input_root):
        raise ValueError("Output must not overwrite frozen seed_loso_full.")
    for name in PROTECTED_EXPERIMENTS[1:]:
        protected = ROOT / "experiments" / name
        if _is_same_or_child(output_root, protected):
            raise ValueError(f"Output must not overwrite {name}.")


def normalized_frozen_config(value: object) -> dict[str, Any]:
    payload = dict(_mapping(value, "neuroonline_config"))
    defaults = asdict(NeuroOnlineConfig())
    for key in (
        "update_scope",
        "update_trigger",
        "min_feedback_per_class",
        "memory_strategy",
        "balanced_memory_per_class",
    ):
        payload.setdefault(key, defaults[key])
    return payload


def validate_frozen_config(summary: Mapping[str, Any]) -> None:
    actual = normalized_frozen_config(summary.get("neuroonline_config"))
    expected = asdict(NeuroOnlineConfig())
    if actual != expected:
        raise RuntimeError(
            "Frozen NeuroOnline config differs from the V1 baseline: "
            f"expected={expected}, actual={actual}."
        )


def _declared_path(summary: Mapping[str, Any], section: str) -> Path:
    payload = _mapping(summary.get(section), section)
    value = payload.get("path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Frozen summary is missing {section}.path.")
    return Path(value)


def arm_config(arm: str) -> NeuroOnlineConfig:
    if arm not in ARM_CONFIGS:
        raise ValueError(f"Unknown buffer ablation arm: {arm!r}.")
    return replace(NeuroOnlineConfig(), **ARM_CONFIGS[arm])


def build_plan(
    *,
    subject: int,
    model: str,
    arm: str,
    input_root: Path,
    output_root: Path,
    session: str,
    device: str,
    evaluator_config: Path,
    require_artifacts: bool,
) -> BufferAblationPlan:
    if session != "S2":
        raise ValueError("SEED buffer ablation is validation-only and requires S2.")
    if model not in VALID_MODELS:
        raise ValueError(f"Unsupported model: {model!r}.")
    config = arm_config(arm)
    subject_dir = input_root / f"subject_{subject:02d}" / model
    frozen_summary = subject_dir / "evaluation" / "summary.json"
    frozen = _load_mapping(frozen_summary)
    validate_frozen_config(frozen)
    data_path = _declared_path(frozen, "data")
    preferred_package = subject_dir / "package"
    declared_package = _declared_path(frozen, "runtime_package")
    package_path = (
        preferred_package
        if preferred_package.exists() or not require_artifacts
        else declared_package
    )
    if require_artifacts:
        if not data_path.is_file():
            raise FileNotFoundError(f"SEED source HDF5 is missing: {data_path}")
        if not package_path.is_dir():
            raise FileNotFoundError(f"Runtime Package is missing: {package_path}")

    output_dir = output_root / f"subject_{subject:02d}" / model / arm
    argv = (
        sys.executable,
        str(ROOT / "scripts" / "evaluate_neuroonline_sequential.py"),
        "--config", str(evaluator_config),
        "--data", str(data_path),
        "--model-package", str(package_path),
        "--session", "S2",
        "--device", device,
        "--online-strategy", "both",
        "--evaluation-order", "persisted",
        "--update-scope", "generator_and_head",
        "--update-trigger", config.update_trigger,
        "--min-feedback-per-class", str(config.min_feedback_per_class),
        "--memory-strategy", config.memory_strategy,
        "--balanced-memory-per-class", str(config.balanced_memory_per_class),
        "--output-dir", str(output_dir),
    )
    return BufferAblationPlan(
        subject=subject,
        model=model,
        arm=arm,
        frozen_summary=frozen_summary,
        output_dir=output_dir,
        config=config,
        argv=argv,
    )


def build_plans(args: argparse.Namespace) -> tuple[BufferAblationPlan, ...]:
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    validate_output_root(input_root, output_root)
    return tuple(
        build_plan(
            subject=subject,
            model=model,
            arm=arm,
            input_root=input_root,
            output_root=output_root,
            session=str(args.session),
            device=str(args.device),
            evaluator_config=Path(args.config),
            require_artifacts=not bool(args.dry_run),
        )
        for subject in normalize_subjects(args.subjects)
        for model in args.models
        for arm in ARM_CONFIGS
    )


def _mode(summary: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return _mapping(summary.get(name), name)


def _overall(summary: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    metrics = _mapping(_mode(summary, name).get("metrics"), f"{name}.metrics")
    return _mapping(metrics.get("overall"), f"{name}.metrics.overall")


def collect_arm(plan: BufferAblationPlan) -> dict[str, Any]:
    summary = _load_mapping(plan.output_dir / "summary.json")
    if summary.get("online_strategy") != "both":
        raise RuntimeError("Ablation run must contain paired static and online modes.")
    protocol = _mapping(summary.get("protocol"), "protocol")
    if protocol.get("evaluation_order", "persisted") != "persisted":
        raise RuntimeError("Ablation run must use persisted evaluation order.")
    if _mapping(summary.get("data"), "data").get("session") != "S2":
        raise RuntimeError("Ablation result is not from validation session S2.")
    actual_config = dict(_mapping(summary.get("neuroonline_config"), "config"))
    if actual_config != asdict(plan.config):
        raise RuntimeError(f"Ablation config mismatch for arm {plan.arm!r}.")
    identity = _mapping(
        summary.get("identity_initialization_check"), "identity_initialization_check"
    )
    if identity.get("equivalent") is not True:
        raise RuntimeError(f"Identity initialization failed for arm {plan.arm!r}.")

    static = _overall(summary, "static")
    online_mode = _mode(summary, "neuroonline")
    online = _overall(summary, "neuroonline")
    updates = _mapping(online_mode.get("updates"), "neuroonline.updates")
    update_rows = updates.get("updates")
    if not isinstance(update_rows, list):
        raise ValueError("neuroonline.updates.updates must be a list.")
    metrics = [
        _mapping(_mapping(row, "update").get("metrics"), "update.metrics")
        for row in update_rows
    ]
    gradient_norms = [float(item["last_gradient_norm"]) for item in metrics]
    clipping_count = sum(int(item["gradient_clipping_count"]) for item in metrics)
    batch_count = sum(int(item["batches"]) for item in metrics)
    imbalances = []
    for item in metrics:
        histogram = _mapping(item["memory_label_histogram"], "memory histogram")
        counts = [int(value) for value in histogram.values()]
        imbalances.append(max(counts) - min(counts))
    first = metrics[0] if metrics else None
    return {
        "subject": plan.subject,
        "model": plan.model,
        "arm": plan.arm,
        "static_balanced_accuracy": float(static["balanced_accuracy"]),
        "online_balanced_accuracy": float(online["balanced_accuracy"]),
        "balanced_accuracy_gain": (
            float(online["balanced_accuracy"]) - float(static["balanced_accuracy"])
        ),
        "accuracy": float(online["accuracy"]),
        "macro_f1": float(online["macro_f1"]),
        "number_of_updates": int(updates["num_updates"]),
        "first_update_ordinal": (
            None if first is None else first["first_update_evaluation_ordinal"]
        ),
        "first_update_label_histogram": (
            None if first is None else first["first_update_label_histogram"]
        ),
        "mean_buffer_class_imbalance": (
            None if not imbalances else sum(imbalances) / len(imbalances)
        ),
        "gradient_norm_mean": (
            None if not gradient_norms else sum(gradient_norms) / len(gradient_norms)
        ),
        "gradient_norm_max": None if not gradient_norms else max(gradient_norms),
        "clipping_count": clipping_count,
        "clipping_rate": 0.0 if batch_count == 0 else clipping_count / batch_count,
        "update_latency": dict(_mapping(updates.get("latency"), "update latency")),
    }


def add_current_comparisons(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["subject"]), str(row["model"])), {})[
            str(row["arm"])
        ] = row
    comparisons = []
    for (subject, model), arms in grouped.items():
        if set(arms) != set(ARM_CONFIGS):
            raise ValueError("Each subject/model must contain all four ablation arms.")
        static_values = {
            float(row["static_balanced_accuracy"])
            for row in arms.values()
        }
        if len(static_values) != 1:
            raise RuntimeError("Static balanced accuracy changed across ablation arms.")
        current = float(arms["current"]["online_balanced_accuracy"])
        arms["current"]["vs_current"] = 0.0
        for arm in ("coverage_only", "balanced_only", "combined"):
            arms[arm]["vs_current"] = (
                float(arms[arm]["online_balanced_accuracy"]) - current
            )
        comparisons.append(
            {
                "subject": subject,
                "model": model,
                "coverage_vs_current": arms["coverage_only"]["vs_current"],
                "balanced_vs_current": arms["balanced_only"]["vs_current"],
                "combined_vs_current": arms["combined"]["vs_current"],
            }
        )
    return comparisons


def write_results(output_root: Path, rows: Sequence[dict[str, Any]]) -> None:
    comparisons = add_current_comparisons(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "diagnostic_only": True,
        "development_session": "S2",
        "evaluation_order": "persisted",
        "results": list(rows),
        "comparisons_to_current": comparisons,
    }
    (output_root / "buffer_ablation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fieldnames = (
        "subject", "model", "arm", "static_balanced_accuracy",
        "online_balanced_accuracy", "balanced_accuracy_gain", "vs_current",
        "accuracy", "macro_f1", "number_of_updates", "first_update_ordinal",
        "first_update_label_histogram", "mean_buffer_class_imbalance",
        "gradient_norm_mean", "gradient_norm_max", "clipping_count",
        "clipping_rate",
    )
    with (output_root / "buffer_ablation_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["first_update_label_histogram"] = json.dumps(
                row["first_update_label_histogram"], sort_keys=True
            )
            writer.writerow(serialized)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plans = build_plans(args)
    for plan in plans:
        print(shlex.join(plan.argv))
    if args.dry_run:
        return 0
    for plan in plans:
        subprocess.run(plan.argv, cwd=ROOT, check=True)
    rows = [collect_arm(plan) for plan in plans]
    write_results(Path(args.output_root), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
