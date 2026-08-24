"""Plan and run the full SEED LOSO experiment with existing CLIs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _bootstrap import ROOT


DEFAULT_SUBJECTS = tuple(range(1, 16))
MODELS = ("labram", "cbramod")


@dataclass(frozen=True, slots=True)
class CommandSpec:
    name: str
    argv: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SubjectPlan:
    target_subject: int
    population_subjects: tuple[int, ...]
    output_dir: Path
    commands: tuple[CommandSpec, ...]


def normalize_subjects(subjects: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(sorted(set(int(subject) for subject in subjects)))
    if not normalized:
        raise ValueError("subjects cannot be empty.")
    if any(subject <= 0 for subject in normalized):
        raise ValueError("subjects must contain positive integers.")
    return normalized


def population_subjects_for(
    subjects: Sequence[int], target_subject: int
) -> tuple[int, ...]:
    normalized = normalize_subjects(subjects)
    target = int(target_subject)
    if target not in normalized:
        raise ValueError("target_subject must occur in subjects.")
    population = tuple(subject for subject in normalized if subject != target)
    if not population:
        raise ValueError("LOSO requires at least one population subject.")
    return population


def _subject_data_path(
    *, data_root: Path, data_pattern: str, subject: int
) -> Path:
    try:
        filename = data_pattern.format(subject=int(subject))
    except (KeyError, ValueError) as error:
        raise ValueError(
            "data_pattern must contain a valid {subject} format field."
        ) from error
    return data_root / filename


def build_subject_plan(
    *,
    subjects: Sequence[int],
    target_subject: int,
    data_root: Path,
    data_pattern: str,
    output_root: Path,
    labram_checkpoint: Path,
    cbramod_checkpoint: Path,
    train_session: str,
    validation_session: str,
    final_test_session: str,
    window_sec: float,
    device: str,
) -> SubjectPlan:
    normalized = normalize_subjects(subjects)
    target = int(target_subject)
    population = population_subjects_for(normalized, target)
    if window_sec <= 0:
        raise ValueError("window_sec must be positive.")

    subject_tag = f"subject_{target:02d}"
    subject_dir = output_root / subject_tag
    data_path = _subject_data_path(
        data_root=data_root,
        data_pattern=data_pattern,
        subject=target,
    )
    common_train = (
        "--data-root",
        str(data_root),
        "--data-pattern",
        data_pattern,
        "--subjects",
        *(str(subject) for subject in normalized),
        "--target-subject",
        str(target),
        "--train-session",
        train_session,
        "--validation-session",
        validation_session,
        "--final-test-session",
        final_test_session,
    )

    labram_dir = subject_dir / "labram"
    labram_head = labram_dir / "head.pt"
    labram_package = labram_dir / "package"
    labram_evaluation = labram_dir / "evaluation"
    cbramod_dir = subject_dir / "cbramod"
    cbramod_head = cbramod_dir / "head.pt"
    cbramod_training = cbramod_dir / "training"
    cbramod_package = cbramod_dir / "package"
    cbramod_evaluation = cbramod_dir / "evaluation"

    commands = (
        CommandSpec(
            "train_labram",
            (
                sys.executable,
                str(ROOT / "scripts" / "train_labram_population_head.py"),
                *common_train,
                "--checkpoint",
                str(labram_checkpoint),
                "--output",
                str(labram_head),
                "--run-dir",
                str(labram_dir / "training"),
                "--device",
                device,
                "--window-sec",
                f"{window_sec:g}",
            ),
        ),
        CommandSpec(
            "export_labram",
            (
                sys.executable,
                str(ROOT / "scripts" / "export_labram_model_package.py"),
                "--data",
                str(data_path),
                "--checkpoint",
                str(labram_checkpoint),
                "--classifier",
                str(labram_head),
                "--output",
                str(labram_package),
                "--device",
                device,
                "--session",
                final_test_session,
            ),
        ),
        CommandSpec(
            "evaluate_labram",
            (
                sys.executable,
                str(ROOT / "scripts" / "evaluate_neuroonline_sequential.py"),
                "--data",
                str(data_path),
                "--model-package",
                str(labram_package),
                "--session",
                final_test_session,
                "--device",
                device,
                "--online-strategy",
                "both",
                "--output-dir",
                str(labram_evaluation),
            ),
        ),
        CommandSpec(
            "train_cbramod",
            (
                sys.executable,
                str(ROOT / "scripts" / "train_cbramod_population_head.py"),
                *common_train,
                "--checkpoint",
                str(cbramod_checkpoint),
                "--output-head",
                str(cbramod_head),
                "--run-dir",
                str(cbramod_training),
                "--device",
                device,
                "--window-sec",
                f"{window_sec:g}",
            ),
        ),
        CommandSpec(
            "export_cbramod",
            (
                sys.executable,
                str(ROOT / "scripts" / "export_cbramod_model_package.py"),
                "--data",
                str(data_path),
                "--checkpoint",
                str(cbramod_checkpoint),
                "--classifier",
                str(cbramod_head),
                "--training-report",
                str(cbramod_training / "training_report.json"),
                "--output",
                str(cbramod_package),
                "--device",
                device,
                "--session",
                final_test_session,
            ),
        ),
        CommandSpec(
            "evaluate_cbramod",
            (
                sys.executable,
                str(ROOT / "scripts" / "evaluate_neuroonline_sequential.py"),
                "--data",
                str(data_path),
                "--model-package",
                str(cbramod_package),
                "--session",
                final_test_session,
                "--device",
                device,
                "--online-strategy",
                "both",
                "--output-dir",
                str(cbramod_evaluation),
            ),
        ),
    )
    return SubjectPlan(target, population, subject_dir, commands)


def build_plans(args: argparse.Namespace) -> tuple[SubjectPlan, ...]:
    subjects = normalize_subjects(args.subjects)
    return tuple(
        build_subject_plan(
            subjects=subjects,
            target_subject=target,
            data_root=args.data_root,
            data_pattern=args.data_pattern,
            output_root=args.output_root,
            labram_checkpoint=args.labram_checkpoint,
            cbramod_checkpoint=args.cbramod_checkpoint,
            train_session=args.train_session,
            validation_session=args.validation_session,
            final_test_session=args.final_test_session,
            window_sec=args.window_sec,
            device=args.device,
        )
        for target in subjects
    )


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def _prediction_latency(path: Path, mode: str) -> dict[str, float | None]:
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("mode") == mode:
                values.append(float(row["preprocess_predict_latency_ms"]))
    return {
        "mean_ms": sum(values) / len(values) if values else None,
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
    }


def _required_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Evaluation field {name} must be an object.")
    return value


def collect_result(*, subject: int, model: str, evaluation_dir: Path) -> dict[str, Any]:
    summary_path = evaluation_dir / "summary.json"
    records_path = evaluation_dir / "trial_predictions.csv"
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary, Mapping):
        raise ValueError(f"Evaluation summary must be an object: {summary_path}")

    modes: dict[str, Any] = {}
    for key, output_name in (("static", "static"), ("neuroonline", "neuroonline")):
        mode = _required_mapping(summary.get(key), key)
        metrics = _required_mapping(mode.get("metrics"), f"{key}.metrics")
        overall = _required_mapping(metrics.get("overall"), f"{key}.metrics.overall")
        updates = _required_mapping(mode.get("updates"), f"{key}.updates")
        modes[output_name] = {
            "accuracy": float(overall["accuracy"]),
            "balanced_accuracy": float(overall["balanced_accuracy"]),
            "macro_f1": float(overall["macro_f1"]),
            "update_count": int(updates["num_updates"]),
            "prediction_latency": _prediction_latency(records_path, key if key != "static" else "none"),
            "update_latency": dict(
                _required_mapping(updates.get("latency"), f"{key}.updates.latency")
            ),
        }
    gains = _required_mapping(summary.get("gains"), "gains")
    return {
        "subject": subject,
        "model": model,
        **modes,
        "gain": dict(_required_mapping(gains.get("overall"), "gains.overall")),
    }


def write_summary(plans: Sequence[SubjectPlan], output_root: Path) -> Path:
    rows = [
        collect_result(
            subject=plan.target_subject,
            model=model,
            evaluation_dir=plan.output_dir / model / "evaluation",
        )
        for plan in plans
        for model in MODELS
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "loso_summary.json"
    path.write_text(
        json.dumps({"dataset": "seed", "results": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/seed"))
    parser.add_argument("--data-pattern", default="subject_{subject:02d}.h5")
    parser.add_argument("--subjects", nargs="+", type=int, default=list(DEFAULT_SUBJECTS))
    parser.add_argument("--train-session", default="S1")
    parser.add_argument("--validation-session", default="S2")
    parser.add_argument("--final-test-session", default="S3")
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--labram-checkpoint", type=Path, required=True)
    parser.add_argument("--cbramod-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("experiments/seed_loso")
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plans = build_plans(args)
    for plan in plans:
        print(
            f"# subject_{plan.target_subject:02d} population="
            + ",".join(str(subject) for subject in plan.population_subjects)
        )
        for command in plan.commands:
            print(f"[{command.name}] {subprocess.list2cmdline(command.argv)}")
            if not args.dry_run:
                subprocess.run(command.argv, cwd=ROOT, check=True)

    if args.dry_run:
        print(f"Dry-run complete: {len(plans)} subjects, no commands executed.")
        return 0

    summary_path = write_summary(plans, args.output_root)
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
