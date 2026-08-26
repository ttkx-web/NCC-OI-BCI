"""Plan and run the full Workload LOSO experiment with existing CLIs.

The entry point deliberately only orchestrates the established population-head,
package-export and sequential-evaluation commands.  Each held-out subject gets
its own directory below ``--output-root`` so model artifacts cannot be reused
between LOSO folds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from _bootstrap import ROOT
from run_seed_loso_experiment import (
    MODELS,
    CommandSpec,
    SubjectPlan,
    collect_result,
    normalize_subjects,
    population_subjects_for,
)


DEFAULT_SUBJECTS = tuple(range(1, 16))


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
    """Build one isolated Workload LOSO fold with separated sessions."""
    normalized = normalize_subjects(subjects)
    target = int(target_subject)
    population = population_subjects_for(normalized, target)
    if window_sec <= 0:
        raise ValueError("window_sec must be positive.")

    subject_dir = output_root / f"subject_{target:02d}"
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
    cbramod_dir = subject_dir / "cbramod"
    cbramod_head = cbramod_dir / "head.pt"
    cbramod_training = cbramod_dir / "training"
    cbramod_package = cbramod_dir / "package"

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
                str(labram_dir / "evaluation"),
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
                str(cbramod_dir / "evaluation"),
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
        json.dumps({"dataset": "workload", "results": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/processed/workload")
    )
    parser.add_argument("--data-pattern", default="subject_{subject:02d}.h5")
    parser.add_argument(
        "--subjects", nargs="+", type=int, default=list(DEFAULT_SUBJECTS)
    )
    parser.add_argument("--train-session", default="S1")
    parser.add_argument("--validation-session", default="S2")
    parser.add_argument("--final-test-session", default="S2")
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--labram-checkpoint", type=Path, required=True)
    parser.add_argument("--cbramod-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("experiments/workload_loso_full"),
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
        print(
            "  "
            f"train_session={args.train_session} "
            f"validation_session={args.validation_session} "
            f"final_test_session={args.final_test_session}"
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
