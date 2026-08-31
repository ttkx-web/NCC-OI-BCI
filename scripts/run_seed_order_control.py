"""Run fixed-seed SEED evaluation-order diagnostics without retraining."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _bootstrap import ROOT


DEFAULT_SUBJECTS = (1, 2, 3)
VALID_MODELS = ("labram", "cbramod")
DEFAULT_ORDER_SEED = 20260826
STATIC_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class OrderControlPlan:
    subject: int
    model: str
    original_summary: Path
    data_path: Path
    package_path: Path
    output_dir: Path
    argv: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-seed random-order SEED static and NeuroOnline "
            "evaluations using existing LOSO packages."
        )
    )
    parser.add_argument("--subjects", nargs="+", type=int, default=DEFAULT_SUBJECTS)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("experiments/seed_loso_full"),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--session", default="S3")
    parser.add_argument(
        "--models", nargs="+", choices=VALID_MODELS, default=VALID_MODELS
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--order-seed", type=int, default=DEFAULT_ORDER_SEED)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage0/day1_bnci_s01.yaml"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def normalize_subjects(subjects: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(sorted(set(int(subject) for subject in subjects)))
    if not normalized or any(subject <= 0 for subject in normalized):
        raise ValueError("subjects must contain positive integers.")
    return normalized


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


def _safe_output_root(input_root: Path, output_root: Path) -> None:
    source = input_root.resolve()
    target = output_root.resolve()
    if target == source or source in target.parents:
        raise ValueError(
            "Order-control output_root must not be seed_loso_full or a child "
            "of the frozen input_root."
        )


def _declared_path(summary: Mapping[str, Any], section: str) -> Path:
    payload = _mapping(summary.get(section), section)
    value = payload.get("path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Frozen summary is missing {section}.path.")
    return Path(value)


def build_plan(
    *,
    subject: int,
    model: str,
    input_root: Path,
    output_root: Path,
    session: str,
    device: str,
    order_seed: int,
    config: Path,
    require_artifacts: bool,
) -> OrderControlPlan:
    if model not in VALID_MODELS:
        raise ValueError(f"Unsupported model: {model!r}.")
    subject_dir = input_root / f"subject_{int(subject):02d}" / model
    original_summary = subject_dir / "evaluation" / "summary.json"
    frozen = _load_mapping(original_summary)
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

    output_dir = output_root / f"subject_{int(subject):02d}" / model / "evaluation"
    argv = (
        sys.executable,
        str(ROOT / "scripts" / "evaluate_neuroonline_sequential.py"),
        "--config",
        str(config),
        "--data",
        str(data_path),
        "--model-package",
        str(package_path),
        "--session",
        str(session),
        "--device",
        str(device),
        "--online-strategy",
        "both",
        "--evaluation-order",
        "random_permutation",
        "--order-seed",
        str(int(order_seed)),
        "--output-dir",
        str(output_dir),
    )
    return OrderControlPlan(
        subject=int(subject),
        model=model,
        original_summary=original_summary,
        data_path=data_path,
        package_path=package_path,
        output_dir=output_dir,
        argv=argv,
    )


def build_plans(args: argparse.Namespace) -> tuple[OrderControlPlan, ...]:
    input_root = Path(args.input_root)
    output_root = Path(args.output_root) if args.output_root is not None else Path(
        "experiments/seed_order_control"
    ) / f"seed_{int(args.order_seed)}"
    _safe_output_root(input_root, output_root)
    return tuple(
        build_plan(
            subject=subject,
            model=model,
            input_root=input_root,
            output_root=output_root,
            session=str(args.session),
            device=str(args.device),
            order_seed=int(args.order_seed),
            config=Path(args.config),
            require_artifacts=not bool(args.dry_run),
        )
        for subject in normalize_subjects(args.subjects)
        for model in args.models
    )


def _overall(summary: Mapping[str, Any], mode: str) -> Mapping[str, Any]:
    mode_payload = _mapping(summary.get(mode), mode)
    metrics = _mapping(mode_payload.get("metrics"), f"{mode}.metrics")
    return _mapping(metrics.get("overall"), f"{mode}.metrics.overall")


def check_static_invariance(
    original_summary: Mapping[str, Any],
    shuffled_summary: Mapping[str, Any],
    *,
    tolerance: float = STATIC_TOLERANCE,
) -> dict[str, Any]:
    original = _overall(original_summary, "static")
    shuffled = _overall(shuffled_summary, "static")
    differences: dict[str, float] = {}
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        difference = abs(float(original[metric]) - float(shuffled[metric]))
        differences[metric] = difference
        if not math.isclose(
            float(original[metric]),
            float(shuffled[metric]),
            rel_tol=0.0,
            abs_tol=float(tolerance),
        ):
            raise RuntimeError(
                "Static order invariance failed for "
                f"{metric}: original={original[metric]}, "
                f"shuffled={shuffled[metric]}, tolerance={tolerance}."
            )
    if original["confusion_matrix"] != shuffled["confusion_matrix"]:
        raise RuntimeError("Static order invariance failed for confusion_matrix.")
    return {
        "passed": True,
        "tolerance": float(tolerance),
        "absolute_differences": differences,
        "confusion_matrix_equal": True,
    }


def collect_result(plan: OrderControlPlan) -> dict[str, Any]:
    original = _load_mapping(plan.original_summary)
    shuffled = _load_mapping(plan.output_dir / "summary.json")
    order_control = _mapping(shuffled.get("order_control"), "order_control")
    if order_control.get("evaluation_order") != "random_permutation":
        raise RuntimeError("Shuffled summary lacks random_permutation provenance.")
    if original.get("neuroonline_config") != shuffled.get("neuroonline_config"):
        raise RuntimeError("NeuroOnline config differs from the frozen benchmark.")

    invariance = check_static_invariance(original, shuffled)
    original_static = float(_overall(original, "static")["balanced_accuracy"])
    original_online = float(_overall(original, "neuroonline")["balanced_accuracy"])
    shuffled_static = float(_overall(shuffled, "static")["balanced_accuracy"])
    shuffled_online = float(_overall(shuffled, "neuroonline")["balanced_accuracy"])
    gain_original = original_online - original_static
    gain_shuffled = shuffled_online - shuffled_static
    return {
        "subject": int(plan.subject),
        "model": plan.model,
        "status": "passed",
        "static_invariance": invariance,
        "order_control": dict(order_control),
        "balanced_accuracy": {
            "static_original": original_static,
            "neuroonline_original": original_online,
            "static_shuffled": shuffled_static,
            "neuroonline_shuffled": shuffled_online,
            "gain_original": gain_original,
            "gain_shuffled": gain_shuffled,
            "order_dependency": gain_original - gain_shuffled,
        },
    }


def _write_results(output_root: Path, rows: Sequence[Mapping[str, Any]], seed: int) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "diagnostic_only": True,
        "evaluation_order": "random_permutation",
        "order_seed": int(seed),
        "results": list(rows),
    }
    (output_root / "seed_order_control_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_root / "seed_order_control_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "subject",
                "model",
                "gain_original",
                "gain_shuffled",
                "order_dependency",
                "static_invariance_passed",
                "status",
            ),
        )
        writer.writeheader()
        for row in rows:
            metrics = _mapping(row["balanced_accuracy"], "balanced_accuracy")
            invariance = _mapping(row["static_invariance"], "static_invariance")
            writer.writerow(
                {
                    "subject": row["subject"],
                    "model": row["model"],
                    "gain_original": metrics["gain_original"],
                    "gain_shuffled": metrics["gain_shuffled"],
                    "order_dependency": metrics["order_dependency"],
                    "static_invariance_passed": invariance["passed"],
                    "status": row["status"],
                }
            )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plans = build_plans(args)
    for plan in plans:
        print(shlex.join(plan.argv))
    if args.dry_run:
        return 0

    results: list[dict[str, Any]] = []
    for plan in plans:
        subprocess.run(plan.argv, cwd=ROOT, check=True)
        results.append(collect_result(plan))

    output_root = Path(args.output_root) if args.output_root is not None else Path(
        "experiments/seed_order_control"
    ) / f"seed_{int(args.order_seed)}"
    _write_results(output_root, results, int(args.order_seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
