"""Run SEED NeuroOnline update-scope diagnostics without retraining."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from _bootstrap import ROOT

from bci_dayloop.inference.neuroonline_strategy import NeuroOnlineConfig


DEFAULT_SUBJECTS = (1, 2, 3)
VALID_MODELS = ("labram", "cbramod")
ABLATION_SCOPES = ("generator_only", "head_only")


@dataclass(frozen=True, slots=True)
class ScopeAblationPlan:
    subject: int
    model: str
    update_scope: str
    frozen_summary: Path
    output_dir: Path
    argv: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run persisted-order SEED Generator-only and head-only "
            "NeuroOnline diagnostics using frozen LOSO packages."
        )
    )
    parser.add_argument("--subjects", nargs="+", type=int, default=DEFAULT_SUBJECTS)
    parser.add_argument(
        "--input-root", type=Path, default=Path("experiments/seed_loso_full")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/seed_update_scope_ablation"),
    )
    parser.add_argument("--session", default="S3")
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
    resolved_path = path.resolve()
    resolved_parent = parent.resolve()
    return resolved_path == resolved_parent or resolved_parent in resolved_path.parents


def validate_output_root(input_root: Path, output_root: Path) -> None:
    if _is_same_or_child(output_root, input_root):
        raise ValueError("Ablation output must not overwrite frozen seed_loso_full.")
    order_control = ROOT / "experiments" / "seed_order_control"
    if _is_same_or_child(output_root, order_control):
        raise ValueError("Ablation output must not overwrite seed_order_control.")


def normalized_neuroonline_config(value: object) -> dict[str, Any]:
    payload = dict(_mapping(value, "neuroonline_config"))
    payload.setdefault("update_scope", "generator_and_head")
    return payload


def validate_frozen_current_config(summary: Mapping[str, Any]) -> dict[str, Any]:
    actual = normalized_neuroonline_config(summary.get("neuroonline_config"))
    expected = asdict(NeuroOnlineConfig())
    if actual != expected:
        raise RuntimeError(
            "Frozen A1 NeuroOnline config does not match the current formal "
            f"baseline: expected={expected}, actual={actual}."
        )
    return actual


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
    update_scope: str,
    input_root: Path,
    output_root: Path,
    session: str,
    device: str,
    config: Path,
    require_artifacts: bool,
) -> ScopeAblationPlan:
    if model not in VALID_MODELS:
        raise ValueError(f"Unsupported model: {model!r}.")
    if update_scope not in ABLATION_SCOPES:
        raise ValueError(f"Unsupported ablation scope: {update_scope!r}.")
    subject_dir = input_root / f"subject_{subject:02d}" / model
    frozen_summary = subject_dir / "evaluation" / "summary.json"
    frozen = _load_mapping(frozen_summary)
    validate_frozen_current_config(frozen)
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

    output_dir = (
        output_root
        / f"subject_{subject:02d}"
        / model
        / update_scope
    )
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
        session,
        "--device",
        device,
        "--online-strategy",
        "both",
        "--evaluation-order",
        "persisted",
        "--update-scope",
        update_scope,
        "--output-dir",
        str(output_dir),
    )
    return ScopeAblationPlan(
        subject=subject,
        model=model,
        update_scope=update_scope,
        frozen_summary=frozen_summary,
        output_dir=output_dir,
        argv=argv,
    )


def build_plans(args: argparse.Namespace) -> tuple[ScopeAblationPlan, ...]:
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    validate_output_root(input_root, output_root)
    return tuple(
        build_plan(
            subject=subject,
            model=model,
            update_scope=scope,
            input_root=input_root,
            output_root=output_root,
            session=str(args.session),
            device=str(args.device),
            config=Path(args.config),
            require_artifacts=not bool(args.dry_run),
        )
        for subject in normalize_subjects(args.subjects)
        for model in args.models
        for scope in ABLATION_SCOPES
    )


def _mode(summary: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return _mapping(summary.get(name), name)


def _overall(summary: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    metrics = _mapping(_mode(summary, name).get("metrics"), f"{name}.metrics")
    return _mapping(metrics.get("overall"), f"{name}.metrics.overall")


def _update_diagnostics(
    mode: Mapping[str, Any],
    *,
    max_grad_norm: float,
) -> dict[str, Any]:
    updates = _mapping(mode.get("updates"), "neuroonline.updates")
    records = updates.get("updates")
    if not isinstance(records, list):
        raise ValueError("neuroonline.updates.updates must be a list.")
    gradients = [
        float(_mapping(_mapping(item, "update").get("metrics"), "update.metrics")[
            "last_gradient_norm"
        ])
        for item in records
    ]
    clipped = sum(value > float(max_grad_norm) for value in gradients)
    return {
        "num_updates": int(updates["num_updates"]),
        "update_latency": dict(
            _mapping(updates.get("latency"), "updates.latency")
        ),
        "gradient_norm": {
            "mean": None if not gradients else sum(gradients) / len(gradients),
            "max": None if not gradients else max(gradients),
            "clipping_count": clipped,
            "clipping_rate": 0.0 if not gradients else clipped / len(gradients),
        },
    }


def _scope_result(summary: Mapping[str, Any], scope: str) -> dict[str, Any]:
    config = normalized_neuroonline_config(summary.get("neuroonline_config"))
    frozen_baseline = asdict(NeuroOnlineConfig())
    expected = {**frozen_baseline, "update_scope": scope}
    if config != expected:
        raise RuntimeError(f"{scope} run changed non-scope NeuroOnline settings.")
    identity = _mapping(
        summary.get("identity_initialization_check"),
        "identity_initialization_check",
    )
    if identity.get("equivalent") is not True:
        raise RuntimeError(f"{scope} identity initialization gate failed.")
    online = _mode(summary, "neuroonline")
    audit = _mapping(online.get("parameter_audit"), "parameter_audit")
    if audit.get("update_scope") != scope:
        raise RuntimeError(f"{scope} run is missing its parameter audit.")
    metrics = _overall(summary, "neuroonline")
    return {
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        **_update_diagnostics(online, max_grad_norm=float(config["max_grad_norm"])),
        "parameter_audit": dict(audit),
    }


def collect_subject_model(plans: Sequence[ScopeAblationPlan]) -> dict[str, Any]:
    if {plan.update_scope for plan in plans} != set(ABLATION_SCOPES):
        raise ValueError("Exactly one plan per ablation scope is required.")
    first = plans[0]
    frozen = _load_mapping(first.frozen_summary)
    validate_frozen_current_config(frozen)
    static = _overall(frozen, "static")
    current = _overall(frozen, "neuroonline")
    frozen_config = validate_frozen_current_config(frozen)
    frozen_online = _mode(frozen, "neuroonline")
    arms = {
        plan.update_scope: _scope_result(
            _load_mapping(plan.output_dir / "summary.json"), plan.update_scope
        )
        for plan in plans
    }
    static_ba = float(static["balanced_accuracy"])
    current_ba = float(current["balanced_accuracy"])
    generator_ba = float(arms["generator_only"]["balanced_accuracy"])
    head_ba = float(arms["head_only"]["balanced_accuracy"])
    return {
        "subject": first.subject,
        "model": first.model,
        "evaluation_order": "persisted",
        "static": {
            key: float(static[key])
            for key in ("accuracy", "balanced_accuracy", "macro_f1")
        },
        "current": {
            **{
                key: float(current[key])
                for key in ("accuracy", "balanced_accuracy", "macro_f1")
            },
            **_update_diagnostics(
                frozen_online,
                max_grad_norm=float(frozen_config["max_grad_norm"]),
            ),
            "parameter_audit": {
                "update_scope": "generator_and_head",
                "status": "not_recorded_in_frozen_legacy_summary",
            },
        },
        **arms,
        "balanced_accuracy_comparison": {
            "static_ba": static_ba,
            "current_ba": current_ba,
            "generator_only_ba": generator_ba,
            "head_only_ba": head_ba,
            "current_gain": current_ba - static_ba,
            "generator_only_gain": generator_ba - static_ba,
            "head_only_gain": head_ba - static_ba,
            "generator_only_vs_current": generator_ba - current_ba,
            "head_only_vs_current": head_ba - current_ba,
        },
    }


def write_results(output_root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "diagnostic_only": True,
        "evaluation_order": "persisted",
        "results": list(rows),
    }
    (output_root / "update_scope_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_root / "update_scope_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = (
            "subject", "model", "static_ba", "current_ba",
            "generator_only_ba", "head_only_ba", "current_gain",
            "generator_only_gain", "head_only_gain",
            "generator_only_vs_current", "head_only_vs_current",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            comparison = _mapping(
                row["balanced_accuracy_comparison"], "balanced_accuracy_comparison"
            )
            writer.writerow({"subject": row["subject"], "model": row["model"], **comparison})


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plans = build_plans(args)
    for plan in plans:
        print(shlex.join(plan.argv))
    if args.dry_run:
        return 0

    for plan in plans:
        subprocess.run(plan.argv, cwd=ROOT, check=True)

    rows = []
    for subject in normalize_subjects(args.subjects):
        for model in args.models:
            rows.append(
                collect_subject_model(
                    [
                        plan
                        for plan in plans
                        if plan.subject == subject and plan.model == model
                    ]
                )
            )
    write_results(Path(args.output_root), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
