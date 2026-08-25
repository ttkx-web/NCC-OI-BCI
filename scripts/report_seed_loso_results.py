"""Build strict aggregate reports for the completed SEED LOSO experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SUBJECTS = tuple(range(1, 16))
MODELS = ("labram", "cbramod")
METRICS = ("balanced_accuracy", "accuracy", "macro_f1")
LATENCY_FIELDS = ("mean_ms", "p50_ms", "p95_ms")
GAIN_EPSILON = 1e-12

CSV_FIELDS = (
    "subject",
    "model",
    "static_accuracy",
    "static_balanced_accuracy",
    "static_macro_f1",
    "neuroonline_accuracy",
    "neuroonline_balanced_accuracy",
    "neuroonline_macro_f1",
    "accuracy_gain",
    "balanced_accuracy_gain",
    "macro_f1_gain",
    "update_count",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Required evaluation summary is missing: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Evaluation summary is not valid JSON: {path}: {error.msg}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation summary root must be an object: {path}")
    return payload


def _mapping(value: object, *, name: str, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: field {name} must be an object.")
    return value


def _finite(value: object, *, name: str, path: Path) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path}: field {name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path}: field {name} must be a finite number."
        ) from error
    if not math.isfinite(result):
        raise ValueError(f"{path}: field {name} must be a finite number.")
    return result


def _optional_finite(value: object, *, name: str, path: Path) -> float | None:
    if value is None:
        return None
    return _finite(value, name=name, path=path)


def _mode_metrics(
    summary: Mapping[str, Any], *, mode: str, path: Path
) -> tuple[dict[str, float], Mapping[str, Any]]:
    mode_payload = _mapping(summary.get(mode), name=mode, path=path)
    metrics = _mapping(
        mode_payload.get("metrics"), name=f"{mode}.metrics", path=path
    )
    overall = _mapping(
        metrics.get("overall"), name=f"{mode}.metrics.overall", path=path
    )
    values = {
        metric: _finite(
            overall.get(metric),
            name=f"{mode}.metrics.overall.{metric}",
            path=path,
        )
        for metric in METRICS
    }
    return values, mode_payload


def _subject_result(*, subject: int, model: str, path: Path) -> dict[str, Any]:
    summary = _load_json(path)
    static, _ = _mode_metrics(summary, mode="static", path=path)
    neuroonline, online_payload = _mode_metrics(
        summary, mode="neuroonline", path=path
    )
    updates = _mapping(
        online_payload.get("updates"),
        name="neuroonline.updates",
        path=path,
    )
    try:
        update_count = int(updates["num_updates"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{path}: field neuroonline.updates.num_updates must be an integer."
        ) from error
    if update_count < 0:
        raise ValueError(
            f"{path}: field neuroonline.updates.num_updates must be non-negative."
        )
    latency = _mapping(
        updates.get("latency"),
        name="neuroonline.updates.latency",
        path=path,
    )
    latency_values = {
        key: _optional_finite(
            latency.get(key),
            name=f"neuroonline.updates.latency.{key}",
            path=path,
        )
        for key in LATENCY_FIELDS
    }

    computed_gains = {
        metric: neuroonline[metric] - static[metric] for metric in METRICS
    }
    gains = _mapping(summary.get("gains"), name="gains", path=path)
    overall_gains = _mapping(
        gains.get("overall"), name="gains.overall", path=path
    )
    for metric, computed in computed_gains.items():
        declared = _finite(
            overall_gains.get(f"{metric}_gain"),
            name=f"gains.overall.{metric}_gain",
            path=path,
        )
        if not math.isclose(declared, computed, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"{path}: declared {metric} gain {declared} does not match "
                f"NeuroOnline - static ({computed})."
            )

    return {
        "subject": subject,
        "model": model,
        **{f"static_{metric}": static[metric] for metric in METRICS},
        **{
            f"neuroonline_{metric}": neuroonline[metric]
            for metric in METRICS
        },
        **{f"{metric}_gain": computed_gains[metric] for metric in METRICS},
        "update_count": update_count,
        "latency_mean_ms": latency_values["mean_ms"],
        "latency_p50_ms": latency_values["p50_ms"],
        "latency_p95_ms": latency_values["p95_ms"],
    }


def load_subject_results(input_root: Path) -> list[dict[str, Any]]:
    if not input_root.is_dir():
        raise ValueError(f"SEED LOSO input root does not exist: {input_root}")

    expected_names = {f"subject_{subject:02d}" for subject in SUBJECTS}
    actual_names = {
        child.name
        for child in input_root.iterdir()
        if child.is_dir() and re.fullmatch(r"subject_\d+", child.name)
    }
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        raise ValueError(
            "SEED LOSO subject directories must be exactly subject_01..subject_15; "
            f"missing={missing}, unexpected={unexpected}."
        )

    return [
        _subject_result(
            subject=subject,
            model=model,
            path=(
                input_root
                / f"subject_{subject:02d}"
                / model
                / "evaluation"
                / "summary.json"
            ),
        )
        for subject in SUBJECTS
        for model in MODELS
    ]


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty value sequence.")
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "std": statistics.pstdev(numeric),
        "median": statistics.median(numeric),
    }


def _gain_counts(values: Sequence[float]) -> dict[str, int]:
    return {
        "positive": sum(value > GAIN_EPSILON for value in values),
        "zero": sum(abs(value) <= GAIN_EPSILON for value in values),
        "negative": sum(value < -GAIN_EPSILON for value in values),
    }


def build_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected_rows = len(SUBJECTS) * len(MODELS)
    if len(results) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} subject/model rows, got {len(results)}."
        )

    model_summaries: dict[str, Any] = {}
    for model in MODELS:
        model_rows = [row for row in results if row["model"] == model]
        if len(model_rows) != len(SUBJECTS):
            raise ValueError(
                f"Expected 15 rows for model {model}, got {len(model_rows)}."
            )
        metric_summaries: dict[str, Any] = {}
        for metric in METRICS:
            gains = [float(row[f"{metric}_gain"]) for row in model_rows]
            metric_summaries[metric] = {
                "static": _distribution(
                    [float(row[f"static_{metric}"]) for row in model_rows]
                ),
                "neuroonline": _distribution(
                    [
                        float(row[f"neuroonline_{metric}"])
                        for row in model_rows
                    ]
                ),
                "gain": {
                    **_distribution(gains),
                    "subject_counts": _gain_counts(gains),
                },
            }

        model_summaries[model] = {
            "subject_count": len(model_rows),
            "primary_metric": "balanced_accuracy",
            "metrics": metric_summaries,
            "update_count": _distribution(
                [float(row["update_count"]) for row in model_rows]
            ),
            "latency_ms": {
                key.removesuffix("_ms").removeprefix("latency_"): (
                    _distribution(
                        [
                            float(row[key])
                            for row in model_rows
                            if row[key] is not None
                        ]
                    )
                    if any(row[key] is not None for row in model_rows)
                    else None
                )
                for key in (
                    "latency_mean_ms",
                    "latency_p50_ms",
                    "latency_p95_ms",
                )
            },
        }

    return {
        "schema_version": 1,
        "dataset": "seed",
        "subjects": list(SUBJECTS),
        "models": list(MODELS),
        "primary_metric": "balanced_accuracy",
        "subject_results": [dict(row) for row in results],
        "model_summaries": model_summaries,
    }


def _fmt(value: object) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.6f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    rows = report["subject_results"]
    model_summaries = report["model_summaries"]
    lines = [
        "# SEED Full LOSO Results",
        "",
        "Primary comparison metric: **balanced accuracy**.",
        "",
        "## Per-subject results",
        "",
        "| Subject | Model | Static Acc | Static BA | Static Macro-F1 | "
        "Online Acc | Online BA | Online Macro-F1 | Acc Gain | BA Gain | "
        "Macro-F1 Gain | Updates | Latency Mean/P50/P95 (ms) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {subject:02d} | {model} | {static_accuracy} | "
            "{static_balanced_accuracy} | {static_macro_f1} | "
            "{neuroonline_accuracy} | {neuroonline_balanced_accuracy} | "
            "{neuroonline_macro_f1} | {accuracy_gain} | "
            "{balanced_accuracy_gain} | {macro_f1_gain} | {update_count} | "
            "{latency_mean_ms}/{latency_p50_ms}/{latency_p95_ms} |".format(
                subject=int(row["subject"]),
                model=row["model"],
                **{
                    key: _fmt(row[key])
                    for key in CSV_FIELDS
                    if key not in {"subject", "model", "update_count"}
                },
                update_count=row["update_count"],
            )
        )

    lines.extend(
        (
            "",
            "## Model summaries",
            "",
            "| Model | Metric | Static mean ± std | Static median | "
            "Online mean ± std | Online median | Gain mean | Gain median | "
            "+ / 0 / − subjects |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for model in MODELS:
        metrics = model_summaries[model]["metrics"]
        for metric in METRICS:
            item = metrics[metric]
            counts = item["gain"]["subject_counts"]
            lines.append(
                f"| {model} | {metric} | "
                f"{_fmt(item['static']['mean'])} ± {_fmt(item['static']['std'])} | "
                f"{_fmt(item['static']['median'])} | "
                f"{_fmt(item['neuroonline']['mean'])} ± "
                f"{_fmt(item['neuroonline']['std'])} | "
                f"{_fmt(item['neuroonline']['median'])} | "
                f"{_fmt(item['gain']['mean'])} | "
                f"{_fmt(item['gain']['median'])} | "
                f"{counts['positive']} / {counts['zero']} / "
                f"{counts['negative']} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_reports(
    *, results: Sequence[Mapping[str, Any]], report: Mapping[str, Any], output_dir: Path
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "seed_loso_subject_results.csv"
    json_path = output_dir / "seed_loso_summary.json"
    markdown_path = output_dir / "seed_loso_report.md"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return csv_path, json_path, markdown_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        results = load_subject_results(args.input_root)
        report = build_summary(results)
        outputs = write_reports(
            results=results, report=report, output_dir=args.output_dir
        )
    except ValueError as error:
        parser.error(str(error))
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
