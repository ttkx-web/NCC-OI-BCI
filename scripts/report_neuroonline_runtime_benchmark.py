"""Compare static and NeuroOnline sequential-evaluation summaries."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
LATENCY_KEYS = ("mean_ms", "p50_ms", "p95_ms")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static-summary",
        required=True,
        type=Path,
        help="summary.json from the static none run.",
    )
    parser.add_argument(
        "--neuroonline-summary",
        required=True,
        type=Path,
        help="summary.json from the neuroonline_80 run.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for runtime_benchmark_report.json and .md.",
    )
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Summary does not exist or is not a file: {path!s}")
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"Summary is not valid JSON: {path!s}: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Summary root must be an object: {path!s}")
    return payload


def _required_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Summary field {name} must be an object.")
    return value


def _finite_metric(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Summary field {name} must be a finite number.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Summary field {name} must be a finite number.") from error
    if not math.isfinite(numeric):
        raise ValueError(f"Summary field {name} must be a finite number.")
    return numeric


def _optional_latency(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _finite_metric(value, name=name)


def _mode_result(summary: Mapping[str, Any], *, mode_key: str) -> dict[str, Any]:
    mode = _required_mapping(summary.get(mode_key), name=mode_key)
    metrics = _required_mapping(mode.get("metrics"), name=f"{mode_key}.metrics")
    overall = _required_mapping(metrics.get("overall"), name=f"{mode_key}.metrics.overall")
    updates = _required_mapping(mode.get("updates"), name=f"{mode_key}.updates")
    latency = _required_mapping(updates.get("latency"), name=f"{mode_key}.updates.latency")

    try:
        update_count = int(updates["num_updates"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Summary field {mode_key}.updates.num_updates must be an integer.") from error
    if update_count < 0:
        raise ValueError(f"Summary field {mode_key}.updates.num_updates must be non-negative.")

    return {
        "accuracy": _finite_metric(overall.get("accuracy"), name=f"{mode_key}.metrics.overall.accuracy"),
        "balanced_accuracy": _finite_metric(
            overall.get("balanced_accuracy"),
            name=f"{mode_key}.metrics.overall.balanced_accuracy",
        ),
        "macro_f1": _finite_metric(overall.get("macro_f1"), name=f"{mode_key}.metrics.overall.macro_f1"),
        "update_count": update_count,
        "latency_ms": {
            key: _optional_latency(latency.get(key), name=f"{mode_key}.updates.latency.{key}")
            for key in LATENCY_KEYS
        },
    }


def build_report(
    *,
    static_summary: Mapping[str, Any],
    neuroonline_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a report from a static-none and neuroonline_80 summary."""
    static = _mode_result(static_summary, mode_key="static")
    neuroonline = _mode_result(neuroonline_summary, mode_key="neuroonline")
    return {
        "schema_version": 1,
        "static_none": static,
        "neuroonline_80": neuroonline,
        "gain": {
            metric: neuroonline[metric] - static[metric]
            for metric in METRICS
        },
    }


def _format_number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    rows = (
        ("static_none", _required_mapping(report["static_none"], name="static_none")),
        ("neuroonline_80", _required_mapping(report["neuroonline_80"], name="neuroonline_80")),
    )
    lines = [
        "# NeuroOnline Runtime Benchmark Report",
        "",
        "| Mode | Accuracy | Balanced Accuracy | Macro-F1 | Updates | Latency Mean (ms) | Latency P50 (ms) | Latency P95 (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in rows:
        latency = _required_mapping(result["latency_ms"], name=f"{name}.latency_ms")
        lines.append(
            "| {name} | {accuracy} | {balanced_accuracy} | {macro_f1} | {update_count} | {mean} | {p50} | {p95} |".format(
                name=name,
                accuracy=_format_number(result["accuracy"]),
                balanced_accuracy=_format_number(result["balanced_accuracy"]),
                macro_f1=_format_number(result["macro_f1"]),
                update_count=result["update_count"],
                mean=_format_number(latency["mean_ms"]),
                p50=_format_number(latency["p50_ms"]),
                p95=_format_number(latency["p95_ms"]),
            )
        )

    gain = _required_mapping(report["gain"], name="gain")
    lines.extend(
        (
            "",
            "## NeuroOnline − Static Gain",
            "",
            f"- Accuracy: {_format_number(gain['accuracy'])}",
            f"- Balanced Accuracy: {_format_number(gain['balanced_accuracy'])}",
            f"- Macro-F1: {_format_number(gain['macro_f1'])}",
            "",
        )
    )
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], *, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "runtime_benchmark_report.json"
    markdown_path = output_dir / "runtime_benchmark_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_report(
            static_summary=_load_json(args.static_summary),
            neuroonline_summary=_load_json(args.neuroonline_summary),
        )
        json_path, markdown_path = write_report(report, output_dir=args.output_dir)
    except ValueError as error:
        build_parser().error(str(error))

    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
