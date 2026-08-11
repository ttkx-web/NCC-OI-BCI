from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_METRICS = (
    "preprocessing_ms",
    "inference_ms",
    "output_materialization_ms",
    "compute_total_ms",
)

OPTIONAL_LIVE_METRICS = (
    "window_ready_to_prediction_ms",
    "last_sample_received_to_prediction_ms",
)

MODEL_DISPLAY_NAMES = {
    "labram": "LaBraM",
    "model_50m": "50M",
    "cbramod": "CBraMod",
}

MODEL_SORT_ORDER = {
    "labram": 0,
    "model_50m": 1,
    "cbramod": 2,
}


@dataclass(frozen=True, slots=True)
class LatencyStats:
    count: int
    mean: float
    minimum: float
    p50: float
    p95: float
    maximum: float


@dataclass(frozen=True, slots=True)
class SummaryRow:
    candidate_id: str
    model_name: str
    model_type: str
    package_path: str
    package_sha256: str | None
    window_sec: float
    step_sec: float
    device: str
    source_mode: str
    warmup_windows: int
    measured_windows: int
    num_records: int
    metrics: dict[str, LatencyStats]


def require_mapping(
    value: object,
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(
            f"{name} must be a JSON object, got {type(value).__name__}."
        )
    return dict(value)


def require_string(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> str:
    value = payload.get(key)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{source}.{key} must be a non-empty string."
        )

    return value.strip()


def require_positive_float(
    payload: dict[str, Any],
    key: str,
    *,
    source: str,
) -> float:
    value = payload.get(key)

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{source}.{key} must be a number, got {value!r}."
        ) from error

    if not math.isfinite(number) or number <= 0:
        raise ValueError(
            f"{source}.{key} must be finite and positive, "
            f"got {number!r}."
        )

    return number


def parse_latency_stats(
    value: object,
    *,
    source: str,
) -> LatencyStats:
    payload = require_mapping(value, name=source)

    required_keys = (
        "count",
        "mean",
        "min",
        "p50",
        "p95",
        "max",
    )

    parsed: dict[str, float] = {}

    for key in required_keys:
        raw_value = payload.get(key)

        try:
            number = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{source}.{key} must be numeric, got {raw_value!r}."
            ) from error

        if not math.isfinite(number):
            raise ValueError(
                f"{source}.{key} must be finite, got {number!r}."
            )

        parsed[key] = number

    if parsed["count"] <= 0:
        raise ValueError(
            f"{source}.count must be positive, got {parsed['count']}."
        )

    return LatencyStats(
        count=int(parsed["count"]),
        mean=parsed["mean"],
        minimum=parsed["min"],
        p50=parsed["p50"],
        p95=parsed["p95"],
        maximum=parsed["max"],
    )


def read_summary_rows(summary_path: Path) -> list[SummaryRow]:
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    root = require_mapping(payload, name=str(summary_path))

    if int(root.get("schema_version", -1)) != 1:
        raise ValueError(
            f"{summary_path}: expected schema_version=1."
        )

    raw_candidates = root.get("candidates")

    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError(
            f"{summary_path}: candidates must be a non-empty list."
        )

    rows: list[SummaryRow] = []

    for index, raw_item in enumerate(raw_candidates):
        source = f"{summary_path}: candidates[{index}]"
        item = require_mapping(raw_item, name=source)

        candidate = require_mapping(
            item.get("candidate"),
            name=f"{source}.candidate",
        )

        candidate_id = require_string(
            candidate,
            "candidate_id",
            source=f"{source}.candidate",
        )

        metrics: dict[str, LatencyStats] = {}

        for metric_name in BASE_METRICS:
            metrics[metric_name] = parse_latency_stats(
                item.get(metric_name),
                source=f"{source}.{metric_name}",
            )

        # Replay 时这两个字段通常为 null；真实设备 benchmark 才会有。
        for metric_name in OPTIONAL_LIVE_METRICS:
            raw_metric = item.get(metric_name)

            if raw_metric is not None:
                metrics[metric_name] = parse_latency_stats(
                    raw_metric,
                    source=f"{source}.{metric_name}",
                )

        rows.append(
            SummaryRow(
                candidate_id=candidate_id,
                model_name=require_string(
                    candidate,
                    "model_name",
                    source=f"{source}.candidate",
                ),
                model_type=require_string(
                    candidate,
                    "model_type",
                    source=f"{source}.candidate",
                ),
                package_path=require_string(
                    candidate,
                    "package_path",
                    source=f"{source}.candidate",
                ),
                package_sha256=(
                    str(candidate["package_sha256"])
                    if candidate.get("package_sha256") is not None
                    else None
                ),
                window_sec=require_positive_float(
                    candidate,
                    "window_sec",
                    source=f"{source}.candidate",
                ),
                step_sec=require_positive_float(
                    candidate,
                    "step_sec",
                    source=f"{source}.candidate",
                ),
                device=require_string(
                    candidate,
                    "device",
                    source=f"{source}.candidate",
                ),
                source_mode=require_string(
                    candidate,
                    "source_mode",
                    source=f"{source}.candidate",
                ),
                warmup_windows=int(
                    candidate["warmup_windows"]
                ),
                measured_windows=int(
                    candidate["measured_windows"]
                ),
                num_records=int(item["num_records"]),
                metrics=metrics,
            )
        )

    if len({row.candidate_id for row in rows}) != len(rows):
        raise ValueError(
            "summary.json contains duplicated candidate_id values."
        )

    return rows


def model_display_name(model_type: str) -> str:
    return MODEL_DISPLAY_NAMES.get(
        model_type,
        model_type,
    )


def model_sort_key(model_type: str) -> tuple[int, str]:
    return (
        MODEL_SORT_ORDER.get(model_type, 999),
        model_type,
    )


def markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|")


def format_ms(stat: LatencyStats | None) -> str:
    if stat is None:
        return "—"

    return f"{stat.p50:.3f} / {stat.p95:.3f}"


def markdown_table(
    headers: list[str],
    rows: list[list[object]],
) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                markdown_escape(cell)
                for cell in row
            )
            + " |"
        )

    return "\n".join(lines) + "\n"


def build_long_markdown(
    rows: list[SummaryRow],
    *,
    summary_path: Path,
) -> str:
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            model_sort_key(row.model_type),
            row.window_sec,
            row.candidate_id,
        ),
    )

    headers = [
        "模型",
        "模型类型",
        "滑窗 (s)",
        "步长 (s)",
        "设备",
        "样本数",
        "预处理 P50 / P95 (ms)",
        "推理 P50 / P95 (ms)",
        "结果搬运 P50 / P95 (ms)",
        "总计算 P50 / P95 (ms)",
        "窗口就绪至预测 P50 / P95 (ms)",
        "最后样本至预测 P50 / P95 (ms)",
        "Candidate ID",
    ]

    table_rows: list[list[object]] = []

    for row in ordered_rows:
        table_rows.append(
            [
                row.model_name,
                model_display_name(row.model_type),
                f"{row.window_sec:g}",
                f"{row.step_sec:g}",
                row.device,
                row.num_records,
                format_ms(
                    row.metrics["preprocessing_ms"]
                ),
                format_ms(
                    row.metrics["inference_ms"]
                ),
                format_ms(
                    row.metrics[
                        "output_materialization_ms"
                    ]
                ),
                format_ms(
                    row.metrics["compute_total_ms"]
                ),
                format_ms(
                    row.metrics.get(
                        "window_ready_to_prediction_ms"
                    )
                ),
                format_ms(
                    row.metrics.get(
                        "last_sample_received_to_prediction_ms"
                    )
                ),
                row.candidate_id,
            ]
        )

    return (
        "# 滑窗延迟 Benchmark：长表\n\n"
        f"来源：`{summary_path}`\n\n"
        "表中所有数值格式均为 **P50 / P95（ms）**。"
        "Replay 模式只衡量模型计算侧耗时，因此真实设备专属字段显示为 `—`。\n\n"
        + markdown_table(headers, table_rows)
    )


def build_wide_markdown(
    rows: list[SummaryRow],
    *,
    metric_name: str,
    summary_path: Path,
) -> str:
    usable_rows = [
        row
        for row in rows
        if metric_name in row.metrics
    ]

    if not usable_rows:
        raise ValueError(
            f"No candidate contains metric {metric_name!r}."
        )

    # 当前实验中，每个模型类型在每个窗口长度下只能有一个候选。
    # 如果未来引入 population/personal 两个 50M 头，需要先扩展
    # BenchmarkCandidate 的 model_variant 字段，不能静默覆盖其中一个结果。
    seen_pairs: dict[tuple[str, float], SummaryRow] = {}

    for row in usable_rows:
        key = (row.model_type, row.window_sec)

        if key in seen_pairs:
            previous = seen_pairs[key]
            raise ValueError(
                "Cannot build wide table because multiple candidates share "
                "the same (model_type, window_sec): "
                f"{previous.candidate_id!r} and {row.candidate_id!r}. "
                "Add an explicit model_variant field before comparing "
                "multiple heads of the same backbone."
            )

        seen_pairs[key] = row

    model_types = sorted(
        {row.model_type for row in usable_rows},
        key=model_sort_key,
    )
    window_lengths = sorted(
        {row.window_sec for row in usable_rows}
    )

    headers = ["滑窗 (s)"]

    for model_type in model_types:
        display_name = model_display_name(model_type)
        headers.extend(
            [
                f"{display_name} P50 (ms)",
                f"{display_name} P95 (ms)",
            ]
        )

    table_rows: list[list[object]] = []

    for window_sec in window_lengths:
        table_row: list[object] = [f"{window_sec:g}"]

        for model_type in model_types:
            row = seen_pairs.get(
                (model_type, window_sec)
            )

            if row is None:
                table_row.extend(["—", "—"])
                continue

            stat = row.metrics[metric_name]
            table_row.extend(
                [
                    f"{stat.p50:.3f}",
                    f"{stat.p95:.3f}",
                ]
            )

        table_rows.append(table_row)

    metric_title = metric_name.replace("_", " ")

    return (
        f"# 滑窗延迟 Benchmark：{metric_title} 宽表\n\n"
        f"来源：`{summary_path}`\n\n"
        "每一行表示同一个滑窗长度下，三个基线模型的 P50/P95 对比。"
        "单位均为毫秒（ms）。\n\n"
        + markdown_table(headers, table_rows)
    )


def write_long_csv(
    rows: list[SummaryRow],
    *,
    path: Path,
) -> None:
    fieldnames = [
        "candidate_id",
        "model_name",
        "model_type",
        "package_path",
        "package_sha256",
        "window_sec",
        "step_sec",
        "device",
        "source_mode",
        "warmup_windows",
        "measured_windows",
        "num_records",
    ]

    for metric_name in (
        *BASE_METRICS,
        *OPTIONAL_LIVE_METRICS,
    ):
        for stat_name in (
            "count",
            "mean",
            "min",
            "p50",
            "p95",
            "max",
        ):
            fieldnames.append(
                f"{metric_name}_{stat_name}"
            )

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in sorted(
            rows,
            key=lambda item: (
                model_sort_key(item.model_type),
                item.window_sec,
                item.candidate_id,
            ),
        ):
            flat_row: dict[str, object] = {
                "candidate_id": row.candidate_id,
                "model_name": row.model_name,
                "model_type": row.model_type,
                "package_path": row.package_path,
                "package_sha256": row.package_sha256,
                "window_sec": row.window_sec,
                "step_sec": row.step_sec,
                "device": row.device,
                "source_mode": row.source_mode,
                "warmup_windows": row.warmup_windows,
                "measured_windows": row.measured_windows,
                "num_records": row.num_records,
            }

            for metric_name in (
                *BASE_METRICS,
                *OPTIONAL_LIVE_METRICS,
            ):
                stat = row.metrics.get(metric_name)

                if stat is None:
                    continue

                flat_row.update(
                    {
                        f"{metric_name}_count": stat.count,
                        f"{metric_name}_mean": stat.mean,
                        f"{metric_name}_min": stat.minimum,
                        f"{metric_name}_p50": stat.p50,
                        f"{metric_name}_p95": stat.p95,
                        f"{metric_name}_max": stat.maximum,
                    }
                )

            writer.writerow(flat_row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render long and wide Markdown/CSV latency tables "
            "from a window benchmark summary.json."
        )
    )

    parser.add_argument(
        "--summary",
        required=True,
        help=(
            "Path to summary.json generated by "
            "run_window_latency_benchmark.py."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Default: "
            "<summary parent>/rendered_tables."
        ),
    )

    parser.add_argument(
        "--wide-metric",
        default="compute_total_ms",
        choices=(
            *BASE_METRICS,
            *OPTIONAL_LIVE_METRICS,
        ),
        help=(
            "Metric shown in the wide comparison table. "
            "Default: compute_total_ms."
        ),
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    summary_path = Path(args.summary).expanduser().resolve()

    if not summary_path.is_file():
        raise FileNotFoundError(
            f"summary.json was not found: {summary_path}"
        )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else summary_path.parent / "rendered_tables"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_summary_rows(summary_path)

    long_markdown_path = (
        output_dir / "window_latency_long.md"
    )
    wide_markdown_path = (
        output_dir / "window_latency_wide.md"
    )
    long_csv_path = (
        output_dir / "window_latency_summary_long.csv"
    )

    long_markdown_path.write_text(
        build_long_markdown(
            rows,
            summary_path=summary_path,
        ),
        encoding="utf-8",
    )

    wide_markdown_path.write_text(
        build_wide_markdown(
            rows,
            metric_name=args.wide_metric,
            summary_path=summary_path,
        ),
        encoding="utf-8",
    )

    write_long_csv(
        rows,
        path=long_csv_path,
    )

    print("Rendered latency tables:")
    print("  long markdown:", long_markdown_path)
    print("  wide markdown:", wide_markdown_path)
    print("  long csv:", long_csv_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())