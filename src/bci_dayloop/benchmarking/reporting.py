from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from bci_dayloop.benchmarking.core import (
    WindowBenchmarkRecord,
)


@dataclass(frozen=True, slots=True)
class BenchmarkCandidate:
    """
    一个实际参与 benchmark 的 Runtime Model Package。

    这些字段会重复写入每一条 window record，
    使 CSV 脱离 summary.json 也能独立分析。
    """
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


@dataclass(frozen=True, slots=True)
class CandidateBenchmarkSummary:
    candidate: BenchmarkCandidate

    num_records: int

    preprocessing_ms: dict[str, float]
    inference_ms: dict[str, float]
    output_materialization_ms: dict[str, float]
    compute_total_ms: dict[str, float]

    window_ready_to_prediction_ms: (
        dict[str, float] | None
    )
    last_sample_received_to_prediction_ms: (
        dict[str, float] | None
    )

    created_at_utc: str

    deadline_ms: float | None = None
    deadline_miss_count: int | None = None
    deadline_miss_rate: float | None = None
    expected_windows: int | None = None
    completed_windows: int | None = None
    failed_windows: int | None = None
    source_integrity: dict[str, int] | None = None
    status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


WINDOW_RECORD_FIELDNAMES = [
    # 本次 benchmark 的候选模型信息。
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

    # 当前滑窗在原始流中的定位。
    "sequence_index",
    "window_id",
    "trial_id",
    "source_start_sample",
    "source_end_sample_exclusive",

    # 当前预测。
    "prediction",
    "confidence",

    # 三段计算延迟。
    "preprocessing_ms",
    "inference_ms",
    "output_materialization_ms",
    "compute_total_ms",

    # 真实设备模式才有；Replay 中为空。
    "window_ready_to_prediction_ms",
    "last_sample_received_to_prediction_ms",
    "deadline_missed",
]


def _number_summary(
    values: Sequence[float],
) -> dict[str, float]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.ndim != 1 or len(array) == 0:
        raise ValueError(
            "Cannot summarize an empty latency array."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "Latency array contains NaN or Inf."
        )

    return {
        "count": float(len(array)),
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "p50": float(
            np.quantile(
                array,
                0.50,
                method="linear",
            )
        ),
        "p95": float(
            np.quantile(
                array,
                0.95,
                method="linear",
            )
        ),
        "max": float(np.max(array)),
    }


def _optional_number_summary(
    values: Sequence[float | None],
) -> dict[str, float] | None:
    available = [
        float(value)
        for value in values
        if value is not None
    ]

    if not available:
        return None

    return _number_summary(available)


def flatten_window_record(
    *,
    candidate: BenchmarkCandidate,
    record: WindowBenchmarkRecord,
    deadline_ms: float | None = None,
) -> dict[str, object]:
    """
    将“候选模型信息 + 单个窗口结果”拍平成一行 CSV。
    """
    if candidate.source_mode != record.source_mode:
        raise ValueError(
            "Candidate source_mode and record source_mode differ: "
            f"{candidate.source_mode!r} != {record.source_mode!r}."
        )

    return {
        "candidate_id": candidate.candidate_id,
        "model_name": candidate.model_name,
        "model_type": candidate.model_type,
        "package_path": candidate.package_path,
        "package_sha256": candidate.package_sha256,
        "window_sec": candidate.window_sec,
        "step_sec": candidate.step_sec,
        "device": candidate.device,
        "source_mode": record.source_mode,
        "warmup_windows": candidate.warmup_windows,
        "measured_windows": candidate.measured_windows,

        "sequence_index": record.sequence_index,
        "window_id": record.window_id,
        "trial_id": record.trial_id,
        "source_start_sample": record.source_start_sample,
        "source_end_sample_exclusive": (
            record.source_end_sample_exclusive
        ),

        "prediction": record.prediction,
        "confidence": record.confidence,

        "preprocessing_ms": record.preprocessing_ms,
        "inference_ms": record.inference_ms,
        "output_materialization_ms": (
            record.output_materialization_ms
        ),
        "compute_total_ms": record.compute_total_ms,

        "window_ready_to_prediction_ms": (
            record.window_ready_to_prediction_ms
        ),
        "last_sample_received_to_prediction_ms": (
            record.last_sample_received_to_prediction_ms
        ),
        "deadline_missed": (
            None
            if deadline_ms is None
            or record.last_sample_received_to_prediction_ms is None
            else record.last_sample_received_to_prediction_ms > deadline_ms
        ),
    }


def build_candidate_summary(
    *,
    candidate: BenchmarkCandidate,
    records: Sequence[WindowBenchmarkRecord],
    deadline_ms: float | None = None,
    expected_windows: int | None = None,
    failed_windows: int | None = None,
    source_integrity: dict[str, int] | None = None,
    status: str | None = None,
) -> CandidateBenchmarkSummary:
    if not records:
        raise ValueError(
            "Cannot build a summary from zero records."
        )

    last_sample_latencies = [
        record.last_sample_received_to_prediction_ms
        for record in records
    ]
    deadline_miss_count = None
    deadline_miss_rate = None
    if deadline_ms is not None:
        if deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive when provided")
        if any(value is None for value in last_sample_latencies):
            raise ValueError(
                "device deadline statistics require last-sample latency for every record"
            )
        deadline_miss_count = sum(
            float(value) > deadline_ms
            for value in last_sample_latencies
            if value is not None
        )
        deadline_miss_rate = deadline_miss_count / len(records)

    return CandidateBenchmarkSummary(
        candidate=candidate,
        num_records=len(records),

        preprocessing_ms=_number_summary(
            [
                record.preprocessing_ms
                for record in records
            ]
        ),
        inference_ms=_number_summary(
            [
                record.inference_ms
                for record in records
            ]
        ),
        output_materialization_ms=_number_summary(
            [
                record.output_materialization_ms
                for record in records
            ]
        ),
        compute_total_ms=_number_summary(
            [
                record.compute_total_ms
                for record in records
            ]
        ),

        window_ready_to_prediction_ms=(
            _optional_number_summary(
                [
                    record.window_ready_to_prediction_ms
                    for record in records
                ]
            )
        ),
        last_sample_received_to_prediction_ms=(
            _optional_number_summary(
                [
                    record
                    .last_sample_received_to_prediction_ms
                    for record in records
                ]
            )
        ),

        deadline_ms=deadline_ms,
        deadline_miss_count=deadline_miss_count,
        deadline_miss_rate=deadline_miss_rate,
        expected_windows=expected_windows,
        completed_windows=len(records),
        failed_windows=failed_windows,
        source_integrity=(
            dict(source_integrity)
            if source_integrity is not None
            else None
        ),
        status=status,

        created_at_utc=datetime.now(
            timezone.utc
        ).isoformat(),
    )


def write_window_records_csv(
    *,
    path: str | Path,
    rows: Sequence[dict[str, object]],
) -> Path:
    target = Path(path)
    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise ValueError(
            "Refusing to write an empty window-record CSV."
        )

    with target.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=WINDOW_RECORD_FIELDNAMES,
            extrasaction="raise",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    return target


def write_benchmark_summary_json(
    *,
    path: str | Path,
    summaries: Sequence[CandidateBenchmarkSummary],
    benchmark_metadata: dict[str, Any],
) -> Path:
    target = Path(path)
    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not summaries:
        raise ValueError(
            "Refusing to write an empty benchmark summary."
        )

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "benchmark_metadata": benchmark_metadata,
        "candidates": [
            summary.to_dict()
            for summary in summaries
        ],
    }

    target.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return target
