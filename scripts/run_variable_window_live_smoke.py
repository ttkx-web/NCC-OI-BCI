"""Run the nine approved 1–3 second Stage 2B live smoke combinations.

This is an orchestrator only: each combination delegates device ingestion,
windowing, package preparation, and prediction to
``probe_neuracle_runtime_inference.py``.  It never writes EEG samples or a
JellyFish endpoint to its outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

try:
    from _bootstrap import ROOT
except ModuleNotFoundError:
    from scripts._bootstrap import ROOT

from bci_dayloop.utils.config import load_yaml

# ``python scripts/<name>.py`` puts ``scripts/`` (rather than the repository
# root) on sys.path.  Prefer the sibling import for that supported CLI form,
# then retain the package import for ``python -m scripts.<name>``.
try:
    from probe_neuracle_runtime_inference import main as run_runtime_probe
except ModuleNotFoundError as exc:
    if exc.name != "probe_neuracle_runtime_inference":
        raise
    from scripts.probe_neuracle_runtime_inference import main as run_runtime_probe


HOST_ENV = "NEURACLE_JELLYFISH_HOST"
DURATION_SECONDS = 20.0
STEP_SECONDS = 0.5
CBRAMOD_MISSING = ("CPz", "P1", "P2")


@dataclass(frozen=True, slots=True)
class SmokeCombination:
    identifier: str
    model: str
    window_seconds: float
    package_relative_path: str
    expected_prepared_shape: tuple[int, ...]


def _combinations() -> tuple[SmokeCombination, ...]:
    values: list[SmokeCombination] = []
    for seconds in (1, 2, 3):
        values.append(
            SmokeCombination(
                f"model_50m_{seconds}s",
                "model_50m",
                float(seconds),
                (
                    "model_packages/stage1/bnci2014_001/subject_01/population/"
                    f"{seconds}s_flatten/experimentB/1e-3/v1"
                ),
                (1, 64, seconds * 100),
            )
        )
    for seconds in (1, 2, 3):
        values.append(
            SmokeCombination(
                f"labram_{seconds}s",
                "labram",
                float(seconds),
                (
                    "model_packages/stage1/bnci2014_001/subject_01/population/"
                    f"{seconds}s_labram_live19/v1"
                ),
                (1, 19, seconds, 200),
            )
        )
    for seconds in (1, 2, 3):
        values.append(
            SmokeCombination(
                f"cbramod_{seconds}s",
                "cbramod",
                float(seconds),
                (
                    "model_packages/stage1/bnci2014_001/subject_01/population/"
                    f"{seconds}s_cbramod_live19_spline22/v1"
                ),
                (1, 22, seconds, 200),
            )
        )
    return tuple(values)


SMOKE_COMBINATIONS = _combinations()


def _safe_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return f"<external>/{path.name}"


def _read_cbramod_completion_sha(package_path: Path) -> str:
    package_yaml = package_path / "package.yaml"
    if not package_yaml.is_file():
        raise ValueError("CBRaMod Package package.yaml is unavailable")
    payload = load_yaml(package_yaml)
    runtime = payload.get("runtime")
    completion = runtime.get("channel_completion") if isinstance(runtime, Mapping) else None
    value = (
        completion.get("completion_matrix_sha256")
        if isinstance(completion, Mapping)
        else None
    )
    if not isinstance(value, str) or not value.strip():
        raise ValueError("CBRaMod Package completion_matrix_sha256 is unavailable")
    return value


def _integer(summary: Mapping[str, object], key: str) -> int:
    value = summary.get(key, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _health_integer(summary: Mapping[str, object], key: str) -> int:
    health = summary.get("pre_disconnect_health")
    if not isinstance(health, Mapping):
        return -1
    try:
        return int(health.get(key, 0))
    except (TypeError, ValueError):
        return -1


def _gate_result(
    combination: SmokeCombination,
    summary: Mapping[str, object],
    *,
    expected_completion_sha: str | None,
) -> tuple[bool, list[str]]:
    """Validate the fixed metadata-only live smoke contract."""
    failures: list[str] = []
    if summary.get("status") != "passed":
        failures.append("status")
    if summary.get("model_type") != combination.model:
        failures.append("model_type")
    if summary.get("compatibility_status") != "passed":
        failures.append("compatibility_status")
    if summary.get("is_test_head") is not False:
        failures.append("is_test_head")
    if list(summary.get("prepared_shape") or ()) != list(
        combination.expected_prepared_shape
    ):
        failures.append("prepared_shape")

    received_samples = _integer(summary, "received_samples")
    emitted_windows = _integer(summary, "emitted_windows")
    if received_samples <= 0:
        failures.append("received_samples")
    if emitted_windows <= 0:
        failures.append("emitted_windows")
    for key in (
        "model_input_safe_count",
        "prediction_success_count",
    ):
        if _integer(summary, key) != emitted_windows:
            failures.append(key)
    for key in (
        "failed_windows",
        "pipeline_failed_windows",
        "model_input_failure_count",
        "prediction_failure_count",
        "missing_packets",
        "duplicate_packets",
        "out_of_order_packets",
        "gap_count",
    ):
        if _integer(summary, key) != 0:
            failures.append(key)
    for key in ("malformed_packets", "reconnect_count"):
        if _health_integer(summary, key) != 0:
            failures.append(key)
    if summary.get("waveforms_saved") is not False:
        failures.append("waveforms_saved")
    if summary.get("last_error") is not None:
        failures.append("last_error")

    if combination.model == "cbramod":
        if _integer(summary, "observed_channel_count") != 19:
            failures.append("observed_channel_count")
        if tuple(summary.get("missing_channel_names") or ()) != CBRAMOD_MISSING:
            failures.append("missing_channel_names")
        if summary.get("completion_policy") != "spherical_spline":
            failures.append("completion_policy")
        if summary.get("completion_matrix_sha256") != expected_completion_sha:
            failures.append("completion_matrix_sha256")
    return not failures, failures


def _row(
    combination: SmokeCombination,
    summary: Mapping[str, object],
    *,
    passed: bool,
    failures: Sequence[str],
) -> dict[str, object]:
    data_errors = sum(
        int(name in failures)
        for name in (
            "missing_packets",
            "duplicate_packets",
            "out_of_order_packets",
            "malformed_packets",
            "gap_count",
        )
    )
    return {
        "model": combination.model,
        "window_sec": combination.window_seconds,
        "prepared_shape": list(summary.get("prepared_shape") or ()),
        "packets": _integer(summary, "received_packets"),
        "samples": _integer(summary, "received_samples"),
        "windows": _integer(summary, "emitted_windows"),
        "predictions": _integer(summary, "prediction_success_count"),
        "data_errors": data_errors,
        "compatibility": summary.get("compatibility_status"),
        "status": "passed" if passed else "failed",
        "failure_reasons": list(failures),
        "package_path": combination.package_relative_path,
    }


CSV_FIELDS = (
    "model",
    "window_sec",
    "prepared_shape",
    "packets",
    "samples",
    "windows",
    "predictions",
    "data_errors",
    "compatibility",
    "status",
    "failure_reasons",
    "package_path",
)


def _write_reports(output_dir: Path, rows: Sequence[Mapping[str, object]]) -> None:
    summary = {
        "schema_version": 1,
        "duration_sec": DURATION_SECONDS,
        "step_sec": STEP_SECONDS,
        "combination_count": len(rows),
        "passed_count": sum(row["status"] == "passed" for row in rows),
        "failed_count": sum(row["status"] != "passed" for row in rows),
        "all_passed": all(row["status"] == "passed" for row in rows),
        "waveforms_saved": False,
        "combinations": [dict(row) for row in rows],
    }
    (output_dir / "variable_window_smoke_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "variable_window_smoke_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            values = dict(row)
            values["prepared_shape"] = json.dumps(values["prepared_shape"])
            values["failure_reasons"] = ";".join(values["failure_reasons"])
            writer.writerow(values)
    markdown_rows = [
        "| Model | Window | Prepared shape | Packets | Samples | Windows | Predictions | Data errors | Compatibility | Status |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        markdown_rows.append(
            "| {model} | {window_sec:g}s | `{shape}` | {packets} | {samples} | "
            "{windows} | {predictions} | {data_errors} | {compatibility} | {status} |".format(
                shape=json.dumps(row["prepared_shape"]), **row
            )
        )
    (output_dir / "variable_window_smoke_summary.md").write_text(
        "\n".join(markdown_rows) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "stage2b" / "variable_window_smoke",
    )
    parser.add_argument("--port", type=int, default=8712)
    parser.add_argument("--duration-sec", type=float, default=DURATION_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not math.isclose(args.duration_sec, DURATION_SECONDS, abs_tol=0.0):
        raise ValueError("variable-window live smoke duration must be exactly 20 seconds")
    host = os.environ.get(HOST_ENV)
    if not host:
        raise ValueError(
            "Neuracle live host must be supplied through the "
            "NEURACLE_JELLYFISH_HOST environment variable"
        )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for combination in SMOKE_COMBINATIONS:
        package_path = ROOT / combination.package_relative_path
        summary_path = output_dir / combination.identifier / "runtime_inference_summary.json"
        expected_completion_sha: str | None = None
        stage = "package_preflight"
        try:
            if not package_path.is_dir():
                raise ValueError("Runtime Package directory is unavailable")
            if combination.model == "cbramod":
                expected_completion_sha = _read_cbramod_completion_sha(package_path)
            stage = "probe_invocation"
            exit_code = run_runtime_probe(
                [
                    "--package", str(package_path),
                    "--device", "cuda",
                    "--duration-sec", str(DURATION_SECONDS),
                    "--host", host,
                    "--port", str(args.port),
                    "--window-sec", str(combination.window_seconds),
                    "--step-sec", str(STEP_SECONDS),
                    "--output-dir", str(summary_path.parent),
                    "--no-save-waveform",
                ]
            )
            stage = "probe_summary"
            if not summary_path.is_file():
                raise ValueError("runtime probe did not write its summary")
            probe_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(probe_summary, Mapping):
                raise ValueError("runtime probe summary is not an object")
            stage = "gate_validation"
            passed, failures = _gate_result(
                combination,
                probe_summary,
                expected_completion_sha=expected_completion_sha,
            )
            if exit_code != 0:
                failures = [*failures, "probe_exit_code"]
                passed = False
            rows.append(_row(combination, probe_summary, passed=passed, failures=failures))
        except Exception as exc:
            # Continue deliberately so the report identifies every failing
            # combination.  The detail is a fixed sanitized error category.
            rows.append(
                {
                    "model": combination.model,
                    "window_sec": combination.window_seconds,
                    "prepared_shape": [],
                    "packets": 0,
                    "samples": 0,
                    "windows": 0,
                    "predictions": 0,
                    "data_errors": 0,
                    "compatibility": "not_checked",
                    "status": "failed",
                    "failure_reasons": [f"{stage}_{type(exc).__name__}"],
                    "package_path": _safe_relative_path(package_path),
                }
            )
    _write_reports(output_dir, rows)
    return 0 if all(row["status"] == "passed" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
