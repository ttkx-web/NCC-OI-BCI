"""Benchmark approved live Neuracle windows with Runtime Model Packages.

The script only orchestrates the existing source, selector, realtime window
pipeline, policy bridge, and model-agnostic benchmark core.  It never writes
EEG samples or host connection details.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

try:
    from _bootstrap import ROOT
except ModuleNotFoundError:
    from scripts._bootstrap import ROOT

from bci_dayloop.benchmarking.core import RuntimeBenchmarkCore
from bci_dayloop.benchmarking.reporting import (
    BenchmarkCandidate,
    build_candidate_summary,
    flatten_window_record,
    write_benchmark_summary_json,
    write_window_records_csv,
)
from bci_dayloop.benchmarking.windows import DeviceWindowProvider
from bci_dayloop.packages.loader import load_runtime_package
from bci_dayloop.realtime.neuracle_jellyfish import (
    NeuracleJellyFishConfig,
    NeuracleJellyFishSource,
)
from bci_dayloop.realtime.pipeline import RealtimeEEGWindowPipeline
from bci_dayloop.realtime.runtime_bridge import RealtimeRuntimeBridge
from bci_dayloop.realtime.runtime_policy import RealtimeModelPolicyRegistry
from bci_dayloop.realtime.window_contract import (
    APPROVED_REALTIME_WINDOW_SECONDS,
    NEURACLE_SOURCE_SAMPLING_RATE,
    REALTIME_STEP_SECONDS,
    validate_approved_realtime_window_contract,
)
from bci_dayloop.utils.config import load_yaml, resolve_path


def _require_mapping(payload: Mapping[str, object], name: str) -> dict[str, object]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"benchmark.{name} must be a mapping")
    return dict(value)


def _safe_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return f"<external>/{path.name}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _release_model(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _source_integrity_passes(integrity: Mapping[str, int]) -> bool:
    return all(
        int(integrity.get(name, 0)) == 0
        for name in (
            "missing_packets",
            "duplicate_packets",
            "out_of_order_packets",
            "malformed_packets",
            "reconnect_count",
            "gap_count",
            "buffer_overflow_count",
            "dropped_window_count",
        )
    )


def _candidate_status(summary: object, integrity: Mapping[str, int]) -> str:
    completed = int(getattr(summary, "completed_windows") or 0)
    expected = int(getattr(summary, "expected_windows") or 0)
    failed = int(getattr(summary, "failed_windows") or 0)
    latency = getattr(summary, "last_sample_received_to_prediction_ms")
    deadline = getattr(summary, "deadline_ms")
    p95 = None if latency is None else latency.get("p95")
    if (
        completed == expected
        and failed == 0
        and _source_integrity_passes(integrity)
        and isinstance(p95, float)
        and isinstance(deadline, float)
        and p95 < deadline
    ):
        return "PASS"
    return "FAIL"


def _schedule_contract(
    schedule: Mapping[str, object],
    candidates: object,
) -> tuple[float | None, tuple[float, ...]]:
    """Parse either the frozen 4 s schedule or package-driven grid mode."""
    if not isinstance(candidates, list):
        raise ValueError("device benchmark candidates must be a list")
    step_sec = float(schedule.get("step_sec", REALTIME_STEP_SECONDS))
    if step_sec != REALTIME_STEP_SECONDS:
        raise ValueError("approved device benchmark step_sec must be 0.5")
    if schedule.get("package_driven_windows") is True:
        allowed_raw = schedule.get("allowed_window_sec")
        if not isinstance(allowed_raw, list):
            raise ValueError("package-driven benchmark must declare allowed_window_sec")
        allowed = tuple(float(value) for value in allowed_raw)
        if allowed != APPROVED_REALTIME_WINDOW_SECONDS:
            raise ValueError("allowed_window_sec must be exactly [1.0, 2.0, 3.0, 4.0]")
        if len(candidates) != 12:
            raise ValueError("package-driven benchmark must contain exactly 12 candidates")
        return None, allowed

    window_sec = float(schedule.get("window_sec", 4.0))
    if window_sec != 4.0 or len(candidates) != 3:
        raise ValueError("frozen device benchmark is fixed at 4.0s / three candidates")
    return window_sec, (window_sec,)


def _prepared_contract(loaded: object) -> tuple[int, ...]:
    """Return the policy-validated prepared shape for manifest audit only."""
    model_type = str(getattr(loaded, "model_type"))
    runtime_model = getattr(loaded, "runtime_model")
    contract = runtime_model.input_contract
    window_sec = validate_approved_realtime_window_contract(
        contract, sampling_rate=float(contract.sample_rate)
    )
    if model_type == "model_50m":
        return (1, 64, int(window_sec * 100))
    if model_type == "labram":
        return (1, 19, int(window_sec), 200)
    if model_type == "cbramod":
        return (1, 22, int(window_sec), 200)
    raise ValueError(f"unsupported benchmark model_type: {model_type!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/benchmarks/window_latency_live_4s.yaml",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--no-verify-hashes", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = resolve_path(args.config)
    payload = load_yaml(config_path)
    benchmark = _require_mapping(payload, "benchmark")
    if benchmark.get("mode") != "device":
        raise ValueError("live benchmark config must set benchmark.mode=device")
    source_config = _require_mapping(benchmark, "source")
    schedule = _require_mapping(benchmark, "schedule")
    output = _require_mapping(benchmark, "output")
    host_env = str(source_config.get("host_env", "")).strip()
    host = os.environ.get(host_env) if host_env else None
    if not host:
        raise ValueError(
            f"live device host is required via environment variable {host_env!r}"
        )
    device = args.device or str(benchmark.get("device", "cuda"))
    candidates = benchmark.get("candidates")
    fixed_window_sec, allowed_window_seconds = _schedule_contract(
        schedule, candidates
    )
    step_sec = float(schedule.get("step_sec", REALTIME_STEP_SECONDS))
    warmup_windows = int(schedule.get("warmup_windows", 20))
    measured_windows = int(schedule.get("measured_windows", 200))
    duration_sec = float(
        args.duration_sec if args.duration_sec is not None else source_config.get("duration_sec", 150.0)
    )
    if warmup_windows != 20 or measured_windows != 200:
        raise ValueError("this approved device benchmark is fixed at warmup=20 / measured=200")
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    if float(source_config.get("expected_sampling_rate", 0.0)) != NEURACLE_SOURCE_SAMPLING_RATE:
        raise ValueError("approved device benchmark source must be 1000 Hz")
    assert isinstance(candidates, list)
    output_root = (
        resolve_path(args.output_root)
        if args.output_root is not None
        else resolve_path(str(output.get("root_dir")))
    )
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / run_id
    if output_dir.exists():
        raise FileExistsError(f"benchmark output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "schema_version": 1,
        "mode": "device",
        "device": device,
        "config_path": _safe_path(config_path),
        "config_sha256": _sha256(config_path),
        "source_contract": {
            "eeg_channels": 59,
            "sampling_rate": NEURACLE_SOURCE_SAMPLING_RATE,
            "window_sec": fixed_window_sec,
            "allowed_window_sec": list(allowed_window_seconds),
            "window_source": (
                "fixed_schedule" if fixed_window_sec is not None else "runtime_package"
            ),
            "step_sec": step_sec,
            "unit": "uV",
            "unit_evidence_level": "vendor_confirmed",
        },
        "warmup_windows": warmup_windows,
        "measured_windows": measured_windows,
        "waveforms_saved": False,
        "candidate_contracts": [],
    }
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    all_rows: list[dict[str, object]] = []
    summaries = []
    total_windows = warmup_windows + measured_windows
    for raw_candidate in candidates:
        if not isinstance(raw_candidate, Mapping):
            raise ValueError("candidate must be a mapping")
        candidate_id = str(raw_candidate.get("id", "")).strip()
        package_path = resolve_path(str(raw_candidate.get("package", "")))
        if not candidate_id or not package_path.is_dir():
            raise ValueError("candidate id and package path must be valid")
        loaded = None
        try:
            loaded = load_runtime_package(
                package_path,
                device=device,
                verify_hashes=not args.no_verify_hashes,
            )
            if loaded.is_test_head:
                raise ValueError("test heads are forbidden for device benchmark")
            package_window_sec = validate_approved_realtime_window_contract(
                loaded.runtime_model.input_contract,
                sampling_rate=float(loaded.runtime_model.input_contract.sample_rate),
            )
            if package_window_sec not in allowed_window_seconds:
                raise ValueError("package window contract is not approved for this benchmark")
            if fixed_window_sec is not None and package_window_sec != fixed_window_sec:
                raise ValueError("package window contract does not match frozen benchmark")
            if loaded.step_sec != step_sec:
                raise ValueError("package step_sec does not match device benchmark")
            policy = RealtimeModelPolicyRegistry.create(loaded)
            bridge = RealtimeRuntimeBridge(loaded.runtime_model, policy=policy)
            source = NeuracleJellyFishSource(
                NeuracleJellyFishConfig(
                    host=host,
                    port=int(source_config.get("port", 8712)),
                    expected_sampling_rate=float(source_config.get("expected_sampling_rate", 1000.0)),
                )
            )
            pipeline = RealtimeEEGWindowPipeline.from_runtime_input_contract(
                loaded.runtime_model.input_contract,
                sampling_rate=NEURACLE_SOURCE_SAMPLING_RATE,
                step_seconds=step_sec,
            )
            manifest["candidate_contracts"].append(
                {
                    "candidate_id": candidate_id,
                    "model_type": loaded.model_type,
                    "package_path": _safe_path(package_path),
                    "window_sec": package_window_sec,
                    "source_shape": [59, round(package_window_sec * 1000)],
                    "prepared_shape": list(_prepared_contract(loaded)),
                }
            )
            provider = DeviceWindowProvider(
                source=source,
                pipeline=pipeline,
                bridge=bridge,
                duration_sec=duration_sec,
                maximum_windows=total_windows,
            )
            records = RuntimeBenchmarkCore(
                runtime_model=loaded.runtime_model,
                device=device,
            ).run(
                provider=provider,
                warmup_windows=warmup_windows,
                measured_windows=measured_windows,
            )
            integrity = provider.source_integrity
            candidate = BenchmarkCandidate(
                candidate_id=candidate_id,
                model_name=loaded.model_name,
                model_type=loaded.model_type,
                package_path=_safe_path(package_path),
                package_sha256=_sha256(package_path / "package.yaml"),
                window_sec=package_window_sec,
                step_sec=step_sec,
                device=device,
                source_mode="device",
                warmup_windows=warmup_windows,
                measured_windows=measured_windows,
            )
            summary = build_candidate_summary(
                candidate=candidate,
                records=records,
                deadline_ms=step_sec * 1000.0,
                expected_windows=measured_windows,
                failed_windows=integrity["pipeline_failed_windows"],
                source_integrity=integrity,
            )
            summary = replace(
                summary,
                status=_candidate_status(summary, integrity),
            )
            if summary.status != "PASS":
                raise RuntimeError(f"device benchmark candidate failed PASS gate: {candidate_id}")
            all_rows.extend(
                flatten_window_record(
                    candidate=candidate,
                    record=record,
                    deadline_ms=step_sec * 1000.0,
                )
                for record in records
            )
            summaries.append(summary)
        finally:
            del loaded
            _release_model(device)

    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    write_window_records_csv(path=output_dir / "window_records.csv", rows=all_rows)
    write_benchmark_summary_json(
        path=output_dir / "summary.json",
        summaries=summaries,
        benchmark_metadata={
            "mode": "device",
            "warmup_windows": warmup_windows,
            "measured_windows": measured_windows,
            "deadline_ms": step_sec * 1000.0,
            "waveforms_saved": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
