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
    window_sec = float(schedule.get("window_sec", 4.0))
    step_sec = float(schedule.get("step_sec", 0.5))
    warmup_windows = int(schedule.get("warmup_windows", 20))
    measured_windows = int(schedule.get("measured_windows", 200))
    duration_sec = float(
        args.duration_sec if args.duration_sec is not None else source_config.get("duration_sec", 150.0)
    )
    if window_sec != 4.0 or step_sec != 0.5:
        raise ValueError("this approved device benchmark is fixed at 4.0s / 0.5s")
    if warmup_windows != 20 or measured_windows != 200:
        raise ValueError("this approved device benchmark is fixed at warmup=20 / measured=200")
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    candidates = benchmark.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise ValueError("device benchmark must contain the three fixed candidates")
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
            "sampling_rate": 1000.0,
            "window_sec": window_sec,
            "step_sec": step_sec,
            "unit": "uV",
            "unit_evidence_level": "vendor_confirmed",
        },
        "warmup_windows": warmup_windows,
        "measured_windows": measured_windows,
        "waveforms_saved": False,
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
            if loaded.window_sec != window_sec or loaded.step_sec != step_sec:
                raise ValueError("package window/step contract does not match device benchmark")
            policy = RealtimeModelPolicyRegistry.create(loaded)
            bridge = RealtimeRuntimeBridge(loaded.runtime_model, policy=policy)
            source = NeuracleJellyFishSource(
                NeuracleJellyFishConfig(
                    host=host,
                    port=int(source_config.get("port", 8712)),
                    expected_sampling_rate=float(source_config.get("expected_sampling_rate", 1000.0)),
                )
            )
            pipeline = RealtimeEEGWindowPipeline(
                sampling_rate=1000.0,
                window_seconds=window_sec,
                step_seconds=step_sec,
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
                window_sec=window_sec,
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
