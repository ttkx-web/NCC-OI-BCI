"""Run the approved Neuracle realtime path through a Runtime Package.

The probe never writes EEG samples or device-identifying connection metadata.
It uses the existing source, unit selector, timestamped window pipeline, and
Runtime prepare bridge.  Prediction is attempted only after the prepared-input
gate has marked a window safe.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np

try:
    from _bootstrap import ROOT
except ModuleNotFoundError:
    from scripts._bootstrap import ROOT

from bci_dayloop.packages.loader import LoadedRuntimePackage, load_runtime_package
from bci_dayloop.realtime.channel_units import select_verified_eeg_channels
from bci_dayloop.realtime.neuracle_jellyfish import (
    NeuracleJellyFishConfig,
    NeuracleJellyFishSource,
    NeuracleSourceError,
)
from bci_dayloop.realtime.pipeline import RealtimeEEGWindowPipeline, RealtimePipelineError
from bci_dayloop.realtime.runtime_bridge import RealtimeRuntimeBridge
from bci_dayloop.realtime.runtime_policy import (
    RealtimeModelPolicy,
    RealtimeModelPolicyRegistry,
    RealtimePolicyError,
)
from bci_dayloop.realtime.window_contract import REALTIME_STEP_SECONDS


def _health_summary(health: Mapping[str, object]) -> dict[str, object]:
    """Keep operational counters but never copy backend identity or endpoint data."""
    return {
        "state": health.get("state"),
        "connected": bool(health.get("connected", False)),
        "metadata_ready": bool(health.get("metadata_ready", False)),
        "received_packets": int(health.get("received_packets", 0)),
        "missing_packets": int(health.get("missing_packets", 0)),
        "duplicate_packets": int(health.get("duplicate_packets", 0)),
        "out_of_order_packets": int(health.get("out_of_order_packets", 0)),
        "malformed_packets": int(health.get("malformed_packets", 0)),
        "reconnect_count": int(health.get("reconnect_count", 0)),
        "last_error_present": health.get("last_error") is not None,
    }


def _safe_package_info(package: LoadedRuntimePackage) -> dict[str, object]:
    """Expose package identity without writing an absolute local path."""
    package_metadata = package.package_metadata.get("package", {})
    package_id = package_metadata.get("id") if isinstance(package_metadata, Mapping) else None
    try:
        package_path = package.package_path.relative_to(ROOT).as_posix()
    except ValueError:
        package_path = f"<external>/{package.package_path.name}"
    return {
        "package_id": str(package_id) if package_id is not None else None,
        "package_path": package_path,
        "model_type": package.model_type,
        "model_name": package.model_name,
        "is_test_head": package.is_test_head,
    }


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "max_ms": float(array.max()),
    }


def _prediction_record(
    *,
    prepared_summary: Mapping[str, object],
    predicted_class: int,
    predicted_name: str,
    confidence: float,
    probabilities: list[float],
    inference_latency_ms: float,
    model_type: str,
    model_name: str,
    package_id: str | None,
    realtime_policy_id: str,
) -> dict[str, object]:
    """Create the intentionally metadata-only output for one successful window."""
    prepare_latency = prepared_summary.get("prepare_latency_ms")
    record = {
        "window_id": prepared_summary["window_id"],
        "continuous_segment_id": prepared_summary["continuous_segment_id"],
        "source_shape": prepared_summary["source_shape"],
        "prepared_shape": prepared_summary["prepared_signal_shape"],
        "valid_channel_count": prepared_summary["valid_channel_count"],
        "model_input_safe": True,
        "predicted_class": predicted_class,
        "predicted_name": predicted_name,
        "confidence": confidence,
        "probabilities": probabilities,
        "prepare_latency_ms": prepare_latency,
        "inference_latency_ms": inference_latency_ms,
        "total_model_latency_ms": float(prepare_latency) + inference_latency_ms,
        "marker_summary": prepared_summary["marker_summary"],
        "model_type": model_type,
        "model_name": model_name,
        "package_id": package_id,
        "realtime_policy_id": realtime_policy_id,
    }
    policy_metadata = prepared_summary.get("policy_metadata")
    if isinstance(policy_metadata, Mapping):
        for key in (
            "observed_channel_count",
            "missing_channel_names",
            "completion_policy",
            "completion_matrix_sha256",
        ):
            if key in policy_metadata:
                record[key] = policy_metadata[key]
    return record


def _prediction_values(output: object, *, class_names: tuple[str, ...]) -> tuple[int, str, float, list[float]]:
    predicted_class = int(getattr(output, "predicted_class"))
    confidence = float(getattr(output, "confidence"))
    tensor = getattr(output, "probabilities")
    if not hasattr(tensor, "detach"):
        raise ValueError("Runtime prediction probabilities must be a tensor")
    probabilities = tensor.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    if len(probabilities) != len(class_names) or not np.isfinite(probabilities).all():
        raise ValueError("Runtime prediction probability vector is invalid")
    if not math.isfinite(confidence) or not 0 <= predicted_class < len(class_names):
        raise ValueError("Runtime prediction class or confidence is invalid")
    return predicted_class, class_names[predicted_class], confidence, [float(value) for value in probabilities]


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path, help="Runtime Package directory")
    parser.add_argument("--device", default="cpu", help="Runtime Package device, e.g. cpu or cuda")
    parser.add_argument("--duration-sec", default=10.0, type=float)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8712, type=int)
    parser.add_argument("--expected-sfreq", default=1000.0, type=float)
    parser.add_argument(
        "--window-sec",
        default=None,
        type=float,
        help="Optional assertion; must exactly match the Runtime Package window_sec.",
    )
    parser.add_argument("--step-sec", default=REALTIME_STEP_SECONDS, type=float)
    parser.add_argument("--output-dir", default=ROOT / "runs" / "stage2b" / "neuracle_runtime_inference", type=Path)
    parser.add_argument("--no-save-waveform", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration_sec) or args.duration_sec <= 0:
        parser.error("--duration-sec must be positive and finite")

    source: NeuracleJellyFishSource | None = None
    package: LoadedRuntimePackage | None = None
    policy: RealtimeModelPolicy | None = None
    pipeline: RealtimeEEGWindowPipeline | None = None
    prepared_latencies: list[float] = []
    inference_latencies: list[float] = []
    total_latencies: list[float] = []
    exit_code = 0
    summary: dict[str, object] = {
        "status": "failed",
        "duration_sec": args.duration_sec,
        "window_sec": None,
        "step_sec": args.step_sec,
        "package": None,
        "package_id": None,
        "model_type": None,
        "model_name": None,
        "realtime_policy_id": None,
        "prepared_shape": None,
        "compatibility_status": "not_checked",
        "compatibility_error": None,
        "observed_channel_count": None,
        "missing_channel_names": None,
        "completion_policy": None,
        "completion_matrix_sha256": None,
        "is_test_head": None,
        "received_packets": 0,
        "received_samples": 0,
        "missing_packets": 0,
        "emitted_windows": 0,
        "failed_windows": 0,
        "pipeline_failed_windows": 0,
        "gap_count": 0,
        "duplicate_packets": 0,
        "out_of_order_packets": 0,
        "model_input_safe_count": 0,
        "model_input_failure_count": 0,
        "prediction_success_count": 0,
        "prediction_failure_count": 0,
        "prepare_latency": _latency_summary(prepared_latencies),
        "inference_latency": _latency_summary(inference_latencies),
        "total_model_latency": _latency_summary(total_latencies),
        "max_latency_ms": None,
        "waveforms_saved": False,
        "last_error": None,
    }

    try:
        package = load_runtime_package(args.package, device=args.device)
        package_info = _safe_package_info(package)
        summary["package"] = package_info
        summary["package_id"] = package_info["package_id"]
        summary["model_type"] = package.model_type
        summary["model_name"] = package.model_name
        summary["is_test_head"] = package.is_test_head
        if package.is_test_head:
            raise ValueError("Runtime Package test head is not allowed for a live inference probe")
        policy = RealtimeModelPolicyRegistry.create(package)
        summary["realtime_policy_id"] = policy.policy_id
        package_window_sec = package.runtime_model.input_contract.window_sec
        summary["window_sec"] = package_window_sec
        if (
            args.window_sec is not None
            and not math.isclose(args.window_sec, package_window_sec, abs_tol=0.0)
        ):
            raise RealtimePolicyError(
                "--window-sec must match the Runtime Package window_sec. BLOCKED."
            )
        if not math.isclose(args.step_sec, REALTIME_STEP_SECONDS, abs_tol=0.0):
            raise RealtimePolicyError(
                "Stage 2B realtime --step-sec must be exactly 0.5. BLOCKED."
            )
        if not math.isclose(package.step_sec, REALTIME_STEP_SECONDS, abs_tol=0.0):
            raise RealtimePolicyError(
                "Runtime Package step_sec must match --step-sec. BLOCKED."
            )
        pipeline = RealtimeEEGWindowPipeline.from_runtime_input_contract(
            package.runtime_model.input_contract,
            sampling_rate=args.expected_sfreq,
            step_seconds=args.step_sec,
        )
        summary["compatibility_status"] = "passed"

        bridge = RealtimeRuntimeBridge(
            package.runtime_model,
            policy=policy,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = args.output_dir / "runtime_predictions.jsonl"
        source = NeuracleJellyFishSource(
            NeuracleJellyFishConfig(
                host=args.host,
                port=args.port,
                expected_sampling_rate=args.expected_sfreq,
            )
        )
        source.connect()
        deadline = time.monotonic() + args.duration_sec
        while time.monotonic() < deadline:
            raw_chunk = source.read_chunk()
            if raw_chunk is None:
                time.sleep(0.001)
                continue
            markers = []
            marker = source.read_event()
            while marker is not None:
                markers.append(marker)
                marker = source.read_event()
            eeg_chunk = select_verified_eeg_channels(raw_chunk)
            for result in pipeline.process(eeg_chunk, markers):
                if result.window is None:
                    continue
                prepared = bridge.prepare(result.window)
                if not prepared.model_input_safe or prepared.prepared_input is None:
                    summary["model_input_failure_count"] = int(summary["model_input_failure_count"]) + 1
                    summary["last_error"] = "prepared_input_gate_failed"
                    continue
                summary["model_input_safe_count"] = int(summary["model_input_safe_count"]) + 1
                summary["prepared_shape"] = (
                    list(prepared.prepared_signal_shape)
                    if prepared.prepared_signal_shape is not None
                    else None
                )
                for key, value in prepared.policy_metadata.items():
                    if key in {
                        "observed_channel_count",
                        "missing_channel_names",
                        "completion_policy",
                        "completion_matrix_sha256",
                    }:
                        summary[key] = value
                prepare_latency = prepared.prepare_latency_ms
                assert prepare_latency is not None
                prepared_latencies.append(prepare_latency)
                inference_started = time.perf_counter()
                try:
                    output = package.runtime_model.predict_prepared(prepared.prepared_input)
                    inference_latency = (time.perf_counter() - inference_started) * 1000.0
                    predicted_class, predicted_name, confidence, probabilities = _prediction_values(
                        output,
                        class_names=package.class_names,
                    )
                except Exception:
                    summary["prediction_failure_count"] = int(summary["prediction_failure_count"]) + 1
                    summary["last_error"] = "runtime_prediction_failed"
                    continue
                inference_latencies.append(inference_latency)
                total_latency = prepare_latency + inference_latency
                total_latencies.append(total_latency)
                _append_jsonl(
                    predictions_path,
                    _prediction_record(
                        prepared_summary=prepared.to_summary(),
                        predicted_class=predicted_class,
                        predicted_name=predicted_name,
                        confidence=confidence,
                        probabilities=probabilities,
                        inference_latency_ms=inference_latency,
                        model_type=package.model_type,
                        model_name=package.model_name,
                        package_id=(
                            str(summary["package_id"])
                            if summary["package_id"] is not None
                            else None
                        ),
                        realtime_policy_id=policy.policy_id,
                    ),
                )
                summary["prediction_success_count"] = int(summary["prediction_success_count"]) + 1
    except KeyboardInterrupt:
        exit_code = 130
        summary["last_error"] = "interrupted"
    except RealtimePolicyError as exc:
        exit_code = 2
        summary["compatibility_status"] = "blocked"
        summary["compatibility_error"] = str(exc)
        summary["last_error"] = "realtime_policy_blocked"
    except (NeuracleSourceError, RealtimePipelineError, ValueError, FileNotFoundError):
        exit_code = 2
        summary["last_error"] = "probe_failed"
    finally:
        pre_health: dict[str, object] = {}
        if source is not None:
            pre_health = _health_summary(source.health())
            summary["pre_disconnect_health"] = pre_health
            try:
                source.disconnect()
            except Exception:
                exit_code = 2
                summary["last_error"] = "disconnect_failed"
            finally:
                summary["final_health"] = _health_summary(source.health())
        summary.update(
            {
                "received_packets": int(pre_health.get("received_packets", 0)),
                "received_samples": (
                    pipeline.accepted_eeg_sample_count if pipeline else 0
                ),
                "missing_packets": int(pre_health.get("missing_packets", 0)),
                "emitted_windows": pipeline.emitted_windows if pipeline else 0,
                "pipeline_failed_windows": pipeline.failed_windows if pipeline else 0,
                "failed_windows": (
                    (pipeline.failed_windows if pipeline else 0)
                    + int(summary["model_input_failure_count"])
                    + int(summary["prediction_failure_count"])
                ),
                "gap_count": pipeline.timestamp_gap_count if pipeline else 0,
                "duplicate_packets": int(pre_health.get("duplicate_packets", 0)),
                "out_of_order_packets": int(pre_health.get("out_of_order_packets", 0)),
                "prepare_latency": _latency_summary(prepared_latencies),
                "inference_latency": _latency_summary(inference_latencies),
                "total_model_latency": _latency_summary(total_latencies),
                "max_latency_ms": max(total_latencies) if total_latencies else None,
            }
        )
        success = (
            exit_code == 0
            and package is not None
            and policy is not None
            and package.is_test_head is False
            and pipeline is not None
            and pipeline.failed_windows == 0
            and pipeline.timestamp_gap_count == 0
            and pre_health.get("missing_packets", 0) == 0
            and pre_health.get("duplicate_packets", 0) == 0
            and pre_health.get("out_of_order_packets", 0) == 0
            and int(summary["model_input_failure_count"]) == 0
            and int(summary["prediction_failure_count"]) == 0
            and int(summary["model_input_safe_count"]) == pipeline.emitted_windows
            and int(summary["prediction_success_count"]) == pipeline.emitted_windows
            and summary.get("final_health", {}).get("state") == "stopped"
            and summary.get("final_health", {}).get("connected") is False
        )
        summary["status"] = "passed" if success else "failed"
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "runtime_inference_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        finally:
            if pipeline is not None:
                pipeline.close()
    return exit_code if summary["status"] == "passed" else exit_code or 2


if __name__ == "__main__":
    raise SystemExit(main())
