"""Probe the verified JellyFish EEG window pipeline without persisting EEG samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

try:
    from _bootstrap import ROOT
except ModuleNotFoundError:
    from scripts._bootstrap import ROOT

from bci_dayloop.realtime.channel_units import unit_status_from_metadata
from bci_dayloop.realtime.channel_units import select_verified_eeg_channels
from bci_dayloop.realtime.neuracle_jellyfish import (
    NeuracleJellyFishConfig,
    NeuracleJellyFishSource,
    NeuracleSourceError,
)
from bci_dayloop.realtime.pipeline import RealtimeEEGWindowPipeline, RealtimePipelineError


def _health_summary(health: Mapping[str, object]) -> dict[str, object]:
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8712, type=int)
    parser.add_argument("--duration-sec", default=10.0, type=float)
    parser.add_argument("--expected-sfreq", default=1000.0, type=float)
    parser.add_argument("--window-sec", default=4.0, type=float)
    parser.add_argument("--step-sec", default=0.5, type=float)
    parser.add_argument("--output-dir", default=ROOT / "runs" / "stage2b" / "neuracle_window_probe", type=Path)
    parser.add_argument("--no-save-waveform", action="store_true")
    args = parser.parse_args(argv)
    if args.duration_sec <= 0:
        parser.error("--duration-sec must be positive")

    source: NeuracleJellyFishSource | None = None
    pipeline = RealtimeEEGWindowPipeline(
        sampling_rate=args.expected_sfreq,
        window_seconds=args.window_sec,
        step_seconds=args.step_sec,
    )
    marker_summaries: list[dict[str, object]] = []
    emitted_ids: set[int] = set()
    duplicate_window_ids = 0
    unit_status: dict[str, object] = {
        "stream_unit": "mixed",
        "eeg_unit": "unknown",
        "unit_evidence_level": None,
        "raw_model_safe": False,
        "eeg_model_safe": False,
    }
    summary: dict[str, object] = {
        "status": "failed",
        "duration_sec": args.duration_sec,
        "packet_count": 0,
        "accepted_eeg_sample_count": 0,
        "eeg_channel_count": 0,
        "sampling_rate": args.expected_sfreq,
        "stream_unit": unit_status["stream_unit"],
        "eeg_unit": unit_status["eeg_unit"],
        "unit_evidence_level": unit_status["unit_evidence_level"],
        "raw_model_safe": unit_status["raw_model_safe"],
        "eeg_model_safe": unit_status["eeg_model_safe"],
        "expected_windows": 0,
        "emitted_windows": 0,
        "failed_windows": 0,
        "window_completion_rate": None,
        "duplicate_window_ids": 0,
        "window_samples": pipeline.window_samples,
        "step_samples": pipeline.step_samples,
        "contiguous_segment_count": 0,
        "timestamp_gap_count": 0,
        "buffer_peak_samples": 0,
        "buffer_overflow_count": 0,
        "unique_trigger_count": 0,
        "marker_window_association_count": 0,
        "marker_summaries": marker_summaries,
        "waveforms_saved": False,
        "last_error": None,
        "failure_reason": None,
    }
    exit_code = 0
    try:
        source = NeuracleJellyFishSource(
            NeuracleJellyFishConfig(
                host=args.host,
                port=args.port,
                expected_sampling_rate=args.expected_sfreq,
            )
        )
        source.connect()
        metadata = source.metadata or {}
        unit_status = unit_status_from_metadata(metadata)
        summary.update(
            {
                "eeg_channel_count": sum(
                    str(channel_type).strip().casefold() == "eeg"
                    for channel_type in metadata.get("channel_types", ())
                ),
                "stream_unit": unit_status["stream_unit"],
                "eeg_unit": unit_status["eeg_unit"],
                "unit_evidence_level": unit_status["unit_evidence_level"],
                "raw_model_safe": unit_status["raw_model_safe"],
                "eeg_model_safe": unit_status["eeg_model_safe"],
            }
        )
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
            results = pipeline.process(eeg_chunk, markers)
            summary["packet_count"] = int(summary["packet_count"]) + 1
            for result in results:
                if result.window_id in emitted_ids:
                    duplicate_window_ids += 1
                emitted_ids.add(result.window_id)
                if result.window is None:
                    continue
                for event in result.window.markers:
                    if len(marker_summaries) >= 64:
                        break
                    marker_summaries.append(
                        {
                            "code": event.code,
                            "window_id": result.window_id,
                            "window_relative_offset_ms": round(
                                (event.timestamp - float(result.window.timestamps[0])) * 1000.0,
                                6,
                            ),
                        }
                    )
    except KeyboardInterrupt:
        exit_code = 130
        summary["last_error"] = "present"
    except (NeuracleSourceError, RealtimePipelineError, ValueError):
        exit_code = 2
        summary["last_error"] = "present"
    finally:
        pre_health: dict[str, object] = {}
        if source is not None:
            pre_health = _health_summary(source.health())
            summary["pre_disconnect_health"] = pre_health
            try:
                source.disconnect()
            except Exception:
                exit_code = 2
                summary["last_error"] = "present"
            finally:
                summary["final_health"] = _health_summary(source.health())
        summary.update(
            {
                "accepted_eeg_sample_count": pipeline.accepted_eeg_sample_count,
                "expected_windows": pipeline.expected_windows,
                "emitted_windows": pipeline.emitted_windows,
                "failed_windows": pipeline.failed_windows,
                "duplicate_window_ids": duplicate_window_ids,
                "contiguous_segment_count": pipeline.contiguous_segment_count,
                "timestamp_gap_count": pipeline.timestamp_gap_count,
                "buffer_peak_samples": pipeline.buffer_peak_samples,
                "buffer_overflow_count": pipeline.buffer_overflow_count,
                "unique_trigger_count": pipeline.unique_trigger_count,
                "marker_window_association_count": pipeline.marker_window_association_count,
                "failure_reason": pipeline.last_failure_reason,
            }
        )
        expected = pipeline.expected_windows
        summary["window_completion_rate"] = (
            pipeline.emitted_windows / expected if expected else None
        )
        source_counters = pre_health
        success = (
            exit_code == 0
            and pipeline.expected_windows == pipeline.emitted_windows
            and pipeline.failed_windows == 0
            and duplicate_window_ids == 0
            and pipeline.buffer_overflow_count == 0
            and pipeline.timestamp_gap_count == 0
            and unit_status["eeg_unit"] == "uV"
            and unit_status["unit_evidence_level"] == "vendor_confirmed"
            and unit_status["eeg_model_safe"] is True
            and source_counters.get("missing_packets", 0) == 0
            and source_counters.get("duplicate_packets", 0) == 0
            and source_counters.get("out_of_order_packets", 0) == 0
            and source_counters.get("malformed_packets", 0) == 0
            and summary.get("final_health", {}).get("state") == "stopped"
            and summary.get("final_health", {}).get("connected") is False
        )
        summary["status"] = "passed" if success else "failed"
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "window_pipeline_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        finally:
            pipeline.close()
    return exit_code if not summary["status"] == "failed" or exit_code else 2


if __name__ == "__main__":
    raise SystemExit(main())
