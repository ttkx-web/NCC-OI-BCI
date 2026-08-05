"""Probe an authorized Neuracle JellyFish backend without persisting EEG waveforms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Sequence

try:
    from _bootstrap import ROOT
except ModuleNotFoundError:  # Imported as scripts.probe_neuracle_realtime in tests.
    from scripts._bootstrap import ROOT

from bci_dayloop.realtime.neuracle_jellyfish import (
    NeuracleJellyFishConfig,
    NeuracleJellyFishSource,
    NeuracleSourceError,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8712, type=int)
    parser.add_argument("--duration-sec", default=10.0, type=float)
    parser.add_argument("--output-dir", default=ROOT / "runs" / "stage2b" / "neuracle_probe", type=Path)
    parser.add_argument("--expected-sfreq", type=float)
    parser.add_argument("--no-save-waveform", action="store_true")
    args = parser.parse_args(argv)
    if args.duration_sec <= 0:
        parser.error("--duration-sec must be positive")

    source = NeuracleJellyFishSource(
        NeuracleJellyFishConfig(
            host=args.host,
            port=args.port,
            expected_sampling_rate=args.expected_sfreq,
        )
    )
    summary: dict[str, object] = {
        "connection_state": None,
        "module_type": None,
        "channel_count": None,
        "channel_names": [],
        "channel_types": [],
        "sample_rates": [],
        "packet_count": 0,
        "sample_count": 0,
        "raw_timestamp_range": None,
        "timestamp_continuity": {"gaps": 0, "duplicate": 0, "out_of_order": 0},
        "trigger_count": 0,
        "reconnect_count": 0,
        "unit_status": {"raw_unit": "unknown", "unit_evidence_level": "realtime_unverified", "model_safe": False},
        "waveforms_saved": False,
    }
    exit_code = 0
    try:
        source.connect()
        metadata = source.metadata or {}
        summary.update(
            {
                "module_type": metadata.get("module_type"),
                "channel_count": metadata.get("forwarded_channel_count"),
                "channel_names": list(metadata.get("channel_names", ())),
                "channel_types": list(metadata.get("channel_types", ())),
                "sample_rates": list(metadata.get("sample_rates", ())),
            }
        )
        deadline = time.monotonic() + args.duration_sec
        timestamps: list[int] = []
        while time.monotonic() < deadline:
            chunk = source.read_chunk()
            if chunk is None:
                time.sleep(0.01)
                continue
            summary["packet_count"] = int(summary["packet_count"]) + 1
            summary["sample_count"] = int(summary["sample_count"]) + chunk.samples.shape[1]
            timestamps.extend(
                [
                    int(chunk.metadata["raw_start_timestamp"]),
                    int(chunk.metadata["raw_start_timestamp"]) + int(chunk.metadata["raw_timestamp_length"]),
                ]
            )
            while source.read_event() is not None:
                summary["trigger_count"] = int(summary["trigger_count"]) + 1
        if timestamps:
            summary["raw_timestamp_range"] = [min(timestamps), max(timestamps)]
    except NeuracleSourceError as exc:
        exit_code = 2
        summary["error"] = str(exc)
    finally:
        health = source.health()
        summary["connection_state"] = health["state"]
        summary["timestamp_continuity"] = {
            "gaps": health["missing_packets"],
            "duplicate": health["duplicate_packets"],
            "out_of_order": health["out_of_order_packets"],
        }
        summary["reconnect_count"] = health["reconnect_count"]
        summary["unit_status"] = {
            "raw_unit": source.config.raw_unit,
            "unit_evidence_level": health["unit_evidence_level"],
            "model_safe": health["model_safe"],
        }
        source.disconnect()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "probe_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
