"""Probe an authorized Neuracle JellyFish backend without persisting EEG waveforms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

try:
    from _bootstrap import ROOT
except ModuleNotFoundError:  # Imported as scripts.probe_neuracle_realtime in tests.
    from scripts._bootstrap import ROOT

from bci_dayloop.realtime.neuracle_jellyfish import (
    NeuracleJellyFishConfig,
    NeuracleJellyFishSource,
    NeuracleSourceError,
)


_SAFE_MODULE_TYPES = frozenset({"eeg", "neuracle", "jellyfish"})


def _safe_module_type(value: object) -> str | None:
    """Keep only known generic module types; META module names are never exported."""
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if normalized.casefold() in _SAFE_MODULE_TYPES else None


def _summary_health(health: Mapping[str, object]) -> dict[str, object]:
    """Copy only non-identifying operational fields from a Source health report."""
    return {
        "state": health.get("state"),
        "connected": bool(health.get("connected", False)),
        "metadata_ready": bool(health.get("metadata_ready", False)),
        "received_packets": int(health.get("received_packets", 0)),
        "malformed_packets": int(health.get("malformed_packets", 0)),
        "missing_packets": int(health.get("missing_packets", 0)),
        "duplicate_packets": int(health.get("duplicate_packets", 0)),
        "out_of_order_packets": int(health.get("out_of_order_packets", 0)),
        "reconnect_count": int(health.get("reconnect_count", 0)),
        "unit_evidence_level": health.get("unit_evidence_level"),
        "model_safe": bool(health.get("model_safe", False)),
        "last_error_present": health.get("last_error") is not None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8712, type=int)
    parser.add_argument("--duration-sec", default=10.0, type=float)
    parser.add_argument("--output-dir", default=ROOT / "runs" / "stage2b" / "neuracle_probe", type=Path)
    parser.add_argument("--expected-sfreq", type=float)
    parser.add_argument("--no-save-waveform", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args(argv)
    if args.duration_sec <= 0:
        parser.error("--duration-sec must be positive")

    source: NeuracleJellyFishSource | None = None
    summary: dict[str, object] = {
        "connection_state": None,
        "metadata_ready": False,
        "module_name": None,  # Never persist META module names: they can embed a device identifier.
        "module_type": None,
        "anonymized_serial_hash": None,
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
        "last_error": None,
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
        summary.update(
            {
                "metadata_ready": True,
                "module_name": None,
                "module_type": _safe_module_type(metadata.get("module_type")),
                "anonymized_serial_hash": metadata.get("serial_number_hash"),
                "channel_count": metadata.get("forwarded_channel_count"),
                "channel_names": list(metadata.get("channel_names", ())),
                "channel_types": list(metadata.get("channel_types", ())),
                "sample_rates": list(metadata.get("sample_rates", ())),
            }
        )
        if not args.metadata_only:
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
    except KeyboardInterrupt:
        exit_code = 130
        summary["error"] = "probe_interrupted"
    except (NeuracleSourceError, ValueError):
        exit_code = 2
        summary["error"] = "probe_failed"
    except Exception:
        exit_code = 2
        summary["error"] = "probe_failed"
    finally:
        if source is not None:
            try:
                pre_disconnect_health = _summary_health(source.health())
            except Exception:
                pre_disconnect_health = _summary_health({})
                pre_disconnect_health["state"] = "unknown"
                pre_disconnect_health["last_error_present"] = True
            summary["pre_disconnect_health"] = pre_disconnect_health
            try:
                source.disconnect()
            except Exception:
                exit_code = 2
                summary["error"] = "probe_cleanup_failed"
            finally:
                try:
                    final_health = _summary_health(source.health())
                except Exception:
                    final_health = _summary_health({})
                    final_health["state"] = "unknown"
                    final_health["last_error_present"] = True
            summary["connection_state"] = final_health["state"]
            summary["metadata_ready"] = final_health["metadata_ready"]
            summary["timestamp_continuity"] = {
                "gaps": pre_disconnect_health["missing_packets"],
                "duplicate": pre_disconnect_health["duplicate_packets"],
                "out_of_order": pre_disconnect_health["out_of_order_packets"],
            }
            summary["reconnect_count"] = pre_disconnect_health["reconnect_count"]
            summary["last_error"] = "present" if pre_disconnect_health["last_error_present"] else None
            summary["final_health"] = final_health
            summary["unit_status"] = {
                "raw_unit": source.config.raw_unit,
                "unit_evidence_level": final_health["unit_evidence_level"],
                "model_safe": final_health["model_safe"],
            }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "probe_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
