"""Privacy-conscious JSONL runtime logging that deliberately omits EEG samples."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from .contracts import EEGChunk, EventMarker, WindowResult
from .sync import EventSampleAlignment


class RealtimeRunLogger:
    """Write parseable operational logs without raw EEG or direct identifiers."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_chunk(self, chunk: EEGChunk) -> None:
        self._append(
            "chunks.jsonl",
            {
                "sequence_id": chunk.sequence_id,
                "sample_count": chunk.samples.shape[1],
                "channel_count": chunk.samples.shape[0],
                "sampling_rate": chunk.sampling_rate,
                "unit": chunk.unit,
                "timestamp_start": float(chunk.timestamps[0]),
                "timestamp_end": float(chunk.timestamps[-1]),
                "received_at": chunk.received_at,
                "device_id_hash": _anonymous_id(chunk.device_id),
            },
        )

    def log_event(self, event: EventMarker, alignment: EventSampleAlignment | None = None) -> None:
        payload: dict[str, object] = {
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "code": event.code,
            "sequence_id": event.sequence_id,
        }
        if alignment is not None:
            payload.update(
                {
                    "nearest_sample_index": alignment.sample_index,
                    "nearest_sequence_id": alignment.sequence_id,
                    "eeg_timestamp": alignment.eeg_timestamp,
                    "error_seconds": alignment.error_seconds,
                }
            )
        self._append("events.jsonl", payload)

    def log_window(self, result: WindowResult) -> None:
        payload: dict[str, object] = {
            "window_id": result.window_id,
            "status": result.status,
            "reason": result.reason,
            "emitted_at": result.emitted_at,
        }
        if result.window is not None:
            window = result.window
            payload.update(
                {
                    "start_sample_index": window.start_sample_index,
                    "end_sample_index": window.end_sample_index,
                    "sample_count": window.samples.shape[1],
                    "timestamp_start": float(window.timestamps[0]),
                    "timestamp_end": float(window.timestamps[-1]),
                    "source_sequence_start": window.source_sequence_start,
                    "source_sequence_end": window.source_sequence_end,
                    "event_count": len(window.markers),
                }
            )
        self._append("windows.jsonl", payload)

    def write_summary(self, summary: Mapping[str, object]) -> Path:
        target = self.output_dir / "runtime_summary.json"
        target.write_text(json.dumps(dict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def _append(self, filename: str, payload: Mapping[str, object]) -> None:
        with (self.output_dir / filename).open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _anonymous_id(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
