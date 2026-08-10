import json
from pathlib import Path

import numpy as np
import pytest

from bci_dayloop.realtime.contracts import EEGChunk, EventMarker
from scripts.probe_neuracle_window_pipeline import main


class FakeWindowSource:
    instances: list["FakeWindowSource"] = []

    def __init__(self, config: object) -> None:
        self.config = config
        self.metadata = {
            "channel_types": ("EEG", "Trigger"),
            "channel_units": ("uV", "code"),
            "unit_evidence_level": "vendor_confirmed",
        }
        self.state = "ready"
        self.connected = False
        self.emitted = False
        self.events = [
            EventMarker(1.0, "trigger", code=4, metadata={"raw_device_timestamp": 1000}),
        ]
        type(self).instances.append(self)

    def connect(self) -> None:
        self.connected = True

    def read_chunk(self) -> EEGChunk | None:
        self.state = "streaming"
        if self.emitted:
            return None
        self.emitted = True
        return EEGChunk(
            samples=np.zeros((2, 4000), dtype=np.float32),
            channel_names=("C3", "Trigger"),
            sampling_rate=1000.0,
            unit="mixed",
            timestamps=np.arange(4000, dtype=np.float64) / 1000.0,
            sequence_id=0,
            device_id=None,
            received_at=0.0,
            metadata={
                "channel_types": ("EEG", "Trigger"),
                "channel_units": ("uV", "code"),
                "unit_evidence_level": "vendor_confirmed",
                "model_safe": False,
                "raw_start_timestamp": 0,
            },
        )

    def read_event(self) -> EventMarker | None:
        return self.events.pop(0) if self.events else None

    def health(self) -> dict[str, object]:
        return {
            "state": self.state,
            "connected": self.connected,
            "metadata_ready": self.connected,
            "received_packets": 1 if self.emitted else 0,
            "missing_packets": 0,
            "duplicate_packets": 0,
            "out_of_order_packets": 0,
            "malformed_packets": 0,
            "reconnect_count": 0,
            "last_error": None,
        }

    def disconnect(self) -> None:
        self.connected = False
        self.state = "stopped"


def test_window_probe_writes_only_anonymized_window_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    FakeWindowSource.instances.clear()
    monkeypatch.setattr(
        "scripts.probe_neuracle_window_pipeline.NeuracleJellyFishSource", FakeWindowSource
    )

    result = main(["--duration-sec", "0.01", "--output-dir", str(tmp_path), "--no-save-waveform"])
    summary = json.loads((tmp_path / "window_pipeline_summary.json").read_text(encoding="utf-8"))

    assert result == 0
    assert summary["status"] == "passed"
    assert summary["accepted_eeg_sample_count"] == 4000
    assert summary["eeg_channel_count"] == 1
    assert summary["stream_unit"] == "mixed"
    assert summary["eeg_unit"] == "uV"
    assert summary["unit_evidence_level"] == "vendor_confirmed"
    assert summary["expected_windows"] == summary["emitted_windows"] == 1
    assert summary["window_samples"] == 4000
    assert summary["step_samples"] == 500
    assert summary["marker_summaries"] == [
        {"code": 4, "window_id": 0, "window_relative_offset_ms": 1000.0}
    ]
    assert summary["final_health"]["state"] == "stopped"
    assert summary["final_health"]["connected"] is False
    assert summary["waveforms_saved"] is False
    assert "channel_names" not in summary
    assert set(summary["marker_summaries"][0]) == {"code", "window_id", "window_relative_offset_ms"}
    assert not list(tmp_path.glob("*.npy"))
