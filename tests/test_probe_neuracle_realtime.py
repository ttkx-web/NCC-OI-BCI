import json
from pathlib import Path
import subprocess
import sys

import pytest

from bci_dayloop.realtime.neuracle_jellyfish import NeuracleSourceError
from scripts.probe_neuracle_realtime import main


def test_probe_cli_help_runs_directly() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/probe_neuracle_realtime.py", "--help"],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--no-save-waveform" in result.stdout


def test_probe_without_forwarder_fails_safely_and_logs_no_sensitive_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class UnavailableSource:
        def __init__(self, config: object) -> None:
            self.config = config

        def connect(self) -> None:
            raise NeuracleSourceError("forwarder unavailable")

        def health(self) -> dict[str, object]:
            return {
                "state": "failed",
                "metadata_ready": False,
                "missing_packets": 0,
                "duplicate_packets": 0,
                "out_of_order_packets": 0,
                "reconnect_count": 0,
                "unit_evidence_level": "realtime_unverified",
                "model_safe": False,
                "last_error": "forwarder unavailable",
            }

        def disconnect(self) -> None:
            pass

    monkeypatch.setattr("scripts.probe_neuracle_realtime.NeuracleJellyFishSource", UnavailableSource)
    result = main(["--duration-sec", "0.1", "--output-dir", str(tmp_path)])
    summary = json.loads((tmp_path / "probe_summary.json").read_text(encoding="utf-8"))

    assert result == 2
    assert summary["waveforms_saved"] is False
    assert summary["unit_status"]["model_safe"] is False
    encoded = json.dumps(summary)
    assert "personName" not in encoded
    assert "serialNumber" not in encoded


def test_probe_metadata_only_writes_anonymized_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class MetadataOnlySource:
        read_called = False

        def __init__(self, config: object) -> None:
            self.config = config
            self.metadata = {
                "module_name": "JellyFish",
                "module_type": "EEG",
                "serial_number_hash": "c0ffee123456",
                "forwarded_channel_count": 2,
                "channel_names": ("C3", "C4"),
                "channel_types": ("EEG", "Trigger"),
                "sample_rates": (250.0, 250.0),
            }

        def connect(self) -> None:
            pass

        def read_chunk(self) -> object:
            type(self).read_called = True
            raise AssertionError("metadata-only must not read EEG")

        def health(self) -> dict[str, object]:
            return {
                "state": "ready",
                "metadata_ready": True,
                "missing_packets": 0,
                "duplicate_packets": 0,
                "out_of_order_packets": 0,
                "reconnect_count": 0,
                "unit_evidence_level": "realtime_unverified",
                "model_safe": False,
                "last_error": None,
            }

        def disconnect(self) -> None:
            pass

    monkeypatch.setattr("scripts.probe_neuracle_realtime.NeuracleJellyFishSource", MetadataOnlySource)
    result = main(["--metadata-only", "--output-dir", str(tmp_path)])
    summary = json.loads((tmp_path / "probe_summary.json").read_text(encoding="utf-8"))

    assert result == 0
    assert MetadataOnlySource.read_called is False
    assert summary["metadata_ready"] is True
    assert summary["anonymized_serial_hash"] == "c0ffee123456"
    assert summary["channel_names"] == ["C3", "C4"]
    assert "personName" not in json.dumps(summary)
