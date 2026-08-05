import json
from pathlib import Path
import subprocess
import sys

import pytest

from bci_dayloop.realtime.neuracle_jellyfish import NeuracleSourceError
from scripts.probe_neuracle_realtime import main


class FakeProbeSource:
    instances: list["FakeProbeSource"] = []

    def __init__(self, config: object) -> None:
        self.config = config
        self.metadata = {
            "module_name": "device-serial-123456789",
            "module_type": "EEG",
            "serial_number_hash": "c0ffee123456",
            "forwarded_channel_count": 2,
            "channel_names": ("C3", "C4"),
            "channel_types": ("EEG", "Trigger"),
            "sample_rates": (250.0, 250.0),
        }
        self.state = "ready"
        self.connected = False
        self.read_called = False
        self.disconnect_calls = 0
        type(self).instances.append(self)

    def connect(self) -> None:
        self.connected = True

    def read_chunk(self) -> object | None:
        self.read_called = True
        self.state = "streaming"
        return None

    def read_event(self) -> None:
        return None

    def health(self) -> dict[str, object]:
        return {
            "state": self.state,
            "connected": self.connected,
            "metadata_ready": self.connected,
            "received_packets": 0,
            "malformed_packets": 0,
            "missing_packets": 0,
            "duplicate_packets": 0,
            "out_of_order_packets": 0,
            "reconnect_count": 0,
            "unit_evidence_level": "realtime_unverified",
            "model_safe": False,
            "last_error": None,
        }

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False
        self.state = "stopped"


def _patch_source(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeProbeSource.instances.clear()
    monkeypatch.setattr("scripts.probe_neuracle_realtime.NeuracleJellyFishSource", FakeProbeSource)


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
    assert "--metadata-only" in result.stdout


def test_probe_metadata_timeout_is_redacted_and_stopped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class UnavailableSource(FakeProbeSource):
        def connect(self) -> None:
            self.connected = True
            raise NeuracleSourceError("META timeout at 192.0.2.1 for serial 123456")

    monkeypatch.setattr("scripts.probe_neuracle_realtime.NeuracleJellyFishSource", UnavailableSource)
    result = main(["--duration-sec", "0.1", "--output-dir", str(tmp_path)])
    summary = json.loads((tmp_path / "probe_summary.json").read_text(encoding="utf-8"))

    assert result == 2
    assert summary["error"] == "probe_failed"
    assert summary["final_health"]["state"] == "stopped"
    assert summary["final_health"]["connected"] is False
    encoded = json.dumps(summary)
    assert "personName" not in encoded
    assert "serialNumber" not in encoded
    assert "192.0.2.1" not in encoded
    assert "123456" not in encoded


def test_metadata_only_is_anonymized_and_disconnects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_source(monkeypatch)
    result = main(["--metadata-only", "--output-dir", str(tmp_path)])
    summary = json.loads((tmp_path / "probe_summary.json").read_text(encoding="utf-8"))

    source = FakeProbeSource.instances[-1]
    assert result == 0
    assert source.read_called is False
    assert source.disconnect_calls == 1
    assert summary["module_name"] is None
    assert summary["module_type"] == "EEG"
    assert summary["anonymized_serial_hash"] == "c0ffee123456"
    assert summary["pre_disconnect_health"]["state"] == "ready"
    assert summary["pre_disconnect_health"]["connected"] is True
    assert summary["final_health"]["state"] == "stopped"
    assert summary["final_health"]["connected"] is False
    assert summary["unit_status"] == {
        "raw_unit": "unknown",
        "unit_evidence_level": "realtime_unverified",
        "model_safe": False,
    }
    encoded = json.dumps(summary)
    assert "device-serial-123456789" not in encoded
    assert "personName" not in encoded


def test_duration_probe_closes_after_streaming_without_waveform_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_source(monkeypatch)
    result = main(["--duration-sec", "0.01", "--no-save-waveform", "--output-dir", str(tmp_path)])
    summary = json.loads((tmp_path / "probe_summary.json").read_text(encoding="utf-8"))

    source = FakeProbeSource.instances[-1]
    assert result == 0
    assert source.read_called is True
    assert source.disconnect_calls == 1
    assert summary["pre_disconnect_health"]["state"] == "streaming"
    assert summary["pre_disconnect_health"]["connected"] is True
    assert summary["final_health"] == {
        "state": "stopped",
        "connected": False,
        "metadata_ready": False,
        "received_packets": 0,
        "malformed_packets": 0,
        "missing_packets": 0,
        "duplicate_packets": 0,
        "out_of_order_packets": 0,
        "reconnect_count": 0,
        "unit_evidence_level": "realtime_unverified",
        "model_safe": False,
        "last_error_present": False,
    }
    assert summary["waveforms_saved"] is False
    assert not list(tmp_path.glob("*.npy"))


def test_ctrl_c_path_disconnects_before_writing_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class InterruptedSource(FakeProbeSource):
        def read_chunk(self) -> object | None:
            self.read_called = True
            self.state = "streaming"
            raise KeyboardInterrupt

    monkeypatch.setattr("scripts.probe_neuracle_realtime.NeuracleJellyFishSource", InterruptedSource)
    result = main(["--duration-sec", "0.1", "--output-dir", str(tmp_path)])
    summary = json.loads((tmp_path / "probe_summary.json").read_text(encoding="utf-8"))

    assert result == 130
    assert summary["error"] == "probe_interrupted"
    assert summary["final_health"]["state"] == "stopped"
    assert InterruptedSource.instances[-1].disconnect_calls == 1


def test_parser_error_creates_no_source_or_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_source(monkeypatch)

    with pytest.raises(SystemExit):
        main(["--duration-sec", "0"])

    assert FakeProbeSource.instances == []


def test_summary_write_failure_still_disconnects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_source(monkeypatch)

    def fail_write(self: Path, *_args: object, **_kwargs: object) -> int:
        raise OSError("output unavailable")

    monkeypatch.setattr(Path, "write_text", fail_write)
    with pytest.raises(OSError, match="output unavailable"):
        main(["--metadata-only", "--output-dir", str(tmp_path)])

    assert FakeProbeSource.instances[-1].disconnect_calls == 1
