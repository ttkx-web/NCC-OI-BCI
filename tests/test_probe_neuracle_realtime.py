import json
from pathlib import Path
import subprocess
import sys

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


def test_probe_without_authorized_backend_fails_safely_and_logs_no_sensitive_values(tmp_path: Path) -> None:
    result = main(["--duration-sec", "0.1", "--output-dir", str(tmp_path)])
    summary = json.loads((tmp_path / "probe_summary.json").read_text(encoding="utf-8"))

    assert result == 2
    assert summary["waveforms_saved"] is False
    assert summary["unit_status"]["model_safe"] is False
    encoded = json.dumps(summary)
    assert "personName" not in encoded
    assert "serialNumber" not in encoded
