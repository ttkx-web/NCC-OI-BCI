import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from bci_dayloop.data.records import EEGEvent, RawEEGRecord, UnitEvidence
from scripts.audit_neuracle_dataset import main


_FIELDNAMES = [
    "subject",
    "session",
    "block",
    "trial",
    "class",
    "event_code",
    "event_name",
    "flip_time",
    "lsl_timestamp",
    "trigger_transport",
]


class FakeBDFReader:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def load(self, path: Path) -> RawEEGRecord:
        name = path.parent.name
        if name == "z_good":
            classes = ((2, "right_hand"), (4, "tongue"))
        else:
            classes = ((1, "left_hand"), (3, "feet"))
        events: list[EEGEvent] = []
        for index, (code, label) in enumerate(classes):
            start = index * 40
            end = start + (45 if name == "trial_failure" and index == 1 else 40)
            events.extend((EEGEvent(start, "imagery", code=code, label=label), EEGEvent(end, "rest", code=20)))
        return RawEEGRecord(
            eeg=np.zeros((2, 160)),
            channel_names=("C3", "C4"),
            sampling_rate=10.0,
            unit_evidence=UnitEvidence("uV", None, "vendor_confirmed"),
            events=tuple(events),
        )


def _write_bdf(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "1.bdf").write_bytes(b"placeholder")


def _write_csv(directory: Path, *, first_code: str = "1", extra: str = "") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"sub-001{extra}.csv"
    classes = ((first_code, "left_hand", 1, 1), ("3", "both_feet", 2, 1))
    if directory.name == "z_good":
        classes = (("2", "right_hand", 1, 1), ("4", "tongue", 2, 1))
    rows: list[dict[str, str]] = []
    for code, label, block, trial in classes:
        rows.extend(
            (
                {
                    "subject": "sub-001",
                    "session": "ses-01",
                    "block": str(block),
                    "trial": str(trial),
                    "class": label,
                    "event_code": code,
                    "event_name": "imagery",
                    "flip_time": "1.0",
                    "lsl_timestamp": "2.0",
                    "trigger_transport": "serial",
                },
                {
                    "subject": "sub-001",
                    "session": "ses-01",
                    "block": str(block),
                    "trial": str(trial),
                    "class": "",
                    "event_code": "20",
                    "event_name": "rest",
                    "flip_time": "3.0",
                    "lsl_timestamp": "4.0",
                    "trigger_transport": "serial",
                },
            )
        )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _paired_session(root: Path, name: str, **csv_kwargs: str) -> Path:
    directory = root / name
    _write_bdf(directory)
    _write_csv(directory, **csv_kwargs)
    return directory


def test_audit_multiple_sessions_uses_stable_relative_order_and_metadata_only_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("scripts.audit_neuracle_dataset.NeuracleBDFReader", FakeBDFReader)
    _paired_session(tmp_path, "z_good")
    _paired_session(tmp_path, "a_good")
    output_path = tmp_path / "nested" / "audit.json"

    result = main(["--root", str(tmp_path), "--output", str(output_path)])

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert result == 0
    assert report["discovered_directories"] == ["a_good", "z_good"]
    assert report["paired_sessions"] == 2
    assert report["passed_sessions"] == 2
    assert report["failed_sessions"] == 0
    assert report["total_trials"] == 4
    assert report["total_trials_by_label"] == {
        "left_hand": 1,
        "right_hand": 1,
        "feet": 1,
        "tongue": 1,
    }
    assert report["subjects"] == ["sub-001"]
    assert report["sessions"] == ["ses-01"]
    assert report["all_sessions_passed"] is True
    assert "eeg" not in report
    assert all("/" not in item["relative_directory"] for item in report["session_reports"])
    assert (output_path.parent / "session_qc.json").is_file()
    assert (output_path.parent / "dataset_summary.json").is_file()
    assert (output_path.parent / "trial_inventory.csv").is_file()
    assert (output_path.parent / "channel_metrics.csv").is_file()
    assert (output_path.parent / "event_alignment.csv").is_file()
    assert (output_path.parent / "export_manifest.json").is_file()


def test_audit_cli_runs_directly_without_pythonpath() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/audit_neuracle_dataset.py", "--help"],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--endpoint-tolerance-seconds" in result.stdout


def test_audit_records_discovery_and_session_failures_but_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("scripts.audit_neuracle_dataset.NeuracleBDFReader", FakeBDFReader)
    _paired_session(tmp_path, "a_good")
    _paired_session(tmp_path, "align_failure", first_code="2")
    _paired_session(tmp_path, "trial_failure")
    _write_csv(tmp_path / "missing_bdf")
    _write_bdf(tmp_path / "missing_csv")
    _write_bdf(tmp_path / "multiple_csv")
    _write_csv(tmp_path / "multiple_csv")
    _write_csv(tmp_path / "multiple_csv", extra="-second")
    output_path = tmp_path / "audit.json"

    result = main(["--root", str(tmp_path), "--output", str(output_path)])

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert result == 1
    assert report["discovered_directories"] == [
        "a_good",
        "align_failure",
        "missing_bdf",
        "missing_csv",
        "multiple_csv",
        "trial_failure",
    ]
    assert report["paired_sessions"] == 3
    assert report["passed_sessions"] == 1
    assert report["failed_sessions"] == 5
    assert report["total_trials"] == 2
    assert len(report["failures"]) == 5
    errors = {failure["relative_directory"]: failure["error"] for failure in report["failures"]}
    assert "Marker code mismatch" in errors["align_failure"]
    assert "Endpoint QC failed" in errors["trial_failure"]
    assert "found 0" in errors["missing_bdf"]
    assert "found 0" in errors["missing_csv"]
    assert "found 2" in errors["multiple_csv"]
