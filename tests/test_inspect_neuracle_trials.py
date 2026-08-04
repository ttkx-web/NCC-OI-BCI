import csv
import json
from pathlib import Path

import numpy as np
import pytest

from bci_dayloop.data.records import EEGEvent, RawEEGRecord, UnitEvidence
from scripts.inspect_neuracle_trials import main


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

    def load(self, _path: Path) -> RawEEGRecord:
        events: list[EEGEvent] = []
        for index, (code, label, block_id, trial_id) in enumerate(
            (
                (1, "left_hand", 1, 1),
                (2, "right_hand", 1, 2),
                (3, "feet", 2, 1),
                (4, "tongue", 2, 2),
            )
        ):
            start = index * 40
            events.extend(
                [
                    EEGEvent(start, "imagery", code=code, label=label),
                    EEGEvent(start + 40, "rest", code=20),
                ]
            )
        return RawEEGRecord(
            eeg=np.zeros((2, 200)),
            channel_names=("C3", "C4"),
            sampling_rate=10.0,
            unit_evidence=UnitEvidence("uV", None, "vendor_confirmed"),
            events=tuple(events),
        )


def _write_csv(
    tmp_path: Path, *, first_code: str = "1", final_rest_offset: int = 0
) -> Path:
    path = tmp_path / "collect.csv"
    rows: list[dict[str, str]] = []
    for code, label, block_id, trial_id in (
        (first_code, "left_hand", 1, 1),
        ("2", "right_hand", 1, 2),
        ("3", "both_feet", 2, 1),
        ("4", "tongue", 2, 2),
    ):
        rows.extend(
            [
                {
                    "subject": "sub-001",
                    "session": "ses-01",
                    "block": str(block_id),
                    "trial": str(trial_id),
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
                    "block": str(block_id),
                    "trial": str(trial_id),
                    "class": "",
                    "event_code": "20",
                    "event_name": "rest",
                    "flip_time": "3.0",
                    "lsl_timestamp": "4.0",
                    "trigger_transport": "serial",
                },
            ]
        )
    if final_rest_offset:
        rows[-1]["event_code"] = str(20 + final_rest_offset)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_cli_writes_trial_qc_report_and_creates_output_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("scripts.inspect_neuracle_trials.NeuracleBDFReader", FakeBDFReader)
    output_path = tmp_path / "nested" / "qc-report.json"

    result = main(
        [
            "--bdf",
            str(tmp_path / "placeholder.bdf"),
            "--csv",
            str(_write_csv(tmp_path)),
            "--output",
            str(output_path),
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert result == 0
    assert report["bdf_event_count"] == 8
    assert report["csv_row_count"] == 8
    assert report["aligned_event_count"] == 8
    assert report["total_trials"] == 4
    assert report["label_trial_counts"] == {
        "left_hand": 1,
        "right_hand": 1,
        "feet": 1,
        "tongue": 1,
    }
    assert report["block_trial_counts"] == {"1": 2, "2": 2}
    assert report["sampling_rate"] == 10.0
    assert report["channel_count"] == 2
    assert report["duration_seconds"] == {"min": 4.0, "max": 4.0, "mean": 4.0}
    assert report["sample_count"] == {"min": 40, "max": 40, "mean": 40.0}
    assert report["first_trial"] == {
        "label": "left_hand",
        "block_id": 1,
        "trial_id": 1,
        "start_sample": 0,
        "end_sample": 40,
        "sample_count": 40,
        "duration_seconds": 4.0,
    }
    assert report["last_trial"] == {
        "label": "tongue",
        "block_id": 2,
        "trial_id": 2,
        "start_sample": 120,
        "end_sample": 160,
        "sample_count": 40,
        "duration_seconds": 4.0,
    }
    assert report["has_nan"] is False
    assert report["has_inf"] is False
    assert report["extraction_passed"] is True
    assert "eeg" not in report


def test_alignment_failure_writes_error_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("scripts.inspect_neuracle_trials.NeuracleBDFReader", FakeBDFReader)
    output_path = tmp_path / "alignment-failure.json"

    result = main(
        [
            "--bdf",
            str(tmp_path / "placeholder.bdf"),
            "--csv",
            str(_write_csv(tmp_path, first_code="2")),
            "--output",
            str(output_path),
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert result == 1
    assert report["extraction_passed"] is False
    assert "Marker code mismatch" in report["error"]


def test_trial_extraction_failure_writes_error_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class WrongDurationReader(FakeBDFReader):
        def load(self, path: Path) -> RawEEGRecord:
            record = super().load(path)
            events = list(record.events)
            events[-1] = EEGEvent(165, "rest", code=20)
            return RawEEGRecord(
                eeg=record.eeg,
                channel_names=record.channel_names,
                sampling_rate=record.sampling_rate,
                unit_evidence=record.unit_evidence,
                events=tuple(events),
            )

    monkeypatch.setattr("scripts.inspect_neuracle_trials.NeuracleBDFReader", WrongDurationReader)
    output_path = tmp_path / "extraction-failure.json"

    result = main(
        [
            "--bdf",
            str(tmp_path / "placeholder.bdf"),
            "--csv",
            str(_write_csv(tmp_path)),
            "--output",
            str(output_path),
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert result == 1
    assert report["extraction_passed"] is False
    assert "duration" in report["error"]
