import csv
import json
from pathlib import Path

import numpy as np
import pytest

from bci_dayloop.data.records import EEGEvent, RawEEGRecord, UnitEvidence
from scripts.align_neuracle_collect import main


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
        self.paths: list[Path] = []

    def load(self, path: Path) -> RawEEGRecord:
        self.paths.append(path)
        return RawEEGRecord(
            eeg=np.zeros((1, 4)),
            channel_names=("C3",),
            sampling_rate=250.0,
            unit_evidence=UnitEvidence("uV", None, "vendor_confirmed"),
            events=(
                EEGEvent(0, "fixation", code=10, label="left_hand"),
                EEGEvent(1, "imagery", code=1, label="left_hand"),
                EEGEvent(2, "rest", code=20, label="left_hand"),
                EEGEvent(3, "imagery", code=3, label="feet"),
            ),
        )


def _write_csv(
    tmp_path: Path, event_codes: tuple[str, str, str, str] = ("10", "1", "20", "3")
) -> Path:
    path = tmp_path / "collect.csv"
    rows = [
        {
            "subject": "sub-001",
            "session": "ses-01",
            "block": "1",
            "trial": "1",
            "class": "left_hand",
            "event_code": event_codes[0],
            "event_name": "fixation",
            "flip_time": "1.0",
            "lsl_timestamp": "2.0",
            "trigger_transport": "serial",
        },
        {
            "subject": "sub-001",
            "session": "ses-01",
            "block": "1",
            "trial": "1",
            "class": "left_hand",
            "event_code": event_codes[1],
            "event_name": "imagery",
            "flip_time": "3.0",
            "lsl_timestamp": "4.0",
            "trigger_transport": "serial",
        },
        {
            "subject": "sub-001",
            "session": "ses-01",
            "block": "1",
            "trial": "1",
            "class": "left_hand",
            "event_code": event_codes[2],
            "event_name": "rest",
            "flip_time": "5.0",
            "lsl_timestamp": "6.0",
            "trigger_transport": "serial",
        },
        {
            "subject": "sub-001",
            "session": "ses-01",
            "block": "2",
            "trial": "1",
            "class": "both_feet",
            "event_code": event_codes[3],
            "event_name": "imagery",
            "flip_time": "7.0",
            "lsl_timestamp": "8.0",
            "trigger_transport": "serial",
        },
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_cli_writes_metadata_only_report_for_strict_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("scripts.align_neuracle_collect.NeuracleBDFReader", FakeBDFReader)
    output_path = tmp_path / "report.json"

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
    assert report == {
        "bdf_event_count": 4,
        "csv_row_count": 4,
        "matched": True,
        "mismatch_count": 0,
        "mismatch_details": [],
        "marker_counts": {"10": 1, "1": 1, "20": 1, "3": 1},
        "event_label_counts": {"left_hand": 3, "feet": 1},
        "imagery_trial_counts": {
            "left_hand": 1,
            "right_hand": 0,
            "feet": 1,
            "tongue": 0,
        },
        "block_event_counts": {"1": 3, "2": 1},
        "block_trial_counts": {"1": 1, "2": 1},
        "total_imagery_trials": 2,
    }
    assert "eeg" not in report


def test_cli_writes_failure_report_and_raises_clear_alignment_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("scripts.align_neuracle_collect.NeuracleBDFReader", FakeBDFReader)
    output_path = tmp_path / "failed-report.json"

    with pytest.raises(ValueError, match="BDF and Collect CSV alignment failed"):
        main(
            [
                "--bdf",
                str(tmp_path / "placeholder.bdf"),
                "--csv",
                str(_write_csv(tmp_path, event_codes=("2", "1", "20", "3"))),
                "--output",
                str(output_path),
            ]
        )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["matched"] is False
    assert report["mismatch_count"] == 1
    assert "index 0" in report["mismatch_details"][0]["error"]
    assert "event_label_counts" in report
    assert "imagery_trial_counts" in report
    assert "block_event_counts" in report
    assert "block_trial_counts" in report
