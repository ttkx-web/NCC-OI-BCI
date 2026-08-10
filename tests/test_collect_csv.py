import csv
from pathlib import Path

import pytest

from bci_dayloop.data.collect_csv import CollectCSV, read_collect_csv


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


def _write_csv(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    path = tmp_path / "collect.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "subject": "sub-001",
        "session": "ses-01",
        "block": "2",
        "trial": "7",
        "class": "both_feet",
        "event_code": "3",
        "event_name": "imagery_start",
        "flip_time": "12.5",
        "lsl_timestamp": "13.25",
        "trigger_transport": "serial",
    }
    row.update(overrides)
    return row


def test_read_collect_csv_parses_utf8_sig_and_normalizes_fields(tmp_path: Path) -> None:
    parsed = read_collect_csv(_write_csv(tmp_path, [_row()]))

    assert parsed.subject == "sub-001"
    assert parsed.session == "ses-01"
    assert len(parsed.rows) == 1
    row = parsed.rows[0]
    assert row.block_id == 2
    assert row.trial_id == 7
    assert row.label == "feet"
    assert row.event_code == 3
    assert row.event_name == "imagery_start"
    assert row.flip_time == 12.5
    assert row.lsl_timestamp == 13.25
    assert row.trigger_transport == "serial"


def test_to_alignment_rows_only_contains_existing_alignment_fields(tmp_path: Path) -> None:
    parsed = CollectCSV.from_file(_write_csv(tmp_path, [_row()]))

    assert parsed.to_alignment_rows() == (
        {
            "event_code": 3,
            "block_id": 2,
            "trial_id": 7,
            "label": "feet",
            "flip_time": 12.5,
            "lsl_timestamp": 13.25,
        },
    )


def test_empty_block_trial_and_class_become_none(tmp_path: Path) -> None:
    parsed = read_collect_csv(_write_csv(tmp_path, [_row(block="", trial="", **{"class": ""})]))

    row = parsed.rows[0]
    assert row.block_id is None
    assert row.trial_id is None
    assert row.label is None


@pytest.mark.parametrize("field", ["subject", "session"])
def test_subject_and_session_must_be_consistent(field: str, tmp_path: Path) -> None:
    second = _row()
    second[field] = "different"

    with pytest.raises(ValueError, match="subject/session mismatch"):
        read_collect_csv(_write_csv(tmp_path, [_row(), second]))


@pytest.mark.parametrize("field", ["flip_time", "lsl_timestamp"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_non_finite_timestamps_are_rejected(field: str, value: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=field):
        read_collect_csv(_write_csv(tmp_path, [_row(**{field: value})]))


def test_unknown_class_and_non_integer_event_code_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown class"):
        read_collect_csv(_write_csv(tmp_path, [_row(**{"class": "unknown"})]))
    with pytest.raises(ValueError, match="event_code"):
        read_collect_csv(_write_csv(tmp_path, [_row(event_code="S 3")]))
