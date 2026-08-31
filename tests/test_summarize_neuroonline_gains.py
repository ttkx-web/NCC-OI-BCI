from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts import summarize_neuroonline_gains as gains


def summary_payload(offset: float = 0.0) -> dict[str, object]:
    return {
        "gains": {
            "overall": {
                "accuracy_gain": 0.052083333333333315 + offset,
                "balanced_accuracy_gain": 0.052083333333333315 + offset,
                "macro_f1_gain": 0.057972564604249266 + offset,
            },
            "post_warmup": {
                "accuracy_gain": 0.05859375 + offset,
                "balanced_accuracy_gain": 0.05951845533498756 + offset,
                "macro_f1_gain": 0.06901967007478343 + offset,
            },
            "after_first_update": {
                "accuracy_gain": 0.05859375 + offset,
                "balanced_accuracy_gain": 0.05951845533498756 + offset,
                "macro_f1_gain": 0.06901967007478343 + offset,
            },
        }
    }


def write_summary(path: Path, payload: dict[str, object] | None = None) -> Path:
    if payload is None:
        payload = summary_payload()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def run_main(inputs: list[str], output: Path) -> int:
    args = [item for input_spec in inputs for item in ("--input", input_spec)]
    return gains.main([*args, "--output", str(output)])


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def test_one_input_generates_one_row_and_creates_output_directory(tmp_path: Path) -> None:
    summary = write_summary(tmp_path / "labram.json")
    output = tmp_path / "new" / "results" / "gains.csv"

    assert run_main([f"labram={summary}"], output) == 0

    columns, rows = read_csv(output)
    assert columns == list(gains.WIDE_COLUMNS)
    assert len(rows) == 1
    assert rows[0]["name"] == "labram"


@pytest.mark.parametrize("count", [2, 4])
def test_any_number_of_inputs_preserves_cli_order(tmp_path: Path, count: int) -> None:
    names = ["first_run", "50m_subject01", "another_model", "cbramod_seed2"][:count]
    inputs = []
    for index, name in enumerate(names):
        summary = write_summary(tmp_path / f"summary_{index}.json", summary_payload(index))
        inputs.append(f"{name}={summary}")

    output = tmp_path / "gains.csv"
    assert run_main(inputs, output) == 0

    _, rows = read_csv(output)
    assert [row["name"] for row in rows] == names


def test_path_with_equals_sign_is_split_only_once(tmp_path: Path) -> None:
    summary = write_summary(tmp_path / "run=one.json")
    output = tmp_path / "gains.csv"

    assert run_main([f"arbitrary_name={summary}"], output) == 0

    _, rows = read_csv(output)
    assert rows[0]["name"] == "arbitrary_name"


def test_csv_values_match_json_without_rounding(tmp_path: Path) -> None:
    payload = summary_payload()
    summary = write_summary(tmp_path / "summary.json", payload)
    output = tmp_path / "gains.csv"

    run_main([f"run={summary}"], output)

    _, rows = read_csv(output)
    assert rows == [
        {
            "name": "run",
            **{
                f"{section}_{metric}": str(payload["gains"][section][metric])
                for section in ("overall", "post_warmup", "after_first_update")
                for metric in (
                    "accuracy_gain",
                    "balanced_accuracy_gain",
                    "macro_f1_gain",
                )
            },
        }
    ]


def test_duplicate_input_name_is_an_argparse_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    summary = write_summary(tmp_path / "summary.json")

    with pytest.raises(SystemExit, match="2"):
        run_main([f"repeat={summary}", f"repeat={summary}"], tmp_path / "gains.csv")

    assert "Duplicate input name: 'repeat'." in capsys.readouterr().err


def test_missing_input_is_an_argparse_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        gains.main(["--output", str(tmp_path / "gains.csv")])


@pytest.mark.parametrize("input_spec", ["missing_equals", "=summary.json", "name="])
def test_invalid_input_format_is_an_argparse_error(
    tmp_path: Path,
    input_spec: str,
) -> None:
    with pytest.raises(SystemExit, match="2"):
        run_main([input_spec], tmp_path / "gains.csv")


def test_missing_file_is_an_argparse_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "missing.json"

    with pytest.raises(SystemExit, match="2"):
        run_main([f"missing={missing}"], tmp_path / "gains.csv")

    error = capsys.readouterr().err
    assert "Input 'missing'" in error
    assert repr(str(missing)) in error
    assert "does not exist" in error


def test_invalid_json_is_an_argparse_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    summary = tmp_path / "invalid.json"
    summary.write_text("not-json", encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        run_main([f"invalid={summary}"], tmp_path / "gains.csv")

    error = capsys.readouterr().err
    assert "Input 'invalid'" in error
    assert "not valid JSON" in error


@pytest.mark.parametrize(
    ("payload", "missing_path"),
    [
        ({}, "gains"),
        ({"gains": {}}, "gains.overall"),
        (
            {
                "gains": {
                    "overall": summary_payload()["gains"]["overall"],
                    "post_warmup": summary_payload()["gains"]["post_warmup"],
                    "after_first_update": {
                        "accuracy_gain": 0.1,
                        "balanced_accuracy_gain": 0.2,
                    },
                }
            },
            "gains.after_first_update.macro_f1_gain",
        ),
    ],
)
def test_missing_gain_fields_identify_input_path_and_field(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    payload: dict[str, object],
    missing_path: str,
) -> None:
    summary = write_summary(tmp_path / "incomplete.json", payload)

    with pytest.raises(SystemExit, match="2"):
        run_main([f"50m={summary}"], tmp_path / "gains.csv")

    error = capsys.readouterr().err
    assert "Input '50m'" in error
    assert repr(str(summary)) in error
    assert missing_path in error
