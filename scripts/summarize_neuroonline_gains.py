"""Collect NeuroOnline summary gains into a CSV wide table."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


WIDE_COLUMNS = (
    "name",
    "overall_accuracy_gain",
    "overall_balanced_accuracy_gain",
    "overall_macro_f1_gain",
    "post_warmup_accuracy_gain",
    "post_warmup_balanced_accuracy_gain",
    "post_warmup_macro_f1_gain",
    "after_first_update_accuracy_gain",
    "after_first_update_balanced_accuracy_gain",
    "after_first_update_macro_f1_gain",
)

_GAIN_SECTIONS = ("overall", "post_warmup", "after_first_update")
_GAIN_METRICS = ("accuracy_gain", "balanced_accuracy_gain", "macro_f1_gain")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="NAME=SUMMARY_JSON",
        help="Experiment name and NeuroOnline summary.json path. Repeat for each run.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="CSV_PATH",
        help="Destination CSV wide table.",
    )
    return parser


def parse_input_spec(value: str) -> tuple[str, Path]:
    """Parse one NAME=SUMMARY_JSON value, preserving equals signs in the path."""
    if "=" not in value:
        raise ValueError(
            f"Invalid --input value {value!r}; expected NAME=SUMMARY_JSON."
        )

    name, path_text = value.split("=", 1)
    name = name.strip()
    path_text = path_text.strip()
    if not name:
        raise ValueError(
            f"Invalid --input value {value!r}; input name before '=' cannot be empty."
        )
    if not path_text:
        raise ValueError(
            f"Invalid --input value {value!r}; summary path after '=' cannot be empty."
        )
    return name, Path(path_text).expanduser()


def _missing_field_error(name: str, path: Path, field_path: str) -> ValueError:
    return ValueError(f"Input {name!r} at {str(path)!r} is missing: {field_path}")


def _load_gains(name: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Input {name!r} at {str(path)!r} does not exist or is not a file.")

    try:
        with path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Input {name!r} at {str(path)!r} is not valid JSON: {error.msg}."
        ) from error
    except OSError as error:
        raise ValueError(f"Could not read input {name!r} at {str(path)!r}: {error}") from error

    if not isinstance(summary, dict) or "gains" not in summary:
        raise _missing_field_error(name, path, "gains")
    gains = summary["gains"]
    if not isinstance(gains, dict):
        raise ValueError(f"Input {name!r} at {str(path)!r} has non-object field: gains")

    row: dict[str, Any] = {"name": name}
    for section in _GAIN_SECTIONS:
        section_path = f"gains.{section}"
        if section not in gains:
            raise _missing_field_error(name, path, section_path)
        values = gains[section]
        if not isinstance(values, dict):
            raise ValueError(
                f"Input {name!r} at {str(path)!r} has non-object field: {section_path}"
            )
        for metric in _GAIN_METRICS:
            metric_path = f"{section_path}.{metric}"
            if metric not in values:
                raise _missing_field_error(name, path, metric_path)
            row[f"{section}_{metric}"] = values[metric]
    return row


def collect_rows(input_specs: Sequence[str]) -> list[dict[str, Any]]:
    """Load requested summaries in CLI order and return their CSV rows."""
    rows: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for input_spec in input_specs:
        name, path = parse_input_spec(input_spec)
        if name in seen_names:
            raise ValueError(f"Duplicate input name: {name!r}.")
        seen_names.add(name)
        rows.append(_load_gains(name, path))
    return rows


def write_wide_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write the fixed-column wide table without rounding JSON metric values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=WIDE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def print_preview(rows: Sequence[dict[str, Any]]) -> None:
    print(f"Collected {len(rows)} summaries.")
    print("Preview:")
    writer = csv.DictWriter(sys.stdout, fieldnames=WIDE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        rows = collect_rows(args.input)
        write_wide_csv(args.output, rows)
    except ValueError as error:
        parser.error(str(error))

    print(f"Collected {len(rows)} summaries.")
    print(f"Output: {args.output}")
    print("Preview:")
    writer = csv.DictWriter(sys.stdout, fieldnames=WIDE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
