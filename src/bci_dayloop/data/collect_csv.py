"""Adapter for parsed event logs emitted by the collection application."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


_REQUIRED_FIELDS = (
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
)
_CLASS_LABELS = {
    "left_hand": "left_hand",
    "right_hand": "right_hand",
    "both_feet": "feet",
    "tongue": "tongue",
}


def _optional_int(value: str) -> int | None:
    return None if not value.strip() else int(value)


def _optional_finite_float(value: str, field: str, row_number: int) -> float | None:
    if not value.strip():
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} at CSV row {row_number} must be finite")
    return parsed


def _class_label(value: str, row_number: int) -> str | None:
    normalized = value.strip()
    if not normalized:
        return None
    try:
        return _CLASS_LABELS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown class at CSV row {row_number}: {value!r}") from exc


@dataclass(frozen=True)
class CollectCSVRow:
    subject: str
    session: str
    block_id: int | None
    trial_id: int | None
    label: str | None
    event_code: int
    event_name: str
    flip_time: float | None
    lsl_timestamp: float | None
    trigger_transport: str


@dataclass(frozen=True)
class CollectCSV:
    """One Collect event log with stable subject/session provenance."""

    rows: tuple[CollectCSVRow, ...]
    subject: str | None
    session: str | None

    @classmethod
    def from_file(cls, path: str | Path) -> CollectCSV:
        source_path = Path(path)
        with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("Collect CSV has no header")
            missing_fields = [field for field in _REQUIRED_FIELDS if field not in reader.fieldnames]
            if missing_fields:
                raise ValueError(f"Collect CSV missing required fields: {', '.join(missing_fields)}")

            parsed_rows: list[CollectCSVRow] = []
            file_subject: str | None = None
            file_session: str | None = None
            for row_number, row in enumerate(reader, start=2):
                subject = row["subject"] or ""
                session = row["session"] or ""
                if file_subject is None:
                    file_subject = subject
                    file_session = session
                elif subject != file_subject or session != file_session:
                    raise ValueError(
                        f"subject/session mismatch at CSV row {row_number}: "
                        f"expected {file_subject!r}/{file_session!r}, got {subject!r}/{session!r}"
                    )

                try:
                    event_code = int(row["event_code"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"event_code at CSV row {row_number} must be an integer") from exc
                try:
                    parsed_rows.append(
                        CollectCSVRow(
                            subject=subject,
                            session=session,
                            block_id=_optional_int(row["block"] or ""),
                            trial_id=_optional_int(row["trial"] or ""),
                            label=_class_label(row["class"] or "", row_number),
                            event_code=event_code,
                            event_name=row["event_name"] or "",
                            flip_time=_optional_finite_float(
                                row["flip_time"] or "", "flip_time", row_number
                            ),
                            lsl_timestamp=_optional_finite_float(
                                row["lsl_timestamp"] or "", "lsl_timestamp", row_number
                            ),
                            trigger_transport=row["trigger_transport"] or "",
                        )
                    )
                except ValueError as exc:
                    if "CSV row" in str(exc):
                        raise
                    raise ValueError(f"Invalid value at CSV row {row_number}") from exc

        return cls(rows=tuple(parsed_rows), subject=file_subject, session=file_session)

    def to_alignment_rows(self) -> tuple[Mapping[str, object], ...]:
        """Return only the CSV fields accepted by ``align_events_with_csv``."""
        return tuple(
            {
                "event_code": row.event_code,
                "block_id": row.block_id,
                "trial_id": row.trial_id,
                "label": row.label,
                "flip_time": row.flip_time,
                "lsl_timestamp": row.lsl_timestamp,
            }
            for row in self.rows
        )


def read_collect_csv(path: str | Path) -> CollectCSV:
    """Read a Collect CSV file using its UTF-8 BOM-compatible encoding."""
    return CollectCSV.from_file(path)
