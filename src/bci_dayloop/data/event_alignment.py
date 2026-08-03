"""Strict positional alignment between BDF events and parsed CSV event rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from bci_dayloop.data.neuracle_bdf import parse_neuracle_marker
from bci_dayloop.data.records import EEGEvent


_CSV_MARKER_FIELDS = ("marker_code", "event_code", "code", "marker")
_CSV_EVENT_FIELDS = ("trial_id", "block_id", "label", "lsl_timestamp", "flip_time")


def _is_present(value: object) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _parse_marker_code(value: object) -> int | None:
    parsed = parse_neuracle_marker(value)
    marker_code = parsed["metadata"]["marker_code"]  # type: ignore[index]
    return marker_code if isinstance(marker_code, int) else None


def _csv_marker_code(row: Mapping[str, object], index: int) -> int:
    for field in _CSV_MARKER_FIELDS:
        if field in row:
            marker_code = _parse_marker_code(row[field])
            if marker_code is None:
                raise ValueError(f"CSV marker code at index {index} cannot be parsed as an integer")
            return marker_code
    raise ValueError(f"CSV row at index {index} has no marker code field")


def align_events_with_csv(
    events: tuple[EEGEvent, ...], csv_rows: Sequence[Mapping[str, object]]
) -> tuple[EEGEvent, ...]:
    """Align BDF and CSV events strictly by their existing positional order."""
    if len(events) != len(csv_rows):
        raise ValueError(
            f"Event count mismatch: BDF has {len(events)} events, CSV has {len(csv_rows)} rows"
        )

    aligned: list[EEGEvent] = []
    for index, (event, row) in enumerate(zip(events, csv_rows, strict=True)):
        bdf_code = _parse_marker_code(event.code)
        if bdf_code is None:
            raise ValueError(
                f"BDF event code at index {index} cannot be parsed as an integer: {event.code!r}"
            )
        csv_code = _csv_marker_code(row, index)
        if bdf_code != csv_code:
            raise ValueError(
                f"Marker code mismatch at index {index}: BDF code {bdf_code}, CSV code {csv_code}"
            )

        csv_label = row.get("label")
        label = event.label
        if _is_present(csv_label):
            csv_label_text = str(csv_label)
            if label is not None and label != csv_label_text:
                raise ValueError(
                    f"Label mismatch at index {index}: BDF label {label!r}, CSV label {csv_label_text!r}"
                )
            label = csv_label_text

        metadata = dict(event.metadata)
        for field in ("lsl_timestamp", "flip_time"):
            if field in row and _is_present(row[field]):
                metadata[field] = row[field]

        replacements: dict[str, object] = {"label": label, "metadata": metadata}
        for field in ("trial_id", "block_id"):
            if field in row and _is_present(row[field]):
                replacements[field] = row[field]
        aligned.append(replace(event, **replacements))

    return tuple(aligned)
