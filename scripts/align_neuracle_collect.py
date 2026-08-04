"""Strictly align a Neuracle BDF recording with a Collect event CSV."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from bci_dayloop.data.collect_csv import CollectCSV as CollectCSVAdapter
from bci_dayloop.data.event_alignment import align_events_with_csv
from bci_dayloop.data.neuracle_bdf import NeuracleBDFReader
from bci_dayloop.data.records import EEGEvent, RawEEGRecord, UnitEvidence


_IMAGERY_LABELS = ("left_hand", "right_hand", "feet", "tongue")


def _counts(events: tuple[EEGEvent, ...]) -> dict[str, object]:
    marker_counts = Counter(str(event.code) for event in events)
    event_label_counts = Counter(event.label for event in events if event.label is not None)
    imagery_trial_counts = Counter(
        event.label
        for event in events
        if event.event_type == "imagery" and event.label in _IMAGERY_LABELS
    )
    block_event_counts = Counter(str(event.block_id) for event in events if event.block_id is not None)
    imagery_trials_by_block: dict[str, set[int | str]] = {}
    all_imagery_trials: set[tuple[int | str, int | str]] = set()
    for event in events:
        if event.event_type != "imagery" or event.block_id is None or event.trial_id is None:
            continue
        block_key = str(event.block_id)
        imagery_trials_by_block.setdefault(block_key, set()).add(event.trial_id)
        all_imagery_trials.add((event.block_id, event.trial_id))

    return {
        "marker_counts": dict(marker_counts),
        "event_label_counts": dict(event_label_counts),
        "imagery_trial_counts": {
            label: imagery_trial_counts.get(label, 0) for label in _IMAGERY_LABELS
        },
        "block_event_counts": dict(block_event_counts),
        "block_trial_counts": {
            block: len(trials) for block, trials in imagery_trials_by_block.items()
        },
        "total_imagery_trials": len(all_imagery_trials),
    }


def build_alignment_report(record: RawEEGRecord, collect_csv: CollectCSVAdapter) -> dict[str, object]:
    """Build a metadata-only alignment report without retaining EEG samples."""
    bdf_events = record.events
    try:
        aligned_events = align_events_with_csv(bdf_events, collect_csv.to_alignment_rows())
    except ValueError as exc:
        return {
            "bdf_event_count": len(bdf_events),
            "csv_row_count": len(collect_csv.rows),
            "matched": False,
            "mismatch_count": 1,
            "mismatch_details": [{"error": str(exc)}],
            **_counts(bdf_events),
        }

    return {
        "bdf_event_count": len(bdf_events),
        "csv_row_count": len(collect_csv.rows),
        "matched": True,
        "mismatch_count": 0,
        "mismatch_details": [],
        **_counts(aligned_events),
    }


def _reader() -> NeuracleBDFReader:
    return NeuracleBDFReader(
        UnitEvidence(
            raw_unit="uV",
            normalized_unit="uV",
            evidence_level="official_reader_verified",
            evidence_source="MNE get_data(..., units='uV')",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bdf", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    record = _reader().load(args.bdf)
    collect_csv = CollectCSVAdapter.from_file(args.csv)
    report = build_alignment_report(record, collect_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not report["matched"]:
        detail = report["mismatch_details"][0]["error"]  # type: ignore[index]
        raise ValueError(f"BDF and Collect CSV alignment failed: {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
