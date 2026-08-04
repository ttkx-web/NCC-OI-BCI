"""Align Neuracle BDF events with Collect CSV and inspect extracted MI trials."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from bci_dayloop.data.collect_csv import CollectCSV as CollectCSVAdapter
from bci_dayloop.data.event_alignment import align_events_with_csv
from bci_dayloop.data.neuracle_bdf import NeuracleBDFReader
from bci_dayloop.data.records import RawEEGRecord, UnitEvidence
from bci_dayloop.data.trial_extraction import EEGTrial, extract_imagery_trials


_IMAGERY_LABELS = ("left_hand", "right_hand", "feet", "tongue")


def _reader() -> NeuracleBDFReader:
    """Use the same explicit uV evidence configuration as the alignment CLI."""
    return NeuracleBDFReader(
        UnitEvidence(
            raw_unit="uV",
            normalized_unit="uV",
            evidence_level="vendor_confirmed",
            evidence_source="MNE get_data(..., units='uV')",
        )
    )


def _trial_summary(trial: EEGTrial) -> dict[str, object]:
    return {
        "label": trial.label,
        "block_id": trial.block_id,
        "trial_id": trial.trial_id,
        "start_sample": trial.start_sample,
        "end_sample": trial.end_sample,
        "sample_count": trial.eeg.shape[1],
        "duration_seconds": trial.duration_seconds,
    }


def build_trial_report(
    record: RawEEGRecord,
    csv_row_count: int,
    trials: tuple[EEGTrial, ...],
    *,
    expected_duration_seconds: float,
    duration_tolerance_seconds: float,
) -> dict[str, object]:
    """Return a metadata-only summary of successfully extracted imagery trials."""
    durations = [trial.duration_seconds for trial in trials]
    sample_counts = [trial.eeg.shape[1] for trial in trials]
    label_counter = Counter(trial.label for trial in trials)
    block_counter = Counter(str(trial.block_id) for trial in trials)
    return {
        "bdf_event_count": len(record.events),
        "csv_row_count": csv_row_count,
        "aligned_event_count": len(record.events),
        "total_trials": len(trials),
        "label_trial_counts": {label: label_counter.get(label, 0) for label in _IMAGERY_LABELS},
        "block_trial_counts": dict(block_counter),
        "sampling_rate": record.sampling_rate,
        "channel_count": len(record.channel_names),
        "expected_duration_seconds": expected_duration_seconds,
        "duration_tolerance_seconds": duration_tolerance_seconds,
        "duration_seconds": {
            "min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
            "mean": float(np.mean(durations)) if durations else None,
        },
        "sample_count": {
            "min": min(sample_counts) if sample_counts else None,
            "max": max(sample_counts) if sample_counts else None,
            "mean": float(np.mean(sample_counts)) if sample_counts else None,
        },
        "first_trial": _trial_summary(trials[0]) if trials else None,
        "last_trial": _trial_summary(trials[-1]) if trials else None,
        "has_nan": bool(np.isnan(record.eeg).any()),
        "has_inf": bool(np.isinf(record.eeg).any()),
        "extraction_passed": True,
    }


def _failure_report(
    *,
    record: RawEEGRecord | None,
    csv_row_count: int | None,
    expected_duration_seconds: float,
    duration_tolerance_seconds: float,
    error: Exception,
) -> dict[str, object]:
    return {
        "bdf_event_count": len(record.events) if record is not None else None,
        "csv_row_count": csv_row_count,
        "aligned_event_count": 0,
        "total_trials": 0,
        "label_trial_counts": {label: 0 for label in _IMAGERY_LABELS},
        "block_trial_counts": {},
        "sampling_rate": record.sampling_rate if record is not None else None,
        "channel_count": len(record.channel_names) if record is not None else None,
        "expected_duration_seconds": expected_duration_seconds,
        "duration_tolerance_seconds": duration_tolerance_seconds,
        "duration_seconds": {"min": None, "max": None, "mean": None},
        "sample_count": {"min": None, "max": None, "mean": None},
        "first_trial": None,
        "last_trial": None,
        "has_nan": bool(np.isnan(record.eeg).any()) if record is not None else None,
        "has_inf": bool(np.isinf(record.eeg).any()) if record is not None else None,
        "extraction_passed": False,
        "error": str(error),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bdf", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-duration-seconds", type=float, default=4.0)
    parser.add_argument("--duration-tolerance-seconds", type=float, default=0.1)
    args = parser.parse_args(argv)

    record: RawEEGRecord | None = None
    csv_row_count: int | None = None
    try:
        record = _reader().load(args.bdf)
        collect_csv = CollectCSVAdapter.from_file(args.csv)
        csv_row_count = len(collect_csv.rows)
        aligned_events = align_events_with_csv(record.events, collect_csv.to_alignment_rows())
        aligned_record = replace(
            record,
            events=aligned_events,
            metadata={**record.metadata, "csv_sha256": collect_csv.source_sha256},
        )
        trials = extract_imagery_trials(
            aligned_record,
            expected_duration_seconds=args.expected_duration_seconds,
            duration_tolerance_seconds=args.duration_tolerance_seconds,
        )
        report = build_trial_report(
            aligned_record,
            csv_row_count,
            trials,
            expected_duration_seconds=args.expected_duration_seconds,
            duration_tolerance_seconds=args.duration_tolerance_seconds,
        )
    except (ValueError, FileNotFoundError) as exc:
        report = _failure_report(
            record=record,
            csv_row_count=csv_row_count,
            expected_duration_seconds=args.expected_duration_seconds,
            duration_tolerance_seconds=args.duration_tolerance_seconds,
            error=exc,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if report["extraction_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
