"""Audit a directory tree of paired Neuracle BDF and Collect CSV sessions."""

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
    return NeuracleBDFReader(
        UnitEvidence(
            raw_unit="uV",
            normalized_unit="uV",
            evidence_level="official_reader_verified",
            evidence_source="MNE get_data(..., units='uV')",
        )
    )


def _summary_metrics(
    trials: tuple[EEGTrial, ...], record: RawEEGRecord | None
) -> dict[str, object]:
    durations = [trial.duration_seconds for trial in trials]
    sample_counts = [trial.eeg.shape[1] for trial in trials]
    label_counts = Counter(trial.label for trial in trials)
    block_counts = Counter(str(trial.block_id) for trial in trials)
    return {
        "total_trials": len(trials),
        "label_trial_counts": {label: label_counts.get(label, 0) for label in _IMAGERY_LABELS},
        "block_trial_counts": dict(block_counts),
        "sampling_rate": record.sampling_rate if record is not None else None,
        "channel_count": len(record.channel_names) if record is not None else None,
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
        "has_nan": bool(np.isnan(record.eeg).any()) if record is not None else None,
        "has_inf": bool(np.isinf(record.eeg).any()) if record is not None else None,
    }


def _session_report(
    *,
    relative_directory: str,
    bdf_path: Path | None,
    csv_path: Path | None,
    expected_duration_seconds: float,
    duration_tolerance_seconds: float,
) -> dict[str, object]:
    base = {
        "relative_directory": relative_directory,
        "bdf_filename": bdf_path.name if bdf_path is not None else None,
        "csv_filename": csv_path.name if csv_path is not None else None,
        "subject": None,
        "session": None,
        "passed": False,
        "error": None,
        "bdf_event_count": None,
        "csv_row_count": None,
        "aligned_event_count": 0,
        **_summary_metrics((), None),
    }
    if bdf_path is None or csv_path is None:
        missing = []
        if bdf_path is None:
            missing.append("1.bdf")
        if csv_path is None:
            missing.append("sub-*.csv")
        base["error"] = f"Missing required session file(s): {', '.join(missing)}"
        return base

    record: RawEEGRecord | None = None
    try:
        record = _reader().load(bdf_path)
        collect_csv = CollectCSVAdapter.from_file(csv_path)
        base["subject"] = collect_csv.subject
        base["session"] = collect_csv.session
        base["bdf_event_count"] = len(record.events)
        base["csv_row_count"] = len(collect_csv.rows)
        aligned_events = align_events_with_csv(record.events, collect_csv.to_alignment_rows())
        aligned_record = replace(record, events=aligned_events)
        base["aligned_event_count"] = len(aligned_events)
        trials = extract_imagery_trials(
            aligned_record,
            expected_duration_seconds=expected_duration_seconds,
            duration_tolerance_seconds=duration_tolerance_seconds,
        )
        base.update(_summary_metrics(trials, aligned_record))
        base["passed"] = True
        return base
    except (ValueError, FileNotFoundError) as exc:
        if record is not None:
            base["bdf_event_count"] = len(record.events)
            base.update(_summary_metrics((), record))
        base["error"] = str(exc)
        return base


def _candidate_directories(root: Path) -> list[Path]:
    directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
    candidates = []
    for directory in directories:
        files = [path for path in directory.iterdir() if path.is_file()]
        if any(path.name.lower() == "1.bdf" for path in files) or any(
            path.name.lower().startswith("sub-") and path.suffix.lower() == ".csv"
            for path in files
        ):
            candidates.append(directory)
    return sorted(candidates, key=lambda path: path.relative_to(root).as_posix())


def audit_dataset(
    root: str | Path,
    *,
    expected_duration_seconds: float = 4.0,
    duration_tolerance_seconds: float = 0.1,
) -> dict[str, object]:
    """Audit every candidate session directory and continue after per-session failures."""
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)

    session_reports: list[dict[str, object]] = []
    paired_sessions = 0
    for directory in _candidate_directories(root_path):
        files = [path for path in directory.iterdir() if path.is_file()]
        bdf_files = [path for path in files if path.name.lower() == "1.bdf"]
        csv_files = [
            path
            for path in files
            if path.name.lower().startswith("sub-") and path.suffix.lower() == ".csv"
        ]
        relative_directory = directory.relative_to(root_path).as_posix()
        if relative_directory == ".":
            relative_directory = ""
        if len(bdf_files) == 1 and len(csv_files) == 1:
            paired_sessions += 1
            report = _session_report(
                relative_directory=relative_directory,
                bdf_path=bdf_files[0],
                csv_path=csv_files[0],
                expected_duration_seconds=expected_duration_seconds,
                duration_tolerance_seconds=duration_tolerance_seconds,
            )
        else:
            errors = []
            if len(bdf_files) != 1:
                errors.append(f"Expected exactly one 1.bdf, found {len(bdf_files)}")
            if len(csv_files) != 1:
                errors.append(f"Expected exactly one sub-*.csv, found {len(csv_files)}")
            report = _session_report(
                relative_directory=relative_directory,
                bdf_path=bdf_files[0] if len(bdf_files) == 1 else None,
                csv_path=csv_files[0] if len(csv_files) == 1 else None,
                expected_duration_seconds=expected_duration_seconds,
                duration_tolerance_seconds=duration_tolerance_seconds,
            )
            report["error"] = "; ".join(errors)
        session_reports.append(report)

    passed_reports = [report for report in session_reports if report["passed"]]
    failed_reports = [report for report in session_reports if not report["passed"]]
    total_by_label = Counter()
    for report in passed_reports:
        total_by_label.update(report["label_trial_counts"])
    subjects = sorted(
        {report["subject"] for report in session_reports if report["subject"] is not None}
    )
    sessions = sorted(
        {report["session"] for report in session_reports if report["session"] is not None}
    )
    return {
        "discovered_directories": [report["relative_directory"] for report in session_reports],
        "paired_sessions": paired_sessions,
        "passed_sessions": len(passed_reports),
        "failed_sessions": len(failed_reports),
        "total_trials": sum(report["total_trials"] for report in passed_reports),
        "total_trials_by_label": {label: total_by_label.get(label, 0) for label in _IMAGERY_LABELS},
        "subjects": subjects,
        "sessions": sessions,
        "all_sessions_passed": not failed_reports,
        "failures": [
            {"relative_directory": report["relative_directory"], "error": report["error"]}
            for report in failed_reports
        ],
        "session_reports": session_reports,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-duration-seconds", type=float, default=4.0)
    parser.add_argument("--duration-tolerance-seconds", type=float, default=0.1)
    args = parser.parse_args(argv)

    try:
        report = audit_dataset(
            args.root,
            expected_duration_seconds=args.expected_duration_seconds,
            duration_tolerance_seconds=args.duration_tolerance_seconds,
        )
    except (ValueError, FileNotFoundError) as exc:
        report = {"all_sessions_passed": False, "error": str(exc), "session_reports": []}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if report["all_sessions_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
