"""Audit a directory tree of paired Neuracle BDF and Collect CSV sessions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from _bootstrap import ROOT
except ModuleNotFoundError:  # Imported as scripts.audit_neuracle_dataset in tests.
    from scripts._bootstrap import ROOT
from bci_dayloop.data.collect_csv import CollectCSV as CollectCSVAdapter
from bci_dayloop.data.event_alignment import align_events_with_csv
from bci_dayloop.data.neuracle_bdf import NeuracleBDFReader
from bci_dayloop.data.records import RawEEGRecord, UnitEvidence
from bci_dayloop.data.trial_extraction import (
    ACCURACY_SCOPE,
    ELIGIBLE_FOR_ACCURACY,
    ELIGIBLE_FOR_PURE_IMAGERY_ACCURACY,
    EXTRACTION_POLICY,
    VISUAL_CUE_DURATION_SECONDS,
    VISUAL_CUE_PRESENT,
    WINDOW_SEMANTICS,
    EEGTrial,
    extract_imagery_trials,
)


_IMAGERY_LABELS = ("left_hand", "right_hand", "feet", "tongue")
def _reader() -> NeuracleBDFReader:
    return NeuracleBDFReader(
        UnitEvidence(
            raw_unit="uV",
            normalized_unit="uV",
            evidence_level="vendor_confirmed",
            evidence_source="MNE get_data(..., units='uV')",
        )
    )


def _summary_metrics(
    trials: tuple[EEGTrial, ...], record: RawEEGRecord | None,
    *, expected_duration_seconds: float = 4.0, endpoint_tolerance_seconds: float = 0.05
) -> dict[str, object]:
    durations = [trial.duration_seconds for trial in trials]
    sample_counts = [trial.eeg.shape[1] for trial in trials]
    label_counts = Counter(trial.label for trial in trials)
    block_counts = Counter(str(trial.block_id) for trial in trials)
    offsets = [trial.rest_offset_samples for trial in trials]
    endpoint_samples = (
        math.ceil(endpoint_tolerance_seconds * record.sampling_rate) if record is not None else None
    )
    return {
        "extraction_policy": EXTRACTION_POLICY,
        "window_semantics": WINDOW_SEMANTICS,
        "eligible_for_accuracy": ELIGIBLE_FOR_ACCURACY,
        "accuracy_scope": ACCURACY_SCOPE,
        "visual_cue_present": VISUAL_CUE_PRESENT,
        "visual_cue_duration_seconds": VISUAL_CUE_DURATION_SECONDS,
        "eligible_for_pure_imagery_accuracy": ELIGIBLE_FOR_PURE_IMAGERY_ACCURACY,
        "endpoint_tolerance_seconds": endpoint_tolerance_seconds,
        "endpoint_tolerance_samples": endpoint_samples,
        "canonical_trial_samples": round(expected_duration_seconds * record.sampling_rate) if record is not None else None,
        "rest_offset_samples": {
            "min": min(offsets) if offsets else None,
            "max": max(offsets) if offsets else None,
            "mean": float(np.mean(offsets)) if offsets else None,
            "p95": float(np.percentile(offsets, 95)) if offsets else None,
        },
        "endpoint_qc_failed_trials": sum(not trial.endpoint_qc_passed for trial in trials),
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
    endpoint_tolerance_seconds: float,
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
        "record_duration_seconds": None,
        "marker_counts": {},
        "bdf_sha256": None,
        "csv_sha256": None,
        "source_format": None,
        "conversion_tool": None,
        "conversion_tool_version": None,
        "reader_name": None,
        "reader_version": None,
        "unit_evidence_level": None,
        **_summary_metrics((), None, expected_duration_seconds=expected_duration_seconds, endpoint_tolerance_seconds=endpoint_tolerance_seconds),
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
        base["record_duration_seconds"] = record.eeg.shape[1] / record.sampling_rate
        base["marker_counts"] = dict(Counter(str(event.code) for event in record.events))
        base["csv_row_count"] = len(collect_csv.rows)
        base["bdf_sha256"] = record.source_sha256
        base["csv_sha256"] = collect_csv.source_sha256
        base["source_format"] = record.metadata.get("source_format")
        base["conversion_tool"] = record.metadata.get("conversion_tool")
        base["conversion_tool_version"] = record.metadata.get("conversion_tool_version")
        base["reader_name"] = record.metadata.get("reader_name")
        base["reader_version"] = record.metadata.get("reader_version")
        base["unit_evidence_level"] = record.unit_evidence.evidence_level
        aligned_events = align_events_with_csv(record.events, collect_csv.to_alignment_rows())
        aligned_record = replace(
            record,
            events=aligned_events,
            metadata={**record.metadata, "csv_sha256": collect_csv.source_sha256},
        )
        base["aligned_event_count"] = len(aligned_events)
        trials = extract_imagery_trials(
            aligned_record,
            expected_duration_seconds=expected_duration_seconds,
            duration_tolerance_seconds=duration_tolerance_seconds,
            endpoint_tolerance_seconds=endpoint_tolerance_seconds,
        )
        base.update(_summary_metrics(trials, aligned_record, expected_duration_seconds=expected_duration_seconds, endpoint_tolerance_seconds=endpoint_tolerance_seconds))
        base["_trial_inventory"] = [
            {
                "relative_directory": relative_directory,
                "label": trial.label,
                "block_id": trial.block_id,
                "trial_id": trial.trial_id,
                "start_sample": trial.start_sample,
                "end_sample": trial.end_sample,
                "duration_seconds": trial.duration_seconds,
                "observed_event_n_samples": trial.observed_event_n_samples,
                "canonical_n_samples": trial.canonical_n_samples,
                "rest_offset_samples": trial.rest_offset_samples,
                "rest_offset_seconds": trial.rest_offset_seconds,
                "endpoint_qc_passed": trial.endpoint_qc_passed,
                "window_semantics": trial.window_semantics,
                "eligible_for_accuracy": trial.eligible_for_accuracy,
                "accuracy_scope": trial.accuracy_scope,
                "visual_cue_present": trial.visual_cue_present,
                "visual_cue_duration_seconds": trial.visual_cue_duration_seconds,
                "eligible_for_pure_imagery_accuracy": trial.eligible_for_pure_imagery_accuracy,
                "bdf_sha256": trial.source_metadata.get("bdf_sha256"),
                "csv_sha256": trial.source_metadata.get("csv_sha256"),
            }
            for trial in trials
        ]
        base["_channel_metrics"] = [
            {
                "relative_directory": relative_directory,
                "channel_name": name,
                "min": float(np.min(aligned_record.eeg[index])),
                "max": float(np.max(aligned_record.eeg[index])),
                "std": float(np.std(aligned_record.eeg[index])),
                "constant": bool(np.all(aligned_record.eeg[index] == aligned_record.eeg[index, 0])),
                "has_nan": bool(np.isnan(aligned_record.eeg[index]).any()),
                "has_inf": bool(np.isinf(aligned_record.eeg[index]).any()),
            }
            for index, name in enumerate(aligned_record.channel_names)
        ]
        base["_event_alignment"] = [
            {
                "relative_directory": relative_directory,
                "index": index,
                "sample_index": event.sample_index,
                "bdf_code": event.code,
                "csv_event_code": row["event_code"],
                "event_type": event.event_type,
                "block_id": event.block_id,
                "trial_id": event.trial_id,
            }
            for index, (event, row) in enumerate(
                zip(aligned_events, collect_csv.to_alignment_rows(), strict=True)
            )
        ]
        base["passed"] = True
        return base
    except (ValueError, FileNotFoundError) as exc:
        if record is not None:
            base["bdf_event_count"] = len(record.events)
            base.update(_summary_metrics((), record, expected_duration_seconds=expected_duration_seconds, endpoint_tolerance_seconds=endpoint_tolerance_seconds))
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
    endpoint_tolerance_seconds: float = 0.05,
) -> dict[str, object]:
    """Audit every candidate session directory and continue after per-session failures."""
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)

    session_reports: list[dict[str, object]] = []
    trial_inventory: list[dict[str, object]] = []
    channel_metrics: list[dict[str, object]] = []
    event_alignment: list[dict[str, object]] = []
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
                endpoint_tolerance_seconds=endpoint_tolerance_seconds,
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
                endpoint_tolerance_seconds=endpoint_tolerance_seconds,
            )
            report["error"] = "; ".join(errors)
        trial_inventory.extend(report.pop("_trial_inventory", []))
        channel_metrics.extend(report.pop("_channel_metrics", []))
        event_alignment.extend(report.pop("_event_alignment", []))
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
        "_qc_trial_inventory": trial_inventory,
        "_qc_channel_metrics": channel_metrics,
        "_qc_event_alignment": event_alignment,
    }


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_qc_outputs(output_path: Path, report: dict[str, object]) -> None:
    """Write the standard metadata-only Stage 2A QC artifact set."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trial_inventory = report.pop("_qc_trial_inventory", [])
    channel_metrics = report.pop("_qc_channel_metrics", [])
    event_alignment = report.pop("_qc_event_alignment", [])
    summary_json = json.dumps(report, indent=2, ensure_ascii=False)
    output_path.write_text(summary_json, encoding="utf-8")
    if output_path.name != "dataset_summary.json":
        (output_path.parent / "dataset_summary.json").write_text(summary_json, encoding="utf-8")
    (output_path.parent / "session_qc.json").write_text(
        json.dumps(report.get("session_reports", []), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(
        output_path.parent / "trial_inventory.csv",
        trial_inventory,
        ["relative_directory", "label", "block_id", "trial_id", "start_sample", "end_sample", "duration_seconds", "observed_event_n_samples", "canonical_n_samples", "rest_offset_samples", "rest_offset_seconds", "endpoint_qc_passed", "window_semantics", "eligible_for_accuracy", "accuracy_scope", "visual_cue_present", "visual_cue_duration_seconds", "eligible_for_pure_imagery_accuracy", "bdf_sha256", "csv_sha256"],
    )
    _write_csv(
        output_path.parent / "channel_metrics.csv",
        channel_metrics,
        ["relative_directory", "channel_name", "min", "max", "std", "constant", "has_nan", "has_inf"],
    )
    _write_csv(
        output_path.parent / "event_alignment.csv",
        event_alignment,
        ["relative_directory", "index", "sample_index", "bdf_code", "csv_event_code", "event_type", "block_id", "trial_id"],
    )
    manifest = {
        "window_semantics": WINDOW_SEMANTICS,
        "eligible_for_accuracy": ELIGIBLE_FOR_ACCURACY,
        "accuracy_scope": ACCURACY_SCOPE,
        "visual_cue_present": VISUAL_CUE_PRESENT,
        "visual_cue_duration_seconds": VISUAL_CUE_DURATION_SECONDS,
        "eligible_for_pure_imagery_accuracy": ELIGIBLE_FOR_PURE_IMAGERY_ACCURACY,
        "extraction_policy": EXTRACTION_POLICY,
        "sessions": [
            {
                key: session.get(key)
                for key in ("relative_directory", "bdf_filename", "csv_filename", "bdf_sha256", "csv_sha256", "source_format", "conversion_tool", "conversion_tool_version", "reader_name", "reader_version", "unit_evidence_level")
            }
            for session in report.get("session_reports", [])
        ],
        "trials": trial_inventory,
    }
    (output_path.parent / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-duration-seconds", type=float, default=4.0)
    parser.add_argument("--duration-tolerance-seconds", type=float, default=0.1)
    parser.add_argument("--endpoint-tolerance-seconds", type=float, default=0.05)
    args = parser.parse_args(argv)

    try:
        report = audit_dataset(
            args.root,
            expected_duration_seconds=args.expected_duration_seconds,
            duration_tolerance_seconds=args.duration_tolerance_seconds,
            endpoint_tolerance_seconds=args.endpoint_tolerance_seconds,
        )
    except (ValueError, FileNotFoundError) as exc:
        report = {"all_sessions_passed": False, "error": str(exc), "session_reports": []}

    write_qc_outputs(args.output, report)
    return 0 if report["all_sessions_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
