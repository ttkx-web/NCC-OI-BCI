from __future__ import annotations

"""Read-only inspection for a proposed SMR-control source H5."""

import argparse
import json
from pathlib import Path

try:
    from _bootstrap import ROOT  # noqa: F401 - makes src importable for script use
except ModuleNotFoundError:
    from scripts._bootstrap import ROOT  # noqa: F401

from bci_dayloop.data.smr_manifest import inspect_h5, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an SMR H5 against a dataset manifest without modifying either file.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--reference-canonical", type=Path, help="Optional QC reference only when the manifest has none.")
    return parser.parse_args()


def _print_report(report: dict) -> None:
    print(f"Input: {Path(report['input_file']).name}\n")
    print("Contract")
    for item in report["checks"]:
        if item["name"] in {"schema", "task", "window", "channels", "50m_mapping"}:
            print(f"  {item['name']:<12}{item['status']:<8} {item['message']}")
    duplicate = report["duplicates"]
    print("\nDuplicates")
    print(f"  New         {duplicate['new_trials']}")
    print(f"  Exact       {duplicate['exact_duplicates']}")
    print("\nSessions")
    for session in report["sessions"]:
        print(f"  {session['source_session_id']:<38} {session['status']:<8} {session['trial_count']:>4} trials {session['class_counts']}")
    print(f"\nOverall: {report['overall_status']}")
    print(f"Recommended action: {report['recommended_action']}")


def main() -> None:
    args = parse_args()
    report = inspect_h5(input_path=args.input.resolve(), manifest_path=args.manifest.resolve(), reference_canonical=args.reference_canonical.resolve() if args.reference_canonical else None)
    write_json_atomic(args.output_report.resolve(), report)
    _print_report(report)


if __name__ == "__main__":
    main()
