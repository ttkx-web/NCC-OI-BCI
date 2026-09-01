"""Convert published preprocessed Workload EEGLAB epochs into grouped HDF5."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import h5py

try:
    from _bootstrap import ROOT  # noqa: F401
except ModuleNotFoundError:  # Imported as scripts.prepare_workload_hdf5 in tests.
    from scripts._bootstrap import ROOT  # noqa: F401

from bci_dayloop.data.workload import WorkloadHDF5, prepare_workload_subject


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--subjects", required=True, nargs="+", type=int)
    parser.add_argument("--sessions", required=True, nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _print_subject_summary(path: Path) -> None:
    dataset = WorkloadHDF5(path)
    with h5py.File(path, "r") as handle:
        print(f"Subject: {handle.attrs['subject_id']}")
        for session_id in dataset.sessions():
            group = handle["sessions"][session_id]
            loaded = dataset.load(session=session_id)
            labels = loaded["labels"]
            print(f"Session: {session_id}")
            print(f"Data shape: {tuple(loaded['data'].shape)}")
            print(f"Easy epoch count: {group.attrs['num_easy_epochs']}")
            print(f"Diff epoch count: {group.attrs['num_diff_epochs']}")
            print(f"Label counts: 0={int((labels == 0).sum())}, 1={int((labels == 1).sum())}")
            print("First 10 labels: " + " ".join(str(value) for value in labels[:10]))
            print("Last 10 labels: " + " ".join(str(value) for value in labels[-10:]))
            print(f"Sample rate: {group.attrs['sample_rate']}")
            print(f"Window duration: {loaded['data'].shape[2] / group.attrs['sample_rate']}")
            print(f"Channel count: {group.attrs['num_channels']}")
            print("First 5 window IDs: " + ", ".join(loaded["window_ids"][:5]))
            print("Last 5 window IDs: " + ", ".join(loaded["window_ids"][-5:]))
        print(f"Output path: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        for subject in args.subjects:
            output = prepare_workload_subject(
                args.data_root,
                args.output_root,
                subject=subject,
                sessions=args.sessions,
                overwrite=args.overwrite,
            )
            _print_subject_summary(output)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
