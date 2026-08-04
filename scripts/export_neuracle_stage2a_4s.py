"""Export one aligned Neuracle/Collect session as fixed 4 s Stage 2A HDF5."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

try:
    from _bootstrap import ROOT
except ModuleNotFoundError:  # Imported as scripts.export_neuracle_stage2a_4s in tests.
    from scripts._bootstrap import ROOT

from bci_dayloop.data.collect_csv import CollectCSV as CollectCSVAdapter
from bci_dayloop.data.event_alignment import align_events_with_csv
from bci_dayloop.data.neuracle_bdf import NeuracleBDFReader
from bci_dayloop.data.records import UnitEvidence
from bci_dayloop.data.stage2a_export import export_stage2a_trials_hdf5
from bci_dayloop.data.trial_extraction import extract_imagery_trials


def _reader() -> NeuracleBDFReader:
    return NeuracleBDFReader(
        UnitEvidence("uV", "uV", "vendor_confirmed", "MNE get_data(..., units='uV')")
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bdf", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-duration-seconds", type=float, default=4.0)
    parser.add_argument("--endpoint-tolerance-seconds", type=float, default=0.05)
    args = parser.parse_args(argv)
    try:
        record = _reader().load(args.bdf)
        collect_csv = CollectCSVAdapter.from_file(args.csv)
        aligned = replace(
            record,
            events=align_events_with_csv(record.events, collect_csv.to_alignment_rows()),
            metadata={**record.metadata, "csv_sha256": collect_csv.source_sha256},
        )
        trials = extract_imagery_trials(
            aligned,
            expected_duration_seconds=args.expected_duration_seconds,
            endpoint_tolerance_seconds=args.endpoint_tolerance_seconds,
        )
        export_stage2a_trials_hdf5(
            args.output,
            trials,
            channel_names=aligned.channel_names,
            sampling_rate=aligned.sampling_rate,
            subject_id=collect_csv.subject,
            session_id=collect_csv.session,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
