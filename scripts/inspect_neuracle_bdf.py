"""Produce a metadata-first health report for a Neuracle-converted BDF file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import mne
import numpy as np

from bci_dayloop.data.neuracle_bdf import parse_neuracle_marker


_EXCLUDED_NAMES = frozenset({"ecg", "heor", "heol", "veou", "veol"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _marker_counts(annotations: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for description in annotations.description:
        parsed = parse_neuracle_marker(description)
        marker_code = parsed["metadata"]["marker_code"]  # type: ignore[index]
        key = str(marker_code) if isinstance(marker_code, int) else "unparsed"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _empty_qc() -> dict[str, object]:
    return {
        "checked": False,
        "has_nan": None,
        "has_inf": None,
        "constant_channel_names": [],
    }


def inspect_neuracle_bdf(path: str | Path, *, max_seconds: float = 0) -> dict[str, object]:
    """Inspect metadata and, optionally, a bounded leading EEG segment in microvolts."""
    if max_seconds < 0:
        raise ValueError("max_seconds must be non-negative")

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    raw = mne.io.read_raw_bdf(str(source_path), preload=False, verbose="ERROR")
    channel_names = tuple(raw.ch_names)
    channel_types = tuple(raw.get_channel_types())
    if len(channel_names) != len(channel_types):
        raise ValueError("BDF channel names and types have different lengths")

    sampling_rate = float(raw.info["sfreq"])
    n_times = int(raw.n_times)
    excluded_names = tuple(
        name for name in channel_names if name.strip().lower() in _EXCLUDED_NAMES
    )
    eeg_indices = [
        index
        for index, (name, channel_type) in enumerate(zip(channel_names, channel_types, strict=True))
        if channel_type == "eeg" and name.strip().lower() not in _EXCLUDED_NAMES
    ]

    checked_samples = 0
    segment_qc = _empty_qc()
    if max_seconds > 0:
        checked_samples = min(n_times, round(max_seconds * sampling_rate))
        segment = raw.get_data(
            picks=eeg_indices,
            start=0,
            stop=checked_samples,
            units="uV",
        )
        constant_channel_names: list[str] = []
        if checked_samples > 0:
            constant_channel_names = [
                channel_names[index]
                for position, index in enumerate(eeg_indices)
                if np.all(segment[position] == segment[position, 0])
            ]
        segment_qc = {
            "checked": True,
            "has_nan": bool(np.isnan(segment).any()),
            "has_inf": bool(np.isinf(segment).any()),
            "constant_channel_names": constant_channel_names,
        }

    return {
        "source_path": str(source_path),
        "sha256": _sha256(source_path),
        "channel_names": list(channel_names),
        "channel_types": list(channel_types),
        "channel_count": len(channel_names),
        "sampling_rate": sampling_rate,
        "n_times": n_times,
        "duration_seconds": n_times / sampling_rate,
        "unit": "uV",
        "event_count": len(raw.annotations.description),
        "marker_counts": _marker_counts(raw.annotations),
        "excluded_channel_names": list(excluded_names),
        "short_segment_qc": segment_qc,
        "checked_seconds": checked_samples / sampling_rate,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-seconds", type=float, default=0)
    args = parser.parse_args(argv)
    if args.max_seconds < 0:
        parser.error("--max-seconds must be non-negative")

    report = inspect_neuracle_bdf(args.input, max_seconds=args.max_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
