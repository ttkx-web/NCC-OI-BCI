from __future__ import annotations

"""Read-only inspection utility for EEG HDF5 files.

The project currently has a canonical trial-level HDF5 layout, but this
script deliberately does not require it.  It always prints the complete HDF5
tree and, when it finds the canonical fields, adds an EEG/trial/session
summary without ever rewriting the source file.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from _bootstrap import ROOT  # noqa: F401
from bci_dayloop.utils.config import resolve_path


_CANONICAL_FIELDS = {
    "data",
    "labels",
    "subject_ids",
    "session_ids",
    "trial_ids",
}


def _decode_value(value: Any) -> Any:
    """Convert HDF5 values into concise, printable Python values."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size <= 20:
            return [_decode_value(item) for item in value.tolist()]
        return f"<array shape={value.shape} dtype={value.dtype}>"
    return value


def _format_value(value: Any) -> str:
    decoded = _decode_value(value)
    if isinstance(decoded, str):
        return decoded
    return json.dumps(decoded, ensure_ascii=False, default=str)


def _print_attributes(attributes: h5py.AttributeManager, *, indent: str) -> None:
    for name in sorted(attributes):
        print(f"{indent}@{name} = {_format_value(attributes[name])}")


def print_h5_tree(handle: h5py.File) -> None:
    """Print every group, dataset, and attribute without reading EEG values."""
    print("=== H5 Structure ===")
    print("/")
    _print_attributes(handle.attrs, indent="  ")

    def visit(group: h5py.Group, *, prefix: str) -> None:
        names = sorted(group.keys())
        for index, name in enumerate(names):
            item = group[name]
            last = index == len(names) - 1
            branch = "└──" if last else "├──"
            child_prefix = prefix + ("    " if last else "│   ")
            if isinstance(item, h5py.Dataset):
                print(
                    f"{prefix}{branch} {item.name} "
                    f"shape={item.shape} dtype={item.dtype}"
                )
                _print_attributes(item.attrs, indent=child_prefix)
            elif isinstance(item, h5py.Group):
                print(f"{prefix}{branch} {item.name}/")
                _print_attributes(item.attrs, indent=child_prefix)
                visit(item, prefix=child_prefix)
            else:
                print(f"{prefix}{branch} {item.name} ({type(item).__name__})")

    visit(handle, prefix="")


def _decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8", errors="replace")
            if isinstance(value, bytes)
            else str(value)
            for value in values
        ],
        dtype=str,
    )


def _parse_json_attribute(handle: h5py.File, name: str) -> Any | None:
    if name not in handle.attrs:
        return None
    value = _decode_value(handle.attrs[name])
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _finite_statistics(dataset: h5py.Dataset) -> dict[str, float | int]:
    """Compute scalar statistics in chunks, avoiding a full EEG allocation."""
    if dataset.ndim == 0 or dataset.dtype.kind not in "fiu":
        return {}

    chunk_size = max(1, min(16, int(dataset.shape[0]))) if dataset.ndim else 1
    finite_count = 0
    value_sum = 0.0
    square_sum = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    nan_count = 0
    inf_count = 0
    for start in range(0, int(dataset.shape[0]), chunk_size):
        values = np.asarray(dataset[start : start + chunk_size], dtype=np.float64)
        finite = np.isfinite(values)
        nan_count += int(np.isnan(values).sum())
        inf_count += int(np.isinf(values).sum())
        selected = values[finite]
        if selected.size == 0:
            continue
        finite_count += int(selected.size)
        value_sum += float(selected.sum())
        square_sum += float(np.square(selected).sum())
        minimum = min(minimum, float(selected.min()))
        maximum = max(maximum, float(selected.max()))

    if finite_count == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "nan_count": nan_count,
            "inf_count": inf_count,
        }
    mean = value_sum / finite_count
    variance = max(square_sum / finite_count - mean**2, 0.0)
    return {
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": float(np.sqrt(variance)),
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


def _format_counts(values: np.ndarray) -> str:
    counts = Counter(values.tolist())
    total = sum(counts.values())
    return ", ".join(
        f"{key}: {count} ({count / total:.1%})"
        for key, count in sorted(counts.items(), key=lambda item: str(item[0]))
    )


def _print_canonical_summary(handle: h5py.File, *, max_trials_preview: int) -> None:
    missing = sorted(_CANONICAL_FIELDS - set(handle.keys()))
    if missing:
        print("=== Semantic Summary ===")
        print(
            "Canonical trial-level summary unavailable; missing root datasets: "
            + ", ".join(missing)
        )
        return

    data = handle["data"]
    labels = np.asarray(handle["labels"], dtype=np.int64)
    subject_ids = np.asarray(handle["subject_ids"], dtype=np.int64)
    session_ids = _decode_strings(np.asarray(handle["session_ids"]))
    trial_ids = np.asarray(handle["trial_ids"], dtype=np.int64)
    arrays = {
        "labels": labels,
        "subject_ids": subject_ids,
        "session_ids": session_ids,
        "trial_ids": trial_ids,
    }
    trial_count = int(data.shape[0]) if data.ndim else 0

    print("=== Semantic Summary ===")
    print("Schema: canonical trial-level root datasets")
    print(f"Trial representation: /data axis 0 (trial count={trial_count})")
    for name, values in arrays.items():
        status = "PASS" if len(values) == trial_count else "FAIL"
        print(f"Alignment /{name}: {len(values)} vs {trial_count} trials ({status})")

    if data.ndim == 3:
        channels = int(data.shape[1])
        samples = int(data.shape[2])
        print("EEG axis convention: axis 0=trial, axis 1=channel, axis 2=time")
        print(f"EEG shape: {data.shape}; dtype: {data.dtype}")
        print(f"Channel count: {channels}; samples per trial: {samples}")
    else:
        print(
            "EEG axis convention: NOT CONFIRMED; expected a rank-3 /data "
            f"dataset, got shape {data.shape}."
        )

    sample_rate = _parse_json_attribute(handle, "sample_rate")
    if sample_rate is None:
        print("Sampling rate: NOT FOUND")
    else:
        print(f"Sampling rate: {sample_rate} Hz")
        if data.ndim == 3 and float(sample_rate) > 0:
            print(f"Trial duration: {data.shape[2] / float(sample_rate):.6g} s")

    channel_names = _parse_json_attribute(handle, "channel_names")
    if isinstance(channel_names, list):
        print(
            f"Channel names ({len(channel_names)}): "
            + ", ".join(str(name) for name in channel_names)
        )
        if data.ndim == 3:
            print(
                "Channel metadata alignment: "
                + ("PASS" if len(channel_names) == data.shape[1] else "FAIL")
            )
    else:
        print("Channel names: NOT FOUND")

    class_names = _parse_json_attribute(handle, "class_names")
    print(f"Class names: {class_names if class_names is not None else 'NOT FOUND'}")
    unit = _parse_json_attribute(handle, "unit")
    print(f"Unit: {unit if unit is not None else 'NOT FOUND'}")
    print(f"Dataset name: {_parse_json_attribute(handle, 'dataset_name') or 'NOT FOUND'}")

    statistics = _finite_statistics(data)
    if statistics:
        formatted = ", ".join(f"{name}={value}" for name, value in statistics.items())
        print(f"EEG value statistics: {formatted}")

    print(f"Subjects: {_format_counts(subject_ids)}")
    unique_sessions = list(dict.fromkeys(session_ids.tolist()))
    print(f"Sessions ({len(unique_sessions)}): {', '.join(unique_sessions)}")
    print(f"Overall labels: {_format_counts(labels)}")
    print(
        "Trial IDs: "
        f"unique={len(np.unique(trial_ids))}/{len(trial_ids)}, "
        f"ordered_non_decreasing={bool(np.all(np.diff(trial_ids) >= 0))}"
    )

    for session_name in unique_sessions:
        mask = session_ids == session_name
        indices = np.flatnonzero(mask)
        session_labels = labels[indices]
        session_trials = trial_ids[indices]
        print(f"--- Session: {session_name} ---")
        print(f"Trials: {len(indices)}; labels: {_format_counts(session_labels)}")
        print(
            "Trial IDs: "
            f"min={int(session_trials.min())}, max={int(session_trials.max())}, "
            f"unique={len(np.unique(session_trials))}/{len(session_trials)}, "
            f"ordered_non_decreasing={bool(np.all(np.diff(session_trials) >= 0))}"
        )
        preview = [
            f"{int(trial_ids[index])}->{int(labels[index])}"
            for index in indices[:max_trials_preview]
        ]
        print("Trial -> label preview: " + ", ".join(preview))


def inspect_h5(path: Path, *, max_trials_preview: int) -> None:
    with h5py.File(path, "r") as handle:
        print(f"File: {path}")
        print_h5_tree(handle)
        _print_canonical_summary(handle, max_trials_preview=max_trials_preview)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only structural and semantic inspection for EEG HDF5 files."
    )
    parser.add_argument("--data", required=True, help="Path to an HDF5 file.")
    parser.add_argument(
        "--max-trials-preview",
        type=int,
        default=20,
        help="Maximum trial_id -> label pairs shown per session (default: 20).",
    )
    args = parser.parse_args()
    if args.max_trials_preview < 0:
        parser.error("--max-trials-preview must be >= 0.")
    return args


def main() -> None:
    args = parse_args()
    path = resolve_path(args.data)
    if not path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    inspect_h5(path, max_trials_preview=args.max_trials_preview)


if __name__ == "__main__":
    main()
