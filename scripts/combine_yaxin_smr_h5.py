from __future__ import annotations

"""Combine the two known yaxin SMR-control H5 files without altering sources.

The result remains flat canonical EEGHDF5.  It deliberately performs no signal
processing: Source S01 is copied verbatim and Source S02 is copied verbatim
after name-validated removal of six non-EEG auxiliary channels.
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

try:  # Script execution.
    from _bootstrap import ROOT
except ModuleNotFoundError:  # Imported by tests.
    from scripts._bootstrap import ROOT

from bci_dayloop.data.hdf5_dataset import EEGHDF5, HDF5Metadata, write_hdf5
from bci_dayloop.data.trial_reader import open_trial_reader
from bci_dayloop.models.model_50m.config import STANDARD_64_CHANNELS
from bci_dayloop.models.model_50m.preprocessing import _align_to_standard_channels


CANONICAL_SUBJECT_ID = 1
CANONICAL_SUBJECT_NAME = "yaxin"
CLASS_NAMES = ["left_hand", "right_hand", "both_hand", "rest"]
SAMPLE_RATE = 1000.0
UNIT = "uV"
DATASET_NAME = "smr_control"
EXPECTED_SHAPE_TAIL = (2000,)
DROP_CHANNELS = ("ECG", "HEOR", "HEOL", "VEOU", "VEOL", "Trigger")

SOURCE_S01_SESSIONS = (
    "neuracle_smr-adapt_S01_0819_163555",
    "neuracle_smr-adapt_S01_0819_164258",
)
SOURCE_S02_SESSIONS = (
    "neuracle_smr-adapt_S02_0819_172732",
    "neuracle_smr-adapt_S02_0819_174527",
    "neuracle_smr-adapt_S02_0819_180141",
    "neuracle_smr-adapt_S02_0819_182838",
)
CANONICAL_SESSION_ORDER = (
    *SOURCE_S01_SESSIONS,
    *SOURCE_S02_SESSIONS,
)
CANONICAL_SESSION_IDS = tuple(f"S{index}" for index in range(1, 7))
SESSION_MAPPING = dict(zip(CANONICAL_SESSION_IDS, CANONICAL_SESSION_ORDER, strict=True))
EXPECTED_SESSION_COUNTS = {"S1": 40, "S2": 80, "S3": 40, "S4": 80, "S5": 40, "S6": 80}
RECOMMENDED_SPLIT = {
    "train_sessions": ["S1", "S2", "S3", "S4"],
    "validation_sessions": ["S5"],
    "test_sessions": ["S6"],
}


@dataclass(frozen=True)
class SourceFile:
    path: Path
    source_file_id: int
    expected_subject_id: int
    expected_sessions: tuple[str, ...]


@dataclass(frozen=True)
class SourcePayload:
    source: SourceFile
    data: np.ndarray
    labels: np.ndarray
    subject_ids: np.ndarray
    session_ids: np.ndarray
    trial_ids: np.ndarray
    channel_names: list[str]


def _string_array(values: Iterable[str]) -> np.ndarray:
    return np.asarray(list(values), dtype=h5py.string_dtype(encoding="utf-8"))


def _decode_strings(values: np.ndarray) -> np.ndarray:
    result: list[str] = []
    for value in np.asarray(values).reshape(-1):
        result.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
    return np.asarray(result, dtype=object)


def _read_json_attr(handle: h5py.File, name: str) -> list[str]:
    if name not in handle.attrs:
        raise KeyError(f"{handle.filename}: missing required root attr {name!r}.")
    value = json.loads(str(handle.attrs[name]))
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{handle.filename}: attr {name!r} must be a JSON string list.")
    return value


def read_source(source: SourceFile) -> SourcePayload:
    if not source.path.is_file():
        raise FileNotFoundError(f"Source H5 is missing: {source.path}")
    with h5py.File(source.path, "r") as handle:
        required = {"data", "labels", "subject_ids", "session_ids", "trial_ids"}
        missing = required - set(handle.keys())
        if missing:
            raise ValueError(f"{source.path}: missing canonical datasets {sorted(missing)}.")
        channel_names = _read_json_attr(handle, "channel_names")
        class_names = _read_json_attr(handle, "class_names")
        if class_names != CLASS_NAMES:
            raise ValueError(f"{source.path}: class semantics differ: {class_names}.")
        if str(handle.attrs.get("dataset_name", "")) != DATASET_NAME:
            raise ValueError(f"{source.path}: expected dataset_name={DATASET_NAME!r}.")
        if float(handle.attrs.get("sample_rate", 0.0)) != SAMPLE_RATE:
            raise ValueError(f"{source.path}: expected sample_rate={SAMPLE_RATE}.")
        if str(handle.attrs.get("unit", "")) != UNIT:
            raise ValueError(f"{source.path}: expected unit={UNIT!r}.")
        data = handle["data"][:].astype(np.float32, copy=False)
        labels = handle["labels"][:].astype(np.int64, copy=False)
        subject_ids = handle["subject_ids"][:].astype(np.int64, copy=False)
        session_ids = _decode_strings(handle["session_ids"][:])
        trial_ids = handle["trial_ids"][:].astype(np.int64, copy=False)
    n_trials = len(data)
    if data.ndim != 3 or data.shape[2:] != EXPECTED_SHAPE_TAIL:
        raise ValueError(f"{source.path}: expected [N,C,2000], got {data.shape}.")
    if len(channel_names) != data.shape[1]:
        raise ValueError(f"{source.path}: channel metadata does not match data shape.")
    if any(len(values) != n_trials for values in (labels, subject_ids, session_ids, trial_ids)):
        raise ValueError(f"{source.path}: trial-level lengths do not match /data.")
    if not np.isfinite(data).all():
        raise ValueError(f"{source.path}: source data contains NaN or Inf.")
    if not np.array_equal(np.unique(subject_ids), np.asarray([source.expected_subject_id])):
        raise ValueError(
            f"{source.path}: expected one original subject ID {source.expected_subject_id}, "
            f"got {np.unique(subject_ids).tolist()}."
        )
    if not np.isin(labels, np.arange(len(CLASS_NAMES))).all():
        raise ValueError(f"{source.path}: labels are outside 0..3.")
    if len(np.unique(trial_ids)) != n_trials or not np.all(np.diff(trial_ids) > 0):
        raise ValueError(f"{source.path}: source trial IDs must be unique and strictly increasing.")
    actual_sessions = tuple(sorted(set(session_ids.tolist())))
    if actual_sessions != tuple(sorted(source.expected_sessions)):
        raise ValueError(
            f"{source.path}: source session names differ. "
            f"Expected {source.expected_sessions}, got {actual_sessions}."
        )
    return SourcePayload(
        source=source,
        data=data,
        labels=labels,
        subject_ids=subject_ids,
        session_ids=session_ids,
        trial_ids=trial_ids,
        channel_names=channel_names,
    )


def filter_s02_auxiliary_channels(
    *,
    s01_channel_names: list[str],
    s02_data: np.ndarray,
    s02_channel_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Drop exactly the declared auxiliary names, then require exact 59-channel order."""
    if len(s02_channel_names) != s02_data.shape[1]:
        raise ValueError("S02 data/channel metadata length mismatch.")
    drop_set = set(DROP_CHANNELS)
    if len(drop_set) != len(DROP_CHANNELS):
        raise RuntimeError("Configured S02 auxiliary channels are not unique.")
    indices = np.asarray(
        [index for index, name in enumerate(s02_channel_names) if name not in drop_set],
        dtype=np.int64,
    )
    dropped = [name for name in s02_channel_names if name in drop_set]
    if tuple(dropped) != DROP_CHANNELS:
        raise ValueError(f"S02 auxiliary channels differ: expected {DROP_CHANNELS}, got {dropped}.")
    if set(s02_channel_names) & drop_set != drop_set:
        raise ValueError("S02 is missing an expected auxiliary channel.")
    filtered_names = [s02_channel_names[index] for index in indices]
    if filtered_names != s01_channel_names:
        raise ValueError("S02 retained channel order differs from S01's 59-channel contract.")
    return s02_data[:, indices, :], indices


def _session_rows(payload: SourcePayload, source_session_id: str) -> np.ndarray:
    rows = np.flatnonzero(payload.session_ids == source_session_id)
    if not len(rows):
        raise ValueError(f"{payload.source.path}: session {source_session_id!r} has no trials.")
    source_trials = payload.trial_ids[rows]
    if not np.all(np.diff(source_trials) > 0):
        raise ValueError(f"{payload.source.path}:{source_session_id}: trial IDs are not ordered.")
    counts = np.bincount(payload.labels[rows], minlength=len(CLASS_NAMES))
    expected = len(rows) // len(CLASS_NAMES)
    if len(rows) not in (40, 80) or not np.array_equal(counts, np.full(4, expected)):
        raise ValueError(
            f"{payload.source.path}:{source_session_id}: expected balanced 40/80 four-class session, "
            f"got {len(rows)} trials and {counts.tolist()}."
        )
    return rows


def build_combined_payload(s01: SourcePayload, s02: SourcePayload) -> dict[str, np.ndarray]:
    filtered_s02_data, kept_s02_indices = filter_s02_auxiliary_channels(
        s01_channel_names=s01.channel_names,
        s02_data=s02.data,
        s02_channel_names=s02.channel_names,
    )
    parts: dict[str, list[np.ndarray]] = {
        "data": [], "labels": [], "session_ids": [], "source_file_ids": [],
        "source_trial_ids": [], "source_subject_ids_original": [], "source_session_ids": [],
    }
    by_session: dict[str, tuple[SourcePayload, np.ndarray, np.ndarray]] = {}
    for source, source_data in ((s01, s01.data), (s02, filtered_s02_data)):
        for source_session in source.source.expected_sessions:
            by_session[source_session] = (source, source_data, _session_rows(source, source_session))
    if tuple(by_session) != CANONICAL_SESSION_ORDER:
        raise RuntimeError("Source sessions are not available in the required canonical time order.")

    for canonical_session, source_session in SESSION_MAPPING.items():
        source, source_data, rows = by_session[source_session]
        parts["data"].append(source_data[rows])
        parts["labels"].append(source.labels[rows])
        parts["session_ids"].append(np.full(len(rows), canonical_session, dtype=object))
        parts["source_file_ids"].append(
            np.full(len(rows), source.source.source_file_id, dtype=np.int64)
        )
        parts["source_trial_ids"].append(source.trial_ids[rows])
        parts["source_subject_ids_original"].append(source.subject_ids[rows])
        parts["source_session_ids"].append(
            np.full(len(rows), source_session, dtype=object)
        )

    data = np.concatenate(parts["data"], axis=0).astype(np.float32, copy=False)
    labels = np.concatenate(parts["labels"], axis=0).astype(np.int64, copy=False)
    sessions = np.concatenate(parts["session_ids"], axis=0)
    n_trials = len(data)
    payload = {
        "data": data,
        "labels": labels,
        "subject_ids": np.full(n_trials, CANONICAL_SUBJECT_ID, dtype=np.int64),
        "session_ids": sessions,
        "trial_ids": np.arange(n_trials, dtype=np.int64),
        "source_file_ids": np.concatenate(parts["source_file_ids"]).astype(np.int64, copy=False),
        "source_trial_ids": np.concatenate(parts["source_trial_ids"]).astype(np.int64, copy=False),
        "source_subject_ids_original": np.concatenate(parts["source_subject_ids_original"]).astype(np.int64, copy=False),
        "source_session_ids": np.concatenate(parts["source_session_ids"]),
        "s02_kept_channel_indices": kept_s02_indices,
    }
    return payload


def _metadata(channel_names: list[str]) -> HDF5Metadata:
    return HDF5Metadata(
        sample_rate=SAMPLE_RATE,
        channel_names=channel_names,
        class_names=list(CLASS_NAMES),
        unit=UNIT,
        dataset_name=DATASET_NAME,
    )


def _write_provenance(
    handle: h5py.File,
    *,
    s01: SourcePayload,
    s02: SourcePayload,
    payload: dict[str, np.ndarray],
) -> None:
    for name in ("source_file_ids", "source_trial_ids", "source_subject_ids_original"):
        handle.create_dataset(name, data=payload[name], dtype="int64")
    handle.create_dataset("source_session_ids", data=_string_array(payload["source_session_ids"]))
    handle.attrs["canonical_subject_name"] = CANONICAL_SUBJECT_NAME
    handle.attrs["canonical_subject_id"] = CANONICAL_SUBJECT_ID
    handle.attrs["source_subject_ids"] = json.dumps(
        {s01.source.path.name: s01.source.expected_subject_id, s02.source.path.name: s02.source.expected_subject_id},
        sort_keys=True,
    )
    handle.attrs["source_file_mapping"] = json.dumps(
        {str(s01.source.source_file_id): s01.source.path.name, str(s02.source.source_file_id): s02.source.path.name},
        sort_keys=True,
    )
    handle.attrs["session_mapping"] = json.dumps(SESSION_MAPPING, sort_keys=True)
    handle.attrs["recommended_split"] = json.dumps(RECOMMENDED_SPLIT, sort_keys=True)
    handle.attrs["channel_normalization"] = json.dumps(
        {
            "source_s01": "59 channels copied unchanged",
            "source_s02_dropped_channels": list(DROP_CHANNELS),
            "signal_preprocessing_applied": False,
        },
        sort_keys=True,
    )
    handle.attrs["trial_id_policy"] = "combined_trial_ids_0_to_359_in_canonical_session_order"


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    return {CLASS_NAMES[index]: int(np.count_nonzero(labels == index)) for index in range(4)}


def verify_combined(
    *,
    output: Path,
    s01: SourcePayload,
    s02: SourcePayload,
    payload: dict[str, np.ndarray],
) -> dict[str, Any]:
    reader = EEGHDF5(output)
    metadata = reader.metadata
    if metadata != _metadata(s01.channel_names):
        raise ValueError(f"{output}: reader metadata differs from the combined contract.")
    expected_sessions = list(CANONICAL_SESSION_IDS)
    if reader.available_sessions() != expected_sessions:
        raise ValueError(f"{output}: reader sessions differ from {expected_sessions}.")
    trial_metadata = open_trial_reader(
        data_reader="eeg", path=output, canonical_subject_id=CANONICAL_SUBJECT_ID
    ).trial_metadata()
    with h5py.File(output, "r") as handle:
        data = handle["data"][:]
        labels = handle["labels"][:]
        subjects = handle["subject_ids"][:]
        sessions = _decode_strings(handle["session_ids"][:])
        trials = handle["trial_ids"][:]
        if data.shape != (360, 59, 2000) or data.dtype != np.dtype("float32"):
            raise ValueError(f"{output}: invalid combined data shape/dtype {data.shape}/{data.dtype}.")
        if not np.array_equal(data, payload["data"]):
            raise ValueError(f"{output}: written data differs from constructed payload.")
        if not np.array_equal(labels, payload["labels"]):
            raise ValueError(f"{output}: labels differ from constructed payload.")
        if not np.array_equal(subjects, np.full(360, CANONICAL_SUBJECT_ID, dtype=np.int64)):
            raise ValueError(f"{output}: subject canonicalization failed.")
        if not np.array_equal(trials, np.arange(360, dtype=np.int64)):
            raise ValueError(f"{output}: combined trial IDs differ from 0..359.")
        if not np.isfinite(data).all():
            raise ValueError(f"{output}: contains NaN or Inf.")
        if not np.array_equal(trial_metadata["trial_ids"], trials):
            raise ValueError(f"{output}: TrialReader trial metadata differs.")
        if json.loads(handle.attrs["session_mapping"]) != SESSION_MAPPING:
            raise ValueError(f"{output}: session mapping provenance differs.")
        if json.loads(handle.attrs["recommended_split"]) != RECOMMENDED_SPLIT:
            raise ValueError(f"{output}: recommended split metadata differs.")
        for session, expected_count in EXPECTED_SESSION_COUNTS.items():
            loaded = reader.load(session)
            if len(loaded["data"]) != expected_count:
                raise ValueError(f"{output}:{session}: expected {expected_count} trials.")
            counts = np.bincount(loaded["labels"], minlength=4)
            if not np.array_equal(counts, np.full(4, expected_count // 4)):
                raise ValueError(f"{output}:{session}: class counts differ: {counts.tolist()}.")

        s01_rows = np.isin(sessions, ("S1", "S2"))
        s02_rows = np.isin(sessions, ("S3", "S4", "S5", "S6"))
        if not np.array_equal(data[s01_rows], s01.data):
            raise ValueError(f"{output}: S01 EEG values are not byte-identical to source.")
        filtered_s02, _indices = filter_s02_auxiliary_channels(
            s01_channel_names=s01.channel_names,
            s02_data=s02.data,
            s02_channel_names=s02.channel_names,
        )
        if not np.array_equal(data[s02_rows], filtered_s02):
            raise ValueError(f"{output}: S02 retained EEG values differ from source.")

        _aligned, mask, _canonical, unknown, duplicates = _align_to_standard_channels(
            np.zeros((len(metadata.channel_names), 2), dtype=np.float32),
            metadata.channel_names,
            0.0,
        )
        missing = [name for name, valid in zip(STANDARD_64_CHANNELS, mask) if not valid]
        return {
            "output_path": str(output),
            "data_shape": list(data.shape),
            "dtype": str(data.dtype),
            "subject_id": CANONICAL_SUBJECT_ID,
            "sessions": {
                session: {
                    "trials": EXPECTED_SESSION_COUNTS[session],
                    "class_counts": _class_counts(reader.load(session)["labels"]),
                }
                for session in expected_sessions
            },
            "total_class_counts": _class_counts(labels),
            "nan_count": int(np.isnan(data).sum()),
            "inf_count": int(np.isinf(data).sum()),
            "s01_equality": True,
            "s02_retained_channel_equality": True,
            "mapped_channels": int(mask.sum()),
            "unknown_channels": list(unknown),
            "missing_standard_channels": missing,
            "duplicate_mappings": int(duplicates),
        }


def combine(
    *,
    source_s01: Path,
    source_s02: Path,
    output: Path,
    overwrite: bool,
) -> dict[str, Any]:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}. Pass --overwrite to replace it.")
    s01 = read_source(SourceFile(source_s01, 0, 1, SOURCE_S01_SESSIONS))
    s02 = read_source(SourceFile(source_s02, 1, 2, SOURCE_S02_SESSIONS))
    payload = build_combined_payload(s01, s02)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        write_hdf5(
            temporary,
            data=payload["data"],
            labels=payload["labels"],
            subject_ids=payload["subject_ids"],
            session_ids=payload["session_ids"].tolist(),
            trial_ids=payload["trial_ids"],
            metadata=_metadata(s01.channel_names),
        )
        with h5py.File(temporary, "a") as handle:
            _write_provenance(handle, s01=s01, s02=s02, payload=payload)
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return verify_combined(output=output, s01=s01, s02=s02, payload=payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine known yaxin SMR-control H5 sources.")
    parser.add_argument(
        "--source-s01", type=Path,
        default=ROOT / "data" / "processed" / "yaxin" / "smr_control_s01_0819.h5",
    )
    parser.add_argument(
        "--source-s02", type=Path,
        default=ROOT / "data" / "processed" / "yaxin" / "smr_control_s02_0819.h5",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data" / "processed" / "yaxin" / "smr_control_yaxin_0819_combined.h5",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output H5.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = combine(
        source_s01=args.source_s01.expanduser().resolve(),
        source_s02=args.source_s02.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
