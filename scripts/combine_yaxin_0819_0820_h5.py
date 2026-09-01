from __future__ import annotations

"""Build the clean 0819 + 0820 yaxin SMR-control canonical HDF5.

This conversion is deliberately lossless for retained EEG samples.  It copies
0819 canonical rows byte-for-byte and appends only the five validated balanced
0820 sessions after removing six explicitly named auxiliary channels.  The
known 0820 partial/anomalous session and the duplicate-only 0821 export are
never read as EEG inputs.
"""

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

try:  # Script execution.
    from _bootstrap import ROOT
except ModuleNotFoundError:  # Test import.
    from scripts._bootstrap import ROOT

from bci_dayloop.data.hdf5_dataset import EEGHDF5, HDF5Metadata
from bci_dayloop.data.trial_reader import open_trial_reader
from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.models.model_50m.preprocessing import Model50MPreprocessor


CANONICAL_SUBJECT_ID = 1
CANONICAL_SUBJECT_NAME = "yaxin"
CLASS_NAMES = ["left_hand", "right_hand", "both_hand", "rest"]
SAMPLE_RATE = 1000.0
UNIT = "uV"
DATASET_NAME = "smr_control"
EXPECTED_0819_SESSIONS = ("S1", "S2", "S3", "S4", "S5", "S6")
EXPECTED_0819_COUNTS = (40, 80, 40, 80, 40, 80)
SOURCE_0820_SESSIONS = (
    "neuracle_smr-adapt_S02_0820_163122",
    "neuracle_smr-adapt_S02_0820_164356",
    "neuracle_smr-adapt_S02_0820_173404",
    "neuracle_smr-adapt_S02_0820_175613",
    "neuracle_smr-adapt_S02_0820_181835",
)
SOURCE_0820_COUNTS = (40, 80, 80, 80, 40)
EXCLUDED_0820_SESSION = "neuracle_smr-adapt_S02_0820_184207"
DROP_CHANNELS = ("ECG", "HEOR", "HEOL", "VEOU", "VEOL", "Trigger")
CANONICAL_SESSIONS = tuple(f"S{index}" for index in range(1, 12))
EXPECTED_SESSION_COUNTS = dict(
    zip(CANONICAL_SESSIONS, (*EXPECTED_0819_COUNTS, *SOURCE_0820_COUNTS), strict=True)
)


@dataclass(frozen=True)
class Source0819:
    path: Path
    data: np.ndarray
    labels: np.ndarray
    subject_ids: np.ndarray
    session_ids: np.ndarray
    trial_ids: np.ndarray
    channel_names: list[str]
    source_file_ids: np.ndarray
    source_trial_ids: np.ndarray
    source_subject_ids_original: np.ndarray
    source_session_ids: np.ndarray
    source_file_mapping: dict[str, str]
    source_subject_ids: dict[str, int]


@dataclass(frozen=True)
class Source0820:
    path: Path
    data: np.ndarray
    labels: np.ndarray
    subject_ids: np.ndarray
    session_ids: np.ndarray
    trial_ids: np.ndarray
    channel_names: list[str]


def _string_array(values: Iterable[str]) -> np.ndarray:
    return np.asarray(list(values), dtype=h5py.string_dtype(encoding="utf-8"))


def _decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


def _json_attr(handle: h5py.File, name: str) -> Any:
    if name not in handle.attrs:
        raise KeyError(f"{handle.filename}: missing root attr {name!r}.")
    return json.loads(str(handle.attrs[name]))


def _validate_base_contract(
    *, handle: h5py.File, data: np.ndarray, labels: np.ndarray,
    subject_ids: np.ndarray, session_ids: np.ndarray, trial_ids: np.ndarray,
    channel_names: list[str],
) -> None:
    if data.ndim != 3 or data.shape[2:] != (2000,):
        raise ValueError(f"{handle.filename}: expected [N,C,2000], got {data.shape}.")
    if data.dtype != np.dtype("float32"):
        raise ValueError(f"{handle.filename}: expected float32 EEG, got {data.dtype}.")
    if len(channel_names) != data.shape[1]:
        raise ValueError(f"{handle.filename}: channel metadata/data mismatch.")
    if any(len(value) != len(data) for value in (labels, subject_ids, session_ids, trial_ids)):
        raise ValueError(f"{handle.filename}: trial-level lengths mismatch.")
    if not np.isfinite(data).all():
        raise ValueError(f"{handle.filename}: contains NaN or Inf.")
    if not np.array_equal(np.unique(labels), np.arange(4, dtype=np.int64)):
        raise ValueError(f"{handle.filename}: expected labels 0..3.")
    if _json_attr(handle, "class_names") != CLASS_NAMES:
        raise ValueError(f"{handle.filename}: class semantics differ.")
    if str(handle.attrs.get("dataset_name", "")) != DATASET_NAME:
        raise ValueError(f"{handle.filename}: dataset_name differs.")
    if float(handle.attrs.get("sample_rate", 0.0)) != SAMPLE_RATE:
        raise ValueError(f"{handle.filename}: sample_rate differs.")
    if str(handle.attrs.get("unit", "")) != UNIT:
        raise ValueError(f"{handle.filename}: unit differs.")


def read_0819(path: Path) -> Source0819:
    with h5py.File(path, "r") as handle:
        required = {
            "data", "labels", "subject_ids", "session_ids", "trial_ids",
            "source_file_ids", "source_trial_ids", "source_subject_ids_original",
            "source_session_ids",
        }
        missing = required - set(handle.keys())
        if missing:
            raise ValueError(f"{path}: missing 0819 provenance datasets {sorted(missing)}.")
        data = handle["data"][:]
        labels = handle["labels"][:].astype(np.int64, copy=False)
        subject_ids = handle["subject_ids"][:].astype(np.int64, copy=False)
        session_ids = _decode_strings(handle["session_ids"][:])
        trial_ids = handle["trial_ids"][:].astype(np.int64, copy=False)
        channel_names = _json_attr(handle, "channel_names")
        _validate_base_contract(
            handle=handle, data=data, labels=labels, subject_ids=subject_ids,
            session_ids=session_ids, trial_ids=trial_ids, channel_names=channel_names,
        )
        if tuple(dict.fromkeys(session_ids.tolist())) != EXPECTED_0819_SESSIONS:
            raise ValueError(f"{path}: expected canonical sessions {EXPECTED_0819_SESSIONS}.")
        for session, expected_count in zip(EXPECTED_0819_SESSIONS, EXPECTED_0819_COUNTS, strict=True):
            _session_rows(
                session_ids=session_ids, labels=labels,
                source_session=session, expected_count=expected_count,
            )
        if not np.array_equal(subject_ids, np.full(360, CANONICAL_SUBJECT_ID, dtype=np.int64)):
            raise ValueError(f"{path}: 0819 canonical subject IDs are invalid.")
        if not np.array_equal(trial_ids, np.arange(360, dtype=np.int64)):
            raise ValueError(f"{path}: 0819 trial IDs are invalid.")
        if json.loads(str(handle.attrs["source_file_mapping"])).keys() != {"0", "1"}:
            raise ValueError(f"{path}: expected two original 0819 source file IDs.")
        return Source0819(
            path=path, data=data, labels=labels, subject_ids=subject_ids,
            session_ids=session_ids, trial_ids=trial_ids, channel_names=channel_names,
            source_file_ids=handle["source_file_ids"][:].astype(np.int64, copy=False),
            source_trial_ids=handle["source_trial_ids"][:].astype(np.int64, copy=False),
            source_subject_ids_original=handle["source_subject_ids_original"][:].astype(np.int64, copy=False),
            source_session_ids=_decode_strings(handle["source_session_ids"][:]),
            source_file_mapping=json.loads(str(handle.attrs["source_file_mapping"])),
            source_subject_ids={
                str(name): int(subject)
                for name, subject in json.loads(str(handle.attrs["source_subject_ids"])).items()
            },
        )


def read_0820(path: Path) -> Source0820:
    with h5py.File(path, "r") as handle:
        required = {"data", "labels", "subject_ids", "session_ids", "trial_ids"}
        missing = required - set(handle.keys())
        if missing:
            raise ValueError(f"{path}: missing canonical datasets {sorted(missing)}.")
        data = handle["data"][:]
        labels = handle["labels"][:].astype(np.int64, copy=False)
        subject_ids = handle["subject_ids"][:].astype(np.int64, copy=False)
        session_ids = _decode_strings(handle["session_ids"][:])
        trial_ids = handle["trial_ids"][:].astype(np.int64, copy=False)
        channel_names = _json_attr(handle, "channel_names")
        _validate_base_contract(
            handle=handle, data=data, labels=labels, subject_ids=subject_ids,
            session_ids=session_ids, trial_ids=trial_ids, channel_names=channel_names,
        )
    if not np.array_equal(np.unique(subject_ids), np.asarray([2], dtype=np.int64)):
        raise ValueError(f"{path}: expected source subject ID 2.")
    return Source0820(path, data, labels, subject_ids, session_ids, trial_ids, channel_names)


def retained_0820_channel_indices(
    *, canonical_channels: list[str], source_channels: list[str],
) -> np.ndarray:
    indices = np.asarray(
        [index for index, name in enumerate(source_channels) if name not in set(DROP_CHANNELS)],
        dtype=np.int64,
    )
    dropped = tuple(name for name in source_channels if name in set(DROP_CHANNELS))
    if dropped != DROP_CHANNELS:
        raise ValueError(f"0820 auxiliary channels differ: got {dropped}, expected {DROP_CHANNELS}.")
    retained_names = [source_channels[index] for index in indices]
    if retained_names != canonical_channels:
        raise ValueError("0820 retained channel names/order do not exactly match 0819 canonical list.")
    return indices


def _session_rows(
    *, session_ids: np.ndarray, labels: np.ndarray, source_session: str, expected_count: int,
) -> np.ndarray:
    rows = np.flatnonzero(session_ids == source_session)
    if len(rows) != expected_count:
        raise ValueError(f"{source_session}: expected {expected_count} trials, got {len(rows)}.")
    counts = np.bincount(labels[rows], minlength=4)
    if not np.array_equal(counts, np.full(4, expected_count // 4, dtype=np.int64)):
        raise ValueError(f"{source_session}: expected balanced labels, got {counts.tolist()}.")
    return rows


def build_payload(s0819: Source0819, s0820: Source0820) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    indices = retained_0820_channel_indices(
        canonical_channels=s0819.channel_names, source_channels=s0820.channel_names,
    )
    source_rows = [
        _session_rows(
            session_ids=s0820.session_ids, labels=s0820.labels,
            source_session=session, expected_count=count,
        )
        for session, count in zip(SOURCE_0820_SESSIONS, SOURCE_0820_COUNTS, strict=True)
    ]
    rows_0820 = np.concatenate(source_rows)
    present_0820 = set(s0820.session_ids.tolist())
    if EXCLUDED_0820_SESSION not in present_0820:
        raise ValueError(f"0820 expected excluded session {EXCLUDED_0820_SESSION!r} is absent.")
    if np.any(s0820.session_ids[rows_0820] == EXCLUDED_0820_SESSION):
        raise RuntimeError("Excluded 0820 session was selected.")

    sessions_0820 = np.concatenate([
        np.full(count, canonical, dtype=object)
        for canonical, count in zip(CANONICAL_SESSIONS[6:], SOURCE_0820_COUNTS, strict=True)
    ])
    payload = {
        "labels": np.concatenate((s0819.labels, s0820.labels[rows_0820])).astype(np.int64, copy=False),
        "subject_ids": np.full(680, CANONICAL_SUBJECT_ID, dtype=np.int64),
        "session_ids": np.concatenate((s0819.session_ids, sessions_0820)),
        "trial_ids": np.arange(680, dtype=np.int64),
        "source_file_ids": np.concatenate((
            s0819.source_file_ids,
            np.full(320, 2, dtype=np.int64),
        )),
        "source_trial_ids": np.concatenate((s0819.source_trial_ids, s0820.trial_ids[rows_0820])),
        "source_subject_ids_original": np.concatenate((
            s0819.source_subject_ids_original, s0820.subject_ids[rows_0820],
        )),
        "source_session_ids": np.concatenate((s0819.source_session_ids, s0820.session_ids[rows_0820])),
        "s0820_retained_channel_indices": indices,
        "s0820_rows": rows_0820,
    }
    mapping = {
        **{
            canonical: str(source)
            for canonical, source in zip(CANONICAL_SESSIONS[:6], s0819.source_session_ids[
                [np.flatnonzero(s0819.session_ids == canonical)[0] for canonical in CANONICAL_SESSIONS[:6]]
            ], strict=True)
        },
        **dict(zip(CANONICAL_SESSIONS[6:], SOURCE_0820_SESSIONS, strict=True)),
    }
    if not np.array_equal(np.bincount(payload["labels"], minlength=4), np.full(4, 170)):
        raise RuntimeError("Constructed labels are not exactly balanced.")
    return payload, mapping


def _metadata(channel_names: list[str]) -> HDF5Metadata:
    return HDF5Metadata(SAMPLE_RATE, channel_names, list(CLASS_NAMES), UNIT, DATASET_NAME)


def _write_streamed_hdf5(
    path: Path, *, s0819: Source0819, s0820: Source0820,
    payload: dict[str, np.ndarray], metadata: HDF5Metadata,
) -> None:
    """Write canonical core datasets without materializing a second 320 MiB tensor."""
    with h5py.File(path, "w") as handle:
        data = handle.create_dataset(
            "data", shape=(680, 59, 2000), dtype="float32",
            compression="gzip", shuffle=True, chunks=(1, 59, 2000),
        )
        for start in range(0, 360, 32):
            stop = min(start + 32, 360)
            data[start:stop] = s0819.data[start:stop]
        output_start = 360
        for source_rows in np.split(payload["s0820_rows"], np.cumsum(SOURCE_0820_COUNTS)[:-1]):
            source_block = s0820.data[source_rows][:, payload["s0820_retained_channel_indices"], :]
            output_stop = output_start + len(source_rows)
            data[output_start:output_stop] = source_block
            output_start = output_stop
            del source_block
        if output_start != 680:
            raise RuntimeError("Streamed write did not cover all 680 trials.")
        handle.create_dataset("labels", data=payload["labels"], dtype="int64")
        handle.create_dataset("subject_ids", data=payload["subject_ids"], dtype="int64")
        handle.create_dataset("session_ids", data=_string_array(payload["session_ids"]))
        handle.create_dataset("trial_ids", data=payload["trial_ids"], dtype="int64")
        handle.attrs["sample_rate"] = float(metadata.sample_rate)
        handle.attrs["channel_names"] = json.dumps(metadata.channel_names, ensure_ascii=False)
        handle.attrs["class_names"] = json.dumps(metadata.class_names, ensure_ascii=False)
        handle.attrs["unit"] = metadata.unit
        handle.attrs["dataset_name"] = metadata.dataset_name


def _write_provenance(
    handle: h5py.File, *, s0819: Source0819, s0820: Source0820,
    payload: dict[str, np.ndarray], session_mapping: dict[str, str],
) -> None:
    for name in ("source_file_ids", "source_trial_ids", "source_subject_ids_original"):
        handle.create_dataset(name, data=payload[name], dtype="int64")
    handle.create_dataset("source_session_ids", data=_string_array(payload["source_session_ids"]))
    source_file_mapping = {**s0819.source_file_mapping, "2": s0820.path.name}
    handle.attrs["canonical_subject_id"] = CANONICAL_SUBJECT_ID
    handle.attrs["canonical_subject_name"] = CANONICAL_SUBJECT_NAME
    handle.attrs["source_file_mapping"] = json.dumps(source_file_mapping, sort_keys=True)
    handle.attrs["source_subject_ids"] = json.dumps(
        {**s0819.source_subject_ids, s0820.path.name: 2},
        sort_keys=True,
    )
    handle.attrs["session_mapping"] = json.dumps(session_mapping, sort_keys=True)
    handle.attrs["source_date_mapping"] = json.dumps(
        {session: ("0819" if int(session[1:]) <= 6 else "0820") for session in CANONICAL_SESSIONS},
        sort_keys=True,
    )
    handle.attrs["clean_sessions"] = json.dumps(list(CANONICAL_SESSIONS))
    handle.attrs["excluded_sessions"] = json.dumps([{
        "source_file": s0820.path.name,
        "source_session": EXCLUDED_0820_SESSION,
        "trials": 13,
        "reason": "partial_class_and_abnormal_signal_qc",
    }], sort_keys=True)
    handle.attrs["ignored_duplicate_exports"] = json.dumps([{
        "source_file": "smr_control_s02_0821.h5",
        "trials": 573,
        "unique_trials": 0,
        "reason": "573_of_573_trials_exact_duplicates_of_0819_or_0820",
    }], sort_keys=True)
    handle.attrs["channel_normalization"] = json.dumps({
        "source_0819": "59 channels copied unchanged",
        "source_0820_dropped_channels": list(DROP_CHANNELS),
        "signal_preprocessing_applied": False,
    }, sort_keys=True)
    handle.attrs["trial_id_policy"] = "canonical_trial_ids_0_to_679_in_chronological_S1_to_S11_order"


def _duplicate_pairs(dataset: h5py.Dataset) -> list[tuple[int, int]]:
    seen: dict[str, int] = {}
    duplicates: list[tuple[int, int]] = []
    for start in range(0, len(dataset), 32):
        block = dataset[start:start + 32]
        for offset, trial in enumerate(block):
            index = start + offset
            digest = hashlib.sha256(np.ascontiguousarray(trial).view(np.uint8)).hexdigest()
            previous = seen.get(digest)
            if previous is None:
                seen[digest] = index
            else:
                duplicates.append((previous, index))
    return duplicates


def _smoke_50m(data: np.ndarray, channel_names: list[str]) -> dict[str, Any]:
    config = Model50MConfig(
        checkpoint_path="unused-for-preprocessing-smoke.pt",
        window_seconds=2.0,
        model_n_time_patches=10,
    )
    processor = Model50MPreprocessor(config)
    result = processor(data, channel_names, SAMPLE_RATE, UNIT)
    if result.shape != (64, 200) or result.mapped_channel_count != 57:
        raise ValueError(f"Unexpected 50M preprocessing result: {result.shape}, {result.mapped_channel_count} mapped.")
    return {"raw_shape": list(data.shape), "mapped_channels": result.mapped_channel_count, "output_shape": list(result.shape)}


def verify_output(
    *, output: Path, s0819: Source0819, s0820: Source0820,
    payload: dict[str, np.ndarray], session_mapping: dict[str, str],
) -> dict[str, Any]:
    reader = EEGHDF5(output)
    if reader.metadata != _metadata(s0819.channel_names):
        raise ValueError("EEGHDF5 metadata differs from the intended contract.")
    available = reader.available_sessions()
    if set(available) != set(CANONICAL_SESSIONS) or "S12" in available:
        raise ValueError(f"Unexpected available sessions: {available}.")
    trial_reader = open_trial_reader(data_reader="eeg", path=output, canonical_subject_id=1)
    trial_metadata = trial_reader.trial_metadata()
    with h5py.File(output, "r") as handle, h5py.File(s0819.path, "r") as source_0819_handle, h5py.File(s0820.path, "r") as source_0820_handle:
        labels = handle["labels"][:]
        subjects = handle["subject_ids"][:]
        sessions = _decode_strings(handle["session_ids"][:])
        trials = handle["trial_ids"][:]
        dataset = handle["data"]
        if dataset.shape != (680, 59, 2000) or dataset.dtype != np.dtype("float32"):
            raise ValueError(f"Invalid output shape/dtype: {dataset.shape}/{dataset.dtype}.")
        source_0819_data = source_0819_handle["data"]
        for start in range(0, 360, 32):
            stop = min(start + 32, 360)
            if not np.array_equal(dataset[start:stop], source_0819_data[start:stop]):
                raise ValueError("0819 EEG values are not byte-identical.")
        output_start = 360
        for source_rows in np.split(payload["s0820_rows"], np.cumsum(SOURCE_0820_COUNTS)[:-1]):
            expected_0820 = source_0820_handle["data"][source_rows][:, payload["s0820_retained_channel_indices"], :]
            output_stop = output_start + len(source_rows)
            if not np.array_equal(dataset[output_start:output_stop], expected_0820):
                raise ValueError("0820 retained EEG values are not byte-identical.")
            output_start = output_stop
            del expected_0820
        if output_start != 680:
            raise RuntimeError("0820 equality verification did not cover all retained output trials.")
        p2p_values: list[np.ndarray] = []
        std_values: list[np.ndarray] = []
        for start in range(0, len(dataset), 32):
            block = dataset[start:start + 32]
            if not np.isfinite(block).all():
                raise ValueError("Output contains NaN or Inf.")
            p2p_values.append(np.ptp(block, axis=(1, 2)))
            std_values.append(np.std(block, axis=(1, 2)))
        if not np.array_equal(np.bincount(labels, minlength=4), np.full(4, 170)):
            raise ValueError("Final class counts differ from 170/class.")
        if not np.array_equal(subjects, np.full(680, 1, dtype=np.int64)):
            raise ValueError("Canonical subject IDs differ.")
        if not np.array_equal(trials, np.arange(680, dtype=np.int64)):
            raise ValueError("Canonical trial IDs differ.")
        if not np.array_equal(trial_metadata["trial_ids"], trials):
            raise ValueError("TrialReader metadata differs from the output.")
        if json.loads(str(handle.attrs["session_mapping"])) != session_mapping:
            raise ValueError("Session provenance mapping differs.")
        if any("0821" in value for value in _decode_strings(handle["source_session_ids"][:])):
            raise ValueError("0821 was unexpectedly included in trial provenance.")
        for session, expected_count in EXPECTED_SESSION_COUNTS.items():
            loaded = reader.load(session)
            if len(loaded["data"]) != expected_count:
                raise ValueError(f"{session}: count differs.")
            if not np.array_equal(np.bincount(loaded["labels"], minlength=4), np.full(4, expected_count // 4)):
                raise ValueError(f"{session}: labels are not balanced.")
        duplicates = _duplicate_pairs(dataset)
        if duplicates:
            raise ValueError(f"Final H5 contains exact duplicate trials: {duplicates[:5]}.")
    p2p = np.concatenate(p2p_values)
    trial_std = np.concatenate(std_values)
    smoke = {
        session: _smoke_50m(reader.load(session)["data"][0], s0819.channel_names)
        for session in ("S1", "S7", "S11")
    }
    return {
        "output_path": str(output), "data_shape": [680, 59, 2000], "dtype": "float32",
        "canonical_subject": CANONICAL_SUBJECT_NAME, "canonical_subject_id": CANONICAL_SUBJECT_ID,
        "sessions": {session: EXPECTED_SESSION_COUNTS[session] for session in CANONICAL_SESSIONS},
        "class_counts": np.bincount(labels, minlength=4).tolist(),
        "nan_count": 0, "inf_count": 0,
        "0819_exact_equality": True, "0820_retained_channel_exact_equality": True,
        "exact_duplicate_trial_pairs": 0,
        "trial_p2p": dict(zip(
            ("median", "p95", "max"), map(float, np.percentile(p2p, (50, 95, 100))), strict=True,
        )),
        "trial_std": dict(zip(
            ("median", "p95", "max"), map(float, np.percentile(trial_std, (50, 95, 100))), strict=True,
        )),
        "reader_smoke": {"EEGHDF5": "PASS", "TrialReader": "PASS", "loaded_sessions": list(CANONICAL_SESSIONS)},
        "preprocessing_smoke": smoke,
    }


def combine(*, source_0819: Path, source_0820: Path, output: Path, summary: Path, overwrite: bool) -> dict[str, Any]:
    if output in {source_0819, source_0820}:
        raise ValueError("Output must not overwrite either source H5.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}. Pass --overwrite to replace it.")
    if summary.exists() and not overwrite:
        raise FileExistsError(f"Summary already exists: {summary}. Pass --overwrite to replace it.")
    s0819 = read_0819(source_0819)
    s0820 = read_0820(source_0820)
    payload, session_mapping = build_payload(s0819, s0820)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        _write_streamed_hdf5(
            temporary, s0819=s0819, s0820=s0820, payload=payload,
            metadata=_metadata(s0819.channel_names),
        )
        with h5py.File(temporary, "a") as handle:
            _write_provenance(handle, s0819=s0819, s0820=s0820, payload=payload, session_mapping=session_mapping)
        temporary.replace(output)
        # Source tensors were needed for the streamed write.  Equality checks
        # reopen the read-only source H5 files in small blocks, avoiding a
        # second full output-sized tensor plus both sources in memory.
        object.__setattr__(s0819, "data", np.empty((0,), dtype=np.float32))
        object.__setattr__(s0820, "data", np.empty((0,), dtype=np.float32))
        verification = verify_output(
            output=output, s0819=s0819, s0820=s0820, payload=payload, session_mapping=session_mapping,
        )
        summary_payload = {
            **verification,
            "source_files": [str(source_0819), str(source_0820)],
            "included_sessions": session_mapping,
            "excluded_sessions": [{"source_file": source_0820.name, "source_session": EXCLUDED_0820_SESSION, "trials": 13, "reason": "partial_class_and_abnormal_signal_qc"}],
            "ignored_duplicate_exports": [{"source_file": "smr_control_s02_0821.h5", "trials": 573, "unique_trials": 0}],
            "channel_canonicalization": {"0819": "copied unchanged", "0820_dropped_auxiliary": list(DROP_CHANNELS)},
            "source_to_canonical_trial_mapping": {"0819_canonical_trials": "0..359 copied", "0820_source_trials": "selected by source_session_ids and remapped to 360..679"},
        }
        summary.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return summary_payload
    except Exception:
        temporary.unlink(missing_ok=True)
        if output.exists() and not output.samefile(source_0819):
            # A completed output only exists here after a failed post-write verification.
            output.unlink()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean yaxin 0819 + 0820 canonical HDF5 (S1-S11).")
    parser.add_argument("--source-0819", type=Path, default=ROOT / "data/processed/yaxin/smr_control_yaxin_0819_combined.h5")
    parser.add_argument("--source-0820", type=Path, default=ROOT / "data/processed/yaxin/smr_control_s02_0820.h5")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/yaxin/smr_control_yaxin_0819_0820_combined.h5")
    parser.add_argument("--summary", type=Path, default=ROOT / "data/processed/yaxin/smr_control_yaxin_0819_0820_combined_summary.json")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output H5 and summary JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = combine(
        source_0819=args.source_0819.resolve(), source_0820=args.source_0820.resolve(),
        output=args.output.resolve(), summary=args.summary.resolve(), overwrite=bool(args.overwrite),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
