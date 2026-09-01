from __future__ import annotations

"""Convert MEMA For-DL MAT trials into source-trial-safe canonical EEGHDF5.

The MAT files remain untouched.  This converter only segments each independent
``*_eegK`` source trial into non-overlapping 2 s epochs; it never concatenates
or resamples source data.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from scipy.io import loadmat, whosmat

try:  # Script execution.
    from _bootstrap import ROOT
except ModuleNotFoundError:  # Imported by tests.
    from scripts._bootstrap import ROOT

from bci_dayloop.data.hdf5_dataset import EEGHDF5, HDF5Metadata, write_hdf5


MEMA_CHANNEL_NAMES = [
    "FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz",
    "FC3", "FC4", "FT7", "FT8", "Cz", "C3", "C4", "T3",
    "T4", "CPz", "CP3", "CP4", "TP7", "TP8", "Pz", "P3",
    "P4", "T5", "T6", "Oz", "O1", "O2", "HEOL", "HEOR",
]
CLASS_NAMES = ["relaxing", "neutral", "concentrating"]
LABEL_MAPPING = {"0": "relaxing", "1": "neutral", "2": "concentrating"}

SAMPLE_RATE = 500.0
WINDOW_SECONDS = 2.0
SAMPLES_PER_EPOCH = int(SAMPLE_RATE * WINDOW_SECONDS)
DATASET_NAME = "mema_attention"
SESSION_POLICY = "source_trials_1_9=S1; source_trials_10_12=S2"
SEGMENTATION_POLICY = "non_overlapping_2s_within_source_trial"
FLOAT32_OVERFLOW_POLICY = "discard_whole_2s_epoch_if_any_value_exceeds_float32_range"


@dataclass(frozen=True)
class SubjectPayload:
    data: np.ndarray
    labels: np.ndarray
    subject_ids: np.ndarray
    session_ids: np.ndarray
    trial_ids: np.ndarray
    source_trial_ids: np.ndarray
    source_epoch_indices: np.ndarray
    source_start_samples: np.ndarray
    source_end_samples: np.ndarray
    source_sample_counts: dict[int, int]
    source_candidate_epoch_counts: dict[int, int]
    source_epoch_counts: dict[int, int]
    source_kept_epoch_indices: dict[int, np.ndarray]
    dropped_float32_overflow_epoch_indices: dict[int, list[int]]
    source_remainder_samples: dict[int, int]


def _source_path(input_root: Path, subject: int) -> Path:
    path = input_root / f"Subject{subject}.mat"
    if not path.is_file():
        raise FileNotFoundError(f"MEMA source subject file is missing: {path}")
    return path


def _eeg_variables(path: Path) -> list[tuple[int, str]]:
    matched: list[tuple[int, str]] = []
    for name, _shape, _dtype in whosmat(path):
        match = re.search(r"_eeg(\d+)$", name, flags=re.IGNORECASE)
        if match:
            matched.append((int(match.group(1)), name))
    matched.sort()
    expected = list(range(1, 13))
    actual = [trial for trial, _name in matched]
    if actual != expected:
        raise ValueError(
            f"{path}: expected exactly *_eeg1 through *_eeg12; found {actual}."
        )
    return matched


def load_attention_labels(input_root: Path) -> np.ndarray:
    path = input_root / "label_attention.mat"
    if not path.is_file():
        raise FileNotFoundError(f"MEMA attention label file is missing: {path}")
    loaded = loadmat(path, variable_names=["label"])
    if "label" not in loaded:
        raise KeyError(f"{path}: expected a 'label' variable.")
    labels = np.asarray(loaded["label"], dtype=np.int64)
    if labels.shape != (20, 12):
        raise ValueError(
            f"{path}: expected attention labels shape (20, 12), got {labels.shape}."
        )
    if not np.isin(labels, (0, 1, 2)).all():
        values = sorted(set(labels.reshape(-1).tolist()))
        raise ValueError(f"{path}: labels must be in {{0, 1, 2}}, got {values}.")
    return labels


def logical_session_for_source_trial(source_trial_id: int) -> str:
    if not 1 <= source_trial_id <= 12:
        raise ValueError(f"source trial ID must be in [1, 12], got {source_trial_id}.")
    return "S1" if source_trial_id <= 9 else "S2"


def build_subject_payload(
    *,
    source_path: Path,
    subject_id: int,
    source_labels: np.ndarray,
) -> SubjectPayload:
    """Read and segment one subject without crossing source-trial boundaries."""
    if source_labels.shape != (12,):
        raise ValueError(
            f"subject {subject_id}: expected 12 source labels, got {source_labels.shape}."
        )

    data_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    session_parts: list[np.ndarray] = []
    source_trial_parts: list[np.ndarray] = []
    source_epoch_parts: list[np.ndarray] = []
    source_start_parts: list[np.ndarray] = []
    source_end_parts: list[np.ndarray] = []
    source_sample_counts: dict[int, int] = {}
    source_candidate_epoch_counts: dict[int, int] = {}
    source_epoch_counts: dict[int, int] = {}
    source_kept_epoch_indices: dict[int, np.ndarray] = {}
    dropped_float32_overflow_epoch_indices: dict[int, list[int]] = {}
    source_remainder_samples: dict[int, int] = {}

    for source_trial_id, variable_name in _eeg_variables(source_path):
        loaded = loadmat(source_path, variable_names=[variable_name])
        trial = np.asarray(loaded[variable_name])
        if trial.ndim != 2 or trial.shape[0] != len(MEMA_CHANNEL_NAMES):
            raise ValueError(
                f"{source_path}:{variable_name}: expected [32,T], got {trial.shape}."
            )
        if not np.isfinite(trial).all():
            raise ValueError(f"{source_path}:{variable_name}: contains NaN or Inf.")
        source_samples = int(trial.shape[1])
        candidate_epochs = source_samples // SAMPLES_PER_EPOCH
        remainder = source_samples % SAMPLES_PER_EPOCH
        if candidate_epochs <= 0:
            raise ValueError(
                f"{source_path}:{variable_name}: shorter than one {WINDOW_SECONDS:g} s epoch."
            )

        usable_samples = candidate_epochs * SAMPLES_PER_EPOCH
        float32_limit = np.finfo(np.float32).max
        # Values are finite but cannot be represented by the required canonical
        # float32 schema.  The approved policy discards the complete affected
        # epoch rather than clipping/scaling any samples or splitting a trial.
        overflow_by_epoch = np.abs(trial[:, :usable_samples]).reshape(
            len(MEMA_CHANNEL_NAMES), candidate_epochs, SAMPLES_PER_EPOCH
        ).max(axis=(0, 2)) > float32_limit
        candidate_epoch_indices = np.arange(candidate_epochs, dtype=np.int64)
        kept_epoch_indices = candidate_epoch_indices[~overflow_by_epoch]
        dropped_epoch_indices = candidate_epoch_indices[overflow_by_epoch]
        # [C,T] -> [N,C,1000]; astype only reduces storage precision, never scales.
        all_epochs = trial[:, :usable_samples].reshape(
            len(MEMA_CHANNEL_NAMES), candidate_epochs, SAMPLES_PER_EPOCH
        ).transpose(1, 0, 2)
        epochs = np.ascontiguousarray(
            all_epochs[kept_epoch_indices],
            dtype=np.float32,
        )
        n_epochs = len(epochs)
        epoch_indices = kept_epoch_indices

        data_parts.append(epochs)
        label_parts.append(
            np.full(n_epochs, int(source_labels[source_trial_id - 1]), dtype=np.int64)
        )
        session_parts.append(
            np.full(n_epochs, logical_session_for_source_trial(source_trial_id), dtype="U2")
        )
        source_trial_parts.append(np.full(n_epochs, source_trial_id, dtype=np.int64))
        source_epoch_parts.append(epoch_indices)
        source_start_parts.append(epoch_indices * SAMPLES_PER_EPOCH)
        source_end_parts.append((epoch_indices + 1) * SAMPLES_PER_EPOCH)
        source_sample_counts[source_trial_id] = source_samples
        source_candidate_epoch_counts[source_trial_id] = candidate_epochs
        source_epoch_counts[source_trial_id] = n_epochs
        source_kept_epoch_indices[source_trial_id] = kept_epoch_indices
        dropped_float32_overflow_epoch_indices[source_trial_id] = (
            dropped_epoch_indices.astype(int).tolist()
        )
        source_remainder_samples[source_trial_id] = remainder

    data = np.concatenate(data_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    sessions = np.concatenate(session_parts, axis=0)
    source_trial_ids = np.concatenate(source_trial_parts, axis=0)
    source_epoch_indices = np.concatenate(source_epoch_parts, axis=0)
    source_start_samples = np.concatenate(source_start_parts, axis=0)
    source_end_samples = np.concatenate(source_end_parts, axis=0)
    n_epochs = len(data)

    return SubjectPayload(
        data=data,
        labels=labels,
        subject_ids=np.full(n_epochs, subject_id, dtype=np.int64),
        session_ids=sessions,
        trial_ids=np.arange(n_epochs, dtype=np.int64),
        source_trial_ids=source_trial_ids,
        source_epoch_indices=source_epoch_indices,
        source_start_samples=source_start_samples,
        source_end_samples=source_end_samples,
        source_sample_counts=source_sample_counts,
        source_candidate_epoch_counts=source_candidate_epoch_counts,
        source_epoch_counts=source_epoch_counts,
        source_kept_epoch_indices=source_kept_epoch_indices,
        dropped_float32_overflow_epoch_indices=dropped_float32_overflow_epoch_indices,
        source_remainder_samples=source_remainder_samples,
    )


def _metadata() -> HDF5Metadata:
    return HDF5Metadata(
        sample_rate=SAMPLE_RATE,
        channel_names=list(MEMA_CHANNEL_NAMES),
        class_names=list(CLASS_NAMES),
        # Model50MPreprocessor accepts V/mV/uV only.  The actual source unit is
        # unknown; the compatibility assumption is preserved separately below.
        unit="uV",
        dataset_name=DATASET_NAME,
    )


def _write_provenance(handle: h5py.File, *, subject_id: int, payload: SubjectPayload) -> None:
    handle.create_dataset("source_trial_ids", data=payload.source_trial_ids, dtype="int64")
    handle.create_dataset(
        "source_epoch_indices", data=payload.source_epoch_indices, dtype="int64"
    )
    handle.create_dataset(
        "source_start_samples", data=payload.source_start_samples, dtype="int64"
    )
    handle.create_dataset(
        "source_end_samples", data=payload.source_end_samples, dtype="int64"
    )
    handle.attrs["subject_id"] = int(subject_id)
    handle.attrs["source_subject_id"] = f"Subject{subject_id}"
    handle.attrs["num_classes"] = len(CLASS_NAMES)
    handle.attrs["label_mapping"] = json.dumps(LABEL_MAPPING, sort_keys=True)
    handle.attrs["label_mapping_status"] = "working_assumption"
    handle.attrs["source_unit"] = "unknown"
    handle.attrs["unit_status"] = "assumed_for_scale_invariant_zscore_pipeline"
    handle.attrs["unit_assumption"] = "uV"
    handle.attrs["window_seconds"] = WINDOW_SECONDS
    handle.attrs["source_samples_per_epoch"] = SAMPLES_PER_EPOCH
    handle.attrs["segmentation"] = SEGMENTATION_POLICY
    handle.attrs["float32_overflow_policy"] = FLOAT32_OVERFLOW_POLICY
    handle.attrs["dropped_float32_overflow_epoch_indices"] = json.dumps(
        {
            str(key): value
            for key, value in payload.dropped_float32_overflow_epoch_indices.items()
            if value
        },
        sort_keys=True,
    )
    handle.attrs["session_type"] = "logical"
    handle.attrs["session_policy"] = SESSION_POLICY
    handle.attrs["original_session_metadata_available"] = False


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        CLASS_NAMES[index]: int(np.count_nonzero(labels == index))
        for index in range(len(CLASS_NAMES))
    }


def verify_mema_hdf5(
    path: Path,
    *,
    subject_id: int,
    source_epoch_counts: dict[int, int],
    source_kept_epoch_indices: dict[int, np.ndarray],
    dropped_float32_overflow_epoch_indices: dict[int, list[int]],
    source_remainder_samples: dict[int, int],
) -> dict[str, Any]:
    """Validate canonical reader compatibility and source-trial containment."""
    reader = EEGHDF5(path)
    metadata = reader.metadata
    if metadata.sample_rate != SAMPLE_RATE:
        raise ValueError(f"{path}: sample rate mismatch: {metadata.sample_rate}.")
    if metadata.channel_names != MEMA_CHANNEL_NAMES:
        raise ValueError(f"{path}: channel order differs from MEMA contract.")
    if metadata.class_names != CLASS_NAMES:
        raise ValueError(f"{path}: class names differ from MEMA contract.")
    if metadata.unit != "uV":
        raise ValueError(f"{path}: expected pipeline compatibility unit 'uV'.")

    with h5py.File(path, "r") as handle:
        required = {
            "data", "labels", "subject_ids", "session_ids", "trial_ids",
            "source_trial_ids", "source_epoch_indices", "source_start_samples",
            "source_end_samples",
        }
        missing = required - set(handle.keys())
        if missing:
            raise ValueError(f"{path}: missing required datasets: {sorted(missing)}.")
        data = handle["data"]
        n_epochs = len(data)
        if data.ndim != 3 or data.shape[1:] != (32, SAMPLES_PER_EPOCH):
            raise ValueError(f"{path}: expected [N,32,1000], got {data.shape}.")
        arrays = {
            name: handle[name][:]
            for name in (
                "labels", "subject_ids", "trial_ids", "source_trial_ids",
                "source_epoch_indices", "source_start_samples", "source_end_samples",
            )
        }
        sessions = handle["session_ids"].asstr()[:]
        if any(len(values) != n_epochs for values in arrays.values()) or len(sessions) != n_epochs:
            raise ValueError(f"{path}: trial-level dataset lengths differ from data.")
        if not np.isfinite(data[:]).all():
            raise ValueError(f"{path}: data contains NaN or Inf.")
        if not np.isin(arrays["labels"], (0, 1, 2)).all():
            raise ValueError(f"{path}: labels are outside {{0,1,2}}.")
        if not np.array_equal(np.unique(arrays["subject_ids"]), np.asarray([subject_id])):
            raise ValueError(f"{path}: subject IDs do not match {subject_id}.")
        if len(np.unique(arrays["trial_ids"])) != n_epochs:
            raise ValueError(f"{path}: trial_ids are not unique.")
        if not set(sessions.tolist()) <= {"S1", "S2"}:
            raise ValueError(f"{path}: sessions must be S1/S2, got {sorted(set(sessions))}.")

        source_ids = arrays["source_trial_ids"]
        if not np.isin(source_ids, np.arange(1, 13)).all():
            raise ValueError(f"{path}: source trial IDs are outside [1,12].")
        for source_trial_id in range(1, 13):
            mask = source_ids == source_trial_id
            actual_count = int(mask.sum())
            expected_count = int(source_epoch_counts[source_trial_id])
            if actual_count != expected_count:
                raise ValueError(
                    f"{path}: source trial {source_trial_id} has {actual_count} epochs, "
                    f"expected {expected_count}."
                )
            expected_session = logical_session_for_source_trial(source_trial_id)
            if actual_count and set(sessions[mask].tolist()) != {expected_session}:
                raise ValueError(
                    f"{path}: source trial {source_trial_id} crosses logical sessions."
                )
            expected_indices = source_kept_epoch_indices[source_trial_id]
            if not np.array_equal(arrays["source_epoch_indices"][mask], expected_indices):
                raise ValueError(f"{path}: source epoch indices are invalid for trial {source_trial_id}.")
            if not np.array_equal(
                arrays["source_start_samples"][mask], expected_indices * SAMPLES_PER_EPOCH
            ):
                raise ValueError(f"{path}: source starts are invalid for trial {source_trial_id}.")
            if not np.array_equal(
                arrays["source_end_samples"][mask], (expected_indices + 1) * SAMPLES_PER_EPOCH
            ):
                raise ValueError(f"{path}: source ends are invalid for trial {source_trial_id}.")

        expected_attrs = {
            "source_subject_id": f"Subject{subject_id}",
            "label_mapping_status": "working_assumption",
            "source_unit": "unknown",
            "unit_status": "assumed_for_scale_invariant_zscore_pipeline",
            "unit_assumption": "uV",
            "segmentation": SEGMENTATION_POLICY,
            "float32_overflow_policy": FLOAT32_OVERFLOW_POLICY,
            "session_type": "logical",
            "session_policy": SESSION_POLICY,
        }
        for key, expected in expected_attrs.items():
            if key not in handle.attrs:
                # Outputs created before the overflow policy was introduced
                # remain valid when they have no dropped overflow epoch.
                if key == "float32_overflow_policy" and not any(
                    dropped_float32_overflow_epoch_indices.values()
                ):
                    continue
                raise ValueError(f"{path}: missing required root attr {key!r}.")
            if handle.attrs[key] != expected:
                raise ValueError(
                    f"{path}: root attr {key!r}={handle.attrs[key]!r}, expected {expected!r}."
                )
        if json.loads(handle.attrs["label_mapping"]) != LABEL_MAPPING:
            raise ValueError(f"{path}: label mapping provenance differs.")
        expected_dropped = {
            str(key): value
            for key, value in dropped_float32_overflow_epoch_indices.items()
            if value
        }
        actual_dropped = (
            json.loads(handle.attrs["dropped_float32_overflow_epoch_indices"])
            if "dropped_float32_overflow_epoch_indices" in handle.attrs
            else {}
        )
        if actual_dropped != expected_dropped:
            raise ValueError(f"{path}: dropped float32-overflow provenance differs.")
        if bool(handle.attrs["original_session_metadata_available"]):
            raise ValueError(f"{path}: logical sessions incorrectly marked original.")

        s1_mask = sessions == "S1"
        s2_mask = sessions == "S2"
        return {
            "subject": subject_id,
            "source_subject_id": f"Subject{subject_id}",
            "total_epochs": n_epochs,
            "S1_epochs": int(s1_mask.sum()),
            "S2_epochs": int(s2_mask.sum()),
            "S1_class_counts": _class_counts(arrays["labels"][s1_mask]),
            "S2_class_counts": _class_counts(arrays["labels"][s2_mask]),
            "total_class_counts": _class_counts(arrays["labels"]),
            "source_trial_epoch_counts": {
                str(key): int(value) for key, value in source_epoch_counts.items()
            },
            "dropped_float32_overflow_epoch_indices": expected_dropped,
            "dropped_float32_overflow_epoch_total": int(
                sum(len(value) for value in dropped_float32_overflow_epoch_indices.values())
            ),
            "discarded_remainder_samples": {
                str(key): int(value) for key, value in source_remainder_samples.items()
            },
            "discarded_remainder_total": int(sum(source_remainder_samples.values())),
        }


def convert_subject(
    *,
    input_root: Path,
    output_root: Path,
    subject_id: int,
    attention_labels: np.ndarray,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Any]:
    source_path = _source_path(input_root, subject_id)
    payload = build_subject_payload(
        source_path=source_path,
        subject_id=subject_id,
        source_labels=np.asarray(attention_labels[subject_id - 1], dtype=np.int64),
    )
    if dry_run:
        return {
            "subject": subject_id,
            "source_subject_id": f"Subject{subject_id}",
            "source_trial_samples": {
                str(key): int(value) for key, value in payload.source_sample_counts.items()
            },
            "source_trial_labels": attention_labels[subject_id - 1].astype(int).tolist(),
            "S1_source_trials": list(range(1, 10)),
            "S2_source_trials": list(range(10, 13)),
            "total_epochs": int(len(payload.data)),
            "S1_epochs": int(np.count_nonzero(payload.session_ids == "S1")),
            "S2_epochs": int(np.count_nonzero(payload.session_ids == "S2")),
            "total_class_counts": _class_counts(payload.labels),
            "discarded_remainder_samples": {
                str(key): int(value) for key, value in payload.source_remainder_samples.items()
            },
            "discarded_remainder_total": int(sum(payload.source_remainder_samples.values())),
            "dropped_float32_overflow_epoch_indices": {
                str(key): value
                for key, value in payload.dropped_float32_overflow_epoch_indices.items()
                if value
            },
            "dropped_float32_overflow_epoch_total": int(
                sum(len(value) for value in payload.dropped_float32_overflow_epoch_indices.values())
            ),
        }

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"subject_{subject_id:02d}.h5"
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        write_hdf5(
            temporary,
            data=payload.data,
            labels=payload.labels,
            subject_ids=payload.subject_ids,
            session_ids=payload.session_ids.tolist(),
            trial_ids=payload.trial_ids,
            metadata=_metadata(),
        )
        with h5py.File(temporary, "a") as handle:
            _write_provenance(handle, subject_id=subject_id, payload=payload)
        summary = verify_mema_hdf5(
            temporary,
            subject_id=subject_id,
            source_epoch_counts=payload.source_epoch_counts,
            source_kept_epoch_indices=payload.source_kept_epoch_indices,
            dropped_float32_overflow_epoch_indices=payload.dropped_float32_overflow_epoch_indices,
            source_remainder_samples=payload.source_remainder_samples,
        )
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    summary["output_path"] = str(output_path)
    return summary


def verify_existing_subject(
    *,
    input_root: Path,
    output_root: Path,
    subject_id: int,
    attention_labels: np.ndarray,
) -> dict[str, Any]:
    """Revalidate an existing conversion and include it in an aggregate summary."""
    payload = build_subject_payload(
        source_path=_source_path(input_root, subject_id),
        subject_id=subject_id,
        source_labels=np.asarray(attention_labels[subject_id - 1], dtype=np.int64),
    )
    output_path = output_root / f"subject_{subject_id:02d}.h5"
    if not output_path.is_file():
        raise FileNotFoundError(f"Converted MEMA output is missing: {output_path}")
    summary = verify_mema_hdf5(
        output_path,
        subject_id=subject_id,
        source_epoch_counts=payload.source_epoch_counts,
        source_kept_epoch_indices=payload.source_kept_epoch_indices,
        dropped_float32_overflow_epoch_indices=payload.dropped_float32_overflow_epoch_indices,
        source_remainder_samples=payload.source_remainder_samples,
    )
    summary["output_path"] = str(output_path)
    return summary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _summary_payload(
    *,
    input_root: Path,
    output_root: Path,
    subjects: Iterable[int],
    dry_run: bool,
    per_subject: list[dict[str, Any]],
) -> dict[str, Any]:
    total_class_counts = {
        name: int(sum(item["total_class_counts"][name] for item in per_subject))
        for name in CLASS_NAMES
    }
    return {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "subjects": list(subjects),
        "dry_run": dry_run,
        "dataset_name": DATASET_NAME,
        "sample_rate": SAMPLE_RATE,
        "channel_names": MEMA_CHANNEL_NAMES,
        "class_names": CLASS_NAMES,
        "label_mapping": LABEL_MAPPING,
        "label_mapping_status": "working_assumption",
        "source_unit": "unknown",
        "unit": "uV",
        "unit_status": "assumed_for_scale_invariant_zscore_pipeline",
        "unit_assumption": "uV",
        "window_seconds": WINDOW_SECONDS,
        "source_samples_per_epoch": SAMPLES_PER_EPOCH,
        "segmentation": SEGMENTATION_POLICY,
        "float32_overflow_policy": FLOAT32_OVERFLOW_POLICY,
        "session_type": "logical",
        "session_policy": SESSION_POLICY,
        "original_session_metadata_available": False,
        "total_epochs": int(sum(item["total_epochs"] for item in per_subject)),
        "total_class_counts": total_class_counts,
        "discarded_remainder_total": int(
            sum(item.get("discarded_remainder_total", 0) for item in per_subject)
        ),
        "dropped_float32_overflow_epoch_total": int(
            sum(item.get("dropped_float32_overflow_epoch_total", 0) for item in per_subject)
        ),
        "per_subject": per_subject,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MEMA For-DL MAT files to canonical 2 s EEGHDF5."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/Volumes/UBUNTU-SERV/data/MEMA"),
        help="Directory containing Subject*.mat and label_attention.mat.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data" / "processed" / "mema",
        help="Directory for subject_XX.h5 outputs and conversion_summary.json.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=list(range(1, 21)),
        help="Canonical MEMA subject IDs to convert. Default: 1 through 20.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing selected output HDF5 file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect segmentation/counts without writing HDF5 or summary files.",
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Revalidate existing selected HDF5 outputs and rewrite conversion_summary.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()
    subjects = sorted(set(int(subject) for subject in args.subjects))
    if not subjects or any(subject < 1 or subject > 20 for subject in subjects):
        raise ValueError(f"--subjects must be unique IDs in [1,20], got {subjects}.")
    if args.dry_run and args.verify_existing:
        raise ValueError("--dry-run and --verify-existing are mutually exclusive.")
    if args.overwrite and args.verify_existing:
        raise ValueError("--overwrite and --verify-existing are mutually exclusive.")

    attention_labels = load_attention_labels(input_root)
    per_subject: list[dict[str, Any]] = []
    for subject_id in subjects:
        if args.verify_existing:
            summary = verify_existing_subject(
                input_root=input_root,
                output_root=output_root,
                subject_id=subject_id,
                attention_labels=attention_labels,
            )
        else:
            summary = convert_subject(
                input_root=input_root,
                output_root=output_root,
                subject_id=subject_id,
                attention_labels=attention_labels,
                overwrite=bool(args.overwrite),
                dry_run=bool(args.dry_run),
            )
        per_subject.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    summary = _summary_payload(
        input_root=input_root,
        output_root=output_root,
        subjects=subjects,
        dry_run=bool(args.dry_run),
        per_subject=per_subject,
    )
    if not args.dry_run:
        _write_json(output_root / "conversion_summary.json", summary)
        print(f"Wrote conversion summary: {output_root / 'conversion_summary.json'}")
    print(json.dumps({key: summary[key] for key in ("total_epochs", "total_class_counts", "discarded_remainder_total")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
