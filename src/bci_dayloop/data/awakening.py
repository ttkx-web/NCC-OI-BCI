"""Awakening-specific validation, segmentation, splitting, and HDF5 output.

The source Lance rows are independent ten-second parent segments.  This module
never joins parent rows: every emitted two-second window is a direct view of one
parent row and keeps that row's provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import h5py
import numpy as np

from bci_dayloop.data.dataset_registry import (
    AWAKENING_2S,
    AWAKENING_SOURCE_CHANNELS,
)


SAMPLE_RATE = 200.0
WINDOW_SECONDS = 2.0
STEP_SECONDS = 2.0
N_CHANNELS = 62
PARENT_SAMPLES = 2_000
WINDOW_SAMPLES = 400
PARENT_ELEMENTS = N_CHANNELS * PARENT_SAMPLES
SUBWINDOWS_PER_PARENT = PARENT_SAMPLES // WINDOW_SAMPLES
MIN_ALL_CHANNEL_ZERO_RUN_SAMPLES = 2

CHANNEL_NAMES = AWAKENING_SOURCE_CHANNELS
CLASS_NAMES = AWAKENING_2S.class_names
LABEL_MAPPING: dict[str, str] = {"0": CLASS_NAMES[0], "1": CLASS_NAMES[1]}
SESSION_SEMANTICS = AWAKENING_2S.session_semantics


@dataclass(frozen=True, slots=True)
class ParentWindowPlan:
    """Lightweight plan retained after inspecting one source Lance row."""

    row_position: int
    global_idx: int
    sample_id: str
    source_subject_id: str
    label: int
    valid_subwindow_indices: tuple[int, ...]
    zero_run_subwindow_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RowInspection:
    """Result of fail-closed validation for one parent row."""

    row_rejection_reason: str | None
    valid_subwindow_indices: tuple[int, ...]
    zero_run_subwindow_indices: tuple[int, ...]
    zero_runs: tuple[tuple[int, int], ...]

    @property
    def is_structurally_valid(self) -> bool:
        return self.row_rejection_reason is None


def validate_fixed_window_contract(
    *, window_seconds: float, step_seconds: float
) -> None:
    """Reject unsupported segmentation settings instead of ignoring them."""

    if not math.isclose(window_seconds, WINDOW_SECONDS, abs_tol=1e-9):
        raise ValueError(
            "Awakening conversion currently supports only "
            f"--window-seconds={WINDOW_SECONDS:g}; got {window_seconds}."
        )
    if not math.isclose(step_seconds, STEP_SECONDS, abs_tol=1e-9):
        raise ValueError(
            "Awakening conversion currently supports only "
            f"--step-seconds={STEP_SECONDS:g}; got {step_seconds}."
        )


def _shape_tuple(value: object) -> tuple[int, ...]:
    if value is None:
        return ()
    try:
        return tuple(int(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ()


def consecutive_true_runs(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return end-exclusive runs of true values in a one-dimensional mask."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError(f"mask must be one-dimensional, got {values.shape}.")
    if not values.any():
        return ()
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return tuple((int(start), int(end)) for start, end in zip(starts, ends))


def inspect_lance_row(
    *,
    data: np.ndarray,
    shape: object,
    original_shape: object,
    valid_length: object,
    channel_names: Sequence[str] | None,
    label: object,
    min_zero_run_samples: int = MIN_ALL_CHANNEL_ZERO_RUN_SAMPLES,
) -> RowInspection:
    """Validate and identify independently usable 2 s children of one row."""

    actual_shape = _shape_tuple(shape)
    source_shape = _shape_tuple(original_shape)
    values = np.asarray(data)

    if actual_shape == (0, PARENT_SAMPLES) or values.size == 0:
        return RowInspection("empty_row", (), (), ())
    if actual_shape != (N_CHANNELS, PARENT_SAMPLES):
        return RowInspection("shape_mismatch", (), (), ())
    if source_shape != (N_CHANNELS, PARENT_SAMPLES):
        return RowInspection("original_shape_mismatch", (), (), ())
    try:
        length = int(valid_length)
    except (TypeError, ValueError):
        return RowInspection("valid_length_mismatch", (), (), ())
    if length != PARENT_ELEMENTS:
        return RowInspection("valid_length_mismatch", (), (), ())
    if values.size != PARENT_ELEMENTS:
        return RowInspection("data_length_mismatch", (), (), ())
    if channel_names is None or len(channel_names) != N_CHANNELS:
        return RowInspection("channel_count_mismatch", (), (), ())
    if tuple(str(name) for name in channel_names) != CHANNEL_NAMES:
        return RowInspection("channel_order_mismatch", (), (), ())
    try:
        canonical_label = int(label)
    except (TypeError, ValueError):
        return RowInspection("invalid_label", (), (), ())
    if canonical_label not in (0, 1):
        return RowInspection("invalid_label", (), (), ())
    if not np.issubdtype(values.dtype, np.number):
        return RowInspection("non_numeric_data", (), (), ())
    if not np.isfinite(values).all():
        return RowInspection("nan_or_inf", (), (), ())

    signal = np.asarray(values, dtype=np.float32).reshape(
        N_CHANNELS, PARENT_SAMPLES
    )
    if np.all(signal == 0):
        return RowInspection("all_zero_row", (), (), ())
    if min_zero_run_samples <= 0:
        raise ValueError("min_zero_run_samples must be positive.")

    simultaneous_zero = np.all(signal == 0, axis=0)
    abnormal_runs = tuple(
        (start, end)
        for start, end in consecutive_true_runs(simultaneous_zero)
        if end - start >= min_zero_run_samples
    )
    affected: set[int] = set()
    for start, end in abnormal_runs:
        first = start // WINDOW_SAMPLES
        last = (end - 1) // WINDOW_SAMPLES
        affected.update(range(first, last + 1))
    affected = {
        index for index in affected if 0 <= index < SUBWINDOWS_PER_PARENT
    }
    valid = tuple(
        index
        for index in range(SUBWINDOWS_PER_PARENT)
        if index not in affected
    )
    return RowInspection(
        row_rejection_reason=None,
        valid_subwindow_indices=valid,
        zero_run_subwindow_indices=tuple(sorted(affected)),
        zero_runs=abnormal_runs,
    )


def extract_subwindows(
    data: np.ndarray, indices: Iterable[int]
) -> np.ndarray:
    """Copy selected non-overlapping children from one validated parent."""

    signal = np.asarray(data, dtype=np.float32)
    if signal.size != PARENT_ELEMENTS:
        raise ValueError(
            f"parent must contain {PARENT_ELEMENTS} values, got {signal.size}."
        )
    signal = signal.reshape(N_CHANNELS, PARENT_SAMPLES)
    selected = tuple(int(index) for index in indices)
    if any(not 0 <= index < SUBWINDOWS_PER_PARENT for index in selected):
        raise ValueError(f"invalid subwindow indices: {selected}.")
    if not selected:
        return np.empty((0, N_CHANNELS, WINDOW_SAMPLES), dtype=np.float32)
    return np.stack(
        [
            signal[:, index * WINDOW_SAMPLES:(index + 1) * WINDOW_SAMPLES]
            for index in selected
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _stable_parent_score(
    *, seed: int, subject_id: str, label: int, global_idx: int
) -> bytes:
    payload = f"{seed}\0{subject_id}\0{label}\0{global_idx}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def assign_logical_sessions(
    parents: Sequence[ParentWindowPlan],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[dict[int, str], list[dict[str, object]]]:
    """Deterministically stratify parent rows within subject and label."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")
    groups: dict[tuple[str, int], list[ParentWindowPlan]] = {}
    for parent in parents:
        if not parent.valid_subwindow_indices:
            continue
        groups.setdefault((parent.source_subject_id, parent.label), []).append(parent)

    assignments: dict[int, str] = {}
    manifest_groups: list[dict[str, object]] = []
    for (subject_id, label), group in sorted(groups.items()):
        ranked = sorted(
            group,
            key=lambda parent: (
                _stable_parent_score(
                    seed=seed,
                    subject_id=subject_id,
                    label=label,
                    global_idx=parent.global_idx,
                ),
                parent.global_idx,
            ),
        )
        count = len(ranked)
        validation_count = int(math.floor(count * validation_fraction + 0.5))
        if count >= 2:
            validation_count = min(max(validation_count, 1), count - 1)
        else:
            validation_count = 0
        validation_ids = {parent.global_idx for parent in ranked[:validation_count]}
        s1_ids: list[int] = []
        s2_ids: list[int] = []
        for parent in sorted(group, key=lambda value: value.global_idx):
            session = "S2" if parent.global_idx in validation_ids else "S1"
            if parent.global_idx in assignments:
                raise ValueError(
                    f"duplicate source_lance_row_id: {parent.global_idx}."
                )
            assignments[parent.global_idx] = session
            (s2_ids if session == "S2" else s1_ids).append(parent.global_idx)
        manifest_groups.append(
            {
                "source_subject_id": subject_id,
                "label": label,
                "parent_rows": count,
                "S1_parent_rows": len(s1_ids),
                "S2_parent_rows": len(s2_ids),
                "S1_source_lance_row_ids": s1_ids,
                "S2_source_lance_row_ids": s2_ids,
            }
        )
    return assignments, manifest_groups


def canonical_subject_mapping(
    parents: Sequence[ParentWindowPlan],
) -> dict[str, int]:
    subjects = sorted({parent.source_subject_id for parent in parents})
    return {source_id: index + 1 for index, source_id in enumerate(subjects)}


class AwakeningHDF5Writer:
    """Preallocated streaming writer for one canonical subject file."""

    def __init__(
        self,
        path: Path,
        *,
        total_windows: int,
        canonical_subject_id: int,
        source_subject_id: str,
        seed: int,
        validation_fraction: float,
    ) -> None:
        if total_windows <= 0:
            raise ValueError("total_windows must be positive.")
        self.path = Path(path)
        self.total_windows = int(total_windows)
        self.canonical_subject_id = int(canonical_subject_id)
        self.source_subject_id = str(source_subject_id)
        self.offset = 0
        self.handle = h5py.File(self.path, "w")
        chunk_rows = min(8, self.total_windows)
        self.data = self.handle.create_dataset(
            "data",
            shape=(self.total_windows, N_CHANNELS, WINDOW_SAMPLES),
            dtype="float32",
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=(chunk_rows, N_CHANNELS, WINDOW_SAMPLES),
        )
        self.labels = self.handle.create_dataset(
            "labels", shape=(self.total_windows,), dtype="int64"
        )
        self.subject_ids = self.handle.create_dataset(
            "subject_ids", shape=(self.total_windows,), dtype="int64"
        )
        string_dtype = h5py.string_dtype(encoding="utf-8")
        self.session_ids = self.handle.create_dataset(
            "session_ids", shape=(self.total_windows,), dtype=string_dtype
        )
        self.trial_ids = self.handle.create_dataset(
            "trial_ids", shape=(self.total_windows,), dtype="int64"
        )
        self.source_lance_row_ids = self.handle.create_dataset(
            "source_lance_row_ids", shape=(self.total_windows,), dtype="int64"
        )
        self.source_subwindow_indices = self.handle.create_dataset(
            "source_subwindow_indices", shape=(self.total_windows,), dtype="int64"
        )
        self.source_sample_ids = self.handle.create_dataset(
            "source_sample_ids", shape=(self.total_windows,), dtype=string_dtype
        )
        attrs = self.handle.attrs
        attrs["sample_rate"] = SAMPLE_RATE
        attrs["window_seconds"] = WINDOW_SECONDS
        attrs["step_seconds"] = STEP_SECONDS
        attrs["samples_per_window"] = WINDOW_SAMPLES
        attrs["channel_names"] = json.dumps(CHANNEL_NAMES, ensure_ascii=False)
        attrs["class_names"] = json.dumps(CLASS_NAMES, ensure_ascii=False)
        attrs["unit"] = "uV"
        attrs["dataset_name"] = "awakening"
        attrs["no_temporal_padding"] = True
        attrs["session_semantics"] = SESSION_SEMANTICS
        attrs["label_mapping"] = json.dumps(LABEL_MAPPING, sort_keys=True)
        attrs["segmentation"] = "five_non_overlapping_2s_windows_within_10s_parent"
        attrs["source_sample_rate"] = SAMPLE_RATE
        attrs["source_samples_per_parent"] = PARENT_SAMPLES
        attrs["source_subject_id"] = self.source_subject_id
        attrs["canonical_subject_id"] = self.canonical_subject_id
        attrs["split_seed"] = int(seed)
        attrs["validation_fraction"] = float(validation_fraction)
        attrs["all_channel_zero_run_min_samples"] = (
            MIN_ALL_CHANNEL_ZERO_RUN_SAMPLES
        )

    def append(
        self,
        *,
        windows: np.ndarray,
        label: int,
        session_id: str,
        source_lance_row_id: int,
        source_subwindow_indices: Sequence[int],
        source_sample_id: str,
    ) -> None:
        values = np.asarray(windows, dtype=np.float32)
        expected = (len(source_subwindow_indices), N_CHANNELS, WINDOW_SAMPLES)
        if values.shape != expected:
            raise ValueError(f"windows shape mismatch: expected {expected}, got {values.shape}.")
        if session_id not in {"S1", "S2"}:
            raise ValueError(f"invalid logical session: {session_id!r}.")
        stop = self.offset + len(values)
        if stop > self.total_windows:
            raise RuntimeError("writer received more windows than preallocated.")
        target = slice(self.offset, stop)
        self.data[target] = values
        self.labels[target] = int(label)
        self.subject_ids[target] = self.canonical_subject_id
        self.session_ids[target] = np.asarray(
            [session_id] * len(values), dtype=h5py.string_dtype("utf-8")
        )
        self.trial_ids[target] = np.arange(self.offset, stop, dtype=np.int64)
        self.source_lance_row_ids[target] = int(source_lance_row_id)
        self.source_subwindow_indices[target] = np.asarray(
            source_subwindow_indices, dtype=np.int64
        )
        self.source_sample_ids[target] = np.asarray(
            [source_sample_id] * len(values), dtype=h5py.string_dtype("utf-8")
        )
        self.offset = stop

    def close(self) -> None:
        if self.handle.id.valid:
            if self.offset != self.total_windows:
                self.handle.close()
                raise RuntimeError(
                    f"wrote {self.offset} windows, expected {self.total_windows}."
                )
            self.handle.flush()
            self.handle.close()

    def abort(self) -> None:
        if self.handle.id.valid:
            self.handle.close()


def validate_awakening_hdf5(
    path: Path,
    *,
    expected_windows: int,
    canonical_subject_id: int,
    source_subject_id: str,
    io_batch_size: int = 128,
) -> dict[str, object]:
    """Validate the persisted public schema and its no-padding invariants."""

    required = {
        "data", "labels", "subject_ids", "session_ids", "trial_ids",
        "source_lance_row_ids", "source_subwindow_indices", "source_sample_ids",
    }
    with h5py.File(path, "r") as handle:
        missing = required - set(handle.keys())
        if missing:
            raise ValueError(f"{path}: missing datasets {sorted(missing)}.")
        data = handle["data"]
        expected_shape = (expected_windows, N_CHANNELS, WINDOW_SAMPLES)
        if data.shape != expected_shape or data.dtype != np.dtype("float32"):
            raise ValueError(
                f"{path}: expected float32 {expected_shape}, got {data.shape} {data.dtype}."
            )
        if data.compression != "gzip" or data.compression_opts != 4 or not data.shuffle:
            raise ValueError(f"{path}: data compression contract differs.")
        for name in (
            "labels", "subject_ids", "trial_ids", "source_lance_row_ids",
            "source_subwindow_indices",
        ):
            if handle[name].shape != (expected_windows,):
                raise ValueError(f"{path}: {name} length differs.")
        if handle["labels"].dtype != np.dtype("int64"):
            raise ValueError(f"{path}: labels must be int64.")
        labels = handle["labels"][:]
        subjects = handle["subject_ids"][:]
        sessions = handle["session_ids"].asstr()[:]
        trials = handle["trial_ids"][:]
        subwindows = handle["source_subwindow_indices"][:]
        if not np.isin(labels, (0, 1)).all():
            raise ValueError(f"{path}: invalid labels.")
        if not np.all(subjects == canonical_subject_id):
            raise ValueError(f"{path}: canonical subject IDs differ.")
        if set(sessions.tolist()) != {"S1", "S2"}:
            raise ValueError(f"{path}: expected both S1 and S2.")
        if not np.array_equal(trials, np.arange(expected_windows, dtype=np.int64)):
            raise ValueError(f"{path}: trial_ids are not unique sequential IDs.")
        if not np.isin(subwindows, np.arange(SUBWINDOWS_PER_PARENT)).all():
            raise ValueError(f"{path}: invalid source subwindow index.")
        if json.loads(handle.attrs["channel_names"]) != list(CHANNEL_NAMES):
            raise ValueError(f"{path}: channel order differs.")
        if json.loads(handle.attrs["class_names"]) != list(CLASS_NAMES):
            raise ValueError(f"{path}: class semantics differ.")
        if str(handle.attrs["source_subject_id"]) != source_subject_id:
            raise ValueError(f"{path}: source subject provenance differs.")
        if not bool(handle.attrs["no_temporal_padding"]):
            raise ValueError(f"{path}: no_temporal_padding is false.")

        for start in range(0, expected_windows, io_batch_size):
            block = data[start:min(start + io_batch_size, expected_windows)]
            if not np.isfinite(block).all():
                raise ValueError(f"{path}: data contains NaN or Inf.")
            if np.any(np.all(block == 0, axis=(1, 2))):
                raise ValueError(f"{path}: data contains an all-zero window.")
            simultaneous_zero = np.all(block == 0, axis=1)
            for row_mask in simultaneous_zero:
                if any(
                    end - begin >= MIN_ALL_CHANNEL_ZERO_RUN_SAMPLES
                    for begin, end in consecutive_true_runs(row_mask)
                ):
                    raise ValueError(
                        f"{path}: data contains an all-channel zero run."
                    )
        return {
            "windows": expected_windows,
            "labels": {
                CLASS_NAMES[index]: int(np.count_nonzero(labels == index))
                for index in range(len(CLASS_NAMES))
            },
            "sessions": {
                session: int(np.count_nonzero(sessions == session))
                for session in ("S1", "S2")
            },
        }


def counts_by_subject_label_session(
    parents: Sequence[ParentWindowPlan],
    assignments: Mapping[int, str],
) -> dict[str, object]:
    result: dict[str, dict[str, dict[str, int]]] = {}
    for parent in parents:
        if not parent.valid_subwindow_indices:
            continue
        session = assignments[parent.global_idx]
        subject = result.setdefault(parent.source_subject_id, {})
        label = subject.setdefault(str(parent.label), {"S1": 0, "S2": 0})
        label[session] += len(parent.valid_subwindow_indices)
    return result
