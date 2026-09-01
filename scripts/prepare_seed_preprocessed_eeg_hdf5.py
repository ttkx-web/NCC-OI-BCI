from __future__ import annotations

"""Convert SEED Preprocessed_EEG MAT sessions into canonical 2 s EEGHDF5.

The converter is deliberately source-trial safe: each 2 s epoch comes from one
``*_eegK`` variable in one original session.  It does not filter, resample,
normalise, rename, or otherwise alter the source signal beyond float32 storage.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy.io import loadmat, whosmat

try:  # Script execution.
    from _bootstrap import ROOT
except ModuleNotFoundError:  # Imported by tests.
    from scripts._bootstrap import ROOT

from bci_dayloop.data.hdf5_dataset import EEGHDF5, HDF5Metadata


SEED_CHANNEL_NAMES = [
    "FP1", "FPZ", "FP2", "AF3", "AF4",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "CB1", "O1", "OZ", "O2", "CB2",
]
CLASS_NAMES = ["negative", "neutral", "positive"]
SOURCE_LABEL_MAPPING = {-1: "negative", 0: "neutral", 1: "positive"}
CANONICAL_LABEL_MAPPING = {0: "negative", 1: "neutral", 2: "positive"}
SOURCE_TO_CANONICAL = {-1: 0, 0: 1, 1: 2}

SAMPLE_RATE = 200.0
WINDOW_SECONDS = 2.0
SAMPLES_PER_EPOCH = int(SAMPLE_RATE * WINDOW_SECONDS)
DATASET_NAME = "seed_emotion"
SEGMENTATION_POLICY = "non_overlapping_2s_within_source_trial"
SOURCE_BANDPASS = "0-75 Hz"
EXPECTED_SUBJECTS = 15
TRIALS_PER_SESSION = 15
SESSIONS_PER_SUBJECT = 3


@dataclass(frozen=True)
class SessionSource:
    canonical_session_id: str
    source_session_id: str
    date: str
    path: Path


@dataclass(frozen=True)
class TrialPlan:
    session: SessionSource
    source_trial_id: int
    source_trial_ordinal: int
    variable_name: str
    source_samples: int
    epoch_count: int
    remainder_samples: int
    source_label: int
    canonical_label: int


def _decode_matlab_string(value: object) -> str:
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_channel_names(data_root: Path) -> list[str]:
    path = data_root / "chn_names.mat"
    if not path.is_file():
        raise FileNotFoundError(f"SEED channel-name MAT is missing: {path}")
    loaded = loadmat(path, variable_names=["chn_names"])
    if "chn_names" not in loaded:
        raise KeyError(f"{path}: expected variable 'chn_names'.")
    names = [_decode_matlab_string(value) for value in loaded["chn_names"].reshape(-1)]
    if names != SEED_CHANNEL_NAMES:
        raise ValueError(
            f"{path}: channel order differs from the approved SEED 62-channel contract."
        )
    return names


def _xlsx_shared_strings(archive: ZipFile) -> list[str]:
    name = "xl/sharedStrings.xml"
    if name not in archive.namelist():
        return []
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(archive.read(name))
    return [
        "".join(node.text or "" for node in item.findall(".//x:t", namespace))
        for item in root.findall("x:si", namespace)
    ]


def _xlsx_column_a_values(archive: ZipFile, worksheet_name: str, shared: list[str]) -> list[str]:
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(archive.read(worksheet_name))
    values: list[str] = []
    for cell in root.findall(".//x:sheetData/x:row/x:c", namespace):
        if not cell.attrib.get("r", "").startswith("A"):
            continue
        raw = cell.findtext("x:v", default="", namespaces=namespace)
        if cell.attrib.get("t") == "s":
            if not raw or int(raw) >= len(shared):
                raise ValueError(f"{worksheet_name}: invalid shared-string index {raw!r}.")
            values.append(shared[int(raw)])
        elif raw:
            values.append(raw)
    return values


def validate_channel_order_workbook(data_root: Path, expected_names: list[str]) -> None:
    """Require every populated workbook sheet to match chn_names.mat exactly."""
    path = data_root / "labels" / "Channel Order.xlsx"
    if not path.is_file():
        raise FileNotFoundError(f"SEED channel-order workbook is missing: {path}")
    with ZipFile(path) as archive:
        shared = _xlsx_shared_strings(archive)
        sheets = sorted(
            name for name in archive.namelist()
            if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)
        )
        if not sheets:
            raise ValueError(f"{path}: no worksheet XML files found.")
        for sheet in sheets:
            values = _xlsx_column_a_values(archive, sheet, shared)
            if values and values != expected_names:
                raise ValueError(f"{path}:{sheet}: channel order differs from chn_names.mat.")


def load_subject_order(data_root: Path) -> list[str]:
    path = data_root / "Subjects_list.txt"
    if not path.is_file():
        raise FileNotFoundError(f"SEED subject list is missing: {path}")
    names = re.findall(r"'([A-Za-z]+_\d{8})'", path.read_text(encoding="utf-8"))
    if len(names) != EXPECTED_SUBJECTS * SESSIONS_PER_SUBJECT:
        raise ValueError(
            f"{path}: expected {EXPECTED_SUBJECTS * SESSIONS_PER_SUBJECT} full session names, "
            f"found {len(names)}."
        )
    subject_order: list[str] = []
    for offset in range(0, len(names), SESSIONS_PER_SUBJECT):
        group = names[offset:offset + SESSIONS_PER_SUBJECT]
        prefixes = {name.rsplit("_", 1)[0] for name in group}
        if len(prefixes) != 1:
            raise ValueError(f"{path}: session group {group} has inconsistent subject names.")
        subject_order.append(next(iter(prefixes)))
    if len(subject_order) != EXPECTED_SUBJECTS or len(set(subject_order)) != EXPECTED_SUBJECTS:
        raise ValueError(f"{path}: invalid published subject order: {subject_order}.")
    return subject_order


def load_public_labels(input_root: Path) -> np.ndarray:
    path = input_root / "label.mat"
    if not path.is_file():
        raise FileNotFoundError(f"SEED public label MAT is missing: {path}")
    loaded = loadmat(path, variable_names=["label"])
    if "label" not in loaded:
        raise KeyError(f"{path}: expected variable 'label'.")
    labels = np.asarray(loaded["label"], dtype=np.int64).reshape(-1)
    if labels.shape != (TRIALS_PER_SESSION,):
        raise ValueError(f"{path}: expected {TRIALS_PER_SESSION} labels, got {labels.shape}.")
    if not np.isin(labels, tuple(SOURCE_TO_CANONICAL)).all():
        raise ValueError(f"{path}: labels must be -1/0/1, got {sorted(set(labels.tolist()))}.")
    return labels


def _eeg_variables(path: Path) -> list[tuple[int, str, tuple[int, ...]]]:
    matched: list[tuple[int, str, tuple[int, ...]]] = []
    for name, shape, _dtype in whosmat(path):
        match = re.search(r"_eeg(\d+)$", name, flags=re.IGNORECASE)
        if match:
            matched.append((int(match.group(1)), name, tuple(shape)))
    matched.sort(key=lambda item: item[0])
    ordinals = [item[0] for item in matched]
    expected = list(range(1, TRIALS_PER_SESSION + 1))
    if ordinals != expected:
        raise ValueError(f"{path}: expected *_eeg1..*_eeg15, found {ordinals}.")
    for ordinal, name, shape in matched:
        if len(shape) != 2 or shape[0] != len(SEED_CHANNEL_NAMES) or shape[1] <= 0:
            raise ValueError(f"{path}:{name}: expected [62,T], got {shape}.")
    return matched


def discover_subject_sessions(input_root: Path, source_subject_id: str) -> list[SessionSource]:
    pattern = re.compile(rf"^{re.escape(source_subject_id)}_(\d{{8}})\.mat$")
    candidates: list[tuple[str, Path]] = []
    for path in input_root.glob(f"{source_subject_id}_*.mat"):
        match = pattern.fullmatch(path.name)
        if match:
            candidates.append((match.group(1), path))
    candidates.sort(key=lambda item: item[0])
    dates = [date for date, _path in candidates]
    if len(candidates) != SESSIONS_PER_SUBJECT or len(set(dates)) != SESSIONS_PER_SUBJECT:
        raise ValueError(
            f"{source_subject_id}: expected exactly three unique YYYYMMDD session MAT files; "
            f"found {[path.name for _date, path in candidates]}."
        )
    sessions: list[SessionSource] = []
    for index, (date, path) in enumerate(candidates, start=1):
        _eeg_variables(path)
        sessions.append(
            SessionSource(
                canonical_session_id=f"S{index}",
                source_session_id=path.stem,
                date=date,
                path=path,
            )
        )
    return sessions


def build_trial_plan(
    *,
    sessions: list[SessionSource],
    source_labels: np.ndarray,
) -> list[TrialPlan]:
    if len(sessions) != SESSIONS_PER_SUBJECT:
        raise ValueError(f"Expected three sessions, got {len(sessions)}.")
    plans: list[TrialPlan] = []
    for session_index, session in enumerate(sessions):
        for ordinal, variable_name, shape in _eeg_variables(session.path):
            source_samples = int(shape[1])
            epoch_count = source_samples // SAMPLES_PER_EPOCH
            if epoch_count <= 0:
                raise ValueError(f"{session.path}:{variable_name}: shorter than one 2 s epoch.")
            source_label = int(source_labels[ordinal - 1])
            plans.append(
                TrialPlan(
                    session=session,
                    # Unique within the subject H5; ordinal alone repeats in every session.
                    source_trial_id=session_index * TRIALS_PER_SESSION + ordinal,
                    source_trial_ordinal=ordinal,
                    variable_name=variable_name,
                    source_samples=source_samples,
                    epoch_count=epoch_count,
                    remainder_samples=source_samples % SAMPLES_PER_EPOCH,
                    source_label=source_label,
                    canonical_label=SOURCE_TO_CANONICAL[source_label],
                )
            )
    return plans


def _string_array(values: Iterable[str]) -> np.ndarray:
    return np.asarray(list(values), dtype=h5py.string_dtype(encoding="utf-8"))


def _write_metadata(
    handle: h5py.File,
    *,
    subject_id: int,
    source_subject_id: str,
    sessions: list[SessionSource],
) -> None:
    handle.attrs["sample_rate"] = SAMPLE_RATE
    handle.attrs["channel_names"] = json.dumps(SEED_CHANNEL_NAMES, ensure_ascii=False)
    handle.attrs["class_names"] = json.dumps(CLASS_NAMES, ensure_ascii=False)
    # Current runtime accepts only V/mV/uV.  No source values are scaled.
    handle.attrs["unit"] = "uV"
    handle.attrs["dataset_name"] = DATASET_NAME
    handle.attrs["canonical_subject_id"] = int(subject_id)
    handle.attrs["source_subject_id"] = source_subject_id
    handle.attrs["num_classes"] = len(CLASS_NAMES)
    handle.attrs["label_mapping"] = json.dumps(
        {str(key): value for key, value in CANONICAL_LABEL_MAPPING.items()}, sort_keys=True
    )
    handle.attrs["source_label_mapping"] = json.dumps(
        {str(key): value for key, value in SOURCE_LABEL_MAPPING.items()}, sort_keys=True
    )
    handle.attrs["source_preprocessed"] = True
    handle.attrs["source_sample_rate"] = SAMPLE_RATE
    handle.attrs["source_bandpass"] = SOURCE_BANDPASS
    handle.attrs["window_seconds"] = WINDOW_SECONDS
    handle.attrs["samples_per_epoch"] = SAMPLES_PER_EPOCH
    handle.attrs["segmentation"] = SEGMENTATION_POLICY
    handle.attrs["source_unit"] = "unknown"
    handle.attrs["unit_status"] = "pipeline_compatibility_assumption"
    handle.attrs["unit_assumption"] = "uV"
    handle.attrs["unit_scaling_applied"] = False
    handle.attrs["session_type"] = "original"
    handle.attrs["source_session_mapping"] = json.dumps(
        {
            session.canonical_session_id: {
                "source_session_id": session.source_session_id,
                "date": session.date,
                "source_file": session.path.name,
            }
            for session in sessions
        },
        sort_keys=True,
    )


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        CLASS_NAMES[index]: int(np.count_nonzero(labels == index))
        for index in range(len(CLASS_NAMES))
    }


def _write_subject_hdf5(
    *,
    output_path: Path,
    subject_id: int,
    source_subject_id: str,
    sessions: list[SessionSource],
    plans: list[TrialPlan],
) -> None:
    total_epochs = sum(plan.epoch_count for plan in plans)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with h5py.File(temporary, "w") as handle:
            data = handle.create_dataset(
                "data",
                shape=(total_epochs, len(SEED_CHANNEL_NAMES), SAMPLES_PER_EPOCH),
                dtype="float32",
            )
            labels = handle.create_dataset("labels", shape=(total_epochs,), dtype="int64")
            subject_ids = handle.create_dataset("subject_ids", shape=(total_epochs,), dtype="int64")
            session_ids = handle.create_dataset(
                "session_ids", shape=(total_epochs,), dtype=h5py.string_dtype(encoding="utf-8")
            )
            trial_ids = handle.create_dataset("trial_ids", shape=(total_epochs,), dtype="int64")
            source_trial_ids = handle.create_dataset(
                "source_trial_ids", shape=(total_epochs,), dtype="int64"
            )
            source_trial_ordinals = handle.create_dataset(
                "source_trial_ordinals", shape=(total_epochs,), dtype="int64"
            )
            source_epoch_indices = handle.create_dataset(
                "source_epoch_indices", shape=(total_epochs,), dtype="int64"
            )
            source_start_samples = handle.create_dataset(
                "source_start_samples", shape=(total_epochs,), dtype="int64"
            )
            source_end_samples = handle.create_dataset(
                "source_end_samples", shape=(total_epochs,), dtype="int64"
            )
            source_labels = handle.create_dataset("source_labels", shape=(total_epochs,), dtype="int64")
            source_session_ids = handle.create_dataset(
                "source_session_ids", shape=(total_epochs,), dtype=h5py.string_dtype(encoding="utf-8")
            )
            _write_metadata(
                handle,
                subject_id=subject_id,
                source_subject_id=source_subject_id,
                sessions=sessions,
            )

            offset = 0
            loaded_session_path: Path | None = None
            loaded_session: dict[str, Any] | None = None
            for plan in plans:
                # Load an original MAT session exactly once.  Re-opening this
                # 300+ MiB source file per trial is needlessly expensive and
                # does not change the source-trial segmentation policy.
                if loaded_session_path != plan.session.path:
                    variable_names = [item[1] for item in _eeg_variables(plan.session.path)]
                    loaded_session = loadmat(plan.session.path, variable_names=variable_names)
                    loaded_session_path = plan.session.path
                assert loaded_session is not None
                trial = np.asarray(loaded_session[plan.variable_name])
                if trial.shape != (len(SEED_CHANNEL_NAMES), plan.source_samples):
                    raise ValueError(
                        f"{plan.session.path}:{plan.variable_name}: shape changed from "
                        f"metadata {(len(SEED_CHANNEL_NAMES), plan.source_samples)} to {trial.shape}."
                    )
                if not np.isfinite(trial).all():
                    raise ValueError(f"{plan.session.path}:{plan.variable_name}: contains NaN or Inf.")
                if np.max(np.abs(trial)) > np.finfo(np.float32).max:
                    raise ValueError(
                        f"{plan.session.path}:{plan.variable_name}: cannot be represented in float32 "
                        "without scaling, which this converter forbids."
                    )
                stop = offset + plan.epoch_count
                epochs = np.ascontiguousarray(
                    trial[:, :plan.epoch_count * SAMPLES_PER_EPOCH]
                    .reshape(len(SEED_CHANNEL_NAMES), plan.epoch_count, SAMPLES_PER_EPOCH)
                    .transpose(1, 0, 2),
                    dtype=np.float32,
                )
                indices = np.arange(plan.epoch_count, dtype=np.int64)
                data[offset:stop] = epochs
                labels[offset:stop] = plan.canonical_label
                subject_ids[offset:stop] = subject_id
                session_ids[offset:stop] = _string_array(
                    [plan.session.canonical_session_id] * plan.epoch_count
                )
                trial_ids[offset:stop] = np.arange(offset, stop, dtype=np.int64)
                source_trial_ids[offset:stop] = plan.source_trial_id
                source_trial_ordinals[offset:stop] = plan.source_trial_ordinal
                source_epoch_indices[offset:stop] = indices
                source_start_samples[offset:stop] = indices * SAMPLES_PER_EPOCH
                source_end_samples[offset:stop] = (indices + 1) * SAMPLES_PER_EPOCH
                source_labels[offset:stop] = plan.source_label
                source_session_ids[offset:stop] = _string_array(
                    [plan.session.source_session_id] * plan.epoch_count
                )
                offset = stop
            if offset != total_epochs:
                raise RuntimeError(f"{output_path}: wrote {offset}, expected {total_epochs} epochs.")
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def verify_seed_hdf5(
    path: Path,
    *,
    subject_id: int,
    source_subject_id: str,
    sessions: list[SessionSource],
    plans: list[TrialPlan],
) -> dict[str, Any]:
    reader = EEGHDF5(path)
    metadata: HDF5Metadata = reader.metadata
    if metadata.sample_rate != SAMPLE_RATE:
        raise ValueError(f"{path}: sample-rate mismatch.")
    if metadata.channel_names != SEED_CHANNEL_NAMES or metadata.class_names != CLASS_NAMES:
        raise ValueError(f"{path}: canonical metadata differs from the SEED contract.")
    if metadata.unit != "uV" or metadata.dataset_name != DATASET_NAME:
        raise ValueError(f"{path}: unit or dataset metadata differs from the SEED contract.")

    expected_by_source = {plan.source_trial_id: plan for plan in plans}
    with h5py.File(path, "r") as handle:
        required = {
            "data", "labels", "subject_ids", "session_ids", "trial_ids",
            "source_trial_ids", "source_trial_ordinals", "source_epoch_indices",
            "source_start_samples", "source_end_samples", "source_labels", "source_session_ids",
        }
        missing = required - set(handle.keys())
        if missing:
            raise ValueError(f"{path}: missing required datasets {sorted(missing)}.")
        data = handle["data"]
        n_epochs = len(data)
        if data.ndim != 3 or data.shape[1:] != (62, SAMPLES_PER_EPOCH) or data.dtype != np.dtype("float32"):
            raise ValueError(f"{path}: expected float32 [N,62,400], got {data.shape} {data.dtype}.")
        values = {name: handle[name][:] for name in required if name not in {"data", "session_ids", "source_session_ids"}}
        canonical_sessions = handle["session_ids"].asstr()[:]
        source_sessions = handle["source_session_ids"].asstr()[:]
        if not np.isfinite(data[:]).all():
            raise ValueError(f"{path}: data contains NaN or Inf.")
        if any(len(value) != n_epochs for value in values.values()) or len(canonical_sessions) != n_epochs:
            raise ValueError(f"{path}: trial-level arrays have inconsistent lengths.")
        if not np.isin(values["labels"], (0, 1, 2)).all():
            raise ValueError(f"{path}: canonical labels are invalid.")
        if not np.array_equal(np.unique(values["subject_ids"]), np.asarray([subject_id])):
            raise ValueError(f"{path}: subject IDs differ from {subject_id}.")
        if len(np.unique(values["trial_ids"])) != n_epochs:
            raise ValueError(f"{path}: epoch-level trial IDs are not unique.")
        if set(canonical_sessions.tolist()) != {"S1", "S2", "S3"}:
            raise ValueError(f"{path}: canonical sessions are invalid: {sorted(set(canonical_sessions))}.")
        if len(set(zip(values["source_trial_ids"].tolist(), values["source_epoch_indices"].tolist()))) != n_epochs:
            raise ValueError(f"{path}: source trial / epoch identity is not unique.")

        for source_id, plan in expected_by_source.items():
            mask = values["source_trial_ids"] == source_id
            if int(mask.sum()) != plan.epoch_count:
                raise ValueError(f"{path}: source trial {source_id} epoch count differs.")
            if set(canonical_sessions[mask].tolist()) != {plan.session.canonical_session_id}:
                raise ValueError(f"{path}: source trial {source_id} crosses canonical sessions.")
            if set(source_sessions[mask].tolist()) != {plan.session.source_session_id}:
                raise ValueError(f"{path}: source trial {source_id} crosses source sessions.")
            if not np.all(values["source_trial_ordinals"][mask] == plan.source_trial_ordinal):
                raise ValueError(f"{path}: source trial ordinal differs for {source_id}.")
            if not np.all(values["source_labels"][mask] == plan.source_label):
                raise ValueError(f"{path}: source labels differ for {source_id}.")
            indices = values["source_epoch_indices"][mask]
            if not np.array_equal(indices, np.arange(plan.epoch_count, dtype=np.int64)):
                raise ValueError(f"{path}: source epoch indices differ for {source_id}.")
            if not np.array_equal(values["source_start_samples"][mask], indices * SAMPLES_PER_EPOCH):
                raise ValueError(f"{path}: source starts differ for {source_id}.")
            if not np.array_equal(values["source_end_samples"][mask], (indices + 1) * SAMPLES_PER_EPOCH):
                raise ValueError(f"{path}: source ends differ for {source_id}.")

        mapping = json.loads(handle.attrs["source_session_mapping"])
        expected_mapping = {
            session.canonical_session_id: {
                "source_session_id": session.source_session_id,
                "date": session.date,
                "source_file": session.path.name,
            }
            for session in sessions
        }
        if mapping != expected_mapping:
            raise ValueError(f"{path}: source session mapping differs.")
        if handle.attrs["canonical_subject_id"] != subject_id or handle.attrs["source_subject_id"] != source_subject_id:
            raise ValueError(f"{path}: subject provenance differs.")
        if bool(handle.attrs["unit_scaling_applied"]):
            raise ValueError(f"{path}: signal values were unexpectedly scaled.")

        session_counts = {
            session: int(np.count_nonzero(canonical_sessions == session))
            for session in ("S1", "S2", "S3")
        }
        return {
            "subject": subject_id,
            "source_subject_id": source_subject_id,
            "session_mapping": mapping,
            "S1_epochs": session_counts["S1"],
            "S2_epochs": session_counts["S2"],
            "S3_epochs": session_counts["S3"],
            "total_epochs": n_epochs,
            "class_counts": _class_counts(values["labels"]),
            "discarded_remainder_samples": int(sum(plan.remainder_samples for plan in plans)),
            "source_trial_epoch_counts": {
                str(plan.source_trial_id): int(plan.epoch_count) for plan in plans
            },
        }


def _build_subject_conversion(
    *,
    input_root: Path,
    data_root: Path,
    subject_id: int,
    subject_order: list[str],
    source_labels: np.ndarray,
) -> tuple[str, list[SessionSource], list[TrialPlan]]:
    if not 1 <= subject_id <= len(subject_order):
        raise ValueError(f"Subject ID must be in [1,{len(subject_order)}], got {subject_id}.")
    source_subject_id = subject_order[subject_id - 1]
    sessions = discover_subject_sessions(input_root, source_subject_id)
    plans = build_trial_plan(sessions=sessions, source_labels=source_labels)
    if len(plans) != SESSIONS_PER_SUBJECT * TRIALS_PER_SESSION:
        raise RuntimeError(f"{source_subject_id}: expected 45 source trials, got {len(plans)}.")
    return source_subject_id, sessions, plans


def convert_subject(
    *,
    input_root: Path,
    data_root: Path,
    output_root: Path,
    subject_id: int,
    subject_order: list[str],
    source_labels: np.ndarray,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Any]:
    source_subject_id, sessions, plans = _build_subject_conversion(
        input_root=input_root,
        data_root=data_root,
        subject_id=subject_id,
        subject_order=subject_order,
        source_labels=source_labels,
    )
    if dry_run:
        labels = np.concatenate([
            np.full(plan.epoch_count, plan.canonical_label, dtype=np.int64) for plan in plans
        ])
        return {
            "subject": subject_id,
            "source_subject_id": source_subject_id,
            "session_mapping": {
                session.canonical_session_id: session.source_session_id for session in sessions
            },
            "S1_epochs": int(sum(plan.epoch_count for plan in plans if plan.session.canonical_session_id == "S1")),
            "S2_epochs": int(sum(plan.epoch_count for plan in plans if plan.session.canonical_session_id == "S2")),
            "S3_epochs": int(sum(plan.epoch_count for plan in plans if plan.session.canonical_session_id == "S3")),
            "total_epochs": int(sum(plan.epoch_count for plan in plans)),
            "class_counts": _class_counts(labels),
            "discarded_remainder_samples": int(sum(plan.remainder_samples for plan in plans)),
        }

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"subject_{subject_id:02d}.h5"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")
    _write_subject_hdf5(
        output_path=output_path,
        subject_id=subject_id,
        source_subject_id=source_subject_id,
        sessions=sessions,
        plans=plans,
    )
    summary = verify_seed_hdf5(
        output_path,
        subject_id=subject_id,
        source_subject_id=source_subject_id,
        sessions=sessions,
        plans=plans,
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
    subjects: list[int],
    per_subject: list[dict[str, Any]],
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "subjects": subjects,
        "dry_run": dry_run,
        "dataset_name": DATASET_NAME,
        "sample_rate": SAMPLE_RATE,
        "channel_names": SEED_CHANNEL_NAMES,
        "class_names": CLASS_NAMES,
        "label_mapping": {str(key): value for key, value in CANONICAL_LABEL_MAPPING.items()},
        "source_label_mapping": {str(key): value for key, value in SOURCE_LABEL_MAPPING.items()},
        "source_preprocessed": True,
        "source_bandpass": SOURCE_BANDPASS,
        "window_seconds": WINDOW_SECONDS,
        "samples_per_epoch": SAMPLES_PER_EPOCH,
        "segmentation": SEGMENTATION_POLICY,
        "source_unit": "unknown",
        "unit": "uV",
        "unit_status": "pipeline_compatibility_assumption",
        "unit_assumption": "uV",
        "unit_scaling_applied": False,
        "total_epochs": int(sum(item["total_epochs"] for item in per_subject)),
        "total_class_counts": {
            name: int(sum(item["class_counts"][name] for item in per_subject))
            for name in CLASS_NAMES
        },
        "discarded_remainder_samples": int(
            sum(item["discarded_remainder_samples"] for item in per_subject)
        ),
        "per_subject": per_subject,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SEED Preprocessed_EEG MAT files to canonical 2 s EEGHDF5."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/Volumes/UBUNTU-SERV/data/SEED/Preprocessed_EEG"),
        help="SEED Preprocessed_EEG directory containing session MAT files and label.mat.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data" / "processed" / "seed",
        help="Directory for subject_XX.h5 and conversion_summary.json.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=list(range(1, EXPECTED_SUBJECTS + 1)),
        help="Canonical SEED subject IDs to convert. Default: 1 through 15.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace selected existing HDF5 outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print counts without writing HDF5.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    data_root = input_root.parent
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()
    subjects = sorted(set(int(subject) for subject in args.subjects))
    if not subjects or any(subject < 1 or subject > EXPECTED_SUBJECTS for subject in subjects):
        raise ValueError(f"--subjects must be unique IDs in [1,{EXPECTED_SUBJECTS}], got {subjects}.")
    if not input_root.is_dir():
        raise NotADirectoryError(f"SEED Preprocessed_EEG directory is missing: {input_root}")

    channel_names = load_channel_names(data_root)
    validate_channel_order_workbook(data_root, channel_names)
    subject_order = load_subject_order(data_root)
    labels = load_public_labels(input_root)
    per_subject = [
        convert_subject(
            input_root=input_root,
            data_root=data_root,
            output_root=output_root,
            subject_id=subject_id,
            subject_order=subject_order,
            source_labels=labels,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
        )
        for subject_id in subjects
    ]
    for item in per_subject:
        print(json.dumps(item, ensure_ascii=False), flush=True)
    summary = _summary_payload(
        input_root=input_root,
        output_root=output_root,
        subjects=subjects,
        per_subject=per_subject,
        dry_run=bool(args.dry_run),
    )
    if not args.dry_run:
        _write_json(output_root / "conversion_summary.json", summary)
        print(f"Wrote conversion summary: {output_root / 'conversion_summary.json'}")
    print(json.dumps({
        "total_epochs": summary["total_epochs"],
        "total_class_counts": summary["total_class_counts"],
        "discarded_remainder_samples": summary["discarded_remainder_samples"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
