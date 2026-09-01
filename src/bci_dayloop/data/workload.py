"""Convert published preprocessed Workload EEGLAB epochs into grouped HDF5."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np

from .hdf5_dataset import HDF5Metadata


DATASET_NAME = "workload_pbci_hackathon"
FORMAT_VERSION = "workload_hdf5_v1"
WINDOW_SEC = 2.0
EASY_CONDITION = "MATBeasy"
DIFF_CONDITION = "MATBdiff"
IGNORED_CONDITIONS = ("MATBmed", "RS", "RSraw")
CLASS_NAMES = ("low_workload", "high_workload")
LABEL_MAP = {EASY_CONDITION: 0, DIFF_CONDITION: 1}


@dataclass(frozen=True)
class WorkloadSourcePaths:
    """The two source EEGLAB files needed for one subject session."""

    easy_set: Path
    diff_set: Path


@dataclass(frozen=True)
class WorkloadCondition:
    """One published, already-preprocessed EEGLAB epoch collection."""

    condition: str
    data: np.ndarray
    channel_names: tuple[str, ...]
    sample_rate: float
    unit: str
    source_set: Path


@dataclass(frozen=True)
class WorkloadStream:
    """A synthetic, balanced easy/diff stream for one session."""

    data: np.ndarray
    labels: np.ndarray
    condition_ids: np.ndarray
    source_epoch_indices: np.ndarray
    trial_ordinals: np.ndarray
    window_ids: np.ndarray


@dataclass(frozen=True)
class WorkloadSession:
    """Validated source metadata and alternating stream for one session."""

    session_id: str
    stream: WorkloadStream
    channel_names: tuple[str, ...]
    sample_rate: float
    unit: str
    easy_source_set: Path
    diff_source_set: Path


def canonical_subject_id(subject: int | str) -> str:
    """Return the on-disk Pxx identifier used by the published dataset."""
    text = str(subject).strip()
    if text.upper().startswith("P"):
        text = text[1:]
    if not text.isdigit():
        raise ValueError(f"Invalid Workload subject {subject!r}; expected an integer or Pxx.")
    return f"P{int(text):02d}"


def locate_workload_sources(
    data_root: str | Path,
    *,
    subject_id: str,
    session_id: str,
) -> WorkloadSourcePaths:
    """Find exactly the MATBeasy and MATBdiff EEGLAB files for one session."""
    eeg_dir = Path(data_root) / subject_id / session_id / "eeg"
    if not eeg_dir.is_dir():
        raise FileNotFoundError(
            f"Workload source directory is missing for subject={subject_id}, "
            f"session={session_id}: {eeg_dir}"
        )

    def find_condition(condition: str) -> Path:
        # macOS/exFAT can materialize AppleDouble sidecars (``._*.set``).
        # They are metadata files, never EEG sources.
        matches = sorted(
            path for path in eeg_dir.glob(f"*{condition}.set")
            if not path.name.startswith("._")
        )
        if not matches:
            raise FileNotFoundError(
                f"Missing Workload source for subject={subject_id}, session={session_id}, "
                f"condition={condition}: expected one '*{condition}.set' under {eeg_dir}"
            )
        if len(matches) != 1:
            rendered = ", ".join(str(path) for path in matches)
            raise ValueError(
                f"Ambiguous Workload source for subject={subject_id}, session={session_id}, "
                f"condition={condition}: {rendered}"
            )
        return matches[0]

    return WorkloadSourcePaths(
        easy_set=find_condition(EASY_CONDITION),
        diff_set=find_condition(DIFF_CONDITION),
    )


def _context(
    *,
    subject_id: str,
    session_id: str,
    condition: str,
    source_set: Path,
) -> str:
    return (
        f"subject={subject_id}, session={session_id}, condition={condition}, "
        f"path={source_set}"
    )


def _mne_epochs_unit(epochs: object, *, context: str) -> str:
    """MNE returns EEG values in volts; reject mixed or non-volt channel units."""
    try:
        from mne.io.constants import FIFF
    except ImportError as error:  # pragma: no cover - guarded by the caller.
        raise RuntimeError("MNE is required to read EEGLAB epochs.") from error

    info = getattr(epochs, "info")
    units = {int(channel["unit"]) for channel in info["chs"]}
    if units != {int(FIFF.FIFF_UNIT_V)}:
        raise ValueError(
            f"Workload EEGLAB units must be volts for {context}; "
            f"actual MNE unit codes={sorted(units)}, expected={[int(FIFF.FIFF_UNIT_V)]}."
        )
    return "V"


def read_eeglab_condition(
    set_path: str | Path,
    *,
    subject_id: str,
    session_id: str,
    condition: str,
) -> WorkloadCondition:
    """Read one pre-epoched EEGLAB condition without changing its signal values."""
    path = Path(set_path)
    context = _context(
        subject_id=subject_id,
        session_id=session_id,
        condition=condition,
        source_set=path,
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing Workload EEGLAB source for {context}.")
    try:
        import mne
    except ImportError as error:
        raise RuntimeError("MNE is required to read Workload EEGLAB .set files.") from error

    try:
        epochs = mne.io.read_epochs_eeglab(path, verbose=False)
    except ImportError as error:
        raise RuntimeError(
            f"Could not load Workload EEGLAB epochs for {context}; "
            "the installed MNE optional dependencies are incompatible or incomplete."
        ) from error
    loaded = WorkloadCondition(
        condition=condition,
        data=np.asarray(epochs.get_data()),
        channel_names=tuple(str(name) for name in epochs.ch_names),
        sample_rate=float(epochs.info["sfreq"]),
        unit=_mne_epochs_unit(epochs, context=context),
        source_set=path,
    )
    validate_condition(loaded, subject_id=subject_id, session_id=session_id)
    return loaded


def validate_condition(
    condition: WorkloadCondition,
    *,
    subject_id: str,
    session_id: str,
) -> None:
    """Validate a single source condition as fixed 2 s finite [N,C,T] epochs."""
    context = _context(
        subject_id=subject_id,
        session_id=session_id,
        condition=condition.condition,
        source_set=condition.source_set,
    )
    data = np.asarray(condition.data)
    if data.ndim != 3:
        raise ValueError(f"Expected [N,C,T] data for {context}; actual shape={data.shape}.")
    if data.shape[0] == 0:
        raise ValueError(f"Expected at least one epoch for {context}; actual epoch count=0.")
    if data.shape[1] != len(condition.channel_names):
        raise ValueError(
            f"Channel count mismatch for {context}; actual data channels={data.shape[1]}, "
            f"expected channel_names length={len(condition.channel_names)}."
        )
    if condition.sample_rate <= 0:
        raise ValueError(
            f"Sample rate must be positive for {context}; actual={condition.sample_rate}."
        )
    expected_samples = int(round(WINDOW_SEC * condition.sample_rate))
    if data.shape[2] != expected_samples:
        raise ValueError(
            f"Expected a {WINDOW_SEC:g} s epoch for {context}; actual samples={data.shape[2]}, "
            f"expected samples={expected_samples} at sample_rate={condition.sample_rate}."
        )
    if not condition.unit:
        raise ValueError(f"Missing data unit for {context}; expected a non-empty unit.")
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite EEG values for {context}; expected all values to be finite.")


def validate_condition_pair(
    easy: WorkloadCondition,
    diff: WorkloadCondition,
    *,
    subject_id: str,
    session_id: str,
) -> None:
    """Validate that easy and diff can form one lossless alternating stream."""
    validate_condition(easy, subject_id=subject_id, session_id=session_id)
    validate_condition(diff, subject_id=subject_id, session_id=session_id)
    if easy.condition != EASY_CONDITION or diff.condition != DIFF_CONDITION:
        raise ValueError(
            f"Expected conditions {EASY_CONDITION}/{DIFF_CONDITION} for subject={subject_id}, "
            f"session={session_id}; actual={easy.condition}/{diff.condition}."
        )
    if easy.data.shape[0] != diff.data.shape[0]:
        raise ValueError(
            f"Unequal Workload epoch counts for subject={subject_id}, session={session_id}: "
            f"easy epochs={easy.data.shape[0]} ({easy.source_set}); "
            f"diff epochs={diff.data.shape[0]} ({diff.source_set})."
        )
    if easy.channel_names != diff.channel_names:
        raise ValueError(
            f"Channel names/order mismatch for subject={subject_id}, session={session_id}: "
            f"easy path={easy.source_set}, diff path={diff.source_set}; "
            f"actual easy={easy.channel_names}, expected diff={diff.channel_names}."
        )
    if easy.sample_rate != diff.sample_rate:
        raise ValueError(
            f"Sample rate mismatch for subject={subject_id}, session={session_id}: "
            f"easy path={easy.source_set} actual={easy.sample_rate}, "
            f"diff path={diff.source_set} expected={diff.sample_rate}."
        )
    if easy.data.shape[2] != diff.data.shape[2]:
        raise ValueError(
            f"Epoch sample count mismatch for subject={subject_id}, session={session_id}: "
            f"easy path={easy.source_set} actual={easy.data.shape[2]}, "
            f"diff path={diff.source_set} expected={diff.data.shape[2]}."
        )
    if easy.unit != diff.unit:
        raise ValueError(
            f"Data unit mismatch for subject={subject_id}, session={session_id}: "
            f"easy path={easy.source_set} actual={easy.unit!r}, "
            f"diff path={diff.source_set} expected={diff.unit!r}."
        )


def interleave_easy_diff(
    easy_data: np.ndarray,
    diff_data: np.ndarray,
    *,
    subject_id: str,
    session_id: str,
) -> WorkloadStream:
    """Create one deterministic easy[0], diff[0], easy[1], diff[1] stream."""
    easy = np.asarray(easy_data)
    diff = np.asarray(diff_data)
    if easy.ndim != 3 or diff.ndim != 3:
        raise ValueError(
            f"Expected [N,C,T] easy/diff arrays for subject={subject_id}, session={session_id}; "
            f"actual easy={easy.shape}, diff={diff.shape}."
        )
    if easy.shape[0] != diff.shape[0]:
        raise ValueError(
            f"Unequal Workload epoch counts for subject={subject_id}, session={session_id}: "
            f"easy epochs={easy.shape[0]}, diff epochs={diff.shape[0]}."
        )
    if easy.shape[1:] != diff.shape[1:]:
        raise ValueError(
            f"Easy/diff feature shape mismatch for subject={subject_id}, session={session_id}: "
            f"easy={easy.shape[1:]}, diff={diff.shape[1:]}."
        )

    epochs_per_condition = easy.shape[0]
    num_trials = 2 * epochs_per_condition
    data = np.empty((num_trials, easy.shape[1], easy.shape[2]), dtype=np.float32)
    data[0::2] = easy.astype(np.float32, copy=False)
    data[1::2] = diff.astype(np.float32, copy=False)
    labels = np.empty(num_trials, dtype=np.int64)
    labels[0::2] = LABEL_MAP[EASY_CONDITION]
    labels[1::2] = LABEL_MAP[DIFF_CONDITION]
    condition_ids = labels.astype(np.int8, copy=True)
    source_epoch_indices = np.repeat(
        np.arange(epochs_per_condition, dtype=np.int64), 2
    )
    trial_ordinals = np.arange(1, num_trials + 1, dtype=np.int64)
    window_ids = np.asarray(
        [
            f"{subject_id}:{session_id}:{condition}:{source_index:06d}"
            for source_index in range(epochs_per_condition)
            for condition in (EASY_CONDITION, DIFF_CONDITION)
        ],
        dtype=object,
    )
    stream = WorkloadStream(
        data=data,
        labels=labels,
        condition_ids=condition_ids,
        source_epoch_indices=source_epoch_indices,
        trial_ordinals=trial_ordinals,
        window_ids=window_ids,
    )
    _validate_stream(stream, subject_id=subject_id, session_id=session_id)
    return stream


def build_workload_session(
    easy: WorkloadCondition,
    diff: WorkloadCondition,
    *,
    subject_id: str,
    session_id: str,
) -> WorkloadSession:
    """Validate one source pair and build its only permitted stream order."""
    validate_condition_pair(easy, diff, subject_id=subject_id, session_id=session_id)
    stream = interleave_easy_diff(
        easy.data,
        diff.data,
        subject_id=subject_id,
        session_id=session_id,
    )
    return WorkloadSession(
        session_id=session_id,
        stream=stream,
        channel_names=easy.channel_names,
        sample_rate=easy.sample_rate,
        unit=easy.unit,
        easy_source_set=easy.source_set,
        diff_source_set=diff.source_set,
    )


def load_workload_session(
    data_root: str | Path,
    *,
    subject_id: str,
    session_id: str,
) -> WorkloadSession:
    """Locate, read, validate, and interleave exactly the two target conditions."""
    sources = locate_workload_sources(
        data_root, subject_id=subject_id, session_id=session_id
    )
    easy = read_eeglab_condition(
        sources.easy_set,
        subject_id=subject_id,
        session_id=session_id,
        condition=EASY_CONDITION,
    )
    diff = read_eeglab_condition(
        sources.diff_set,
        subject_id=subject_id,
        session_id=session_id,
        condition=DIFF_CONDITION,
    )
    return build_workload_session(
        easy, diff, subject_id=subject_id, session_id=session_id
    )


def _validate_stream(stream: WorkloadStream, *, subject_id: str, session_id: str) -> None:
    data = stream.data
    num_trials = data.shape[0]
    if data.ndim != 3 or num_trials == 0 or num_trials % 2:
        raise ValueError(
            f"Invalid alternating stream for subject={subject_id}, session={session_id}; "
            f"actual data shape={data.shape}, expected non-empty even [N,C,T]."
        )
    trial_arrays = (
        stream.labels,
        stream.condition_ids,
        stream.source_epoch_indices,
        stream.trial_ordinals,
        stream.window_ids,
    )
    if any(len(values) != num_trials for values in trial_arrays):
        raise ValueError(
            f"Mismatched stream metadata length for subject={subject_id}, session={session_id}."
        )
    expected_labels = np.tile(np.asarray([0, 1], dtype=np.int64), num_trials // 2)
    if not np.array_equal(stream.labels, expected_labels):
        raise ValueError(
            f"Alternating labels must be 0,1,... for subject={subject_id}, session={session_id}."
        )
    if not np.array_equal(stream.condition_ids, expected_labels.astype(np.int8)):
        raise ValueError(
            f"Alternating condition IDs must be 0,1,... for subject={subject_id}, session={session_id}."
        )
    if not np.array_equal(
        stream.source_epoch_indices,
        np.repeat(np.arange(num_trials // 2, dtype=np.int64), 2),
    ):
        raise ValueError(
            f"Source epoch indices must be 0,0,1,1,... for subject={subject_id}, session={session_id}."
        )
    if not np.array_equal(stream.trial_ordinals, np.arange(1, num_trials + 1)):
        raise ValueError(
            f"Trial ordinals must be 1-based and contiguous for subject={subject_id}, session={session_id}."
        )
    if len(set(str(value) for value in stream.window_ids)) != num_trials:
        raise ValueError(
            f"Window IDs must be unique for subject={subject_id}, session={session_id}."
        )
    if not np.isfinite(data).all():
        raise ValueError(
            f"Alternating stream contains non-finite values for subject={subject_id}, session={session_id}."
        )


def _relative_source_path(path: Path, data_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(data_root.resolve()))
    except ValueError:
        return str(path)


def _validate_subject_sessions(
    sessions: Sequence[WorkloadSession], *, subject_id: str
) -> None:
    if not sessions:
        raise ValueError(f"No Workload sessions supplied for subject={subject_id}.")
    session_ids = [session.session_id for session in sessions]
    if len(set(session_ids)) != len(session_ids):
        raise ValueError(f"Duplicate Workload session IDs for subject={subject_id}: {session_ids}")
    reference = sessions[0]
    _validate_stream(reference.stream, subject_id=subject_id, session_id=reference.session_id)
    for session in sessions[1:]:
        _validate_stream(session.stream, subject_id=subject_id, session_id=session.session_id)
        if session.channel_names != reference.channel_names:
            raise ValueError(
                f"Channel names/order differs across sessions for subject={subject_id}: "
                f"session={session.session_id} actual={session.channel_names} "
                f"({session.easy_source_set}, {session.diff_source_set}); "
                f"expected session={reference.session_id} values={reference.channel_names} "
                f"({reference.easy_source_set}, {reference.diff_source_set})."
            )
        if session.sample_rate != reference.sample_rate:
            raise ValueError(
                f"Sample rate differs across sessions for subject={subject_id}: "
                f"session={session.session_id} actual={session.sample_rate} "
                f"({session.easy_source_set}, {session.diff_source_set}); "
                f"expected session={reference.session_id} value={reference.sample_rate} "
                f"({reference.easy_source_set}, {reference.diff_source_set})."
            )
        if session.stream.data.shape[2] != reference.stream.data.shape[2]:
            raise ValueError(
                f"Epoch sample count differs across sessions for subject={subject_id}: "
                f"session={session.session_id} actual={session.stream.data.shape[2]} "
                f"({session.easy_source_set}, {session.diff_source_set}); "
                f"expected session={reference.session_id} value={reference.stream.data.shape[2]} "
                f"({reference.easy_source_set}, {reference.diff_source_set})."
            )
        if session.unit != reference.unit:
            raise ValueError(
                f"Data unit differs across sessions for subject={subject_id}: "
                f"session={session.session_id} actual={session.unit!r} "
                f"({session.easy_source_set}, {session.diff_source_set}); "
                f"expected session={reference.session_id} value={reference.unit!r} "
                f"({reference.easy_source_set}, {reference.diff_source_set})."
            )


def _string_array(values: Iterable[object]) -> np.ndarray:
    return np.asarray(
        [str(value) for value in values], dtype=h5py.string_dtype(encoding="utf-8")
    )


def _write_workload_hdf5(
    path: Path,
    sessions: Sequence[WorkloadSession],
    *,
    subject_id: str,
    data_root: Path,
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["format_version"] = FORMAT_VERSION
        handle.attrs["dataset_name"] = DATASET_NAME
        handle.attrs["subject_id"] = subject_id
        handle.attrs["window_sec"] = WINDOW_SEC
        handle.attrs["input_is_preprocessed"] = True
        handle.attrs["unit"] = sessions[0].unit
        handle.attrs["class_names"] = json.dumps(CLASS_NAMES)
        handle.attrs["label_map"] = json.dumps(LABEL_MAP, sort_keys=True)
        handle.attrs["ignored_conditions"] = json.dumps(IGNORED_CONDITIONS)
        handle.attrs["stream_construction"] = "synthetic_alternating_easy_diff"
        handle.attrs["preserves_within_condition_order"] = True
        handle.attrs["preserves_original_cross_condition_timeline"] = False
        handle.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        grouped_sessions = handle.create_group("sessions")
        for session in sessions:
            group = grouped_sessions.create_group(session.session_id)
            stream = session.stream
            group.create_dataset(
                "data", data=stream.data, dtype="float32", compression="gzip", shuffle=True
            )
            group.create_dataset("labels", data=stream.labels, dtype="int64")
            group.create_dataset("condition_ids", data=stream.condition_ids, dtype="int8")
            group.create_dataset(
                "source_epoch_indices", data=stream.source_epoch_indices, dtype="int64"
            )
            group.create_dataset("trial_ordinals", data=stream.trial_ordinals, dtype="int64")
            group.create_dataset("window_ids", data=_string_array(stream.window_ids))
            group.attrs["session_id"] = session.session_id
            group.attrs["sample_rate"] = session.sample_rate
            group.attrs["channel_names"] = json.dumps(session.channel_names)
            group.attrs["num_channels"] = len(session.channel_names)
            group.attrs["num_samples_per_window"] = stream.data.shape[2]
            group.attrs["num_easy_epochs"] = stream.data.shape[0] // 2
            group.attrs["num_diff_epochs"] = stream.data.shape[0] // 2
            group.attrs["num_trials"] = stream.data.shape[0]
            group.attrs["easy_source_set"] = _relative_source_path(
                session.easy_source_set, data_root
            )
            group.attrs["diff_source_set"] = _relative_source_path(
                session.diff_source_set, data_root
            )
            group.attrs["shuffle"] = False
            group.attrs["alternating"] = True
            group.attrs["odd_trial_condition"] = EASY_CONDITION
            group.attrs["even_trial_condition"] = DIFF_CONDITION


def verify_workload_hdf5(path: str | Path, *, expected_sessions: Sequence[str]) -> None:
    """Reopen an export and assert its grouped stream invariants before publication."""
    target = Path(path)
    required_root_attrs = (
        "format_version",
        "dataset_name",
        "subject_id",
        "window_sec",
        "input_is_preprocessed",
        "unit",
        "class_names",
        "label_map",
        "ignored_conditions",
        "stream_construction",
        "preserves_within_condition_order",
        "preserves_original_cross_condition_timeline",
        "created_at",
    )
    required_session_attrs = (
        "session_id",
        "sample_rate",
        "channel_names",
        "num_channels",
        "num_samples_per_window",
        "num_easy_epochs",
        "num_diff_epochs",
        "num_trials",
        "easy_source_set",
        "diff_source_set",
        "shuffle",
        "alternating",
        "odd_trial_condition",
        "even_trial_condition",
    )
    with h5py.File(target, "r") as handle:
        missing_root = [name for name in required_root_attrs if name not in handle.attrs]
        if missing_root:
            raise ValueError(f"Workload HDF5 {target} is missing root attributes: {missing_root}")
        expected_root_values = {
            "format_version": FORMAT_VERSION,
            "dataset_name": DATASET_NAME,
            "window_sec": WINDOW_SEC,
            "input_is_preprocessed": True,
            "stream_construction": "synthetic_alternating_easy_diff",
            "preserves_within_condition_order": True,
            "preserves_original_cross_condition_timeline": False,
        }
        for name, expected in expected_root_values.items():
            actual = handle.attrs[name]
            if actual != expected:
                raise ValueError(
                    f"Workload HDF5 {target} root attribute {name!r} differs; "
                    f"actual={actual!r}, expected={expected!r}."
                )
        if "sessions" not in handle:
            raise ValueError(f"Workload HDF5 {target} is missing /sessions.")
        actual_sessions = sorted(handle["sessions"].keys())
        if actual_sessions != sorted(expected_sessions):
            raise ValueError(
                f"Workload HDF5 {target} session groups differ; "
                f"actual={actual_sessions}, expected={sorted(expected_sessions)}."
            )
        subject_id = str(handle.attrs["subject_id"])
        for session_id in expected_sessions:
            group = handle["sessions"][session_id]
            missing_attrs = [name for name in required_session_attrs if name not in group.attrs]
            if missing_attrs:
                raise ValueError(
                    f"Workload HDF5 {target} /sessions/{session_id} is missing attributes: "
                    f"{missing_attrs}"
                )
            required_datasets = (
                "data",
                "labels",
                "condition_ids",
                "source_epoch_indices",
                "trial_ordinals",
                "window_ids",
            )
            missing_datasets = [name for name in required_datasets if name not in group]
            if missing_datasets:
                raise ValueError(
                    f"Workload HDF5 {target} /sessions/{session_id} is missing datasets: "
                    f"{missing_datasets}"
                )
            data = group["data"][:]
            labels = group["labels"][:]
            condition_ids = group["condition_ids"][:]
            source_epoch_indices = group["source_epoch_indices"][:]
            trial_ordinals = group["trial_ordinals"][:]
            window_ids = group["window_ids"].asstr()[:]
            stream = WorkloadStream(
                data=data,
                labels=labels,
                condition_ids=condition_ids,
                source_epoch_indices=source_epoch_indices,
                trial_ordinals=trial_ordinals,
                window_ids=window_ids,
            )
            _validate_stream(stream, subject_id=subject_id, session_id=session_id)
            expected_window_ids = np.asarray(
                [
                    f"{subject_id}:{session_id}:{condition}:{source_index:06d}"
                    for source_index in range(len(window_ids) // 2)
                    for condition in (EASY_CONDITION, DIFF_CONDITION)
                ],
                dtype=object,
            )
            if not np.array_equal(window_ids, expected_window_ids):
                raise ValueError(
                    f"Workload HDF5 {target} /sessions/{session_id} window_ids do not "
                    "preserve the required alternating source order."
                )

    reader = WorkloadHDF5(target)
    for session_id in expected_sessions:
        loaded = reader.load(session=session_id)
        if loaded["data"].shape[0] != len(loaded["labels"]):
            raise ValueError(
                f"WorkloadHDF5.load() returned mismatched data/labels for "
                f"{target} session={session_id}."
            )


def write_workload_hdf5(
    path: str | Path,
    sessions: Sequence[WorkloadSession],
    *,
    subject_id: str,
    data_root: str | Path,
    overwrite: bool = False,
) -> Path:
    """Atomically write grouped session streams, preserving any existing target on failure."""
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"Workload output already exists: {target}. Pass --overwrite to replace it."
        )
    _validate_subject_sessions(sessions, subject_id=subject_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        _write_workload_hdf5(
            temporary,
            sessions,
            subject_id=subject_id,
            data_root=Path(data_root),
        )
        verify_workload_hdf5(
            temporary, expected_sessions=[session.session_id for session in sessions]
        )
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def prepare_workload_subject(
    data_root: str | Path,
    output_root: str | Path,
    *,
    subject: int | str,
    sessions: Sequence[str],
    overwrite: bool = False,
) -> Path:
    """Prepare all requested sessions for one subject into one grouped HDF5 file."""
    subject_id = canonical_subject_id(subject)
    session_ids = [str(session).strip() for session in sessions]
    if not session_ids or any(not session for session in session_ids):
        raise ValueError("At least one non-empty Workload session ID is required.")
    if len(set(session_ids)) != len(session_ids):
        raise ValueError(f"Duplicate Workload session IDs requested: {session_ids}")
    prepared_sessions = [
        load_workload_session(
            data_root, subject_id=subject_id, session_id=session_id
        )
        for session_id in session_ids
    ]
    output_path = Path(output_root) / f"subject_{int(subject_id[1:]):02d}.h5"
    return write_workload_hdf5(
        output_path,
        prepared_sessions,
        subject_id=subject_id,
        data_root=data_root,
        overwrite=overwrite,
    )


class WorkloadHDF5:
    """Process-safe reader for one Workload subject file with separate session groups."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Workload HDF5 not found: {self.path}")

    def sessions(self) -> list[str]:
        with h5py.File(self.path, "r") as handle:
            if "sessions" not in handle:
                raise ValueError(f"Workload HDF5 is missing /sessions: {self.path}")
            return sorted(handle["sessions"].keys())

    def available_sessions(self) -> list[str]:
        """Return session names through the shared trial-reader contract."""
        return self.sessions()

    @property
    def source_subject_id(self) -> str:
        with h5py.File(self.path, "r") as handle:
            return str(handle.attrs["subject_id"])

    @property
    def canonical_subject_id(self) -> int:
        """Return the integer subject identity used by population trainers."""
        source_id = canonical_subject_id(self.source_subject_id)
        return int(source_id[1:])

    @property
    def metadata(self) -> HDF5Metadata:
        """Expose grouped Workload metadata in the common reader shape."""
        with h5py.File(self.path, "r") as handle:
            sessions = sorted(handle["sessions"].keys())
            if not sessions:
                raise ValueError(f"Workload HDF5 has no sessions: {self.path}")
            first = handle["sessions"][sessions[0]]
            return HDF5Metadata(
                sample_rate=float(first.attrs["sample_rate"]),
                channel_names=json.loads(first.attrs["channel_names"]),
                class_names=json.loads(handle.attrs["class_names"]),
                unit=str(handle.attrs["unit"]),
                dataset_name=str(handle.attrs["dataset_name"]),
            )

    def _trainer_trial_ids(
        self,
        *,
        session: str,
        trial_ordinals: np.ndarray,
    ) -> np.ndarray:
        """Encode (session, trial_ordinal) into a stable file-local int64 ID.

        The population trainer adds the canonical subject in the high 32 bits.
        Workload ordinals repeat across S1/S2, so the lower 32 bits reserve
        12 bits for the numeric S<n> session and 20 bits for the ordinal.
        """
        if not session.startswith("S") or not session[1:].isdigit():
            raise ValueError(
                "Workload session IDs must use the generated S<n> form to "
                f"derive stable trainer trial IDs; got {session!r}."
            )
        session_number = int(session[1:])
        ordinals = np.asarray(trial_ordinals, dtype=np.int64)
        if not 0 < session_number < 2**12:
            raise ValueError(f"Workload session number is out of range: {session!r}.")
        if np.any(ordinals <= 0) or np.any(ordinals >= 2**20):
            raise ValueError("Workload trial_ordinals must be in [1, 2**20).")
        return ((session_number << 20) | ordinals).astype(np.int64, copy=False)

    def trial_metadata(self) -> dict[str, np.ndarray]:
        """Return all grouped sessions in the trainer's canonical trial view."""
        loaded = [self.load(session=session) for session in self.sessions()]
        return {
            key: np.concatenate([item[key] for item in loaded], axis=0)
            for key in ("labels", "subject_ids", "session_ids", "trial_ids")
        }

    def load(self, *, session: str) -> dict[str, np.ndarray]:
        with h5py.File(self.path, "r") as handle:
            if "sessions" not in handle or session not in handle["sessions"]:
                raise ValueError(
                    f"Session {session!r} not found in {self.path}. "
                    f"Available: {self.sessions()}"
                )
            group = handle["sessions"][session]
            trial_ordinals = group["trial_ordinals"][:].astype(
                np.int64, copy=False
            )
            subject_id = int(canonical_subject_id(str(handle.attrs["subject_id"]))[1:])
            n_trials = len(trial_ordinals)
            return {
                "data": group["data"][:].astype(np.float32, copy=False),
                "labels": group["labels"][:].astype(np.int64, copy=False),
                "condition_ids": group["condition_ids"][:].astype(np.int8, copy=False),
                "source_epoch_indices": group["source_epoch_indices"][:].astype(
                    np.int64, copy=False
                ),
                "trial_ordinals": trial_ordinals,
                "window_ids": np.asarray(
                    group["window_ids"].asstr()[:], dtype=object
                ),
                # Canonical in-memory view used by the existing population
                # trainer. These values are intentionally never written back.
                "subject_ids": np.full(n_trials, subject_id, dtype=np.int64),
                "session_ids": np.full(
                    n_trials,
                    session,
                    dtype=f"<U{max(1, len(session))}",
                ),
                "trial_ids": self._trainer_trial_ids(
                    session=session,
                    trial_ordinals=trial_ordinals,
                ),
            }
