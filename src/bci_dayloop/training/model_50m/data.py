"""Data discovery, source-trial splitting, and window construction for 50M training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from bci_dayloop.data.hdf5_dataset import HDF5Metadata
from bci_dayloop.data.splits import (
    WithinSubjectTrialSplit,
    resolve_within_subject_trial_split,
)
from bci_dayloop.data.trial_reader import (
    DataReaderName,
    open_trial_reader,
    reader_identity,
)
from bci_dayloop.training.model_50m.linear_head import (
    WindowSet,
    build_direct_trial_windows,
    build_same_label_concat_windows,
    class_counts,
    limit_windows_per_class,
    validate_labels,
)
from bci_dayloop.training.model_50m.types import SplitBuildResult, WindowBundle
from bci_dayloop.utils.config import project_root

ROOT = project_root()


def normalize_subjects(values: Iterable[int]) -> list[int]:
    subjects = sorted(set(int(value) for value in values))
    if not subjects:
        raise ValueError("At least one subject is required.")
    invalid = [subject for subject in subjects if subject <= 0]
    if invalid:
        raise ValueError(f"Subject IDs must be positive, got {invalid}.")
    return subjects


def class_name_counts(
    labels: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, int]:
    numeric = class_counts(labels, len(class_names))
    return {
        str(class_names[index]): int(numeric[index])
        for index in range(len(class_names))
    }


def encoded_trial_id(subject_id: int, trial_id: int) -> int:
    """
    Encode subject and file-local trial ID into one collision-free int64 key.

    High 32 bits: subject ID
    Low 32 bits: non-negative trial ID
    """
    subject_id = int(subject_id)
    trial_id = int(trial_id)
    if subject_id <= 0:
        raise ValueError(f"subject_id must be positive, got {subject_id}.")
    if trial_id < 0 or trial_id >= 2**32:
        raise ValueError(
            "trial_id must be in [0, 2**32), "
            f"got {trial_id} for subject {subject_id}."
        )
    return (subject_id << 32) | trial_id


def encode_trial_ids(
    subject_id: int,
    trial_ids: np.ndarray,
) -> np.ndarray:
    trial_ids = np.asarray(trial_ids, dtype=np.int64)
    return np.asarray(
        [encoded_trial_id(subject_id, value) for value in trial_ids],
        dtype=np.int64,
    )


def source_id_set(window_set: WindowSet) -> set[int]:
    return {
        int(source_id)
        for source_ids in window_set.source_trial_ids
        for source_id in source_ids
    }


def validate_no_source_leakage(
    left: WindowSet,
    right: WindowSet,
    *,
    left_name: str,
    right_name: str,
) -> None:
    overlap = source_id_set(left) & source_id_set(right)
    if overlap:
        examples = sorted(overlap)[:10]
        raise RuntimeError(
            f"Source-trial leakage between {left_name} and {right_name}. "
            f"Example encoded trial IDs: {examples}."
        )


def validate_metadata_compatibility(
    reference: HDF5Metadata,
    candidate: HDF5Metadata,
    *,
    subject_id: int,
    path: Path,
) -> None:
    mismatches: list[str] = []

    if not np.isclose(reference.sample_rate, candidate.sample_rate):
        mismatches.append(
            "sample_rate "
            f"{candidate.sample_rate} != {reference.sample_rate}"
        )
    if list(reference.channel_names) != list(candidate.channel_names):
        mismatches.append("channel_names differ")
    if list(reference.class_names) != list(candidate.class_names):
        mismatches.append("class_names differ")
    if str(reference.unit) != str(candidate.unit):
        mismatches.append(f"unit {candidate.unit!r} != {reference.unit!r}")
    if str(reference.dataset_name) != str(candidate.dataset_name):
        mismatches.append(
            "dataset_name "
            f"{candidate.dataset_name!r} != {reference.dataset_name!r}"
        )

    if mismatches:
        raise ValueError(
            f"Metadata mismatch for subject {subject_id} at {path}: "
            + "; ".join(mismatches)
        )


def resolve_subject_file(
    *,
    data_root: Path,
    pattern: str,
    subject_id: int,
) -> Path:
    try:
        relative_name = pattern.format(subject=subject_id)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "--data-pattern must be a valid Python format string. It may "
            "use {subject}, for example 'subject_{subject:02d}.h5'."
        ) from exc

    candidates = [
        data_root / relative_name,
        data_root / f"subject_{subject_id:02d}.h5",
        data_root / f"bnci2014_001_s{subject_id:02d}.h5",
        ROOT / "data" / "processed" / f"bnci2014_001_s{subject_id:02d}.h5",
    ]

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        if not resolved.is_absolute():
            resolved = ROOT / resolved
        resolved = resolved.resolve()
        if resolved not in seen:
            unique_candidates.append(resolved)
            seen.add(resolved)

    for candidate in unique_candidates:
        if candidate.is_file():
            return candidate

    formatted = "\n".join(f"  - {path}" for path in unique_candidates)
    raise FileNotFoundError(
        f"Could not find HDF5 data for subject {subject_id}. Tried:\n"
        f"{formatted}"
    )


def build_subject_identities(
    *,
    subject_paths: Mapping[int, Path],
    data_reader: DataReaderName,
) -> dict[str, dict[str, int | str]]:
    """Read stable reader identities for all pre-resolved subject files."""
    return {
        str(subject): reader_identity(
            open_trial_reader(
                data_reader=data_reader,
                path=path,
                canonical_subject_id=subject,
            ),
            data_reader=data_reader,
            canonical_subject_id=subject,
        )
        for subject, path in subject_paths.items()
    }


def validate_loaded_session(
    session_data: Mapping[str, np.ndarray],
    *,
    expected_subject: int,
    expected_session: str,
    allowed_sessions: Sequence[str] | None = None,
    num_classes: int,
    path: Path,
) -> None:
    required = {
        "data",
        "labels",
        "subject_ids",
        "session_ids",
        "trial_ids",
    }
    missing = required - set(session_data)
    if missing:
        raise KeyError(
            f"{path}: loaded session is missing keys {sorted(missing)}."
        )

    n_trials = len(session_data["data"])
    if n_trials <= 0:
        raise ValueError(
            f"{path}: session {expected_session!r} contains no trials."
        )

    for key in ("labels", "subject_ids", "session_ids", "trial_ids"):
        if len(session_data[key]) != n_trials:
            raise ValueError(
                f"{path}: {key} length {len(session_data[key])} "
                f"does not match data length {n_trials}."
            )

    subject_values = sorted(
        set(np.asarray(session_data["subject_ids"], dtype=np.int64).tolist())
    )
    if subject_values != [expected_subject]:
        raise ValueError(
            f"{path}: expected only subject {expected_subject}, "
            f"found {subject_values}."
        )

    session_values = sorted(set(np.asarray(session_data["session_ids"]).astype(str)))
    expected_sessions = (
        tuple(str(session) for session in allowed_sessions)
        if allowed_sessions is not None
        else (expected_session,)
    )
    if not session_values or not set(session_values).issubset(expected_sessions):
        raise ValueError(
            f"{path}: expected session values within {list(expected_sessions)!r}, "
            f"found {session_values}."
        )

    trial_ids = np.asarray(session_data["trial_ids"], dtype=np.int64)
    if len(np.unique(trial_ids)) != len(trial_ids):
        raise ValueError(
            f"{path}: duplicate trial_ids found in session "
            f"{expected_session!r}."
        )

    validate_labels(
        np.asarray(session_data["labels"], dtype=np.int64),
        num_classes=num_classes,
        split_name=f"subject_{expected_subject:02d}/{expected_session}",
    )

    signal = np.asarray(session_data["data"])
    if signal.ndim != 3:
        raise ValueError(
            f"{path}: EEG data must have shape [N,C,T], got {signal.shape}."
        )
    if not np.isfinite(signal).all():
        raise ValueError(
            f"{path}: session {expected_session!r} contains NaN or Inf."
        )


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------
def select_direct_trial_segment(
    *,
    trials: np.ndarray,
    sample_rate: float,
    window_seconds: float,
    anchor: str,
    split_name: str,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    """
    从每个源 trial 中显式选择一个连续窗口。

    不补零、不拼接、不跨 trial；每个输出窗口仍只对应一个源 trial。
    """
    trials = np.asarray(trials, dtype=np.float32)

    if trials.ndim != 3:
        raise ValueError(
            f"{split_name}: trials must have shape [N,C,T], "
            f"got {trials.shape}."
        )
    if sample_rate <= 0:
        raise ValueError(
            f"{split_name}: sample_rate must be positive, "
            f"got {sample_rate}."
        )
    if window_seconds <= 0:
        raise ValueError(
            f"{split_name}: window_seconds must be positive, "
            f"got {window_seconds}."
        )
    if anchor not in {"start", "center", "end"}:
        raise ValueError(
            f"{split_name}: unsupported direct-trial anchor "
            f"{anchor!r}; expected start, center, or end."
        )

    source_samples = int(trials.shape[-1])
    target_samples = int(round(window_seconds * sample_rate))

    if target_samples <= 0:
        raise ValueError(
            f"{split_name}: target window has no samples."
        )

    if target_samples > source_samples:
        raise ValueError(
            f"{split_name}: source trials are only "
            f"{source_samples / sample_rate:.3f}s, but "
            f"--window-sec={window_seconds:.3f}s requires "
            f"{target_samples} samples. Direct-trial mode does not "
            "pad, concatenate, or cross source-trial boundaries."
        )

    if anchor == "start":
        start_sample = 0
    elif anchor == "center":
        start_sample = (source_samples - target_samples) // 2
    else:  # anchor == "end"
        start_sample = source_samples - target_samples

    end_sample = start_sample + target_samples

    return (
        trials[..., start_sample:end_sample],
        {
            "policy": "one_contiguous_window_per_source_trial",
            "anchor": anchor,
            "source_samples": source_samples,
            "source_seconds": source_samples / sample_rate,
            "selected_start_sample": start_sample,
            "selected_end_sample_exclusive": end_sample,
            "selected_start_seconds": start_sample / sample_rate,
            "selected_end_seconds": end_sample / sample_rate,
            "selected_samples": target_samples,
            "selected_seconds": target_samples / sample_rate,
        },
    )

def build_subject_window_bundle(
    *,
    subject_id: int,
    path: Path,
    data_reader: DataReaderName,
    session_name: str,
    reference_metadata: HDF5Metadata | None,
    window_seconds: float,
    stride_seconds: float,
    seed: int,
    shuffle_trials_within_class: bool,
    max_windows_per_class: int | None,
    window_construction: str,
    direct_trial_anchor: str,
) -> tuple[WindowBundle, HDF5Metadata, dict[str, Any]]:
    dataset = open_trial_reader(
        data_reader=data_reader,
        path=path,
        canonical_subject_id=subject_id,
    )
    metadata = dataset.metadata

    if reference_metadata is not None:
        validate_metadata_compatibility(
            reference_metadata,
            metadata,
            subject_id=subject_id,
            path=path,
        )

    session_data = dataset.load(session=session_name)
    bundle, metadata, summary = build_window_bundle_from_session_data(
        subject_id=subject_id,
        path=path,
        session_name=session_name,
        metadata=metadata,
        session_data=session_data,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        seed=seed,
        shuffle_trials_within_class=shuffle_trials_within_class,
        max_windows_per_class=max_windows_per_class,
        window_construction=window_construction,
        direct_trial_anchor=direct_trial_anchor,
    )
    summary.update(
        reader_identity(
            dataset,
            data_reader=data_reader,
            canonical_subject_id=subject_id,
        )
    )
    summary["data_reader"] = data_reader
    return bundle, metadata, summary


def build_window_bundle_from_session_data(
    *,
    subject_id: int,
    path: Path,
    session_name: str,
    allowed_sessions: Sequence[str] | None = None,
    metadata: HDF5Metadata,
    session_data: Mapping[str, np.ndarray],
    class_names: Sequence[str] | None = None,
    window_seconds: float,
    stride_seconds: float,
    seed: int,
    shuffle_trials_within_class: bool,
    max_windows_per_class: int | None,
    window_construction: str,
    direct_trial_anchor: str,
) -> tuple[WindowBundle, HDF5Metadata, dict[str, Any]]:
    """Build windows from an already selected source-trial subset."""
    effective_class_names = list(class_names or metadata.class_names)
    num_classes = len(effective_class_names)
    validate_loaded_session(
        session_data,
        expected_subject=subject_id,
        expected_session=session_name,
        allowed_sessions=allowed_sessions,
        num_classes=num_classes,
        path=path,
    )

    raw_trial_ids = np.asarray(session_data["trial_ids"], dtype=np.int64)
    global_trial_ids = encode_trial_ids(subject_id, raw_trial_ids)

    trials = np.asarray(
        session_data["data"],
        dtype=np.float32,
    )
    labels = np.asarray(
        session_data["labels"],
        dtype=np.int64,
    )

    if window_construction == "direct_trial":
        direct_trials, direct_trial_selection = (
            select_direct_trial_segment(
                trials=trials,
                sample_rate=metadata.sample_rate,
                window_seconds=window_seconds,
                anchor=direct_trial_anchor,
                split_name=(
                    f"subject_{subject_id:02d}/{session_name}"
                ),
            )
        )

        window_set = build_direct_trial_windows(
            trials=direct_trials,
            labels=labels,
            trial_ids=global_trial_ids,
            sample_rate=metadata.sample_rate,
            window_seconds=window_seconds,
            num_classes=num_classes,
            seed=seed,
            split_name=(
                f"subject_{subject_id:02d}/{session_name}"
            ),
        )

    elif window_construction == "same_label_concat":
        direct_trial_selection = None
        window_set = build_same_label_concat_windows(
            trials=trials,
            labels=labels,
            trial_ids=global_trial_ids,
            sample_rate=metadata.sample_rate,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            num_classes=num_classes,
            seed=seed,
            shuffle_trials_within_class=(
                shuffle_trials_within_class
            ),
            split_name=(
                f"subject_{subject_id:02d}/{session_name}"
            ),
        )

    else:
        raise ValueError(
            f"Unsupported window_construction="
            f"{window_construction!r}."
        )

    window_set = limit_windows_per_class(
        window_set,
        max_per_class=max_windows_per_class,
        num_classes=num_classes,
        seed=seed + 1,
    )

    bundle = WindowBundle(
        window_set=window_set,
        window_subject_ids=np.full(
            len(window_set.windows),
            subject_id,
            dtype=np.int64,
        ),
    )

    summary = {
        "subject_id": subject_id,
        "path": str(path),
        "session": session_name,
        "sessions": list(allowed_sessions or (session_name,)),
        "raw_shape": list(np.asarray(session_data["data"]).shape),
        "source_trials_total": int(len(session_data["labels"])),
        "source_trials_per_class": class_name_counts(
            np.asarray(session_data["labels"], dtype=np.int64),
            effective_class_names,
        ),
        "direct_trial_selection": direct_trial_selection,
        "derived_windows_total": int(len(window_set.windows)),
        "derived_windows_per_class": class_name_counts(
            window_set.labels,
            effective_class_names,
        ),
        "unique_source_trials_used": int(len(source_id_set(window_set))),
        "window_seconds": float(window_seconds),
        "stride_seconds": float(stride_seconds),
        "construction": window_set.construction,
    }
    return bundle, metadata, summary


def select_trial_rows(
    trial_data: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return a trial-level subset without changing the underlying HDF5 data."""
    indices = np.asarray(indices, dtype=np.int64)
    return {
        key: np.asarray(values)[indices]
        for key, values in trial_data.items()
    }


def select_trial_ids(
    trial_data: Mapping[str, np.ndarray],
    trial_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Select source trials by ID and fail rather than silently dropping one."""
    requested = np.asarray(trial_ids, dtype=np.int64)
    available = np.asarray(trial_data["trial_ids"], dtype=np.int64)
    mask = np.isin(available, requested)
    selected = select_trial_rows(trial_data, np.flatnonzero(mask))
    selected_ids = np.asarray(selected["trial_ids"], dtype=np.int64)
    if set(selected_ids.tolist()) != set(requested.tolist()):
        missing = sorted(set(requested.tolist()) - set(selected_ids.tolist()))
        raise RuntimeError(
            "Selected source trials are missing from the loaded session: "
            f"{missing[:10]}."
        )
    return selected


def concatenate_session_trial_data(
    session_data: Sequence[Mapping[str, np.ndarray]],
    *,
    sessions: Sequence[str],
) -> dict[str, np.ndarray]:
    """Concatenate canonical trial views while preserving trial identity."""
    if not session_data:
        raise ValueError("At least one train-session payload is required.")
    if len(session_data) != len(sessions):
        raise ValueError(
            "Train-session payload count does not match requested sessions: "
            f"{len(session_data)} != {len(sessions)}."
        )

    required = ("data", "labels", "subject_ids", "session_ids", "trial_ids")
    combined: dict[str, np.ndarray] = {}
    for key in required:
        missing = [index for index, payload in enumerate(session_data) if key not in payload]
        if missing:
            raise KeyError(
                f"Loaded train session payload(s) {missing} are missing {key!r}."
            )
        combined[key] = np.concatenate(
            [np.asarray(payload[key]) for payload in session_data], axis=0
        )

    combined_session_values = set(combined["session_ids"].astype(str).tolist())
    requested_session_values = {str(session) for session in sessions}
    if not combined_session_values.issubset(requested_session_values):
        raise RuntimeError(
            "Concatenated train-session data contains unexpected sessions: "
            f"{sorted(combined_session_values)} not within "
            f"{list(sessions)!r}."
        )
    trial_ids = np.asarray(combined["trial_ids"], dtype=np.int64)
    if len(np.unique(trial_ids)) != len(trial_ids):
        raise ValueError(
            "Within-subject multi-session training requires globally unique "
            "trial_ids across the requested train sessions."
        )
    return combined


def resolve_class_names(
    *,
    metadata: HDF5Metadata,
    explicit_class_names: Sequence[str] | None,
) -> list[str]:
    """Resolve logit semantics without assuming every 4-class task is BNCI."""
    num_classes = len(metadata.class_names)
    if num_classes <= 0:
        raise ValueError("HDF5 metadata class_names must not be empty.")

    if explicit_class_names is not None:
        class_names = [str(name).strip() for name in explicit_class_names]
        source = "--class-names"
    else:
        metadata_names = [str(name).strip() for name in metadata.class_names]
        if len(metadata_names) == num_classes and all(metadata_names):
            class_names = metadata_names
            source = "HDF5 metadata"
        else:
            class_names = [f"class_{index}" for index in range(num_classes)]
            source = "numeric fallback"

    if len(class_names) != num_classes:
        raise ValueError(
            "class_names length must match the HDF5 class count: "
            f"{len(class_names)} != {num_classes}."
        )
    if not all(class_names):
        raise ValueError("class_names must not contain empty values.")
    if len(set(class_names)) != len(class_names):
        raise ValueError(f"class_names must be unique, got {class_names}.")
    print(f"Class semantics source: {source}")
    print("Class semantics:", {index: name for index, name in enumerate(class_names)})
    return class_names


def build_within_subject_splits(
    *,
    subject_id: int,
    path: Path,
    data_reader: DataReaderName = "eeg",
    train_sessions: Sequence[str],
    test_session: str,
    validation_ratio: float,
    seed: int,
    window_seconds: float,
    stride_seconds: float,
    max_windows_per_class: int | None,
    window_construction: str,
    direct_trial_anchor: str,
    explicit_class_names: Sequence[str] | None,
) -> tuple[
    SplitBuildResult,
    SplitBuildResult,
    HDF5Metadata,
    list[str],
    WithinSubjectTrialSplit,
    dict[str, np.ndarray],
]:
    """Resolve a held-out cross-session split before any window construction."""
    dataset = open_trial_reader(
        data_reader=data_reader,
        path=path,
        canonical_subject_id=subject_id,
    )
    metadata = dataset.metadata
    class_names = resolve_class_names(
        metadata=metadata,
        explicit_class_names=explicit_class_names,
    )
    all_trial_metadata = dataset.trial_metadata()
    normalized_train_sessions = tuple(str(session) for session in train_sessions)
    split = resolve_within_subject_trial_split(
        subject_ids=all_trial_metadata["subject_ids"],
        session_ids=all_trial_metadata["session_ids"],
        labels=all_trial_metadata["labels"],
        subject_id=subject_id,
        train_sessions=normalized_train_sessions,
        test_session=test_session,
        validation_ratio=validation_ratio,
        seed=seed,
        num_classes=len(class_names),
    )

    train_source_data = concatenate_session_trial_data(
        [dataset.load(session=session) for session in normalized_train_sessions],
        sessions=normalized_train_sessions,
    )
    train_data = select_trial_ids(
        train_source_data,
        all_trial_metadata["trial_ids"][split.train_indices],
    )
    validation_data = select_trial_ids(
        train_source_data,
        all_trial_metadata["trial_ids"][split.validation_indices],
    )
    train_session_name = ",".join(normalized_train_sessions)
    train_bundle, _, train_summary = build_window_bundle_from_session_data(
        subject_id=subject_id,
        path=path,
        session_name=train_session_name,
        allowed_sessions=normalized_train_sessions,
        metadata=metadata,
        session_data=train_data,
        class_names=class_names,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        seed=seed + 1_000,
        shuffle_trials_within_class=True,
        max_windows_per_class=max_windows_per_class,
        window_construction=window_construction,
        direct_trial_anchor=direct_trial_anchor,
    )
    validation_bundle, _, validation_summary = build_window_bundle_from_session_data(
        subject_id=subject_id,
        path=path,
        session_name=train_session_name,
        allowed_sessions=normalized_train_sessions,
        metadata=metadata,
        session_data=validation_data,
        class_names=class_names,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        seed=seed + 2_000,
        shuffle_trials_within_class=False,
        max_windows_per_class=max_windows_per_class,
        window_construction=window_construction,
        direct_trial_anchor=direct_trial_anchor,
    )

    common_paths = {subject_id: path}
    train_summary["selected_trial_ids"] = train_data["trial_ids"].tolist()
    validation_summary["selected_trial_ids"] = validation_data["trial_ids"].tolist()
    identity = reader_identity(
        dataset,
        data_reader=data_reader,
        canonical_subject_id=subject_id,
    )
    for summary in (train_summary, validation_summary):
        summary.update(identity)
        summary["data_reader"] = data_reader
        summary["train_sessions"] = list(normalized_train_sessions)
    return (
        SplitBuildResult(
            bundle=train_bundle,
            source_trial_summary={f"subject_{subject_id:02d}": train_summary},
            subject_paths=common_paths,
            metadata=metadata,
        ),
        SplitBuildResult(
            bundle=validation_bundle,
            source_trial_summary={f"subject_{subject_id:02d}": validation_summary},
            subject_paths=common_paths,
            metadata=metadata,
        ),
        metadata,
        class_names,
        split,
        all_trial_metadata,
    )


def build_within_subject_test_split(
    *,
    subject_id: int,
    path: Path,
    data_reader: DataReaderName = "eeg",
    metadata: HDF5Metadata,
    class_names: Sequence[str],
    split: WithinSubjectTrialSplit,
    all_trial_metadata: Mapping[str, np.ndarray],
    window_seconds: float,
    stride_seconds: float,
    max_windows_per_class: int | None,
    window_construction: str,
    direct_trial_anchor: str,
) -> SplitBuildResult:
    """Build the held-out test windows only after model selection completes."""
    dataset = open_trial_reader(
        data_reader=data_reader,
        path=path,
        canonical_subject_id=subject_id,
    )
    test_session_data = dataset.load(session=split.test_session)
    test_data = select_trial_ids(
        test_session_data,
        np.asarray(all_trial_metadata["trial_ids"], dtype=np.int64)[
            split.test_indices
        ],
    )
    test_bundle, _, test_summary = build_window_bundle_from_session_data(
        subject_id=subject_id,
        path=path,
        session_name=split.test_session,
        metadata=metadata,
        session_data=test_data,
        class_names=class_names,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        seed=0,
        shuffle_trials_within_class=False,
        max_windows_per_class=max_windows_per_class,
        window_construction=window_construction,
        direct_trial_anchor=direct_trial_anchor,
    )
    test_summary["selected_trial_ids"] = test_data["trial_ids"].tolist()
    test_summary.update(
        reader_identity(
            dataset,
            data_reader=data_reader,
            canonical_subject_id=subject_id,
        )
    )
    test_summary["data_reader"] = data_reader
    return SplitBuildResult(
        bundle=test_bundle,
        source_trial_summary={f"subject_{subject_id:02d}": test_summary},
        subject_paths={subject_id: path},
        metadata=metadata,
    )


def build_within_subject_split_metadata(
    *,
    subject_id: int,
    split: WithinSubjectTrialSplit,
    all_trial_metadata: Mapping[str, np.ndarray],
    class_names: Sequence[str],
    validation_ratio: float,
    seed: int,
) -> dict[str, Any]:
    """Serialize exact within-subject source-trial provenance."""
    labels = np.asarray(all_trial_metadata["labels"], dtype=np.int64)
    trial_ids = np.asarray(all_trial_metadata["trial_ids"], dtype=np.int64)
    return {
        "subject": int(subject_id),
        "train_session": (
            split.train_sessions[0] if len(split.train_sessions) == 1 else None
        ),
        "train_sessions": list(split.train_sessions),
        "test_session": split.test_session,
        "validation_ratio": float(validation_ratio),
        "split_seed": int(seed),
        "available_sessions": list(split.available_sessions),
        "train_trial_ids": trial_ids[split.train_indices].tolist(),
        "validation_trial_ids": trial_ids[split.validation_indices].tolist(),
        "test_trial_ids": trial_ids[split.test_indices].tolist(),
        "train_class_counts": class_name_counts(
            labels[split.train_indices], class_names
        ),
        "validation_class_counts": class_name_counts(
            labels[split.validation_indices], class_names
        ),
        "test_class_counts": class_name_counts(
            labels[split.test_indices], class_names
        ),
    }


def combine_window_bundles(
    bundles: Sequence[WindowBundle],
    *,
    seed: int,
    construction: str,
) -> WindowBundle:
    if not bundles:
        raise ValueError("No window bundles were provided.")

    windows = np.concatenate(
        [bundle.window_set.windows for bundle in bundles],
        axis=0,
    ).astype(np.float32, copy=False)
    labels = np.concatenate(
        [bundle.window_set.labels for bundle in bundles],
        axis=0,
    ).astype(np.int64, copy=False)
    window_subject_ids = np.concatenate(
        [bundle.window_subject_ids for bundle in bundles],
        axis=0,
    ).astype(np.int64, copy=False)

    source_trial_ids: list[tuple[int, ...]] = []
    for bundle in bundles:
        source_trial_ids.extend(bundle.window_set.source_trial_ids)

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(windows))

    combined_set = WindowSet(
        windows=windows[permutation],
        labels=labels[permutation],
        source_trial_ids=tuple(
            source_trial_ids[int(index)] for index in permutation
        ),
        construction=construction,
    )
    return WindowBundle(
        window_set=combined_set,
        window_subject_ids=window_subject_ids[permutation],
    )


def build_population_split(
    *,
    subjects: Sequence[int],
    data_root: Path,
    data_pattern: str,
    data_reader: DataReaderName = "eeg",
    session_name: str,
    window_seconds: float,
    stride_seconds: float,
    base_seed: int,
    shuffle_trials_within_class: bool,
    max_windows_per_class_per_subject: int | None,
    reference_metadata: HDF5Metadata | None = None,
    window_construction: str,
    direct_trial_anchor: str,
) -> SplitBuildResult:
    bundles: list[WindowBundle] = []
    summaries: dict[str, Any] = {}
    paths: dict[int, Path] = {}
    common_metadata = reference_metadata

    for offset, subject_id in enumerate(subjects):
        path = resolve_subject_file(
            data_root=data_root,
            pattern=data_pattern,
            subject_id=subject_id,
        )
        paths[subject_id] = path

        bundle, metadata, summary = build_subject_window_bundle(
            subject_id=subject_id,
            path=path,
            data_reader=data_reader,
            session_name=session_name,
            reference_metadata=common_metadata,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            seed=base_seed + offset * 100,
            shuffle_trials_within_class=shuffle_trials_within_class,
            max_windows_per_class=max_windows_per_class_per_subject,
            window_construction=window_construction,
            direct_trial_anchor=direct_trial_anchor,
        )
        if common_metadata is None:
            common_metadata = metadata

        bundles.append(bundle)
        summaries[f"subject_{subject_id:02d}"] = summary

    assert common_metadata is not None
    combined = combine_window_bundles(
        bundles,
        seed=base_seed + 99_999,
        construction=window_construction,
    )

    return SplitBuildResult(
        bundle=combined,
        source_trial_summary=summaries,
        subject_paths=paths,
        metadata=common_metadata,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
