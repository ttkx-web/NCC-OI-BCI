from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def class_counts(labels: np.ndarray, *, num_classes: int) -> dict[int, int]:
    labels = np.asarray(labels, dtype=np.int64)
    return {
        class_index: int(np.sum(labels == class_index))
        for class_index in range(num_classes)
    }


def validate_label_coverage(
    labels: np.ndarray,
    *,
    num_classes: int,
    split_name: str,
) -> None:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError(
            f"{split_name} labels must be one-dimensional, got {labels.shape}."
        )
    if len(labels) == 0:
        raise ValueError(f"{split_name} contains no trials.")
    counts = class_counts(labels, num_classes=num_classes)
    missing = [index for index, count in counts.items() if count == 0]
    if labels.min() < 0 or labels.max() >= num_classes or missing:
        raise ValueError(
            f"{split_name} must contain every label in [0,{num_classes - 1}]. "
            f"Class counts: {counts}."
        )


def stratified_source_trial_split(
    labels: np.ndarray,
    *,
    val_fraction: float,
    seed: int,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically split source trials before window construction."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(
            f"val_fraction must be in (0,1), got {val_fraction}."
        )

    labels = np.asarray(labels, dtype=np.int64)
    validate_label_coverage(
        labels,
        num_classes=num_classes,
        split_name="source trials",
    )
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []

    for class_index in range(num_classes):
        indices = np.flatnonzero(labels == class_index)
        if len(indices) < 2:
            raise ValueError(
                f"Class {class_index} has only {len(indices)} source trial(s); "
                "at least 2 are required for train/validation splitting."
            )
        rng.shuffle(indices)
        validation_count = max(1, int(round(len(indices) * val_fraction)))
        validation_count = min(validation_count, len(indices) - 1)
        validation_indices.extend(int(value) for value in indices[:validation_count])
        train_indices.extend(int(value) for value in indices[validation_count:])

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    train_array = np.asarray(train_indices, dtype=np.int64)
    validation_array = np.asarray(validation_indices, dtype=np.int64)
    if set(train_array.tolist()) & set(validation_array.tolist()):
        raise RuntimeError("Source-trial leakage between train and validation.")
    return train_array, validation_array


@dataclass(frozen=True, slots=True)
class WithinSubjectTrialSplit:
    subject_id: int
    train_sessions: tuple[str, ...]
    test_session: str
    available_sessions: tuple[str, ...]
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray

    @property
    def train_session(self) -> str:
        """Legacy single-session view retained for existing callers."""
        if len(self.train_sessions) != 1:
            raise AttributeError(
                "train_session is ambiguous for a multi-session split; "
                "use train_sessions instead."
            )
        return self.train_sessions[0]


def resolve_within_subject_trial_split(
    *,
    subject_ids: np.ndarray,
    session_ids: np.ndarray,
    labels: np.ndarray,
    subject_id: int,
    train_sessions: Sequence[str] | None = None,
    train_session: str | None = None,
    test_session: str,
    validation_ratio: float,
    seed: int,
    num_classes: int,
) -> WithinSubjectTrialSplit:
    """Resolve a cross-session, source-trial-level split for one subject."""
    subject_ids = np.asarray(subject_ids, dtype=np.int64)
    session_ids = np.asarray(session_ids).astype(str)
    labels = np.asarray(labels, dtype=np.int64)
    expected = subject_ids.shape
    if session_ids.shape != expected or labels.shape != expected:
        raise ValueError(
            "subject_ids, session_ids, and labels must have identical "
            f"one-dimensional shapes, got {subject_ids.shape}, "
            f"{session_ids.shape}, and {labels.shape}."
        )
    if subject_ids.ndim != 1:
        raise ValueError(
            f"subject_ids must be one-dimensional, got {subject_ids.shape}."
        )
    if int(subject_id) <= 0:
        raise ValueError(f"subject_id must be positive, got {subject_id}.")
    if train_sessions is not None and train_session is not None:
        raise ValueError(
            "Specify either train_sessions or legacy train_session, not both."
        )
    if train_sessions is None:
        normalized_train_sessions = (
            () if train_session is None else (str(train_session),)
        )
    else:
        normalized_train_sessions = tuple(str(session) for session in train_sessions)

    subject_mask = subject_ids == int(subject_id)
    if not np.any(subject_mask):
        available_subjects = sorted(set(subject_ids.tolist()))
        raise ValueError(
            f"Subject {subject_id} is not present. Available subjects: "
            f"{available_subjects}."
        )
    available_sessions = tuple(
        dict.fromkeys(session_ids[subject_mask].tolist())
    )
    available_text = ", ".join(available_sessions)
    context = (
        f"Requested train sessions: {list(normalized_train_sessions)!r}; "
        f"requested test session: {test_session!r}. "
        f"Available sessions: {available_text}."
    )
    if not normalized_train_sessions:
        raise ValueError(
            "within-subject training requires at least one train session. "
            + context
        )
    if any(not session for session in normalized_train_sessions):
        raise ValueError(
            "within-subject train sessions must not contain empty values. "
            + context
        )
    if len(set(normalized_train_sessions)) != len(normalized_train_sessions):
        raise ValueError(
            "within-subject train sessions must be unique. " + context
        )
    if test_session in normalized_train_sessions:
        raise ValueError(
            "within-subject cross-session training requires the held-out test "
            "session not to appear in train sessions. " + context
        )
    requested_sessions = (*normalized_train_sessions, str(test_session))
    missing_sessions = [
        session for session in requested_sessions if session not in available_sessions
    ]
    if missing_sessions:
        raise ValueError(
            f"Subject {subject_id} does not contain session(s) "
            f"{missing_sessions}. {context}"
        )

    source_indices = np.flatnonzero(
        subject_mask & np.isin(session_ids, normalized_train_sessions)
    )
    test_indices = np.flatnonzero(subject_mask & (session_ids == test_session))
    validate_label_coverage(
        labels[source_indices],
        num_classes=num_classes,
        split_name=f"train sessions {list(normalized_train_sessions)!r}",
    )
    validate_label_coverage(
        labels[test_indices],
        num_classes=num_classes,
        split_name=f"test session {test_session!r}",
    )

    train_local, validation_local = stratified_source_trial_split(
        labels[source_indices],
        val_fraction=validation_ratio,
        seed=seed,
        num_classes=num_classes,
    )
    train_indices = source_indices[train_local]
    validation_indices = source_indices[validation_local]
    if (
        set(train_indices.tolist()) & set(validation_indices.tolist())
        or set(train_indices.tolist()) & set(test_indices.tolist())
        or set(validation_indices.tolist()) & set(test_indices.tolist())
    ):
        raise RuntimeError("Source-trial leakage in within-subject split.")

    return WithinSubjectTrialSplit(
        subject_id=int(subject_id),
        train_sessions=normalized_train_sessions,
        test_session=str(test_session),
        available_sessions=available_sessions,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )
