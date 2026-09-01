from __future__ import annotations

import numpy as np
import pytest

from bci_dayloop.data.splits import resolve_within_subject_trial_split


def make_trials() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_labels = np.repeat(np.arange(4, dtype=np.int64), 20)
    test_labels = np.concatenate(
        [
            np.zeros(7, dtype=np.int64),
            np.ones(7, dtype=np.int64),
            np.full(10, 2, dtype=np.int64),
            np.full(10, 3, dtype=np.int64),
        ]
    )
    labels = np.concatenate((train_labels, test_labels))
    subjects = np.ones(len(labels), dtype=np.int64)
    sessions = np.asarray(
        ["train_source"] * len(train_labels)
        + ["held_out_test"] * len(test_labels),
        dtype=str,
    )
    return subjects, sessions, labels


def resolve(seed: int = 42):
    subjects, sessions, labels = make_trials()
    return resolve_within_subject_trial_split(
        subject_ids=subjects,
        session_ids=sessions,
        labels=labels,
        subject_id=1,
        train_sessions=["train_source"],
        test_session="held_out_test",
        validation_ratio=0.2,
        seed=seed,
        num_classes=4,
    )


def test_within_subject_filters_sessions_and_stratifies_trials() -> None:
    subjects, sessions, labels = make_trials()
    split = resolve()

    assert split.available_sessions == ("train_source", "held_out_test")
    assert split.train_sessions == ("train_source",)
    assert np.all(subjects[split.train_indices] == 1)
    assert np.all(sessions[split.train_indices] == "train_source")
    assert np.all(sessions[split.validation_indices] == "train_source")
    assert np.all(sessions[split.test_indices] == "held_out_test")
    assert len(split.train_indices) == 64
    assert len(split.validation_indices) == 16
    assert len(split.test_indices) == 34
    np.testing.assert_array_equal(
        np.bincount(labels[split.train_indices], minlength=4),
        np.asarray([16, 16, 16, 16]),
    )
    np.testing.assert_array_equal(
        np.bincount(labels[split.validation_indices], minlength=4),
        np.asarray([4, 4, 4, 4]),
    )
    np.testing.assert_array_equal(
        np.bincount(labels[split.test_indices], minlength=4),
        np.asarray([7, 7, 10, 10]),
    )


def test_within_subject_source_trial_sets_do_not_overlap() -> None:
    split = resolve()
    train_ids = set(split.train_indices.tolist())
    validation_ids = set(split.validation_indices.tolist())
    test_ids = set(split.test_indices.tolist())

    assert not train_ids & validation_ids
    assert not train_ids & test_ids
    assert not validation_ids & test_ids


def test_within_subject_split_is_deterministic_for_a_seed() -> None:
    left = resolve(seed=7)
    right = resolve(seed=7)

    np.testing.assert_array_equal(left.train_indices, right.train_indices)
    np.testing.assert_array_equal(left.validation_indices, right.validation_indices)


def test_within_subject_different_seed_preserves_class_counts() -> None:
    subjects, sessions, labels = make_trials()
    left = resolve(seed=7)
    right = resolve(seed=8)

    assert not np.array_equal(left.validation_indices, right.validation_indices)
    np.testing.assert_array_equal(
        np.bincount(labels[left.train_indices], minlength=4),
        np.bincount(labels[right.train_indices], minlength=4),
    )
    np.testing.assert_array_equal(
        np.bincount(labels[left.validation_indices], minlength=4),
        np.bincount(labels[right.validation_indices], minlength=4),
    )
    assert np.all(subjects[right.test_indices] == 1)
    assert np.all(sessions[right.test_indices] == "held_out_test")


def test_missing_class_is_rejected_with_class_counts() -> None:
    subjects, sessions, labels = make_trials()
    sessions = sessions.copy()
    labels = labels.copy()
    sessions[labels == 3] = "other"

    with pytest.raises(ValueError, match="Class counts"):
        resolve_within_subject_trial_split(
            subject_ids=subjects,
            session_ids=sessions,
            labels=labels,
            subject_id=1,
            train_sessions=["train_source"],
            test_session="held_out_test",
            validation_ratio=0.2,
            seed=42,
            num_classes=4,
        )


def test_invalid_session_lists_available_sessions() -> None:
    subjects, sessions, labels = make_trials()

    with pytest.raises(ValueError, match="Available sessions: train_source, held_out_test"):
        resolve_within_subject_trial_split(
            subject_ids=subjects,
            session_ids=sessions,
            labels=labels,
            subject_id=1,
            train_sessions=["missing"],
            test_session="held_out_test",
            validation_ratio=0.2,
            seed=42,
            num_classes=4,
        )


def test_same_train_and_test_session_is_rejected() -> None:
    subjects, sessions, labels = make_trials()

    with pytest.raises(ValueError, match="not to appear in train sessions"):
        resolve_within_subject_trial_split(
            subject_ids=subjects,
            session_ids=sessions,
            labels=labels,
            subject_id=1,
            train_sessions=["train_source"],
            test_session="train_source",
            validation_ratio=0.2,
            seed=42,
            num_classes=4,
        )


def test_multiple_train_sessions_are_concatenated_before_one_split() -> None:
    labels_per_session = np.repeat(np.arange(4, dtype=np.int64), 10)
    labels = np.concatenate(
        [labels_per_session, labels_per_session, labels_per_session, labels_per_session]
    )
    subjects = np.ones(len(labels), dtype=np.int64)
    sessions = np.asarray(
        ["S1"] * 40 + ["S2"] * 40 + ["S3"] * 40 + ["S6"] * 40,
        dtype=str,
    )

    split = resolve_within_subject_trial_split(
        subject_ids=subjects,
        session_ids=sessions,
        labels=labels,
        subject_id=1,
        train_sessions=["S1", "S2", "S3"],
        test_session="S6",
        validation_ratio=0.2,
        seed=42,
        num_classes=4,
    )

    assert split.train_sessions == ("S1", "S2", "S3")
    assert len(split.train_indices) == 96
    assert len(split.validation_indices) == 24
    assert len(split.test_indices) == 40
    np.testing.assert_array_equal(
        np.bincount(labels[split.train_indices], minlength=4),
        np.asarray([24, 24, 24, 24]),
    )
    np.testing.assert_array_equal(
        np.bincount(labels[split.validation_indices], minlength=4),
        np.asarray([6, 6, 6, 6]),
    )
    assert set(sessions[split.train_indices]) <= {"S1", "S2", "S3"}
    assert set(sessions[split.validation_indices]) <= {"S1", "S2", "S3"}


def test_duplicate_or_held_out_train_sessions_are_rejected() -> None:
    subjects, sessions, labels = make_trials()

    with pytest.raises(ValueError, match="must be unique"):
        resolve_within_subject_trial_split(
            subject_ids=subjects,
            session_ids=sessions,
            labels=labels,
            subject_id=1,
            train_sessions=["train_source", "train_source"],
            test_session="held_out_test",
            validation_ratio=0.2,
            seed=42,
            num_classes=4,
        )

    with pytest.raises(ValueError, match="not to appear in train sessions"):
        resolve_within_subject_trial_split(
            subject_ids=subjects,
            session_ids=sessions,
            labels=labels,
            subject_id=1,
            train_sessions=["train_source", "held_out_test"],
            test_session="held_out_test",
            validation_ratio=0.2,
            seed=42,
            num_classes=4,
        )
