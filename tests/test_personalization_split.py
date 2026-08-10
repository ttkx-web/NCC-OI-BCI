from __future__ import annotations

import numpy as np
import pytest

from bci_dayloop.personalization import (
    build_loso_subject_split,
    build_personal_trial_split,
    encode_subject_trial_id,
    encode_subject_trial_ids,
    select_rows,
    validate_nested_budgets,
    validate_three_way_trial_split,
)


CLASS_NAMES = ("left_hand", "right_hand", "feet", "tongue")


def _balanced_trial_rows(
    trials_per_class: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.repeat(
        np.arange(len(CLASS_NAMES), dtype=np.int64),
        trials_per_class,
    )
    trial_ids = np.arange(1_000, 1_000 + len(labels), dtype=np.int64)
    return labels, trial_ids


def _build_split(
    *,
    trials_per_class: int,
    validation_trials_per_class: int = 2,
    validation_seed: int = 2026,
    personalization_seed: int = 42,
):
    labels, trial_ids = _balanced_trial_rows()
    return build_personal_trial_split(
        labels=labels,
        trial_ids=trial_ids,
        class_names=CLASS_NAMES,
        trials_per_class=trials_per_class,
        validation_trials_per_class=validation_trials_per_class,
        validation_seed=validation_seed,
        personalization_seed=personalization_seed,
    )


def test_loso_split_excludes_target_and_normalizes_subjects() -> None:
    split = build_loso_subject_split(
        subjects=[9, 1, 3, 2, 3, 4, 5, 6, 7, 8],
        target_subject=1,
    )

    assert split.target_subject == 1
    assert split.population_subjects == (2, 3, 4, 5, 6, 7, 8, 9)
    assert split.to_dict() == {
        "target_subject": 1,
        "population_subjects": [2, 3, 4, 5, 6, 7, 8, 9],
    }


def test_loso_split_rejects_missing_target_subject() -> None:
    with pytest.raises(ValueError, match="is not in subjects"):
        build_loso_subject_split(
            subjects=[2, 3, 4],
            target_subject=1,
        )


def test_personal_split_is_balanced_and_has_no_source_trial_leakage() -> None:
    labels, trial_ids = _balanced_trial_rows()
    split = build_personal_trial_split(
        labels=labels,
        trial_ids=trial_ids,
        class_names=CLASS_NAMES,
        trials_per_class=5,
        validation_trials_per_class=2,
        validation_seed=2026,
        personalization_seed=42,
    )

    assert len(split.train_indices) == 5 * len(CLASS_NAMES)
    assert len(split.validation_indices) == 2 * len(CLASS_NAMES)
    assert len(split.pool_indices) == 10 * len(CLASS_NAMES)

    train_ids = set(trial_ids[split.train_indices].tolist())
    validation_ids = set(trial_ids[split.validation_indices].tolist())
    pool_ids = set(trial_ids[split.pool_indices].tolist())

    assert train_ids.isdisjoint(validation_ids)
    assert validation_ids.isdisjoint(pool_ids)
    assert train_ids <= pool_ids

    for class_index, class_name in enumerate(CLASS_NAMES):
        assert len(split.train_trial_ids_by_class[class_name]) == 5
        assert len(split.validation_trial_ids_by_class[class_name]) == 2
        assert len(split.pool_trial_ids_by_class[class_name]) == 10

        assert all(
            labels[np.flatnonzero(trial_ids == trial_id)[0]] == class_index
            for trial_id in split.train_trial_ids_by_class[class_name]
        )


def test_personal_split_is_reproducible_for_fixed_seeds() -> None:
    first = _build_split(trials_per_class=5)
    second = _build_split(trials_per_class=5)

    np.testing.assert_array_equal(first.train_indices, second.train_indices)
    np.testing.assert_array_equal(
        first.validation_indices,
        second.validation_indices,
    )
    np.testing.assert_array_equal(first.pool_indices, second.pool_indices)
    assert first.to_dict() == second.to_dict()


def test_validation_set_is_fixed_across_budgets_and_personalization_seeds() -> None:
    small = _build_split(
        trials_per_class=3,
        personalization_seed=42,
    )
    large = _build_split(
        trials_per_class=8,
        personalization_seed=99,
    )

    np.testing.assert_array_equal(
        np.sort(small.validation_indices),
        np.sort(large.validation_indices),
    )
    assert (
        small.validation_trial_ids_by_class
        == large.validation_trial_ids_by_class
    )


def test_training_trials_are_nested_for_increasing_budgets() -> None:
    splits = {
        budget: _build_split(trials_per_class=budget)
        for budget in (2, 5, 8)
    }

    validate_nested_budgets(splits)

    for class_name in CLASS_NAMES:
        ids_2 = set(splits[2].train_trial_ids_by_class[class_name])
        ids_5 = set(splits[5].train_trial_ids_by_class[class_name])
        ids_8 = set(splits[8].train_trial_ids_by_class[class_name])
        assert ids_2 < ids_5 < ids_8


def test_personal_split_rejects_insufficient_trials_per_class() -> None:
    labels, trial_ids = _balanced_trial_rows(trials_per_class=4)

    with pytest.raises(ValueError, match="validation.*personalization"):
        build_personal_trial_split(
            labels=labels,
            trial_ids=trial_ids,
            class_names=CLASS_NAMES,
            trials_per_class=3,
            validation_trials_per_class=2,
            validation_seed=2026,
            personalization_seed=42,
        )


def test_personal_split_rejects_duplicate_trial_ids() -> None:
    labels, trial_ids = _balanced_trial_rows()
    trial_ids[1] = trial_ids[0]

    with pytest.raises(ValueError, match="trial_ids contains duplicates"):
        build_personal_trial_split(
            labels=labels,
            trial_ids=trial_ids,
            class_names=CLASS_NAMES,
            trials_per_class=5,
            validation_trials_per_class=2,
            validation_seed=2026,
            personalization_seed=42,
        )


def test_three_way_split_detects_trial_leakage() -> None:
    validate_three_way_trial_split(
        train_trial_ids=[1, 2],
        validation_trial_ids=[3, 4],
        test_trial_ids=[5, 6],
    )

    with pytest.raises(RuntimeError, match="Trial leakage"):
        validate_three_way_trial_split(
            train_trial_ids=[1, 2],
            validation_trial_ids=[3, 4],
            test_trial_ids=[4, 5],
        )


def test_select_rows_applies_identical_indices_to_every_array() -> None:
    rows = {
        "data": np.arange(5 * 2 * 3).reshape(5, 2, 3),
        "labels": np.array([0, 1, 2, 3, 0]),
        "trial_ids": np.array([10, 11, 12, 13, 14]),
    }
    indices = np.array([4, 1, 3], dtype=np.int64)

    selected = select_rows(rows, indices)

    np.testing.assert_array_equal(selected["data"], rows["data"][indices])
    np.testing.assert_array_equal(selected["labels"], rows["labels"][indices])
    np.testing.assert_array_equal(
        selected["trial_ids"],
        rows["trial_ids"][indices],
    )


def test_subject_trial_encoding_is_unique_across_subjects() -> None:
    assert encode_subject_trial_id(1, 7) != encode_subject_trial_id(2, 7)

    encoded = encode_subject_trial_ids(3, [0, 1, 2])

    assert encoded.dtype == np.int64
    np.testing.assert_array_equal(
        encoded,
        np.array(
            [
                encode_subject_trial_id(3, 0),
                encode_subject_trial_id(3, 1),
                encode_subject_trial_id(3, 2),
            ],
            dtype=np.int64,
        ),
    )
