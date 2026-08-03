from __future__ import annotations

"""Data-splitting utilities for Stage-1 subject adaptation.

The functions in this module operate at the *source-trial* level.  Window or
feature construction must happen only after these splits are frozen; otherwise
windows derived from the same source trial may leak across train/validation.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


ArrayMapping = Mapping[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class LeaveOneSubjectOutSplit:
    """Subject-level split used to train a population model."""

    target_subject: int
    population_subjects: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.target_subject <= 0:
            raise ValueError("target_subject must be positive.")
        if not self.population_subjects:
            raise ValueError("At least one population subject is required.")
        if self.target_subject in self.population_subjects:
            raise ValueError(
                "The target subject must not appear in population_subjects."
            )
        if len(set(self.population_subjects)) != len(self.population_subjects):
            raise ValueError("population_subjects contains duplicates.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_subject": self.target_subject,
            "population_subjects": list(self.population_subjects),
        }


@dataclass(frozen=True, slots=True)
class PersonalTrialSplit:
    """Target-subject source-trial split.

    ``validation_indices`` are selected first with ``validation_seed`` and are
    fixed across all personalization budgets.  ``pool_indices`` are the
    remaining trials.  ``train_indices`` are a prefix of a seed-specific pool
    permutation, so 5/10/20/40-trial experiments are nested when the seeds are
    held constant.
    """

    train_indices: np.ndarray
    validation_indices: np.ndarray
    pool_indices: np.ndarray
    train_trial_ids_by_class: dict[str, list[int]]
    validation_trial_ids_by_class: dict[str, list[int]]
    pool_trial_ids_by_class: dict[str, list[int]]
    trials_per_class: int
    validation_trials_per_class: int
    validation_seed: int
    personalization_seed: int

    def __post_init__(self) -> None:
        for name, values in (
            ("train_indices", self.train_indices),
            ("validation_indices", self.validation_indices),
            ("pool_indices", self.pool_indices),
        ):
            if values.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional.")
            if values.dtype.kind not in {"i", "u"}:
                raise TypeError(f"{name} must contain integer indices.")
            if len(np.unique(values)) != len(values):
                raise ValueError(f"{name} contains duplicate indices.")

        train_set = set(int(value) for value in self.train_indices)
        validation_set = set(int(value) for value in self.validation_indices)
        pool_set = set(int(value) for value in self.pool_indices)

        if train_set & validation_set:
            raise RuntimeError(
                "Personal train and validation source indices overlap."
            )
        if validation_set & pool_set:
            raise RuntimeError(
                "Personal validation indices overlap the personalization pool."
            )
        if not train_set.issubset(pool_set):
            raise RuntimeError(
                "Selected personal train indices are not a subset of the pool."
            )
        if self.trials_per_class <= 0:
            raise ValueError("trials_per_class must be positive.")
        if self.validation_trials_per_class <= 0:
            raise ValueError("validation_trials_per_class must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_indices": self.train_indices.tolist(),
            "validation_indices": self.validation_indices.tolist(),
            "pool_indices": self.pool_indices.tolist(),
            "train_trial_ids_by_class": self.train_trial_ids_by_class,
            "validation_trial_ids_by_class": (
                self.validation_trial_ids_by_class
            ),
            "pool_trial_ids_by_class": self.pool_trial_ids_by_class,
            "trials_per_class": self.trials_per_class,
            "validation_trials_per_class": (
                self.validation_trials_per_class
            ),
            "validation_seed": self.validation_seed,
            "personalization_seed": self.personalization_seed,
        }


def normalize_subjects(subjects: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(sorted(set(int(subject) for subject in subjects)))
    if not normalized:
        raise ValueError("At least one subject is required.")
    invalid = [subject for subject in normalized if subject <= 0]
    if invalid:
        raise ValueError(f"Subject IDs must be positive, got {invalid}.")
    return normalized


def build_loso_subject_split(
    *,
    subjects: Sequence[int],
    target_subject: int,
) -> LeaveOneSubjectOutSplit:
    """Build a leave-one-subject-out split.

    Example for BNCI2014_001 with target subject 1::

        population_subjects = (2, 3, 4, 5, 6, 7, 8, 9)
        target_subject = 1
    """

    normalized = normalize_subjects(subjects)
    target_subject = int(target_subject)
    if target_subject not in normalized:
        raise ValueError(
            f"target_subject={target_subject} is not in subjects={normalized}."
        )
    return LeaveOneSubjectOutSplit(
        target_subject=target_subject,
        population_subjects=tuple(
            subject for subject in normalized if subject != target_subject
        ),
    )


def _validate_labels(
    labels: np.ndarray,
    *,
    num_classes: int,
    split_name: str,
) -> None:
    if labels.ndim != 1:
        raise ValueError(
            f"{split_name}: labels must be one-dimensional, got {labels.shape}."
        )
    if len(labels) == 0:
        raise ValueError(f"{split_name}: labels are empty.")
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one.")
    minimum = int(labels.min())
    maximum = int(labels.max())
    if minimum < 0 or maximum >= num_classes:
        raise ValueError(
            f"{split_name}: labels must be in [0, {num_classes - 1}], "
            f"got min={minimum}, max={maximum}."
        )
    missing = [
        class_index
        for class_index in range(num_classes)
        if not np.any(labels == class_index)
    ]
    if missing:
        raise ValueError(
            f"{split_name}: missing class indices {missing}."
        )


def _class_rng(seed: int, class_index: int) -> np.random.Generator:
    # A class-specific SeedSequence makes the selected trials independent of
    # iteration order and stable if classes are inspected separately.
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int(class_index)])
    )


def build_personal_trial_split(
    *,
    labels: np.ndarray,
    trial_ids: np.ndarray,
    class_names: Sequence[str],
    trials_per_class: int,
    validation_trials_per_class: int,
    validation_seed: int,
    personalization_seed: int,
) -> PersonalTrialSplit:
    """Split one target subject's calibration session at trial level.

    The validation set is selected first and remains fixed.  Training trials
    are then selected from the remaining personalization pool.  For a fixed
    ``personalization_seed``, increasing ``trials_per_class`` yields nested
    training subsets.
    """

    labels = np.asarray(labels, dtype=np.int64)
    trial_ids = np.asarray(trial_ids, dtype=np.int64)
    class_names = tuple(str(name) for name in class_names)

    if trial_ids.shape != labels.shape:
        raise ValueError(
            "labels and trial_ids must have identical shapes, got "
            f"{labels.shape} and {trial_ids.shape}."
        )
    if len(np.unique(trial_ids)) != len(trial_ids):
        raise ValueError("trial_ids contains duplicates within the session.")
    if trials_per_class <= 0:
        raise ValueError("trials_per_class must be positive.")
    if validation_trials_per_class <= 0:
        raise ValueError("validation_trials_per_class must be positive.")

    num_classes = len(class_names)
    _validate_labels(
        labels,
        num_classes=num_classes,
        split_name="target personalization source session",
    )

    train_indices: list[int] = []
    validation_indices: list[int] = []
    pool_indices: list[int] = []
    train_ids_by_class: dict[str, list[int]] = {}
    validation_ids_by_class: dict[str, list[int]] = {}
    pool_ids_by_class: dict[str, list[int]] = {}

    for class_index, class_name in enumerate(class_names):
        class_indices = np.flatnonzero(labels == class_index).astype(
            np.int64,
            copy=False,
        )
        required = validation_trials_per_class + trials_per_class
        if len(class_indices) < required:
            raise ValueError(
                f"Class {class_index} ({class_name}) has "
                f"{len(class_indices)} trials, but {required} are required "
                f"({validation_trials_per_class} validation + "
                f"{trials_per_class} personalization)."
            )

        validation_order = class_indices.copy()
        _class_rng(validation_seed, class_index).shuffle(validation_order)
        class_validation = validation_order[:validation_trials_per_class]
        validation_members = set(int(index) for index in class_validation)

        class_pool = np.asarray(
            [
                int(index)
                for index in class_indices
                if int(index) not in validation_members
            ],
            dtype=np.int64,
        )
        personalization_order = class_pool.copy()
        _class_rng(personalization_seed, class_index).shuffle(
            personalization_order
        )
        class_train = personalization_order[:trials_per_class]

        train_indices.extend(int(index) for index in class_train)
        validation_indices.extend(int(index) for index in class_validation)
        pool_indices.extend(int(index) for index in class_pool)

        train_ids_by_class[class_name] = [
            int(trial_ids[index]) for index in class_train
        ]
        validation_ids_by_class[class_name] = [
            int(trial_ids[index]) for index in class_validation
        ]
        # Preserve the seed-specific pool order so larger budgets can reuse
        # the exact prefix and remain nested.
        pool_ids_by_class[class_name] = [
            int(trial_ids[index]) for index in personalization_order
        ]

    train_array = np.asarray(train_indices, dtype=np.int64)
    validation_array = np.asarray(validation_indices, dtype=np.int64)
    pool_array = np.asarray(pool_indices, dtype=np.int64)

    # Shuffle row order without changing set membership.
    np.random.default_rng(personalization_seed + 100_003).shuffle(train_array)
    np.random.default_rng(validation_seed + 100_003).shuffle(
        validation_array
    )

    return PersonalTrialSplit(
        train_indices=train_array,
        validation_indices=validation_array,
        pool_indices=pool_array,
        train_trial_ids_by_class=train_ids_by_class,
        validation_trial_ids_by_class=validation_ids_by_class,
        pool_trial_ids_by_class=pool_ids_by_class,
        trials_per_class=int(trials_per_class),
        validation_trials_per_class=int(validation_trials_per_class),
        validation_seed=int(validation_seed),
        personalization_seed=int(personalization_seed),
    )


def select_rows(
    rows: ArrayMapping,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Select identical rows from every array in a loaded session mapping."""

    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("indices must be one-dimensional.")
    selected: dict[str, np.ndarray] = {}
    expected_length: int | None = None
    for key, value in rows.items():
        array = np.asarray(value)
        if array.ndim == 0:
            raise ValueError(f"rows[{key!r}] is scalar and cannot be indexed.")
        if expected_length is None:
            expected_length = len(array)
        elif len(array) != expected_length:
            raise ValueError(
                f"rows[{key!r}] length {len(array)} does not match "
                f"{expected_length}."
            )
        selected[key] = array[indices]
    return selected


def trial_id_set(values: Sequence[int] | np.ndarray) -> set[int]:
    return {int(value) for value in np.asarray(values).reshape(-1)}


def validate_disjoint_trial_ids(
    *,
    left_trial_ids: Sequence[int] | np.ndarray,
    right_trial_ids: Sequence[int] | np.ndarray,
    left_name: str,
    right_name: str,
) -> None:
    overlap = trial_id_set(left_trial_ids) & trial_id_set(right_trial_ids)
    if overlap:
        raise RuntimeError(
            f"Trial leakage between {left_name} and {right_name}. "
            f"Examples: {sorted(overlap)[:10]}."
        )


def validate_three_way_trial_split(
    *,
    train_trial_ids: Sequence[int] | np.ndarray,
    validation_trial_ids: Sequence[int] | np.ndarray,
    test_trial_ids: Sequence[int] | np.ndarray,
) -> None:
    validate_disjoint_trial_ids(
        left_trial_ids=train_trial_ids,
        right_trial_ids=validation_trial_ids,
        left_name="train",
        right_name="validation",
    )
    validate_disjoint_trial_ids(
        left_trial_ids=train_trial_ids,
        right_trial_ids=test_trial_ids,
        left_name="train",
        right_name="test",
    )
    validate_disjoint_trial_ids(
        left_trial_ids=validation_trial_ids,
        right_trial_ids=test_trial_ids,
        left_name="validation",
        right_name="test",
    )


def validate_nested_budgets(
    splits_by_budget: Mapping[int, PersonalTrialSplit],
) -> None:
    """Verify that train trial IDs are nested as the budget increases."""

    previous_budget: int | None = None
    previous_by_class: dict[str, set[int]] | None = None
    for budget in sorted(int(value) for value in splits_by_budget):
        split = splits_by_budget[budget]
        current_by_class = {
            class_name: set(int(value) for value in trial_ids)
            for class_name, trial_ids in split.train_trial_ids_by_class.items()
        }
        if previous_by_class is not None:
            for class_name, previous_ids in previous_by_class.items():
                current_ids = current_by_class.get(class_name, set())
                if not previous_ids.issubset(current_ids):
                    raise RuntimeError(
                        f"Personalization budgets are not nested for class "
                        f"{class_name!r}: budget {previous_budget} is not a "
                        f"subset of budget {budget}."
                    )
        previous_budget = budget
        previous_by_class = current_by_class


def encode_subject_trial_id(subject_id: int, trial_id: int) -> int:
    """Encode a positive subject ID and file-local trial ID into int64."""

    subject_id = int(subject_id)
    trial_id = int(trial_id)
    if subject_id <= 0:
        raise ValueError("subject_id must be positive.")
    if trial_id < 0 or trial_id >= 2**32:
        raise ValueError("trial_id must be in [0, 2**32).")
    return (subject_id << 32) | trial_id


def encode_subject_trial_ids(
    subject_id: int,
    trial_ids: Sequence[int] | np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [encode_subject_trial_id(subject_id, value) for value in trial_ids],
        dtype=np.int64,
    )
