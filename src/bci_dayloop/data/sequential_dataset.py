"""Strict, causal-order dataset adapters for sequential evaluation.

The adapters expose one trial at a time in the order persisted by the source
HDF5.  They deliberately do not concatenate trials, pad, crop, shuffle, or
otherwise manufacture a window.  A caller may explicitly request a prefix via
``max_trials`` for a bounded evaluation run.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import h5py
import numpy as np

from bci_dayloop.data.hdf5_dataset import EEGHDF5
from bci_dayloop.data.workload import WorkloadHDF5


@dataclass(frozen=True, slots=True)
class SequentialDatasetMetadata:
    """Dataset-level contract required before a Runtime Package is used."""

    sample_rate: float
    channel_names: tuple[str, ...]
    class_names: tuple[str, ...]
    unit: str
    dataset_name: str
    window_sec: float


@dataclass(frozen=True, slots=True)
class SequentialDataset:
    """One ordered source-trial stream, independent of HDF5 layout."""

    metadata: SequentialDatasetMetadata
    data: np.ndarray
    labels: np.ndarray
    subject_ids: np.ndarray
    session_ids: np.ndarray
    trial_ids: np.ndarray
    trial_ordinals: np.ndarray
    window_ids: np.ndarray

    @property
    def num_trials(self) -> int:
        return int(self.data.shape[0])


@dataclass(frozen=True, slots=True)
class PopulationSplitPlan:
    train_subjects: tuple[int, ...]
    validation_subjects: tuple[int, ...]
    final_test_subjects: tuple[int, ...]
    train_session: str
    validation_session: str
    final_test_session: str


def resolve_population_split_plan(
    subjects: object, target_subject: int, train_session: str,
    validation_session: str, final_test_session: str,
) -> PopulationSplitPlan:
    normalized = tuple(sorted(set(int(value) for value in subjects)))
    if not normalized or target_subject not in normalized:
        raise ValueError("target_subject must occur in a non-empty subjects list.")
    if any(not str(value).strip() for value in (train_session, validation_session, final_test_session)):
        raise ValueError("Population split sessions must be non-empty.")
    population = tuple(value for value in normalized if value != target_subject)
    if not population:
        raise ValueError("Population split requires at least one non-target subject.")
    return PopulationSplitPlan(population, population, (target_subject,), str(train_session), str(validation_session), str(final_test_session))


def source_trial_identities(dataset: SequentialDataset) -> tuple[tuple[str, str, str, str], ...]:
    return tuple((dataset.metadata.dataset_name, str(subject), str(session), str(window)) for subject, session, window in zip(dataset.subject_ids, dataset.session_ids, dataset.window_ids, strict=True))


def _as_trial_vector(values: object, *, name: str, expected: int) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    if array.shape != (expected,):
        raise ValueError(
            f"{name} length does not match trial count: "
            f"{array.shape} != ({expected},)."
        )
    return array


def _metadata_from_eeg(
    *,
    sample_rate: float,
    channel_names: object,
    class_names: object,
    unit: object,
    dataset_name: object,
    num_samples: int,
) -> SequentialDatasetMetadata:
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("HDF5 sample_rate must be finite and positive.")
    if num_samples <= 0:
        raise ValueError("Sequential dataset must contain at least one sample per trial.")

    window_sec = num_samples / sample_rate
    if not math.isfinite(window_sec) or window_sec <= 0:
        raise ValueError("Sequential dataset window_sec must be finite and positive.")

    return SequentialDatasetMetadata(
        sample_rate=float(sample_rate),
        channel_names=tuple(str(value) for value in channel_names),
        class_names=tuple(str(value) for value in class_names),
        unit=str(unit),
        dataset_name=str(dataset_name),
        # Legacy EEGHDF5 has no window attribute. This is still an explicit
        # adapter contract derived from its persisted sample-rate and shape.
        window_sec=float(window_sec),
    )


def _window_suffix(window_sec: float) -> str:
    rounded = int(round(float(window_sec)))
    if np.isclose(float(window_sec), rounded, atol=1e-9, rtol=0.0):
        return f"{rounded}s"
    return f"{float(window_sec):g}s"


def _validate_dataset(dataset: SequentialDataset) -> None:
    data = np.asarray(dataset.data)
    if data.ndim != 3:
        raise ValueError(f"Sequential data must have shape [N,C,T], got {data.shape}.")

    num_trials, num_channels, num_samples = data.shape
    if num_trials <= 0:
        raise ValueError("Sequential dataset must contain at least one trial.")
    if num_channels != len(dataset.metadata.channel_names):
        raise ValueError(
            "Sequential channel dimension does not match metadata channel_names: "
            f"{num_channels} != {len(dataset.metadata.channel_names)}."
        )
    if not dataset.metadata.class_names:
        raise ValueError("Sequential dataset class_names must not be empty.")
    if (
        not math.isfinite(dataset.metadata.sample_rate)
        or dataset.metadata.sample_rate <= 0
    ):
        raise ValueError("Sequential dataset sample_rate must be finite and positive.")
    if (
        not math.isfinite(dataset.metadata.window_sec)
        or dataset.metadata.window_sec <= 0
    ):
        raise ValueError("Sequential dataset window_sec must be finite and positive.")

    expected_samples = int(round(dataset.metadata.window_sec * dataset.metadata.sample_rate))
    if num_samples != expected_samples:
        duration = num_samples / dataset.metadata.sample_rate
        raise ValueError(
            "Source trial duration does not match dataset window contract: "
            f"samples={num_samples}, sample_rate={dataset.metadata.sample_rate}, "
            f"duration={duration:.9f}, window_sec={dataset.metadata.window_sec:.9f}, "
            f"expected_samples={expected_samples}."
        )

    for name, values in (
        ("labels", dataset.labels),
        ("subject_ids", dataset.subject_ids),
        ("session_ids", dataset.session_ids),
        ("trial_ids", dataset.trial_ids),
        ("trial_ordinals", dataset.trial_ordinals),
        ("window_ids", dataset.window_ids),
    ):
        _as_trial_vector(values, name=name, expected=num_trials)

    labels = np.asarray(dataset.labels, dtype=np.int64).reshape(-1)
    invalid_labels = labels[(labels < 0) | (labels >= len(dataset.metadata.class_names))]
    if invalid_labels.size:
        raise ValueError(
            "Sequential labels are outside the class range: "
            f"{np.unique(invalid_labels).tolist()}."
        )

    expected_ordinals = np.arange(1, num_trials + 1, dtype=np.int64)
    actual_ordinals = np.asarray(dataset.trial_ordinals, dtype=np.int64).reshape(-1)
    if not np.array_equal(actual_ordinals, expected_ordinals):
        raise ValueError(
            "trial_ordinals must preserve the persisted causal sequence "
            "1..N without sorting or gaps."
        )

    window_ids = np.asarray(dataset.window_ids, dtype=str).reshape(-1)
    if np.any(window_ids == "") or len(set(window_ids.tolist())) != num_trials:
        raise ValueError("window_ids must be non-empty and unique per source trial.")
    if not np.all(np.isfinite(data)):
        raise ValueError("Sequential signal contains NaN or Inf.")


def _limit_prefix(dataset: SequentialDataset, max_trials: int | None) -> SequentialDataset:
    if max_trials is None:
        return dataset
    if max_trials <= 0:
        raise ValueError("max_trials must be positive when provided.")
    limit = min(int(max_trials), dataset.num_trials)
    # This is a caller-requested bounded prefix, not a silent crop.
    limited = SequentialDataset(
        metadata=dataset.metadata,
        data=dataset.data[:limit],
        labels=dataset.labels[:limit],
        subject_ids=dataset.subject_ids[:limit],
        session_ids=dataset.session_ids[:limit],
        trial_ids=dataset.trial_ids[:limit],
        trial_ordinals=dataset.trial_ordinals[:limit],
        window_ids=dataset.window_ids[:limit],
    )
    _validate_dataset(limited)
    return limited


def _load_eeg_hdf5(path: Path, *, session: str) -> SequentialDataset:
    reader = EEGHDF5(path)
    metadata = reader.metadata
    payload = reader.load(session)
    data = np.asarray(payload["data"], dtype=np.float32)
    num_trials = int(data.shape[0]) if data.ndim >= 1 else 0
    common_metadata = _metadata_from_eeg(
        sample_rate=float(metadata.sample_rate),
        channel_names=metadata.channel_names,
        class_names=metadata.class_names,
        unit=metadata.unit,
        dataset_name=metadata.dataset_name,
        num_samples=int(data.shape[2]) if data.ndim == 3 else 0,
    )
    trial_ids = _as_trial_vector(payload["trial_ids"], name="trial_ids", expected=num_trials)
    session_ids = _as_trial_vector(payload["session_ids"], name="session_ids", expected=num_trials)
    dataset = SequentialDataset(
        metadata=common_metadata,
        data=data,
        labels=_as_trial_vector(payload["labels"], name="labels", expected=num_trials),
        subject_ids=_as_trial_vector(
            payload["subject_ids"], name="subject_ids", expected=num_trials
        ),
        session_ids=session_ids,
        trial_ids=trial_ids,
        trial_ordinals=np.arange(1, num_trials + 1, dtype=np.int64),
        window_ids=np.asarray(
            [
                f"{trial_id}:trial{_window_suffix(common_metadata.window_sec)}"
                for trial_id in trial_ids
            ],
            dtype=str,
        ),
    )
    _validate_dataset(dataset)
    return dataset


def _load_workload_hdf5(path: Path, *, session: str) -> SequentialDataset:
    payload = WorkloadHDF5(path).load(session=session)
    with h5py.File(path, "r") as handle:
        root = handle.attrs
        group = handle["sessions"][session]
        common_metadata = SequentialDatasetMetadata(
            sample_rate=float(group.attrs["sample_rate"]),
            channel_names=tuple(str(value) for value in json.loads(group.attrs["channel_names"])),
            class_names=tuple(str(value) for value in json.loads(root["class_names"])),
            unit=str(root["unit"]),
            dataset_name=str(root["dataset_name"]),
            window_sec=float(root["window_sec"]),
        )
        subject_id = str(root["subject_id"])

    data = np.asarray(payload["data"], dtype=np.float32)
    num_trials = int(data.shape[0]) if data.ndim >= 1 else 0
    window_ids = _as_trial_vector(payload["window_ids"], name="window_ids", expected=num_trials)
    dataset = SequentialDataset(
        metadata=common_metadata,
        data=data,
        labels=_as_trial_vector(payload["labels"], name="labels", expected=num_trials),
        subject_ids=np.full(num_trials, subject_id, dtype=str),
        session_ids=np.full(num_trials, str(session), dtype=str),
        # A Workload window ID is its persisted source identifier.
        trial_ids=window_ids.copy(),
        trial_ordinals=_as_trial_vector(
            payload["trial_ordinals"], name="trial_ordinals", expected=num_trials
        ),
        window_ids=window_ids,
    )
    _validate_dataset(dataset)
    return dataset


def load_sequential_dataset(
    path: str | Path,
    *,
    session: str,
    max_trials: int | None = None,
) -> SequentialDataset:
    """Load either supported HDF5 layout without changing trial order."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Sequential HDF5 not found: {target}")

    with h5py.File(target, "r") as handle:
        has_flat_eeg = "data" in handle
        has_workload_sessions = "sessions" in handle
    if has_flat_eeg == has_workload_sessions:
        raise ValueError(
            "Unsupported sequential HDF5 layout: expected exactly one of "
            "root /data or grouped /sessions."
        )

    dataset = (
        _load_eeg_hdf5(target, session=session)
        if has_flat_eeg
        else _load_workload_hdf5(target, session=session)
    )
    return _limit_prefix(dataset, max_trials)


def validate_package_window_contract(
    dataset: SequentialDataset,
    *,
    package_window_sec: float,
) -> None:
    """Fail closed when a source trial and package use different durations."""
    if not math.isfinite(package_window_sec) or package_window_sec <= 0:
        raise ValueError("Runtime Package window_sec must be finite and positive.")
    if not np.isclose(
        float(dataset.metadata.window_sec),
        float(package_window_sec),
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError(
            "Dataset and Runtime Package window contracts differ: "
            f"dataset={dataset.metadata.window_sec}, package={package_window_sec}."
        )
