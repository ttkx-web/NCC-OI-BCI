"""Registry for persisted EEG dataset layouts used by sequential evaluation.

Adapters are deliberately limited to reading persisted trial streams.  They do
not reorder, crop, concatenate, preprocess, or otherwise alter EEG signals.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import h5py
import numpy as np

from bci_dayloop.data.hdf5_dataset import EEGHDF5
from bci_dayloop.data.sequential_dataset import (
    SequentialDataset,
    SequentialDatasetMetadata,
    _as_trial_vector,
    _decode_json_attribute,
    _metadata_from_eeg,
    _seed_subject_id,
)
from bci_dayloop.data.workload import WorkloadHDF5


@dataclass(frozen=True, slots=True)
class HDF5DatasetDescriptor:
    """The root-level HDF5 facts used for fail-closed adapter selection."""

    dataset_name: str
    has_root_data: bool
    has_sessions_group: bool


class DatasetAdapter(Protocol):
    """Read exactly one persisted HDF5 layout into ``SequentialDataset``."""

    name: str

    def supports(self, descriptor: HDF5DatasetDescriptor) -> bool:
        """Return whether this adapter owns the supplied HDF5 layout."""

    def load(self, path: Path, *, session: str) -> SequentialDataset:
        """Load one persisted session without changing its trial order."""


def inspect_hdf5_dataset(path: str | Path) -> HDF5DatasetDescriptor:
    """Inspect a HDF5 root without selecting or loading an adapter payload."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Sequential HDF5 not found: {target}")
    with h5py.File(target, "r") as handle:
        return HDF5DatasetDescriptor(
            dataset_name=str(handle.attrs.get("dataset_name", "")).strip().lower(),
            has_root_data="data" in handle,
            has_sessions_group="sessions" in handle,
        )


class DatasetAdapterRegistry:
    """Resolve one and only one adapter for a persisted HDF5 descriptor."""

    def __init__(self, adapters: Iterable[DatasetAdapter]) -> None:
        self._adapters = tuple(adapters)
        if not self._adapters:
            raise ValueError("Dataset adapter registry must not be empty.")
        names = [adapter.name for adapter in self._adapters]
        if len(names) != len(set(names)):
            raise ValueError(f"Dataset adapter names must be unique: {names}.")

    @property
    def adapters(self) -> tuple[DatasetAdapter, ...]:
        return self._adapters

    def resolve(self, descriptor: HDF5DatasetDescriptor) -> DatasetAdapter:
        matches = [
            adapter
            for adapter in self._adapters
            if adapter.supports(descriptor)
        ]
        if len(matches) != 1:
            details = (
                f"dataset_name={descriptor.dataset_name!r}, "
                f"has_root_data={descriptor.has_root_data}, "
                f"has_sessions_group={descriptor.has_sessions_group}"
            )
            if not matches:
                raise ValueError(
                    "Unsupported sequential HDF5 layout; no registered "
                    f"dataset adapter matches ({details})."
                )
            raise ValueError(
                "Ambiguous sequential HDF5 layout; matching adapters="
                f"{[adapter.name for adapter in matches]} ({details})."
            )
        return matches[0]

    def load(self, path: str | Path, *, session: str) -> SequentialDataset:
        target = Path(path)
        return self.resolve(inspect_hdf5_dataset(target)).load(
            target,
            session=session,
        )


class LegacyHDF5Adapter:
    """Adapter for the established flat root ``/data`` EEG HDF5 layout."""

    name = "legacy_hdf5"

    def supports(self, descriptor: HDF5DatasetDescriptor) -> bool:
        return descriptor.has_root_data and not descriptor.has_sessions_group

    def load(self, path: Path, *, session: str) -> SequentialDataset:
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
        trial_ids = _as_trial_vector(
            payload["trial_ids"], name="trial_ids", expected=num_trials
        )
        return SequentialDataset(
            metadata=common_metadata,
            data=data,
            labels=_as_trial_vector(
                payload["labels"], name="labels", expected=num_trials
            ),
            subject_ids=_as_trial_vector(
                payload["subject_ids"], name="subject_ids", expected=num_trials
            ),
            session_ids=_as_trial_vector(
                payload["session_ids"], name="session_ids", expected=num_trials
            ),
            trial_ids=trial_ids,
            trial_ordinals=np.arange(1, num_trials + 1, dtype=np.int64),
            window_ids=np.asarray(
                [
                    f"{trial_id}:trial{_window_suffix(common_metadata.window_sec)}"
                    for trial_id in trial_ids
                ],
                dtype=object,
            ),
        )


class WorkloadHDF5Adapter:
    """Adapter for the grouped Workload HDF5 contract."""

    name = "workload_hdf5"

    def supports(self, descriptor: HDF5DatasetDescriptor) -> bool:
        return (
            not descriptor.has_root_data
            and descriptor.has_sessions_group
            and descriptor.dataset_name == "workload_pbci_hackathon"
        )

    def load(self, path: Path, *, session: str) -> SequentialDataset:
        payload = WorkloadHDF5(path).load(session=session)
        with h5py.File(path, "r") as handle:
            root = handle.attrs
            group = handle["sessions"][session]
            metadata = SequentialDatasetMetadata(
                sample_rate=float(group.attrs["sample_rate"]),
                channel_names=tuple(
                    str(value) for value in _decode_json_attribute(
                        group.attrs, "channel_names"
                    )
                ),
                class_names=tuple(
                    str(value)
                    for value in _decode_json_attribute(root, "class_names")
                ),
                unit=str(root["unit"]),
                dataset_name=str(root["dataset_name"]),
                window_sec=float(root["window_sec"]),
            )

        data = np.asarray(payload["data"], dtype=np.float32)
        num_trials = int(data.shape[0]) if data.ndim >= 1 else 0
        window_ids = _as_trial_vector(
            payload["window_ids"], name="window_ids", expected=num_trials
        )
        return SequentialDataset(
            metadata=metadata,
            data=data,
            labels=_as_trial_vector(
                payload["labels"], name="labels", expected=num_trials
            ),
            # Keep the canonical numeric subject identity for trainers. The
            # sequential public trial identity remains the persisted window ID
            # (the evaluator reports it verbatim); WorkloadHDF5.load() still
            # exposes its compact numeric trial_ids to direct consumers.
            subject_ids=_as_trial_vector(
                payload["subject_ids"], name="subject_ids", expected=num_trials
            ),
            session_ids=_as_trial_vector(
                payload["session_ids"], name="session_ids", expected=num_trials
            ),
            trial_ids=window_ids.copy(),
            trial_ordinals=_as_trial_vector(
                payload["trial_ordinals"],
                name="trial_ordinals",
                expected=num_trials,
            ),
            window_ids=window_ids,
        )


class SeedHDF5Adapter:
    """Adapter for the grouped SEED HDF5 contract."""

    name = "seed_hdf5"

    def supports(self, descriptor: HDF5DatasetDescriptor) -> bool:
        return (
            not descriptor.has_root_data
            and descriptor.has_sessions_group
            and descriptor.dataset_name == "seed"
        )

    def load(self, path: Path, *, session: str) -> SequentialDataset:
        with h5py.File(path, "r") as handle:
            sessions = handle.get("sessions")
            if not isinstance(sessions, h5py.Group) or session not in sessions:
                raise ValueError(f"SEED HDF5 session is missing: {session!r}.")
            group = sessions[session]
            if not isinstance(group, h5py.Group):
                raise ValueError(f"SEED HDF5 session is not a group: {session!r}.")
            for key in ("data", "labels", "trial_ids", "trial_ordinals"):
                if key not in group:
                    raise ValueError(
                        f"SEED HDF5 session {session!r} is missing required "
                        f"dataset: {key}."
                    )
            if "sample_rate" not in group.attrs:
                raise ValueError(
                    f"SEED HDF5 session {session!r} is missing required "
                    "attribute: sample_rate."
                )
            if "unit" not in handle.attrs:
                raise ValueError("SEED HDF5 is missing required root attribute: unit.")
            if "window_sec" not in handle.attrs:
                raise ValueError(
                    "SEED HDF5 is missing required root attribute: window_sec."
                )

            data = np.asarray(group["data"], dtype=np.float32)
            num_trials = int(data.shape[0]) if data.ndim >= 1 else 0
            subject_id = _seed_subject_id(handle.attrs)
            metadata = SequentialDatasetMetadata(
                sample_rate=float(group.attrs["sample_rate"]),
                channel_names=_decode_json_attribute(group.attrs, "channel_names"),
                class_names=_decode_json_attribute(handle.attrs, "class_names"),
                unit=str(handle.attrs["unit"]),
                dataset_name="seed",
                window_sec=float(handle.attrs["window_sec"]),
            )
            trial_ids = np.asarray(
                _as_trial_vector(
                    group["trial_ids"],
                    name="trial_ids",
                    expected=num_trials,
                ).astype(str),
                dtype=object,
            )
            labels = _as_trial_vector(
                group["labels"], name="labels", expected=num_trials
            )
            trial_ordinals = _as_trial_vector(
                group["trial_ordinals"],
                name="trial_ordinals",
                expected=num_trials,
            )

        return SequentialDataset(
            metadata=metadata,
            data=data,
            labels=labels,
            subject_ids=np.repeat(
                np.asarray([subject_id], dtype=object), num_trials
            ),
            session_ids=np.repeat(
                np.asarray([str(session)], dtype=object), num_trials
            ),
            trial_ids=trial_ids,
            trial_ordinals=trial_ordinals,
            window_ids=np.asarray(
                [f"{subject_id}:{session}:{trial_id}" for trial_id in trial_ids],
                dtype=object,
            ),
        )


def _window_suffix(window_sec: float) -> str:
    rounded = int(round(float(window_sec)))
    if np.isclose(float(window_sec), rounded, atol=1e-9, rtol=0.0):
        return f"{rounded}s"
    return f"{float(window_sec):g}s"


DEFAULT_DATASET_ADAPTER_REGISTRY = DatasetAdapterRegistry(
    (
        LegacyHDF5Adapter(),
        WorkloadHDF5Adapter(),
        SeedHDF5Adapter(),
    )
)
