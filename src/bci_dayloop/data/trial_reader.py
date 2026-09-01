"""Small common reader contract for flat EEG and grouped Workload HDF5."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from .hdf5_dataset import EEGHDF5, HDF5Metadata
from .workload import WorkloadHDF5


DataReaderName = Literal["eeg", "workload"]


class TrialReader(Protocol):
    """The only HDF5 operations required by population-head training."""

    path: Path

    @property
    def metadata(self) -> HDF5Metadata: ...

    def available_sessions(self) -> list[str]: ...

    def trial_metadata(self) -> dict[str, np.ndarray]: ...

    def load(self, session: str) -> dict[str, np.ndarray]: ...


def open_trial_reader(
    *,
    data_reader: DataReaderName,
    path: str | Path,
    canonical_subject_id: int,
) -> TrialReader:
    """Open an explicit reader and validate Workload's Pxx ↔ integer mapping."""
    if data_reader == "eeg":
        return EEGHDF5(path)

    if data_reader == "workload":
        reader = WorkloadHDF5(path)
        if reader.canonical_subject_id != int(canonical_subject_id):
            raise ValueError(
                "Workload subject mapping mismatch: "
                f"requested canonical subject {canonical_subject_id}, but "
                f"{reader.path} declares source subject "
                f"{reader.source_subject_id!r}."
            )
        return reader

    raise ValueError(f"Unsupported data_reader: {data_reader!r}.")


def reader_identity(
    reader: TrialReader,
    *,
    data_reader: DataReaderName,
    canonical_subject_id: int,
) -> dict[str, int | str]:
    """Preserve source identity in run metadata without changing EEG IDs."""
    source_subject_id = (
        reader.source_subject_id  # type: ignore[attr-defined]
        if data_reader == "workload"
        else str(canonical_subject_id)
    )
    return {
        "canonical_subject_id": int(canonical_subject_id),
        "source_subject_id": str(source_subject_id),
    }
