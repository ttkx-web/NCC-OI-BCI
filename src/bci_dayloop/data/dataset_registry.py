"""Static dataset contracts used by training entry points.

The adapter registry answers *how* an HDF5 file is read.  This registry answers
*what* a named dataset is expected to contain.  Keeping those concerns separate
lets Awakening reuse :class:`LegacyHDF5Adapter` without inventing another HDF5
layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

AWAKENING_SOURCE_CHANNELS: tuple[str, ...] = (
    "Iz", "O2", "Oz", "O1", "PO8", "PO4", "POz", "PO3", "PO7",
    "P8", "P6", "P4", "P2", "Pz", "P1", "P3", "P5", "P7",
    "TP10", "TP8", "CP6", "CP4", "CP2", "CPz", "CP1", "CP3",
    "CP5", "TP7", "TP9", "T8", "C6", "C4", "C2", "Cz", "C1",
    "C3", "C5", "T7", "FT8", "FC6", "FC4", "FC2", "FCz",
    "FC1", "FC3", "FC5", "FT7", "F8", "F6", "F4", "F2", "Fz",
    "F1", "F3", "F5", "F7", "AF4", "AFz", "AF3", "Fp2", "Fpz",
    "Fp1",
)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Fail-closed facts for one prepared dataset."""

    name: str
    default_data_root: str
    data_pattern: str
    data_reader: str
    canonical_subjects: tuple[int, ...]
    train_session: str
    validation_session: str
    final_test_session: str
    session_semantics: str
    sample_rate: float
    window_seconds: float
    source_channels: tuple[str, ...]
    class_names: tuple[str, ...]
    unit: str
    channel_profile: str
    ignored_source_channels: tuple[str, ...]
    missing_model_channels: tuple[str, ...]
    positive_class_index: int

AWAKENING_2S = DatasetSpec(
    name="awakening",
    default_data_root="data/processed/awakening_2s",
    data_pattern="subject_{subject:02d}.h5",
    data_reader="eeg",
    canonical_subjects=tuple(range(1, 22)),
    train_session="S1",
    validation_session="S2",
    final_test_session="S2",
    session_semantics="deterministic_group_split_by_source_lance_row_id",
    sample_rate=200.0,
    window_seconds=2.0,
    source_channels=AWAKENING_SOURCE_CHANNELS,
    class_names=("non_awakening", "awakening"),
    unit="uV",
    channel_profile="awakening_62_to_model50m_standard64_v1",
    ignored_source_channels=("TP9", "TP10"),
    missing_model_channels=("AF7", "AF8", "F9", "F10"),
    positive_class_index=1,
)


DATASET_REGISTRY: Mapping[str, DatasetSpec] = MappingProxyType(
    {AWAKENING_2S.name: AWAKENING_2S}
)


def get_dataset_spec(name: str) -> DatasetSpec:
    key = str(name).strip().lower()
    try:
        return DATASET_REGISTRY[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown dataset {name!r}; registered datasets: "
            f"{sorted(DATASET_REGISTRY)}."
        ) from exc
