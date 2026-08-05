"""Immutable provenance records for EEG physical units."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np


_ALLOWED_EVIDENCE_LEVELS = frozenset(
    {
        "unknown",
        "header_candidate",
        "vendor_confirmed",
        "official_reader_verified",
        "calibration_verified",
    }
)
_MODEL_SAFE_EVIDENCE_LEVELS = frozenset(
    {
        "vendor_confirmed",
        "official_reader_verified",
        "calibration_verified",
    }
)


def _normalize_unit(unit: str | None) -> str | None:
    """Return a supported canonical unit without inferring from signal values."""
    if unit is None:
        return None
    if not isinstance(unit, str):
        raise ValueError(f"Unsupported EEG unit: {unit!r}")

    aliases = {"V": "V", "mV": "mV", "uV": "uV", "µV": "uV", "μV": "uV"}
    try:
        return aliases[unit]
    except KeyError as exc:
        raise ValueError(f"Unsupported EEG unit: {unit!r}") from exc


@dataclass(frozen=True)
class UnitEvidence:
    """Declared EEG unit plus the evidence that permits its use by a model."""

    raw_unit: str | None
    normalized_unit: str | None
    evidence_level: str
    evidence_source: str | None = None

    def __post_init__(self) -> None:
        if self.evidence_level not in _ALLOWED_EVIDENCE_LEVELS:
            raise ValueError(f"Unsupported evidence level: {self.evidence_level!r}")

        normalized_raw_unit = _normalize_unit(self.raw_unit)
        normalized_declared_unit = _normalize_unit(self.normalized_unit)
        if (
            normalized_raw_unit is not None
            and normalized_declared_unit is not None
            and normalized_raw_unit != normalized_declared_unit
        ):
            raise ValueError("raw_unit and normalized_unit must describe the same unit")

        object.__setattr__(
            self,
            "normalized_unit",
            normalized_declared_unit if normalized_declared_unit is not None else normalized_raw_unit,
        )
        if self.is_model_safe and self.normalized_unit is None:
            raise ValueError("Model-safe unit evidence requires a normalized_unit")

    @property
    def is_model_safe(self) -> bool:
        return self.evidence_level in _MODEL_SAFE_EVIDENCE_LEVELS


def _immutable_mapping(metadata: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(metadata))


@dataclass(frozen=True)
class EEGEvent:
    """An EEG event whose location is expressed in recording sample indices."""

    sample_index: int
    event_type: str
    code: int | str | None = None
    label: str | None = None
    onset_seconds: float | None = None
    duration_seconds: float = 0.0
    trial_id: int | str | None = None
    block_id: int | str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if not isinstance(self.event_type, str) or not self.event_type.strip():
            raise ValueError("event_type must be non-empty")
        if self.onset_seconds is not None and self.onset_seconds < 0:
            raise ValueError("onset_seconds must be non-negative")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")

        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True)
class RawEEGRecord:
    """Validated, immutable in-memory EEG recording data in ``[channels, time]`` order."""

    eeg: np.ndarray
    channel_names: tuple[str, ...]
    sampling_rate: float
    unit_evidence: UnitEvidence
    timestamps: np.ndarray | None = None
    events: tuple[EEGEvent, ...] = ()
    channel_types: tuple[str, ...] | None = None
    subject_id: str | None = None
    session_id: str | None = None
    device_id: str | None = None
    source_path: str | None = None
    source_sha256: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        eeg = np.asarray(self.eeg, dtype=np.float32).copy()
        if eeg.ndim != 2:
            raise ValueError("eeg must have shape [C, T]")
        if not np.isfinite(eeg).all():
            raise ValueError("eeg must not contain NaN or Inf")

        channel_names = tuple(self.channel_names)
        channel_count, sample_count = eeg.shape
        if channel_count != len(channel_names):
            raise ValueError("eeg channel count must match channel_names")
        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")

        channel_types = None
        if self.channel_types is not None:
            channel_types = tuple(self.channel_types)
            if len(channel_types) != channel_count:
                raise ValueError("channel_types length must match channel count")

        timestamps = None
        if self.timestamps is not None:
            timestamps = np.asarray(self.timestamps).copy()
            if timestamps.ndim != 1 or timestamps.shape[0] != sample_count:
                raise ValueError("timestamps must be one-dimensional with length T")
            if not np.issubdtype(timestamps.dtype, np.number) or not np.isfinite(timestamps).all():
                raise ValueError("timestamps must contain only finite numeric values")
            if not np.all(np.diff(timestamps) > 0):
                raise ValueError("timestamps must be strictly increasing")

        events = tuple(self.events)
        for event in events:
            if not isinstance(event, EEGEvent):
                raise ValueError("events must contain EEGEvent instances")
            if event.sample_index >= sample_count:
                raise ValueError("event sample_index must be smaller than T")

        eeg.setflags(write=False)
        if timestamps is not None:
            timestamps.setflags(write=False)
        object.__setattr__(self, "eeg", eeg)
        object.__setattr__(self, "channel_names", channel_names)
        object.__setattr__(self, "channel_types", channel_types)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))
