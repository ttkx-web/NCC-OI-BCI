"""Channel-aware unit contracts for the verified Neuracle JellyFish TCP source."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .contracts import EEGChunk


MIXED_STREAM_UNIT = "mixed"
EEG_UNIT = "uV"
TRIGGER_UNIT = "code"
UNKNOWN_UNIT = "unknown"
VENDOR_CONFIRMED = "vendor_confirmed"

_PHYSIOLOGICAL_TYPES = frozenset({"eeg", "eog", "ecg", "hecg"})


class ChannelUnitContractError(ValueError):
    """Raised when a channel-aware unit contract cannot safely be established."""


def normalize_channel_type(value: object) -> str:
    """Normalize comparison-only channel type text without changing source metadata."""
    return value.strip().casefold() if isinstance(value, str) else ""


def neuracle_channel_units(channel_types: Sequence[object]) -> tuple[str, ...]:
    """Return vendor-confirmed units only for explicitly covered JellyFish channel types."""
    return tuple(
        EEG_UNIT
        if normalize_channel_type(channel_type) in _PHYSIOLOGICAL_TYPES
        else TRIGGER_UNIT
        if normalize_channel_type(channel_type) == "trigger"
        else UNKNOWN_UNIT
        for channel_type in channel_types
    )


def eeg_channels_model_safe(
    channel_types: Sequence[object],
    channel_units: Sequence[object],
    evidence_level: object,
) -> bool:
    """Return true only when every selected EEG channel has vendor-confirmed uV units."""
    if len(channel_types) != len(channel_units) or evidence_level != VENDOR_CONFIRMED:
        return False
    eeg_indices = [
        index for index, channel_type in enumerate(channel_types) if normalize_channel_type(channel_type) == "eeg"
    ]
    return bool(eeg_indices) and all(channel_units[index] == EEG_UNIT for index in eeg_indices)


def unit_status_from_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    """Build the privacy-safe unit status used by health and probe summaries."""
    channel_types = _metadata_sequence(metadata, "channel_types")
    channel_units = _metadata_sequence(metadata, "channel_units")
    evidence_level = metadata.get("unit_evidence_level")
    eeg_safe = eeg_channels_model_safe(channel_types, channel_units, evidence_level)
    eeg_indices = [
        index for index, channel_type in enumerate(channel_types) if normalize_channel_type(channel_type) == "eeg"
    ]
    eeg_unit = (
        EEG_UNIT
        if eeg_indices and len(channel_types) == len(channel_units) and all(channel_units[index] == EEG_UNIT for index in eeg_indices)
        else UNKNOWN_UNIT
    )
    return {
        "stream_unit": MIXED_STREAM_UNIT,
        "eeg_unit": eeg_unit,
        "unit_evidence_level": evidence_level,
        "raw_model_safe": False,
        "eeg_model_safe": eeg_safe,
    }


def select_verified_eeg_channels(chunk: EEGChunk) -> EEGChunk:
    """Select vendor-confirmed EEG channels without preprocessing or changing timing semantics."""
    metadata = chunk.metadata
    channel_types = _metadata_sequence(metadata, "channel_types")
    channel_units = _metadata_sequence(metadata, "channel_units")
    if len(channel_types) != len(chunk.channel_names) or len(channel_units) != len(chunk.channel_names):
        raise ChannelUnitContractError("channel metadata lengths must match chunk channel_names")
    evidence_level = metadata.get("unit_evidence_level")
    eeg_indices = [
        index for index, channel_type in enumerate(channel_types) if normalize_channel_type(channel_type) == "eeg"
    ]
    if not eeg_indices:
        raise ChannelUnitContractError("chunk has no EEG channels")
    if evidence_level != VENDOR_CONFIRMED:
        raise ChannelUnitContractError("EEG channel selection requires vendor-confirmed unit evidence")
    if any(channel_units[index] != EEG_UNIT for index in eeg_indices):
        raise ChannelUnitContractError("every selected EEG channel must have uV units")

    selected_metadata = dict(metadata)
    selected_metadata.update(
        {
            "channel_types": tuple(channel_types[index] for index in eeg_indices),
            "channel_units": tuple(channel_units[index] for index in eeg_indices),
            "unit_evidence_level": evidence_level,
            "model_safe": True,
        }
    )
    return EEGChunk(
        samples=chunk.samples[eeg_indices, :],
        channel_names=tuple(chunk.channel_names[index] for index in eeg_indices),
        sampling_rate=chunk.sampling_rate,
        unit=EEG_UNIT,
        timestamps=chunk.timestamps,
        sequence_id=chunk.sequence_id,
        device_id=chunk.device_id,
        received_at=chunk.received_at,
        metadata=selected_metadata,
    )


def _metadata_sequence(metadata: Mapping[str, object], name: str) -> tuple[object, ...]:
    value = metadata.get(name)
    if isinstance(value, (str, bytes)) or value is None:
        raise ChannelUnitContractError(f"chunk metadata missing sequence field: {name}")
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ChannelUnitContractError(f"chunk metadata field must be a sequence: {name}") from exc
