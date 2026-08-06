import numpy as np
import pytest

from bci_dayloop.realtime.channel_units import (
    ChannelUnitContractError,
    EEG_UNIT,
    MIXED_STREAM_UNIT,
    TRIGGER_UNIT,
    UNKNOWN_UNIT,
    VENDOR_CONFIRMED,
    neuracle_channel_units,
    select_verified_eeg_channels,
)
from bci_dayloop.realtime.contracts import EEGChunk


def _mixed_chunk(
    *,
    channel_types: tuple[str, ...] = ("ECG", " EEG ", "Trigger", "eeg", "other"),
    channel_units: tuple[str, ...] | None = None,
    evidence_level: str = VENDOR_CONFIRMED,
) -> EEGChunk:
    units = channel_units if channel_units is not None else neuracle_channel_units(channel_types)
    return EEGChunk(
        samples=np.arange(len(channel_types) * 3, dtype=np.float32).reshape(len(channel_types), 3),
        channel_names=tuple(f"channel-{index}" for index in range(len(channel_types))),
        sampling_rate=1000.0,
        unit=MIXED_STREAM_UNIT,
        timestamps=np.array([1.0, 1.001, 1.002]),
        sequence_id=9,
        device_id=None,
        received_at=7.5,
        metadata={
            "channel_types": channel_types,
            "channel_units": units,
            "unit_evidence_level": evidence_level,
            "model_safe": False,
            "raw_start_timestamp": 1000,
        },
    )


def test_neuracle_channel_units_cover_confirmed_physiology_and_trigger_only() -> None:
    assert neuracle_channel_units((" EEG ", "EOG", "HECG", "ecg", "Trigger", "Other")) == (
        EEG_UNIT,
        EEG_UNIT,
        EEG_UNIT,
        EEG_UNIT,
        TRIGGER_UNIT,
        UNKNOWN_UNIT,
    )


def test_eeg_selector_preserves_eeg_order_samples_and_timing() -> None:
    raw = _mixed_chunk()

    selected = select_verified_eeg_channels(raw)

    assert selected.unit == EEG_UNIT
    assert selected.channel_names == ("channel-1", "channel-3")
    assert selected.metadata["channel_types"] == (" EEG ", "eeg")
    assert selected.metadata["channel_units"] == (EEG_UNIT, EEG_UNIT)
    assert selected.metadata["unit_evidence_level"] == VENDOR_CONFIRMED
    assert selected.metadata["model_safe"] is True
    np.testing.assert_array_equal(selected.samples, raw.samples[[1, 3], :])
    np.testing.assert_array_equal(selected.timestamps, raw.timestamps)
    assert selected.sequence_id == raw.sequence_id
    assert selected.received_at == raw.received_at
    assert selected.metadata["raw_start_timestamp"] == raw.metadata["raw_start_timestamp"]
    assert "Trigger" not in selected.metadata["channel_types"]


@pytest.mark.parametrize(
    "channel_types, channel_units, evidence_level, message",
    [
        (("EEG", "Trigger"), (EEG_UNIT,), VENDOR_CONFIRMED, "lengths"),
        (("EEG", "Trigger"), (UNKNOWN_UNIT, TRIGGER_UNIT), VENDOR_CONFIRMED, "uV"),
        (("EEG", "Trigger"), (EEG_UNIT, TRIGGER_UNIT), "realtime_unverified", "vendor-confirmed"),
        (("EOG", "Trigger"), (EEG_UNIT, TRIGGER_UNIT), VENDOR_CONFIRMED, "no EEG"),
    ],
)
def test_eeg_selector_rejects_incomplete_or_unverified_contracts(
    channel_types: tuple[str, ...],
    channel_units: tuple[str, ...],
    evidence_level: str,
    message: str,
) -> None:
    raw = _mixed_chunk(
        channel_types=channel_types,
        channel_units=channel_units,
        evidence_level=evidence_level,
    )

    with pytest.raises(ChannelUnitContractError, match=message):
        select_verified_eeg_channels(raw)
