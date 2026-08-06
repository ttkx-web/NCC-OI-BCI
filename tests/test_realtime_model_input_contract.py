import numpy as np
import pytest

from bci_dayloop.models.model_50m.config import STANDARD_64_CHANNELS
from bci_dayloop.realtime.contracts import RealtimeWindow
from bci_dayloop.realtime.model_input_contract import audit_realtime_window_model_input


def _window(
    *,
    channel_names=STANDARD_64_CHANNELS,
    sampling_rate: float = 100.0,
    sample_count: int = 400,
    unit: str = "uV",
    metadata: dict[str, object] | None = None,
) -> RealtimeWindow:
    names = tuple(channel_names)
    values = np.arange(len(names) * sample_count, dtype=np.float32).reshape(len(names), sample_count)
    window_metadata = {
        "channel_types": tuple("eeg" for _ in names),
        "channel_units": tuple("uV" for _ in names),
        "unit_evidence_level": "vendor_confirmed",
        "model_safe": True,
    }
    if metadata is not None:
        window_metadata.update(metadata)
    return RealtimeWindow(
        window_id=1,
        samples=values,
        channel_names=names,
        sampling_rate=sampling_rate,
        unit=unit,
        timestamps=np.arange(sample_count, dtype=np.float64) / sampling_rate,
        start_sample_index=0,
        end_sample_index=sample_count,
        source_sequence_start=1,
        source_sequence_end=1,
        metadata=window_metadata,
    )


def test_exact_four_second_model_contract_passes_without_changing_samples() -> None:
    window = _window()
    before = window.samples.copy()

    audit = audit_realtime_window_model_input(window)

    assert audit.model_input_safe is True
    assert audit.failure_reasons == ()
    assert audit.matched_channels == STANDARD_64_CHANNELS
    np.testing.assert_array_equal(window.samples, before)


def test_missing_extra_and_ordered_channels_are_rejected_without_repair() -> None:
    missing = audit_realtime_window_model_input(_window(channel_names=STANDARD_64_CHANNELS[:-1]))
    extra = audit_realtime_window_model_input(_window(channel_names=(*STANDARD_64_CHANNELS, "Extra")))
    reordered = audit_realtime_window_model_input(
        _window(channel_names=(STANDARD_64_CHANNELS[1], STANDARD_64_CHANNELS[0], *STANDARD_64_CHANNELS[2:]))
    )

    assert missing.model_input_safe is False
    assert missing.missing_model_channels == (STANDARD_64_CHANNELS[-1],)
    assert extra.model_input_safe is False
    assert extra.unexpected_realtime_channels == ("Extra",)
    assert reordered.model_input_safe is False
    assert any("channel order" in reason for reason in reordered.failure_reasons)


@pytest.mark.parametrize(
    ("changes", "expected_reason"),
    [
        ({"sampling_rate": 200.0, "sample_count": 800}, "sampling_rate mismatch"),
        ({"sample_count": 399}, "window sample count mismatch"),
        ({"unit": "mV"}, "unit mismatch"),
    ],
)
def test_rate_length_and_unit_mismatches_are_rejected(changes, expected_reason) -> None:
    audit = audit_realtime_window_model_input(_window(**changes))

    assert audit.model_input_safe is False
    assert any(expected_reason in reason for reason in audit.failure_reasons)


def test_alias_case_and_duplicate_differences_are_reported_but_not_accepted() -> None:
    names = list(STANDARD_64_CHANNELS)
    names[3] = "af7"
    names[26] = "T3"
    names[-1] = names[-2]

    audit = audit_realtime_window_model_input(_window(channel_names=tuple(names)))

    assert audit.model_input_safe is False
    assert "af7 -> AF7" in audit.case_differences
    assert "T3 -> T7" in audit.alias_differences
    assert names[-1] in audit.duplicate_channel_names


def test_current_59_channel_1000_hz_window_remains_blocked() -> None:
    realtime_names = STANDARD_64_CHANNELS[:59]
    window = _window(channel_names=realtime_names, sampling_rate=1000.0, sample_count=4000)

    audit = audit_realtime_window_model_input(window)

    assert audit.model_input_safe is False
    assert len(audit.missing_model_channels) == 5
    assert any("sampling_rate mismatch" in reason for reason in audit.failure_reasons)
    assert any("window sample count mismatch" in reason for reason in audit.failure_reasons)


@pytest.mark.parametrize(
    "metadata",
    [
        {"model_safe": False},
        {"unit_evidence_level": "unknown"},
        {"channel_units": tuple("unknown" for _ in STANDARD_64_CHANNELS)},
        {"channel_types": tuple("eeg" for _ in STANDARD_64_CHANNELS[:-1])},
    ],
)
def test_unit_and_channel_metadata_must_be_complete_and_safe(metadata) -> None:
    audit = audit_realtime_window_model_input(_window(metadata=metadata))

    assert audit.model_input_safe is False
    assert audit.failure_reasons
