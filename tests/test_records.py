import pytest
import numpy as np

from bci_dayloop.data.records import EEGEvent, RawEEGRecord, UnitEvidence


@pytest.mark.parametrize("raw_unit", ["uV", "µV", "μV"])
def test_microvolt_spellings_normalize_to_uv(raw_unit: str) -> None:
    evidence = UnitEvidence(raw_unit, None, "header_candidate")

    assert evidence.raw_unit == raw_unit
    assert evidence.normalized_unit == "uV"


@pytest.mark.parametrize("unit", ["V", "mV"])
def test_volt_and_millivolt_remain_canonical(unit: str) -> None:
    evidence = UnitEvidence(unit, None, "header_candidate")

    assert evidence.normalized_unit == unit


def test_header_candidate_is_not_model_safe() -> None:
    evidence = UnitEvidence("uV", None, "header_candidate")

    assert evidence.is_model_safe is False


def test_vendor_confirmed_is_model_safe() -> None:
    evidence = UnitEvidence("uV", None, "vendor_confirmed")

    assert evidence.is_model_safe is True


def test_unknown_unit_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported EEG unit"):
        UnitEvidence("nV", None, "header_candidate")


def test_model_safe_evidence_requires_normalized_unit() -> None:
    with pytest.raises(ValueError, match="requires a normalized_unit"):
        UnitEvidence(None, None, "official_reader_verified")


def _unit_evidence() -> UnitEvidence:
    return UnitEvidence("uV", None, "vendor_confirmed")


def test_raw_eeg_record_constructs_and_normalizes_eeg_dtype() -> None:
    record = RawEEGRecord(
        eeg=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16),
        channel_names=("C3", "C4"),
        sampling_rate=250.0,
        unit_evidence=_unit_evidence(),
        timestamps=np.array([0.0, 0.004, 0.008]),
        events=(EEGEvent(2, "imagery", code=1),),
    )

    assert record.eeg.dtype == np.float32
    assert record.eeg.shape == (2, 3)
    assert record.timestamps is not None
    assert record.events[0].event_type == "imagery"


def test_raw_eeg_arrays_are_read_only() -> None:
    record = RawEEGRecord(
        eeg=np.zeros((1, 2)),
        channel_names=("C3",),
        sampling_rate=250.0,
        unit_evidence=_unit_evidence(),
        timestamps=np.array([0.0, 0.004]),
    )

    assert record.eeg.flags.writeable is False
    assert record.timestamps is not None
    assert record.timestamps.flags.writeable is False
    with pytest.raises(ValueError):
        record.eeg[0, 0] = 1.0


def test_non_2d_eeg_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"shape \[C, T\]"):
        RawEEGRecord(np.zeros(3), ("C3",), 250.0, _unit_evidence())


def test_channel_count_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="channel count"):
        RawEEGRecord(np.zeros((2, 3)), ("C3",), 250.0, _unit_evidence())


def test_non_positive_sampling_rate_is_rejected() -> None:
    with pytest.raises(ValueError, match="sampling_rate"):
        RawEEGRecord(np.zeros((1, 3)), ("C3",), 0.0, _unit_evidence())


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_non_finite_eeg_is_rejected(invalid_value: float) -> None:
    with pytest.raises(ValueError, match="NaN or Inf"):
        RawEEGRecord(np.array([[invalid_value]]), ("C3",), 250.0, _unit_evidence())


def test_timestamp_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="length T"):
        RawEEGRecord(
            np.zeros((1, 3)),
            ("C3",),
            250.0,
            _unit_evidence(),
            timestamps=np.array([0.0, 0.004]),
        )


def test_non_increasing_timestamps_are_rejected() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        RawEEGRecord(
            np.zeros((1, 3)),
            ("C3",),
            250.0,
            _unit_evidence(),
            timestamps=np.array([0.0, 0.004, 0.004]),
        )


def test_out_of_bounds_event_is_rejected() -> None:
    with pytest.raises(ValueError, match="smaller than T"):
        RawEEGRecord(
            np.zeros((1, 3)),
            ("C3",),
            250.0,
            _unit_evidence(),
            events=(EEGEvent(3, "imagery"),),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sample_index": -1, "event_type": "imagery"}, "sample_index"),
        ({"sample_index": 0, "event_type": "imagery", "duration_seconds": -0.1}, "duration_seconds"),
        ({"sample_index": 0, "event_type": ""}, "event_type"),
    ],
)
def test_invalid_eeg_event_is_rejected(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EEGEvent(**kwargs)
