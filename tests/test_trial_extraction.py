import numpy as np
import pytest

from bci_dayloop.data.records import EEGEvent, RawEEGRecord, UnitEvidence
from bci_dayloop.data.trial_extraction import extract_imagery_trials


def _event(
    sample_index: int,
    event_type: str,
    *,
    code: int | None = None,
    label: str | None = None,
    block_id: int | None = 1,
    trial_id: int | None = 1,
) -> EEGEvent:
    return EEGEvent(
        sample_index=sample_index,
        event_type=event_type,
        code=code,
        label=label,
        block_id=block_id,
        trial_id=trial_id,
    )


def _record(events: tuple[EEGEvent, ...], *, n_times: int = 200) -> RawEEGRecord:
    return RawEEGRecord(
        eeg=np.arange(2 * n_times).reshape(2, n_times),
        channel_names=("C3", "C4"),
        sampling_rate=10.0,
        unit_evidence=UnitEvidence("uV", None, "vendor_confirmed"),
        events=events,
    )


def test_extracts_four_imagery_classes_using_sample_indices() -> None:
    record = _record(
        (
            _event(0, "imagery", code=1, label="left_hand", trial_id=1),
            _event(40, "rest", code=20, trial_id=1),
            _event(40, "imagery", code=2, label="right_hand", trial_id=2),
            _event(80, "rest", code=20, trial_id=2),
            _event(80, "imagery", code=3, label="feet", trial_id=3),
            _event(120, "rest", code=20, trial_id=3),
            _event(120, "imagery", code=4, label="tongue", trial_id=4),
            _event(160, "rest", code=20, trial_id=4),
        )
    )

    trials = extract_imagery_trials(record)

    assert [trial.label for trial in trials] == ["left_hand", "right_hand", "feet", "tongue"]
    assert trials[0].eeg.shape == (2, 40)
    assert trials[0].eeg.dtype == np.float32
    assert trials[0].eeg.flags.writeable is False
    assert np.array_equal(trials[0].eeg, record.eeg[:, 0:40])
    assert trials[0].start_sample == 0
    assert trials[0].end_sample == 40
    assert trials[0].duration_seconds == 4.0


def test_missing_rest_endpoint_is_rejected() -> None:
    with pytest.raises(ValueError, match="Missing rest endpoint"):
        extract_imagery_trials(_record((_event(0, "imagery", code=1, label="left_hand"),)))


@pytest.mark.parametrize("field", ["block_id", "trial_id"])
def test_missing_imagery_identifiers_are_rejected(field: str) -> None:
    kwargs = {field: None}
    start = _event(0, "imagery", code=1, label="left_hand", **kwargs)
    end = _event(40, "rest", code=20, **kwargs)

    with pytest.raises(ValueError, match="requires block_id and trial_id"):
        extract_imagery_trials(_record((start, end)))


def test_duration_outside_tolerance_is_rejected() -> None:
    with pytest.raises(ValueError, match="duration"):
        extract_imagery_trials(
            _record((_event(0, "imagery", code=1, label="left_hand"), _event(45, "rest", code=20)))
        )


def test_duplicate_trial_identifier_is_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate imagery trial"):
        extract_imagery_trials(
            _record(
                (
                    _event(0, "imagery", code=1, label="left_hand", trial_id=1),
                    _event(40, "rest", code=20, trial_id=1),
                    _event(40, "imagery", code=1, label="left_hand", trial_id=1),
                    _event(80, "rest", code=20, trial_id=1),
                )
            )
        )


def test_overlapping_trials_are_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        extract_imagery_trials(
            _record(
                (
                    _event(0, "imagery", code=1, label="left_hand", trial_id=1),
                    _event(40, "rest", code=20, trial_id=1),
                    _event(20, "imagery", code=2, label="right_hand", trial_id=2),
                    _event(60, "rest", code=20, trial_id=2),
                )
            )
        )


def test_imagery_between_start_and_rest_is_rejected() -> None:
    with pytest.raises(ValueError, match="Another imagery event"):
        extract_imagery_trials(
            _record(
                (
                    _event(0, "imagery", code=1, label="left_hand", trial_id=1),
                    _event(20, "imagery", code=2, label="right_hand", trial_id=2),
                    _event(40, "rest", code=20, trial_id=1),
                )
            )
        )


def test_abort_event_is_rejected() -> None:
    with pytest.raises(ValueError, match="abort"):
        extract_imagery_trials(
            _record(
                (
                    _event(0, "imagery", code=1, label="left_hand"),
                    _event(40, "rest", code=20),
                    _event(41, "abort", code=127),
                )
            )
        )
