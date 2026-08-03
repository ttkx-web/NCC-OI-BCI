import pytest

from bci_dayloop.data.event_alignment import align_events_with_csv
from bci_dayloop.data.records import EEGEvent


def _imagery_event(*, code: int | str = "S 1", label: str | None = "left_hand") -> EEGEvent:
    return EEGEvent(
        sample_index=42,
        event_type="imagery",
        code=code,
        label=label,
        onset_seconds=0.168,
        duration_seconds=3.2,
        metadata={"original_description": "S 1", "marker_code": 1, "original_label": "left_hand"},
    )


def test_strict_alignment_supplements_allowed_fields_without_changing_bdf_timing() -> None:
    event = _imagery_event()
    row = {
        "marker_code": "Stimulus/S 1",
        "trial_id": "trial-7",
        "block_id": 2,
        "label": "left_hand",
        "lsl_timestamp": 123.456,
        "flip_time": 122.5,
        "unapproved_field": "ignored",
    }

    aligned = align_events_with_csv((event,), (row,))

    result = aligned[0]
    assert result is not event
    assert result.sample_index == 42
    assert result.onset_seconds == 0.168
    assert result.duration_seconds == 3.2
    assert result.event_type == "imagery"
    assert result.code == "S 1"
    assert result.trial_id == "trial-7"
    assert result.block_id == 2
    assert result.label == "left_hand"
    assert result.metadata == {
        "original_description": "S 1",
        "marker_code": 1,
        "original_label": "left_hand",
        "lsl_timestamp": 123.456,
        "flip_time": 122.5,
    }


@pytest.mark.parametrize(
    ("field", "csv_code"),
    [
        ("marker_code", "S 1"),
        ("event_code", "S1"),
        ("code", "Stimulus/S 1"),
        ("marker", 1),
    ],
)
def test_supported_csv_marker_fields_and_formats_align(field: str, csv_code: object) -> None:
    aligned = align_events_with_csv((_imagery_event(code=1),), ({field: csv_code},))

    assert aligned[0].code == 1


def test_count_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="Event count mismatch"):
        align_events_with_csv((_imagery_event(),), ())


def test_marker_mismatch_reports_position_and_both_codes() -> None:
    with pytest.raises(
        ValueError, match=r"index 0: BDF code 1, CSV code 2"
    ):
        align_events_with_csv((_imagery_event(code=1),), ({"code": 2},))


def test_csv_row_without_marker_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="no marker code field"):
        align_events_with_csv((_imagery_event(),), ({"trial_id": 1},))


def test_unparseable_csv_marker_is_rejected() -> None:
    with pytest.raises(ValueError, match="CSV marker code"):
        align_events_with_csv((_imagery_event(),), ({"marker": "not-a-code"},))


def test_unparseable_bdf_event_code_is_rejected() -> None:
    with pytest.raises(ValueError, match="BDF event code at index 0"):
        align_events_with_csv((_imagery_event(code="not-a-code"),), ({"marker": 1},))


def test_matching_imagery_label_is_allowed() -> None:
    aligned = align_events_with_csv(
        (_imagery_event(label="left_hand"),), ({"marker": 1, "label": "left_hand"},)
    )

    assert aligned[0].label == "left_hand"


def test_conflicting_imagery_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="Label mismatch"):
        align_events_with_csv(
            (_imagery_event(label="left_hand"),), ({"marker": 1, "label": "right_hand"},)
        )


def test_empty_csv_label_does_not_overwrite_existing_label() -> None:
    event = _imagery_event(label="left_hand")

    aligned = align_events_with_csv((event,), ({"marker": 1, "label": "  "},))

    assert aligned[0].label == "left_hand"


def test_input_event_and_metadata_are_not_modified() -> None:
    event = _imagery_event()
    original_metadata = dict(event.metadata)

    align_events_with_csv(
        (event,), ({"marker": 1, "trial_id": 9, "lsl_timestamp": 1.0},)
    )

    assert event.trial_id is None
    assert event.metadata == original_metadata
