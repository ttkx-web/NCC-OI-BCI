"""Strict extraction of labelled motor-imagery trials from aligned EEG events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bci_dayloop.data.records import EEGEvent, RawEEGRecord


_IMAGERY_LABELS = frozenset({"left_hand", "right_hand", "feet", "tongue"})


@dataclass(frozen=True)
class EEGTrial:
    """One immutable EEG segment delimited by authoritative BDF sample indices."""

    eeg: np.ndarray
    label: str
    block_id: int | str
    trial_id: int | str
    start_sample: int
    end_sample: int
    duration_seconds: float
    start_event: EEGEvent
    end_event: EEGEvent

    def __post_init__(self) -> None:
        eeg = np.asarray(self.eeg, dtype=np.float32).copy()
        if eeg.ndim != 2:
            raise ValueError("trial eeg must have shape [C, T]")
        if not np.isfinite(eeg).all():
            raise ValueError("trial eeg must not contain NaN or Inf")
        if self.label not in _IMAGERY_LABELS:
            raise ValueError(f"Unsupported imagery label: {self.label!r}")
        if self.start_sample < 0 or self.end_sample <= self.start_sample:
            raise ValueError("trial end_sample must be greater than start_sample")
        if self.duration_seconds <= 0:
            raise ValueError("trial duration_seconds must be positive")
        eeg.setflags(write=False)
        object.__setattr__(self, "eeg", eeg)


def _find_end_event(events: tuple[EEGEvent, ...], start_index: int) -> EEGEvent:
    start_event = events[start_index]
    for event in events[start_index + 1 :]:
        if event.event_type == "imagery":
            raise ValueError(
                "Another imagery event appears before the rest endpoint for "
                f"block/trial {start_event.block_id!r}/{start_event.trial_id!r}"
            )
        if (
            event.event_type == "rest"
            and event.code == 20
            and event.block_id == start_event.block_id
            and event.trial_id == start_event.trial_id
        ):
            return event
    raise ValueError(
        "Missing rest endpoint for imagery event at "
        f"block/trial {start_event.block_id!r}/{start_event.trial_id!r}"
    )


def extract_imagery_trials(
    record: RawEEGRecord,
    expected_duration_seconds: float = 4.0,
    duration_tolerance_seconds: float = 0.1,
) -> tuple[EEGTrial, ...]:
    """Extract imagery-to-rest segments without modifying or resampling EEG data."""
    if expected_duration_seconds <= 0 or duration_tolerance_seconds < 0:
        raise ValueError("expected duration must be positive and tolerance non-negative")
    if not np.isfinite(record.eeg).all():
        raise ValueError("record eeg must not contain NaN or Inf")
    if any(event.event_type == "abort" for event in record.events):
        raise ValueError("Cannot extract trials from a record containing an abort event")

    trials: list[EEGTrial] = []
    seen_ids: set[tuple[int | str, int | str]] = set()
    for event_index, start_event in enumerate(record.events):
        if start_event.event_type != "imagery":
            continue
        if start_event.block_id is None or start_event.trial_id is None:
            raise ValueError("Imagery event requires block_id and trial_id")
        if start_event.label not in _IMAGERY_LABELS:
            raise ValueError(f"Imagery event has unsupported label: {start_event.label!r}")

        identifier = (start_event.block_id, start_event.trial_id)
        if identifier in seen_ids:
            raise ValueError(f"Duplicate imagery trial: {identifier!r}")
        seen_ids.add(identifier)

        end_event = _find_end_event(record.events, event_index)
        start_sample = start_event.sample_index
        end_sample = end_event.sample_index
        if end_sample <= start_sample:
            raise ValueError("Rest endpoint must occur after imagery start")
        duration_seconds = (end_sample - start_sample) / record.sampling_rate
        if abs(duration_seconds - expected_duration_seconds) > duration_tolerance_seconds:
            raise ValueError(
                f"Imagery trial duration {duration_seconds} differs from "
                f"expected {expected_duration_seconds} by more than {duration_tolerance_seconds}"
            )

        trials.append(
            EEGTrial(
                eeg=record.eeg[:, start_sample:end_sample],
                label=start_event.label,
                block_id=start_event.block_id,
                trial_id=start_event.trial_id,
                start_sample=start_sample,
                end_sample=end_sample,
                duration_seconds=duration_seconds,
                start_event=start_event,
                end_event=end_event,
            )
        )

    for previous, current in zip(trials, trials[1:]):
        if current.start_sample < previous.end_sample:
            raise ValueError("Imagery trials overlap")
    return tuple(trials)
