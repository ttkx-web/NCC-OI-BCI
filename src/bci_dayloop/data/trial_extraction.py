"""Strict extraction of labelled motor-imagery trials from aligned EEG events."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from bci_dayloop.data.records import EEGEvent, RawEEGRecord


_IMAGERY_LABELS = frozenset({"left_hand", "right_hand", "feet", "tongue"})
EXTRACTION_POLICY = "fixed_duration_from_class_marker"
WINDOW_SEMANTICS = "cue_plus_imagery_4s"
ELIGIBLE_FOR_ACCURACY = True
ACCURACY_SCOPE = "cue_plus_imagery_task_classification"
VISUAL_CUE_PRESENT = True
VISUAL_CUE_DURATION_SECONDS = 0.8
ELIGIBLE_FOR_PURE_IMAGERY_ACCURACY = False


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
    canonical_start_sample: int
    canonical_end_sample: int
    canonical_n_samples: int
    observed_rest_sample: int
    observed_event_n_samples: int
    rest_offset_samples: int
    rest_offset_seconds: float
    endpoint_qc_passed: bool
    extraction_policy: str = EXTRACTION_POLICY
    window_semantics: str = WINDOW_SEMANTICS
    eligible_for_accuracy: bool = ELIGIBLE_FOR_ACCURACY
    accuracy_scope: str = ACCURACY_SCOPE
    visual_cue_present: bool = VISUAL_CUE_PRESENT
    visual_cue_duration_seconds: float = VISUAL_CUE_DURATION_SECONDS
    eligible_for_pure_imagery_accuracy: bool = ELIGIBLE_FOR_PURE_IMAGERY_ACCURACY
    source_metadata: Mapping[str, object] = field(default_factory=dict)

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
        if self.canonical_start_sample != self.start_sample or self.canonical_end_sample != self.end_sample:
            raise ValueError("trial sample bounds must be canonical bounds")
        if self.canonical_n_samples != self.canonical_end_sample - self.canonical_start_sample:
            raise ValueError("canonical_n_samples must match canonical bounds")
        if self.observed_event_n_samples != self.observed_rest_sample - self.canonical_start_sample:
            raise ValueError("observed_event_n_samples does not match observed rest")
        if self.rest_offset_samples != self.observed_rest_sample - self.canonical_end_sample:
            raise ValueError("rest_offset_samples does not match canonical end")
        if self.extraction_policy != EXTRACTION_POLICY:
            raise ValueError("unexpected extraction_policy")
        if (
            self.window_semantics != WINDOW_SEMANTICS
            or self.eligible_for_accuracy is not True
            or self.accuracy_scope != ACCURACY_SCOPE
            or self.visual_cue_present is not True
            or self.visual_cue_duration_seconds != VISUAL_CUE_DURATION_SECONDS
            or self.eligible_for_pure_imagery_accuracy is not False
        ):
            raise ValueError("legacy 4 s trial semantics must remain cue_plus_imagery_4s")
        if self.duration_seconds <= 0:
            raise ValueError("trial duration_seconds must be positive")
        eeg.setflags(write=False)
        object.__setattr__(self, "eeg", eeg)
        object.__setattr__(self, "source_metadata", MappingProxyType(dict(self.source_metadata)))


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
    endpoint_tolerance_seconds: float = 0.05,
) -> tuple[EEGTrial, ...]:
    """Extract fixed-duration continuous EEG windows and retain rest timing observations."""
    if expected_duration_seconds <= 0 or duration_tolerance_seconds < 0 or endpoint_tolerance_seconds < 0:
        raise ValueError("expected duration must be positive and tolerance non-negative")
    if not np.isfinite(record.eeg).all():
        raise ValueError("record eeg must not contain NaN or Inf")
    if any(event.event_type == "abort" for event in record.events):
        raise ValueError("Cannot extract trials from a record containing an abort event")

    trials: list[EEGTrial] = []
    seen_ids: set[tuple[int | str, int | str]] = set()
    canonical_n_samples = round(expected_duration_seconds * record.sampling_rate)
    endpoint_tolerance_samples = math.ceil(endpoint_tolerance_seconds * record.sampling_rate)
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
        end_sample = start_sample + canonical_n_samples
        if start_sample < 0 or end_sample > record.eeg.shape[1]:
            raise ValueError("Canonical trial window is outside the recording")
        for event in record.events[event_index + 1 :]:
            if event.sample_index >= end_sample:
                break
            if event.event_type == "imagery":
                raise ValueError("Another imagery event appears before canonical trial end")
            event_text = f"{event.event_type} {event.metadata.get('original_description', '')}".lower()
            if "boundary" in event_text or "gap" in event_text or "bad_acq_skip" in event_text:
                raise ValueError("Canonical trial window crosses a boundary or gap")
            if event.event_type == "abort":
                raise ValueError("Canonical trial window crosses an abort event")
        observed_rest_sample = end_event.sample_index
        observed_event_n_samples = observed_rest_sample - start_sample
        rest_offset_samples = observed_rest_sample - end_sample
        if abs(rest_offset_samples) > endpoint_tolerance_samples:
            raise ValueError(
                "Endpoint QC failed for "
                f"block/trial {start_event.block_id!r}/{start_event.trial_id!r}: "
                f"rest_offset_samples={rest_offset_samples}, tolerance={endpoint_tolerance_samples}"
            )
        duration_seconds = canonical_n_samples / record.sampling_rate

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
                canonical_start_sample=start_sample,
                canonical_end_sample=end_sample,
                canonical_n_samples=canonical_n_samples,
                observed_rest_sample=observed_rest_sample,
                observed_event_n_samples=observed_event_n_samples,
                rest_offset_samples=rest_offset_samples,
                rest_offset_seconds=rest_offset_samples / record.sampling_rate,
                endpoint_qc_passed=True,
                source_metadata={
                    "bdf_sha256": record.source_sha256,
                    "csv_sha256": record.metadata.get("csv_sha256"),
                    "source_format": record.metadata.get("source_format"),
                    "conversion_tool": record.metadata.get("conversion_tool"),
                    "conversion_tool_version": record.metadata.get("conversion_tool_version"),
                    "reader_name": record.metadata.get("reader_name"),
                    "reader_version": record.metadata.get("reader_version"),
                    "unit_evidence_level": record.unit_evidence.evidence_level,
                    "window_semantics": WINDOW_SEMANTICS,
                    "eligible_for_accuracy": ELIGIBLE_FOR_ACCURACY,
                    "accuracy_scope": ACCURACY_SCOPE,
                    "visual_cue_present": VISUAL_CUE_PRESENT,
                    "visual_cue_duration_seconds": VISUAL_CUE_DURATION_SECONDS,
                    "eligible_for_pure_imagery_accuracy": ELIGIBLE_FOR_PURE_IMAGERY_ACCURACY,
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                },
            )
        )

    for previous, current in zip(trials, trials[1:]):
        if current.start_sample < previous.end_sample:
            raise ValueError("Imagery trials overlap")
    return tuple(trials)
