"""Offline reader for Neuracle recordings converted to BDF."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import hashlib

import mne
import numpy as np

from bci_dayloop.data.records import EEGEvent, RawEEGRecord, UnitEvidence


_EXCLUDED_EEG_NAMES = frozenset({"ecg", "heor", "heol", "veou", "veol"})
_MARKER_PATTERN = re.compile(r"^(?:stimulus/)?s\s*(\d+)$", re.IGNORECASE)
_MARKER_SEMANTICS: dict[int, tuple[str, str | None]] = {
    1: ("imagery", "left_hand"),
    2: ("imagery", "right_hand"),
    3: ("imagery", "feet"),
    4: ("imagery", "tongue"),
    10: ("fixation", None),
    20: ("rest", None),
    90: ("block_start", None),
    91: ("block_end", None),
    100: ("recording_start", None),
    101: ("recording_end", None),
    127: ("abort", None),
}


def parse_neuracle_marker(description: object) -> dict[str, object]:
    """Parse a Neuracle annotation description without discarding unknown events."""
    original_description = str(description)
    marker_code: int | None = None
    if isinstance(description, int) and not isinstance(description, bool):
        marker_code = description
    elif isinstance(description, str):
        normalized = description.strip()
        if normalized.isdecimal():
            marker_code = int(normalized)
        else:
            match = _MARKER_PATTERN.fullmatch(normalized)
            if match is not None:
                marker_code = int(match.group(1))

    metadata: dict[str, object] = {
        "original_description": original_description,
        "marker_code": marker_code,
    }
    if marker_code is None:
        return {
            "event_type": "custom",
            "code": original_description,
            "label": None,
            "metadata": metadata,
        }

    semantics = _MARKER_SEMANTICS.get(marker_code)
    if semantics is None:
        return {
            "event_type": "custom",
            "code": marker_code,
            "label": None,
            "metadata": metadata,
        }

    event_type, label = semantics
    if marker_code == 3:
        metadata["original_label"] = "both_feet"
    return {
        "event_type": event_type,
        "code": marker_code,
        "label": label,
        "metadata": metadata,
    }


def annotations_to_events(
    annotations: object, *, sampling_rate: float, n_times: int
) -> tuple[EEGEvent, ...]:
    """Convert MNE-style annotations to validated, semantically mapped EEG events."""
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive and finite")
    if n_times <= 0:
        raise ValueError("n_times must be positive")

    events: list[EEGEvent] = []
    for onset, duration, description in zip(
        annotations.onset, annotations.duration, annotations.description, strict=True
    ):
        try:
            onset_seconds = float(onset)
            duration_seconds = float(duration)
        except (TypeError, ValueError) as exc:
            raise ValueError("Annotation onset and duration must be finite numbers") from exc
        if not np.isfinite(onset_seconds) or not np.isfinite(duration_seconds):
            raise ValueError("Annotation onset and duration must be finite")
        if onset_seconds < 0:
            raise ValueError("Annotation onset must be non-negative")
        if duration_seconds < 0:
            raise ValueError("Annotation duration must be non-negative")

        sample_index = round(onset_seconds * sampling_rate)
        if sample_index < 0 or sample_index >= n_times:
            raise ValueError("Annotation sample_index is outside the recording")
        parsed = parse_neuracle_marker(description)
        events.append(
            EEGEvent(
                sample_index=sample_index,
                event_type=parsed["event_type"],  # type: ignore[arg-type]
                code=parsed["code"],  # type: ignore[arg-type]
                label=parsed["label"],  # type: ignore[arg-type]
                onset_seconds=onset_seconds,
                duration_seconds=duration_seconds,
                metadata=parsed["metadata"],  # type: ignore[arg-type]
            )
        )
    return tuple(events)


class NeuracleBDFReader:
    """Load a BDF file lazily, retaining only verified EEG channels."""

    reader_name = "neuracle-bdf"

    def __init__(self, unit_evidence: UnitEvidence) -> None:
        if not unit_evidence.is_model_safe:
            raise ValueError("Neuracle BDF reading requires model-safe unit evidence")
        if unit_evidence.normalized_unit != "uV":
            raise ValueError("Neuracle BDF reading requires normalized_unit 'uV'")
        self.unit_evidence = unit_evidence

    def load(
        self,
        path: str | Path,
        *,
        subject_id: str | None = None,
        session_id: str | None = None,
        device_id: str | None = None,
    ) -> RawEEGRecord:
        source_path = Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        digest = hashlib.sha256()
        with source_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        raw = mne.io.read_raw_bdf(str(source_path), preload=False, verbose="ERROR")
        channel_names = tuple(raw.ch_names)
        channel_types = tuple(raw.get_channel_types())
        if len(channel_names) != len(channel_types):
            raise ValueError("BDF channel names and types have different lengths")

        eeg_indices = [
            index
            for index, (name, channel_type) in enumerate(zip(channel_names, channel_types, strict=True))
            if channel_type == "eeg" and name.strip().lower() not in _EXCLUDED_EEG_NAMES
        ]
        if not eeg_indices:
            raise ValueError("BDF recording contains no usable EEG channels")

        selected_names = tuple(channel_names[index] for index in eeg_indices)
        excluded_names = tuple(
            name for index, name in enumerate(channel_names) if index not in eeg_indices
        )
        sampling_rate = float(raw.info["sfreq"])
        n_times = int(raw.n_times)
        eeg = raw.get_data(picks=eeg_indices, start=0, stop=n_times, units="uV")

        events = annotations_to_events(
            raw.annotations, sampling_rate=sampling_rate, n_times=n_times
        )
        metadata = {
            "source_format": "BDF",
            "conversion_tool": "unverified",
            "conversion_tool_version": None,
            "reader_name": self.reader_name,
            "reader_version": "1",
            "unit_evidence_level": self.unit_evidence.evidence_level,
            "window_semantics": "cue_plus_imagery_4s",
            "eligible_for_accuracy": False,
            "start_sample": None,
            "end_sample": None,
            "all_channel_names": channel_names,
            "all_channel_types": channel_types,
            "excluded_channel_names": excluded_names,
            "original_channel_count": int(raw.info["nchan"]),
            "eeg_channel_count": len(eeg_indices),
            "n_times": n_times,
            "measurement_date": self._serialize_measurement_date(raw.info.get("meas_date")),
        }
        timestamps = np.arange(n_times, dtype=np.float64) / sampling_rate
        return RawEEGRecord(
            eeg=eeg,
            channel_names=selected_names,
            sampling_rate=sampling_rate,
            unit_evidence=self.unit_evidence,
            timestamps=timestamps,
            events=events,
            channel_types=tuple("eeg" for _ in eeg_indices),
            subject_id=subject_id,
            session_id=session_id,
            device_id=device_id,
            source_path=source_path.name,
            source_sha256=digest.hexdigest(),
            metadata=metadata,
        )

    @staticmethod
    def _serialize_measurement_date(value: object) -> str | int | float | bool | None:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        isoformat = getattr(value, "isoformat", None)
        return isoformat() if callable(isoformat) else str(value)
