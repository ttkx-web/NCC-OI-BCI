from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import mne
import numpy as np

from bci_dayloop.data.hdf5_dataset import HDF5Metadata, write_hdf5

LOGGER = logging.getLogger(__name__)


def _apply_moabb_windows_path_fix() -> None:
    """Work around MOABB 1.5 sanitizing the colon in an absolute drive path."""
    if os.name != "nt":
        return
    from moabb.datasets import download as moabb_download

    table = {ord(char): "-" for char in ':*?"<>|'}

    def sanitize_without_drive(path: Path) -> Path:
        value = Path(path)
        parts = value.parts
        if value.anchor:
            return Path(value.anchor, *(part.translate(table) for part in parts[1:]))
        return Path(*(part.translate(table) for part in parts))

    moabb_download._sanitize_path = sanitize_without_drive


def prepare_bnci2014_001_subject(
    subject: int,
    output_path: str | Path,
    *,
    trial_tmin_sec: float = 2.0,
    trial_tmax_sec: float = 6.0,
) -> Path:
    """Download through MOABB and write unstandardized EEG trials in volts."""
    from moabb.datasets import BNCI2014_001

    _apply_moabb_windows_path_fix()
    dataset = BNCI2014_001()
    nested = None
    for attempt in range(1, 4):
        try:
            nested = dataset.get_data(subjects=[subject])
            break
        except Exception as exc:  # MOABB wraps requests/pooch errors inconsistently.
            if attempt == 3:
                raise
            LOGGER.warning("BNCI download/load attempt %d/3 failed: %s", attempt, exc)
            time.sleep(float(attempt))
    assert nested is not None
    sessions = nested[subject]
    class_names = list(dataset.event_id)
    expected_codes = {value: index for index, value in enumerate(dataset.event_id.values())}
    all_data: list[np.ndarray] = []
    labels: list[int] = []
    session_ids: list[str] = []
    trial_ids: list[int] = []
    channel_names: list[str] | None = None
    sample_rate: float | None = None
    trial_counter = 0

    for session_name in sorted(sessions):
        for run_name in sorted(sessions[session_name]):
            raw = sessions[session_name][run_name]
            picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude=[])
            names = [raw.ch_names[index] for index in picks]
            if channel_names is None:
                channel_names = names
                sample_rate = float(raw.info["sfreq"])
            elif names != channel_names:
                raise ValueError(f"EEG channel order changed in {session_name}/{run_name}")
            events = mne.find_events(raw, stim_channel="STI", shortest_event=1, verbose=False)
            sfreq = float(raw.info["sfreq"])
            for event_sample, _, event_code in events:
                if int(event_code) not in expected_codes:
                    continue
                start = int(round(event_sample + trial_tmin_sec * sfreq))
                stop = int(round(event_sample + trial_tmax_sec * sfreq))
                trial = raw.get_data(picks=picks, start=start, stop=stop)
                expected_samples = int(round((trial_tmax_sec - trial_tmin_sec) * sfreq))
                if trial.shape[-1] != expected_samples:
                    continue
                all_data.append(trial.astype(np.float32, copy=False))
                labels.append(expected_codes[int(event_code)])
                session_ids.append(session_name)
                trial_ids.append(trial_counter)
                trial_counter += 1

    if not all_data or channel_names is None or sample_rate is None:
        raise RuntimeError("No motor-imagery trials were extracted from BNCI2014_001")
    return write_hdf5(
        output_path,
        np.stack(all_data),
        np.asarray(labels, dtype=np.int64),
        np.full(len(labels), subject, dtype=np.int64),
        session_ids,
        np.asarray(trial_ids, dtype=np.int64),
        HDF5Metadata(sample_rate, channel_names, class_names, "V", "BNCI2014_001"),
    )
