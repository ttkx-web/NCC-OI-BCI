"""Minimal HDF5 trial reader for the isolated demo replay."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from bci_dayloop.data.hdf5_dataset import EEGHDF5


@dataclass(frozen=True, slots=True)
class DemoTrial:
    samples: np.ndarray
    sample_rate: float
    channel_names: list[str]
    unit: str
    session: str
    trial_index: int


def list_demo_sessions(data_path: str | Path) -> list[str]:
    return EEGHDF5(data_path).sessions()


@lru_cache(maxsize=8)
def _load_session(data_path: str, session: str) -> tuple[np.ndarray, float, tuple[str, ...], str]:
    """Keep one demo session in memory during sequential trial replay."""
    dataset = EEGHDF5(data_path)
    payload = dataset.load(session)
    metadata = dataset.metadata
    return payload["data"], metadata.sample_rate, tuple(metadata.channel_names), metadata.unit


def load_demo_trial(data_path: str | Path, *, session: str, trial_index: int) -> DemoTrial:
    resolved_path = str(Path(data_path).expanduser().resolve())
    data, sample_rate, channel_names, unit = _load_session(resolved_path, session)
    if trial_index < 0 or trial_index >= data.shape[0]:
        raise IndexError(f"trial_index must be in [0, {data.shape[0] - 1}]")
    return DemoTrial(
        samples=data[trial_index].astype(np.float32, copy=False),
        sample_rate=sample_rate,
        channel_names=list(channel_names),
        unit=unit,
        session=session,
        trial_index=trial_index,
    )


def window_count(trial: DemoTrial, window_sec: float, step_sec: float) -> int:
    window = int(round(window_sec * trial.sample_rate))
    step = int(round(step_sec * trial.sample_rate))
    if window <= 0 or step <= 0 or step > window:
        raise ValueError("window_sec and step_sec must be positive, with step <= window")
    return max(0, 1 + (trial.samples.shape[1] - window) // step)


def trial_window(trial: DemoTrial, index: int, *, window_sec: float, step_sec: float) -> np.ndarray:
    total = window_count(trial, window_sec, step_sec)
    if index < 0 or index >= total:
        raise IndexError(f"window index must be in [0, {total - 1}]")
    start = int(round(index * step_sec * trial.sample_rate))
    stop = start + int(round(window_sec * trial.sample_rate))
    return trial.samples[:, start:stop]
