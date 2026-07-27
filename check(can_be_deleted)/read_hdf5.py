from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


@dataclass(frozen=True)
class HDF5Metadata:
    sample_rate: float
    channel_names: list[str]
    class_names: list[str]
    unit: str
    dataset_name: str


def _string_array(values: Iterable[str]) -> np.ndarray:
    return np.asarray(list(values), dtype=h5py.string_dtype(encoding="utf-8"))


def write_hdf5(
    path: str | Path,
    data: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    session_ids: Iterable[str],
    trial_ids: np.ndarray,
    metadata: HDF5Metadata,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    subject_ids = np.asarray(subject_ids, dtype=np.int64)
    trial_ids = np.asarray(trial_ids, dtype=np.int64)
    sessions = list(session_ids)
    n_trials = len(data)
    if data.ndim != 3:
        raise ValueError(f"data must have shape [N,C,T], got {data.shape}")
    if not all(len(v) == n_trials for v in (labels, subject_ids, sessions, trial_ids)):
        raise ValueError("All trial-level arrays must have equal length")
    if data.shape[1] != len(metadata.channel_names):
        raise ValueError("channel_names length does not match data channel dimension")

    with h5py.File(target, "w") as handle:
        handle.create_dataset("data", data=data, dtype="float32", compression="gzip", shuffle=True)
        handle.create_dataset("labels", data=labels, dtype="int64")
        handle.create_dataset("subject_ids", data=subject_ids, dtype="int64")
        handle.create_dataset("session_ids", data=_string_array(sessions))
        handle.create_dataset("trial_ids", data=trial_ids, dtype="int64")
        handle.attrs["sample_rate"] = float(metadata.sample_rate)
        handle.attrs["channel_names"] = json.dumps(metadata.channel_names, ensure_ascii=False)
        handle.attrs["class_names"] = json.dumps(metadata.class_names, ensure_ascii=False)
        handle.attrs["unit"] = metadata.unit
        handle.attrs["dataset_name"] = metadata.dataset_name
    return target


class EEGHDF5:
    """Small, process-safe HDF5 reader that opens the file per operation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"EEG HDF5 not found: {self.path}")

    @property
    def metadata(self) -> HDF5Metadata:
        with h5py.File(self.path, "r") as handle:
            return HDF5Metadata(
                sample_rate=float(handle.attrs["sample_rate"]),
                channel_names=json.loads(handle.attrs["channel_names"]),
                class_names=json.loads(handle.attrs["class_names"]),
                unit=str(handle.attrs["unit"]),
                dataset_name=str(handle.attrs["dataset_name"]),
            )

    def sessions(self) -> list[str]:
        with h5py.File(self.path, "r") as handle:
            values = handle["session_ids"].asstr()[:]
        return sorted(set(values.tolist()))

    def load(self, session: str | None = None) -> dict[str, np.ndarray]:
        with h5py.File(self.path, "r") as handle:
            sessions = handle["session_ids"].asstr()[:]
            mask = np.ones(len(sessions), dtype=bool) if session is None else sessions == session
            if not np.any(mask):
                raise ValueError(f"Session '{session}' not found. Available: {sorted(set(sessions))}")
            indices = np.flatnonzero(mask)
            return {
                "data": handle["data"][indices].astype(np.float32, copy=False),
                "labels": handle["labels"][indices].astype(np.int64, copy=False),
                "subject_ids": handle["subject_ids"][indices].astype(np.int64, copy=False),
                "session_ids": sessions[indices],
                "trial_ids": handle["trial_ids"][indices].astype(np.int64, copy=False),
            }



dataset = EEGHDF5("data/processed/bnci2014_001_s01.h5")
metadata = dataset.metadata

print("通道数量:", len(metadata.channel_names))
print("通道名称与顺序:", metadata.channel_names)
print("原始采样率:", metadata.sample_rate)
print("原始单位:", metadata.unit)

session = dataset.load("1test")
print("原始数据 shape:", session["data"].shape)
print("原始数据 dtype:", session["data"].dtype)