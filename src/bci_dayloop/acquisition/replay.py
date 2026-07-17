from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from bci_dayloop.acquisition.base import AbstractAcquirer, AcquirerMetadata, EEGChunk
from bci_dayloop.data.hdf5_dataset import EEGHDF5


class ReplayAcquirer(AbstractAcquirer):
    """Replay one HDF5 session as a paced continuous EEG stream."""

    def __init__(
        self,
        data_path: str | Path,
        session: str,
        speed: float = 1.0,
        loop: bool = False,
        window_sec: float = 4.0,
        step_sec: float = 0.5,
    ) -> None:
        if speed <= 0:
            raise ValueError("Replay speed must be greater than zero")
        if window_sec <= 0 or step_sec <= 0:
            raise ValueError("window_sec and step_sec must be greater than zero")
        self.dataset = EEGHDF5(data_path)
        loaded = self.dataset.load(session)
        info = self.dataset.metadata
        self.session = session
        self.speed = float(speed)
        self.loop = bool(loop)
        self.window_sec = float(window_sec)
        self.step_sec = float(step_sec)
        self._trials = loaded["data"]
        self._labels = loaded["labels"]
        self._trial_ids = loaded["trial_ids"]
        self._samples_per_trial = self._trials.shape[-1]
        self._stream = self._trials.transpose(1, 0, 2).reshape(self._trials.shape[1], -1)
        self._sample_labels = np.repeat(self._labels, self._samples_per_trial)
        self._sample_trial_ids = np.repeat(self._trial_ids, self._samples_per_trial)
        self._cursor = 0
        self._running = False
        self._buffer = np.empty((self._stream.shape[0], 0), dtype=np.float32)
        self.metadata = AcquirerMetadata("replay", info.sample_rate, info.channel_names, info.unit)
        self.current_label: int | None = None
        self.current_trial_id: int | None = None

    @property
    def exhausted(self) -> bool:
        return not self.loop and self._cursor >= self._stream.shape[1]

    def start_stream(self) -> None:
        self._running = True
        self._cursor = 0
        self._buffer = np.empty((self.metadata.n_channels, 0), dtype=np.float32)
        self.current_label = None
        self.current_trial_id = None

    def stop_stream(self) -> None:
        self._running = False

    def _take(self, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        chunks: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        trials: list[np.ndarray] = []
        remaining = count
        while remaining > 0:
            available = self._stream.shape[1] - self._cursor
            if available <= 0:
                if not self.loop:
                    break
                self._cursor = 0
                available = self._stream.shape[1]
            take = min(remaining, available)
            slc = slice(self._cursor, self._cursor + take)
            chunks.append(self._stream[:, slc])
            labels.append(self._sample_labels[slc])
            trials.append(self._sample_trial_ids[slc])
            self._cursor += take
            remaining -= take
        if not chunks:
            return (
                np.empty((self.metadata.n_channels, 0), dtype=np.float32),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )
        return np.concatenate(chunks, axis=1), np.concatenate(labels), np.concatenate(trials)

    def get_new_samples(self) -> EEGChunk:
        if not self._running:
            raise RuntimeError("Replay stream is not running; call start_stream() first")
        count = max(1, round(self.step_sec * self.metadata.sample_rate))
        started = time.perf_counter()
        samples, labels, trials = self._take(count)
        if samples.shape[1] == 0:
            return samples, np.empty(0, dtype=np.float64)
        target_delay = samples.shape[1] / self.metadata.sample_rate / self.speed
        elapsed = time.perf_counter() - started
        if target_delay > elapsed:
            time.sleep(target_delay - elapsed)
        end_index = self._cursor
        start_index = end_index - samples.shape[1]
        timestamps = np.arange(start_index, end_index, dtype=np.float64) / self.metadata.sample_rate
        self.current_label = int(labels[-1])
        self.current_trial_id = int(trials[-1])
        self._buffer = np.concatenate((self._buffer, samples), axis=1)
        max_buffer = max(round(self.window_sec * self.metadata.sample_rate), count) * 2
        self._buffer = self._buffer[:, -max_buffer:]
        return samples, timestamps

    def get_chunk(self, window_sec: float | None = None) -> EEGChunk:
        seconds = self.window_sec if window_sec is None else float(window_sec)
        needed = round(seconds * self.metadata.sample_rate)
        while self._buffer.shape[1] < needed and not self.exhausted:
            samples, _ = self.get_new_samples()
            if samples.shape[1] == 0:
                break
        if self._buffer.shape[1] < needed:
            return np.empty((self.metadata.n_channels, 0), dtype=np.float32), np.empty(0, dtype=np.float64)
        chunk = self._buffer[:, -needed:].copy()
        end = self._cursor / self.metadata.sample_rate
        timestamps = end - np.arange(needed, 0, -1, dtype=np.float64) / self.metadata.sample_rate
        return chunk, timestamps

