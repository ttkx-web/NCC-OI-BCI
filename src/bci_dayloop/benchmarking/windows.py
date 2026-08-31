from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from bci_dayloop.data.hdf5_dataset import EEGHDF5
from bci_dayloop.runtime.types import RawEEGWindow


@dataclass(frozen=True, slots=True)
class BenchmarkWindow:
    """Provider 产出的、尚未送入 RuntimeModel 的原始滑窗。"""

    raw_window: RawEEGWindow

    source_mode: str
    sequence_index: int
    window_id: str

    source_start_sample: int
    source_end_sample_exclusive: int
    trial_id: int | None = None

    # 真实时 provider 使用 host 的 time.perf_counter() 填充；
    # Replay compute benchmark 保持 None。
    window_ready_at_monotonic: float | None = None
    last_sample_received_at_monotonic: float | None = None


class WindowProvider(Protocol):
    """持续提供 BenchmarkWindow 的统一接口。"""

    def __iter__(self) -> Iterator[BenchmarkWindow]:
        ...


class ReplayWindowProvider:
    """
    无 sleep 的 HDF5 伪实时滑窗提供器。

    它保持与 ReplayAcquirer 一致的连续流语义，但不引入 replay_speed
    的等待时间，因此用于测量模型计算延迟。
    """

    def __init__(
        self,
        *,
        data_path: str | Path,
        session: str,
        window_sec: float,
        step_sec: float,
        maximum_windows: int | None = None,
        first_end_sample: int | None = None,
    ) -> None:
        if window_sec <= 0:
            raise ValueError("window_sec must be positive.")

        if step_sec <= 0:
            raise ValueError("step_sec must be positive.")

        if step_sec > window_sec:
            raise ValueError(
                "step_sec must not exceed window_sec."
            )

        if maximum_windows is not None and maximum_windows <= 0:
            raise ValueError(
                "maximum_windows must be positive or None."
            )

        self.data_path = Path(data_path)
        self.session = str(session)
        self.window_sec = float(window_sec)
        self.step_sec = float(step_sec)
        self.maximum_windows = maximum_windows
        self.first_end_sample = first_end_sample

    def __iter__(self) -> Iterator[BenchmarkWindow]:
        dataset = EEGHDF5(self.data_path)
        metadata = dataset.metadata
        loaded = dataset.load(self.session)

        trials = np.asarray(
            loaded["data"],
            dtype=np.float32,
        )
        labels = np.asarray(
            loaded["labels"],
            dtype=np.int64,
        )
        trial_ids = np.asarray(
            loaded["trial_ids"],
            dtype=np.int64,
        )

        if trials.ndim != 3:
            raise ValueError(
                "Replay HDF5 data must have shape [N, C, T], "
                f"got {trials.shape}."
            )

        if labels.shape != (trials.shape[0],):
            raise ValueError(
                "Replay HDF5 labels do not match trial count."
            )

        if trial_ids.shape != (trials.shape[0],):
            raise ValueError(
                "Replay HDF5 trial_ids do not match trial count."
            )

        sample_rate = float(metadata.sample_rate)
        window_samples = int(
            round(self.window_sec * sample_rate)
        )
        step_samples = int(
            round(self.step_sec * sample_rate)
        )

        if window_samples <= 0 or step_samples <= 0:
            raise ValueError(
                "window_sec / step_sec produce zero samples."
            )

        # 与 ReplayAcquirer 的连续数据流语义保持一致。
        stream = np.ascontiguousarray(
            trials.transpose(1, 0, 2).reshape(
                trials.shape[1],
                -1,
            ),
            dtype=np.float32,
        )

        samples_per_trial = int(trials.shape[-1])

        sample_labels = np.repeat(
            labels,
            samples_per_trial,
        )
        sample_trial_ids = np.repeat(
            trial_ids,
            samples_per_trial,
        )

        if stream.shape[1] < window_samples:
            raise ValueError(
                f"Replay stream is only "
                f"{stream.shape[1] / sample_rate:.3f}s, "
                f"shorter than requested window "
                f"{self.window_sec:.3f}s."
            )

        # 为不同窗口长度的模型对齐同一个“决策终点”。
        first_end_sample = (
            window_samples
            if self.first_end_sample is None
            else int(self.first_end_sample)
        )

        if first_end_sample < window_samples:
            raise ValueError(
                "first_end_sample must be >= window_samples."
            )

        if first_end_sample > stream.shape[1]:
            raise ValueError(
                "first_end_sample exceeds replay stream length: "
                f"{first_end_sample} > {stream.shape[1]}."
            )

        emitted = 0

        for end_sample in range(
            first_end_sample,
            stream.shape[1] + 1,
            step_samples,
        ):
            start_sample = end_sample - window_samples

            raw_data = np.ascontiguousarray(
                stream[:, start_sample:end_sample],
                dtype=np.float32,
            )

            source_ids = np.unique(
                sample_trial_ids[start_sample:end_sample]
            )
            source_labels = np.unique(
                sample_labels[start_sample:end_sample]
            )

            # 跨 trial 的滑窗不伪造唯一 trial / label。
            trial_id = (
                int(source_ids[0])
                if len(source_ids) == 1
                else None
            )
            label = (
                int(source_labels[0])
                if len(source_labels) == 1
                else None
            )

            sequence_index = emitted + 1
            window_id = (
                f"replay:{self.session}:"
                f"{sequence_index:08d}:"
                f"{start_sample}:{end_sample}"
            )

            raw_window = RawEEGWindow(
                data=raw_data,
                channel_names=[
                    str(name)
                    for name in metadata.channel_names
                ],
                sample_rate=sample_rate,
                unit=str(metadata.unit),
                layout="CT",
                start_time_sec=start_sample / sample_rate,
                trial_id=(
                    str(trial_id)
                    if trial_id is not None
                    else None
                ),
                window_id=window_id,
                label=label,
                metadata={
                    "source": "replay_window_provider",
                    "session": self.session,
                    "source_start_sample": start_sample,
                    "source_end_sample_exclusive": end_sample,
                    "source_trial_ids": [
                        int(value)
                        for value in source_ids.tolist()
                    ],
                    "crosses_trial_boundary": (
                        len(source_ids) > 1
                    ),
                },
            )

            yield BenchmarkWindow(
                raw_window=raw_window,
                source_mode="replay",
                sequence_index=sequence_index,
                window_id=window_id,
                source_start_sample=start_sample,
                source_end_sample_exclusive=end_sample,
                trial_id=trial_id,
            )

            emitted += 1

            if (
                self.maximum_windows is not None
                and emitted >= self.maximum_windows
            ):
                return