from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from bci_dayloop.data.hdf5_dataset import EEGHDF5
from bci_dayloop.runtime.types import RawEEGWindow
from bci_dayloop.realtime.channel_units import select_verified_eeg_channels
from bci_dayloop.realtime.contracts import EventMarker, RealtimeWindow
from bci_dayloop.realtime.neuracle_jellyfish import NeuracleJellyFishSource
from bci_dayloop.realtime.pipeline import RealtimeEEGWindowPipeline
from bci_dayloop.realtime.runtime_bridge import RealtimeRuntimeBridge


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
    prepare_validator: Callable[[object], None] | None = None


class WindowProvider(Protocol):
    """持续提供 BenchmarkWindow 的统一接口。"""

    def __iter__(self) -> Iterator[BenchmarkWindow]:
        ...


class DeviceWindowProvider:
    """Adapt the approved live source/pipeline/bridge path to benchmark windows.

    No packet decoding, channel mapping, preprocessing, or window construction
    is duplicated here.  Host-side timestamps are captured with
    :func:`time.perf_counter` only and are intentionally not persisted by this
    provider.
    """

    def __init__(
        self,
        *,
        source: NeuracleJellyFishSource,
        pipeline: RealtimeEEGWindowPipeline,
        bridge: RealtimeRuntimeBridge,
        duration_sec: float,
        maximum_windows: int | None = None,
        perf_counter: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if duration_sec <= 0:
            raise ValueError("duration_sec must be positive")
        if maximum_windows is not None and maximum_windows <= 0:
            raise ValueError("maximum_windows must be positive or None")
        import time

        self.source = source
        self.pipeline = pipeline
        self.bridge = bridge
        self.duration_sec = float(duration_sec)
        self.maximum_windows = maximum_windows
        self._perf_counter = perf_counter or time.perf_counter
        self._sleep = sleep or time.sleep
        self._closed = False
        self._started = False
        self._sequence_received_at: dict[int, float] = {}
        self._yielded_windows = 0
        self._pre_disconnect_health: dict[str, object] = {}
        self._final_health: dict[str, object] = {}
        self._pipeline_integrity: dict[str, int] = {}

    def __iter__(self) -> Iterator[BenchmarkWindow]:
        if self._started:
            raise RuntimeError("DeviceWindowProvider may be iterated only once")
        self._started = True
        self.source.connect()
        deadline = self._perf_counter() + self.duration_sec
        try:
            while self._perf_counter() < deadline:
                raw_chunk = self.source.read_chunk()
                if raw_chunk is None:
                    self._sleep(0.001)
                    continue
                received_at = self._perf_counter()
                self._sequence_received_at[raw_chunk.sequence_id] = received_at
                markers = self._drain_events()
                eeg_chunk = select_verified_eeg_channels(raw_chunk)
                results = self.pipeline.process(eeg_chunk, markers)
                for result in results:
                    if result.status != "emitted" or result.window is None:
                        raise RuntimeError(
                            "Realtime window pipeline failed: "
                            f"{result.reason or 'unknown failure'}"
                        )
                    benchmark_window = self._to_benchmark_window(
                        result.window,
                        window_ready_at=self._perf_counter(),
                    )
                    self._yielded_windows += 1
                    yield benchmark_window
                    if (
                        self.maximum_windows is not None
                        and self._yielded_windows >= self.maximum_windows
                    ):
                        return
            raise RuntimeError(
                "Device source ended before enough benchmark windows were "
                f"collected: yielded_windows={self._yielded_windows}, "
                f"maximum_windows={self.maximum_windows}."
            )
        finally:
            self.close()

    def close(self) -> None:
        """Release source and buffer; safe to call after partial iteration."""
        if self._closed:
            return
        try:
            self._pre_disconnect_health = dict(self.source.health())
        finally:
            try:
                self.source.disconnect()
            finally:
                self._final_health = dict(self.source.health())
                self._pipeline_integrity = {
                    "received_samples": int(self.pipeline.accepted_eeg_sample_count),
                    "pipeline_expected_windows": int(self.pipeline.expected_windows),
                    "pipeline_emitted_windows": int(self.pipeline.emitted_windows),
                    "pipeline_failed_windows": int(self.pipeline.failed_windows),
                    "gap_count": int(self.pipeline.timestamp_gap_count),
                    "buffer_overflow_count": int(self.pipeline.buffer_overflow_count),
                }
                self.pipeline.close()
                self._closed = True

    @property
    def source_integrity(self) -> dict[str, int]:
        health = self._pre_disconnect_health
        return {
            "received_packets": int(health.get("received_packets", 0)),
            "received_samples": self._pipeline_integrity.get("received_samples", 0),
            "pipeline_expected_windows": self._pipeline_integrity.get("pipeline_expected_windows", 0),
            "pipeline_emitted_windows": self._pipeline_integrity.get("pipeline_emitted_windows", 0),
            "pipeline_failed_windows": self._pipeline_integrity.get("pipeline_failed_windows", 0),
            "missing_packets": int(health.get("missing_packets", 0)),
            "duplicate_packets": int(health.get("duplicate_packets", 0)),
            "out_of_order_packets": int(health.get("out_of_order_packets", 0)),
            "malformed_packets": int(health.get("malformed_packets", 0)),
            "reconnect_count": int(health.get("reconnect_count", 0)),
            "gap_count": self._pipeline_integrity.get("gap_count", 0),
            "buffer_overflow_count": self._pipeline_integrity.get("buffer_overflow_count", 0),
            "dropped_window_count": self._pipeline_integrity.get("pipeline_failed_windows", 0),
        }

    @property
    def pre_disconnect_health(self) -> Mapping[str, object]:
        return dict(self._pre_disconnect_health)

    @property
    def final_health(self) -> Mapping[str, object]:
        return dict(self._final_health)

    def _drain_events(self) -> tuple[EventMarker, ...]:
        markers: list[EventMarker] = []
        marker = self.source.read_event()
        while marker is not None:
            markers.append(marker)
            marker = self.source.read_event()
        return tuple(markers)

    def _to_benchmark_window(
        self,
        window: RealtimeWindow,
        *,
        window_ready_at: float,
    ) -> BenchmarkWindow:
        last_received = self._sequence_received_at.get(
            window.source_sequence_end
        )
        if last_received is None:
            raise RuntimeError(
                "No host perf_counter timestamp for the window's last "
                "source sequence"
            )
        raw_window = self.bridge.to_raw_window(window)

        def validate_prepared(prepared: object) -> None:
            result = self.bridge.validate_prepared_window(
                window,
                prepared,  # type: ignore[arg-type]
            )
            if not result.model_input_safe:
                raise RuntimeError(
                    "Realtime prepared-input gate failed: "
                    f"{result.failure_reason or 'unknown failure'}"
                )

        return BenchmarkWindow(
            raw_window=raw_window,
            source_mode="device",
            sequence_index=self._yielded_windows + 1,
            window_id=f"device:{window.window_id}",
            source_start_sample=window.start_sample_index,
            source_end_sample_exclusive=window.end_sample_index,
            window_ready_at_monotonic=window_ready_at,
            last_sample_received_at_monotonic=last_received,
            prepare_validator=validate_prepared,
        )


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
