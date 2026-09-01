from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from bci_dayloop.benchmarking.windows import (
    BenchmarkWindow,
    WindowProvider,
)
from bci_dayloop.runtime.model import RuntimeModel


@dataclass(frozen=True, slots=True)
class WindowBenchmarkRecord:
    """一个已完成模型预测的滑窗基准记录。"""

    # 滑窗来源与定位。
    source_mode: str
    sequence_index: int
    window_id: str
    trial_id: int | None
    source_start_sample: int
    source_end_sample_exclusive: int

    # 预测输出。
    prediction: int
    confidence: float
    probabilities: list[float]

    # 纯模型侧耗时。
    preprocessing_ms: float
    inference_ms: float
    output_materialization_ms: float
    compute_total_ms: float

    # 真实设备模式可用；Replay 纯计算 benchmark 中均为 None。
    window_ready_at_monotonic: float | None = None
    last_sample_received_at_monotonic: float | None = None
    prediction_finished_at_monotonic: float | None = None

    # 真实设备模式的端到端时延。
    window_ready_to_prediction_ms: float | None = None
    last_sample_received_to_prediction_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def synchronize_device(device: torch.device) -> None:
    """
    在计时边界同步异步加速器任务。

    CPU 不需要同步；CUDA 和 MPS 需要显式同步，避免异步执行导致
    inference_ms 被低估。
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class RuntimeBenchmarkCore:
    """
    与数据来源无关的 RuntimeModel benchmark 核心。

    ReplayWindowProvider 和未来的 DeviceWindowProvider 都只需要实现
    WindowProvider，并持续产出 BenchmarkWindow。
    """

    def __init__(
        self,
        *,
        runtime_model: RuntimeModel,
        device: str | torch.device,
    ) -> None:
        self.runtime_model = runtime_model
        self.device = torch.device(device)

    def _run_one(
        self,
        item: BenchmarkWindow,
    ) -> WindowBenchmarkRecord:
        # 不计入 provider 取数、模型加载、CSV 写入等非模型侧开销。
        synchronize_device(self.device)
        total_started = time.perf_counter()

        preprocessing_started = time.perf_counter()
        prepared = self.runtime_model.prepare(item.raw_window)
        synchronize_device(self.device)
        preprocessing_finished = time.perf_counter()

        if item.prepare_validator is not None:
            item.prepare_validator(prepared)

        inference_started = time.perf_counter()
        output = self.runtime_model.predict_prepared(prepared)
        synchronize_device(self.device)
        inference_finished = time.perf_counter()

        # 强制将概率搬到 CPU，确保 CUDA/MPS 的实际计算和结果传输完成。
        materialization_started = time.perf_counter()

        probabilities_tensor = output.probabilities.detach()

        if probabilities_tensor.ndim == 2:
            if probabilities_tensor.shape[0] != 1:
                raise ValueError(
                    "One BenchmarkWindow must produce exactly one prediction, "
                    f"but got probability shape "
                    f"{tuple(probabilities_tensor.shape)} for "
                    f"window_id={item.window_id!r}."
                )
            probabilities_tensor = probabilities_tensor[0]
        elif probabilities_tensor.ndim != 1:
            raise ValueError(
                "Model probabilities must have shape [classes] or "
                f"[1, classes], got {tuple(probabilities_tensor.shape)} "
                f"for window_id={item.window_id!r}."
            )

        probabilities_array = (
            probabilities_tensor
            .to(device="cpu", dtype=torch.float32)
            .numpy()
            .reshape(-1)
        )

        synchronize_device(self.device)
        prediction_finished = time.perf_counter()

        if probabilities_array.size == 0:
            raise ValueError(
                f"Window {item.window_id!r} produced empty probabilities."
            )

        if not np.isfinite(probabilities_array).all():
            raise ValueError(
                f"Window {item.window_id!r} produced NaN/Inf probabilities."
            )

        prediction_finished_at_monotonic: float | None = None
        window_ready_to_prediction_ms: float | None = None
        last_sample_received_to_prediction_ms: float | None = None

        # Replay 不提供这些时间戳，因此只统计 compute latency。
        # 真实设备 provider 必须使用 time.perf_counter() 提供时间戳，
        # 以保证和这里的时钟来源一致。
        if item.window_ready_at_monotonic is not None:
            prediction_finished_at_monotonic = prediction_finished
            window_ready_to_prediction_ms = (
                prediction_finished_at_monotonic
                - item.window_ready_at_monotonic
            ) * 1000.0

        if item.last_sample_received_at_monotonic is not None:
            if prediction_finished_at_monotonic is None:
                prediction_finished_at_monotonic = prediction_finished

            last_sample_received_to_prediction_ms = (
                prediction_finished_at_monotonic
                - item.last_sample_received_at_monotonic
            ) * 1000.0

        return WindowBenchmarkRecord(
            source_mode=item.source_mode,
            sequence_index=item.sequence_index,
            window_id=item.window_id,
            trial_id=item.trial_id,
            source_start_sample=item.source_start_sample,
            source_end_sample_exclusive=item.source_end_sample_exclusive,
            prediction=int(output.predicted_class),
            confidence=float(output.confidence),
            probabilities=[
                float(value)
                for value in probabilities_array.tolist()
            ],
            preprocessing_ms=(
                preprocessing_finished - preprocessing_started
            ) * 1000.0,
            inference_ms=(
                inference_finished - inference_started
            ) * 1000.0,
            output_materialization_ms=(
                prediction_finished - materialization_started
            ) * 1000.0,
            compute_total_ms=(
                prediction_finished - total_started
            ) * 1000.0,
            window_ready_at_monotonic=(
                item.window_ready_at_monotonic
            ),
            last_sample_received_at_monotonic=(
                item.last_sample_received_at_monotonic
            ),
            prediction_finished_at_monotonic=(
                prediction_finished_at_monotonic
            ),
            window_ready_to_prediction_ms=(
                window_ready_to_prediction_ms
            ),
            last_sample_received_to_prediction_ms=(
                last_sample_received_to_prediction_ms
            ),
        )

    def run(
        self,
        *,
        provider: WindowProvider,
        warmup_windows: int,
        measured_windows: int,
    ) -> list[WindowBenchmarkRecord]:
        if warmup_windows < 0:
            raise ValueError(
                f"warmup_windows must be >= 0, got {warmup_windows}."
            )

        if measured_windows <= 0:
            raise ValueError(
                f"measured_windows must be > 0, got {measured_windows}."
            )

        records: list[WindowBenchmarkRecord] = []
        required_provider_windows = (
            warmup_windows + measured_windows
        )

        iterator = iter(provider)
        try:
            for provider_index, item in enumerate(iterator):
                record = self._run_one(item)

            # Warmup 已实际执行，但不写入 latency 统计。
                if provider_index < warmup_windows:
                    continue

                records.append(record)

                if len(records) >= measured_windows:
                    break
        finally:
            close = getattr(provider, "close", None)
            if callable(close):
                close()

        if len(records) != measured_windows:
            raise RuntimeError(
                "Window provider ended before enough windows were collected: "
                f"required_provider_windows={required_provider_windows}, "
                f"warmup_windows={warmup_windows}, "
                f"measured_windows={measured_windows}, "
                f"collected_measured_windows={len(records)}."
            )

        return records
