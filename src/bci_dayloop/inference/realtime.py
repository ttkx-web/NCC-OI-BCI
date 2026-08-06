from __future__ import annotations

import time
from dataclasses import (
    asdict,
    dataclass,
    field,
)
from typing import Protocol

import numpy as np

from bci_dayloop.acquisition.base import AbstractAcquirer
from bci_dayloop.control.commands import command_for_prediction
from bci_dayloop.inference.observability import JsonlWindowLogger, LatencyBreakdown, PipelineRunStats
from collections.abc import Callable, Iterator, Sequence

from bci_dayloop.runtime.model import (
    RuntimeModel,
)
from bci_dayloop.runtime.types import (
    RawEEGWindow,
)


class StopEvent(Protocol):
    def is_set(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class DecodeResult:
    prediction: str
    confidence: float
    latency_ms: float
    command: str
    class_id: int
    probabilities: list[float]

    trial_id: int | None = None
    expected_class_id: int | None = None

    preprocessing_latency_ms: float = 0.0
    model_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    preprocessing_trace: tuple[str, ...] = ()
    preprocessing_diagnostics: dict[
        str,
        object,
    ] = field(default_factory=dict)

    model_diagnostics: dict[
        str,
        object,
    ] = field(default_factory=dict)

    model_revision: str = "base"
    online_update_step: int = 0
    online_update_applied: bool = False


class SlidingWindowDecoder:
    def __init__(
        self,
        runtime_model: RuntimeModel,
        class_names: Sequence[str],
        channel_names: Sequence[str],
        *,
        sample_rate: float,
        input_unit: str,
        window_sec: float | None = None,
        step_sec: float = 0.5,
        confidence_threshold: float = 0.55,
        command_map: dict[str, str] | None = None,
        run_stats: PipelineRunStats | None = None,
        jsonl_logger: JsonlWindowLogger | None = None,
    ) -> None:
        self.runtime_model = runtime_model

        self.class_names = tuple(
            str(name) for name in class_names
        )

        self.channel_names = tuple(
            str(name) for name in channel_names
        )

        self.sample_rate = float(sample_rate)
        self.input_unit = str(input_unit)

        if self.sample_rate <= 0:
            raise ValueError(
                "sample_rate must be positive."
            )

        if not self.channel_names:
            raise ValueError(
                "channel_names cannot be empty."
            )

        contract_window_sec = float(
            runtime_model
            .input_contract
            .window_sec
        )

        if window_sec is None:
            resolved_window_sec = (
                contract_window_sec
            )
        else:
            resolved_window_sec = float(
                window_sec
            )

            if not np.isclose(
                resolved_window_sec,
                contract_window_sec,
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError(
                    "Sliding-window duration does not match "
                    "the Runtime Model Package: "
                    f"decoder={resolved_window_sec}, "
                    f"package={contract_window_sec}."
                )

        if step_sec <= 0:
            raise ValueError(
                "step_sec must be positive."
            )

        if step_sec > resolved_window_sec:
            raise ValueError(
                "step_sec must not exceed window_sec."
            )

        self.window_sec = resolved_window_sec
        self.step_sec = float(step_sec)

        # 注意：这里使用数据源采样率切原始窗口，
        # 不是模型目标采样率。
        self.window_samples = int(
            round(
                self.window_sec
                * self.sample_rate
            )
        )

        self.step_samples = int(
            round(
                self.step_sec
                * self.sample_rate
            )
        )

        self.confidence_threshold = float(
            confidence_threshold
        )

        self.command_map = command_map
        self.run_stats = run_stats
        self.jsonl_logger = jsonl_logger

        self._buffer: np.ndarray | None = None
        self._new_since_decode = 0
        self._window_id = 0

    def reset(self) -> None:
        self._buffer = None
        self._new_since_decode = 0
        self._window_id = 0

    def push(
        self,
        samples: np.ndarray,
        *,
        trial_id: int | None = None,
        expected_class_id: int | None = None,
    ) -> DecodeResult | None:
        chunk = np.asarray(
            samples,
            dtype=np.float32,
        )

        if chunk.ndim != 2:
            raise ValueError(
                f"Expected samples [C, T], got {chunk.shape}."
            )

        if chunk.shape[0] != len(self.channel_names):
            raise ValueError(
                "Incoming EEG channel count does not match "
                "channel_names: "
                f"data={chunk.shape[0]}, "
                f"names={len(self.channel_names)}."
            )

        if self.run_stats is not None:
            self.run_stats.record_chunk()

        if self._buffer is None:
            self._buffer = chunk.copy()
        else:
            self._buffer = np.concatenate(
                (self._buffer, chunk),
                axis=1,
            )

        # 只保留生成当前窗口所需的最近数据。
        self._buffer = self._buffer[
                       :,
                       -self.window_samples:,
                       ]

        self._new_since_decode += chunk.shape[1]

        if (
                self._buffer.shape[1] < self.window_samples
                or self._new_since_decode < self.step_samples
        ):
            return None

        self._new_since_decode %= self.step_samples
        self._window_id += 1
        window_id = self._window_id

        total_started = time.perf_counter()

        try:
            raw_window = RawEEGWindow(
                data=self._buffer.copy(),
                channel_names=list(
                    self.channel_names
                ),
                sample_rate=self.sample_rate,
                unit=self.input_unit,
                layout="CT",
                trial_id=(
                    str(trial_id)
                    if trial_id is not None
                    else None
                ),
                window_id=str(window_id),
                label=expected_class_id,
                metadata={
                    "source": (
                        "sliding_window_decoder"
                    ),
                },
            )

            preprocessing_started = (
                time.perf_counter()
            )

            prepared = self.runtime_model.prepare(
                raw_window
            )

            preprocessing_ms = (
                time.perf_counter()
                - preprocessing_started
            ) * 1000.0

            model_started = time.perf_counter()

            output = (
                self.runtime_model.predict_prepared(
                    prepared,
                    return_features=False,
                )
            )

            preprocessing_trace = tuple(
                str(step)
                for step in prepared.preprocessing_trace
            )

            preprocessing_diagnostics = dict(
                prepared.diagnostics
            )

            model_diagnostics = dict(
                output.diagnostics
            )

            model_ms = (
                time.perf_counter()
                - model_started
            ) * 1000.0

            probability_tensor = (
                output.probabilities
                .detach()
                .cpu()
            )

            if probability_tensor.ndim == 2:
                if probability_tensor.shape[0] != 1:
                    raise RuntimeError(
                        "SlidingWindowDecoder expects "
                        "one prediction window, but got "
                        f"batch_size="
                        f"{probability_tensor.shape[0]}."
                    )

                probability_tensor = (
                    probability_tensor[0]
                )

            if probability_tensor.ndim != 1:
                raise RuntimeError(
                    "Expected class probabilities "
                    "with shape [classes], got "
                    f"{tuple(probability_tensor.shape)}."
                )

            probabilities = (
                probability_tensor
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            if probabilities.shape[0] != len(
                self.class_names
            ):
                raise RuntimeError(
                    "Probability count does not match "
                    "class_names: "
                    f"probabilities="
                    f"{probabilities.shape[0]}, "
                    f"classes="
                    f"{len(self.class_names)}."
                )

            class_id = int(
                output.predicted_class
            )

            if not 0 <= class_id < len(
                self.class_names
            ):
                raise RuntimeError(
                    f"Predicted class {class_id} "
                    "is outside class_names range."
                )

            confidence = float(
                output.confidence
            )

            prediction = self.class_names[
                class_id
            ]

            command = command_for_prediction(
                prediction,
                confidence,
                self.confidence_threshold,
                self.command_map,
            )

            total_ms = (
                time.perf_counter()
                - total_started
            ) * 1000.0

            result = DecodeResult(
                prediction=prediction,
                confidence=confidence,
                latency_ms=total_ms,
                command=command,
                class_id=class_id,
                probabilities=probabilities.tolist(),
                trial_id=trial_id,
                expected_class_id=expected_class_id,
                preprocessing_latency_ms=(
                    preprocessing_ms
                ),
                model_latency_ms=model_ms,
                total_latency_ms=total_ms,

                preprocessing_trace=(
                    preprocessing_trace
                ),
                preprocessing_diagnostics=(
                    preprocessing_diagnostics
                ),
                model_diagnostics=(
                    model_diagnostics
                ),
            )

        except Exception as error:
            if self.run_stats is not None:
                self.run_stats.record_failure()

            if self.jsonl_logger is not None:
                try:
                    self.jsonl_logger.log_error(
                        window_id=window_id,
                        error=error,
                    )
                except Exception as logger_error:
                    raise error from logger_error

            raise

        if self.run_stats is not None:
            self.run_stats.record_success(
                LatencyBreakdown(
                    preprocessing_ms=(
                        preprocessing_ms
                    ),
                    model_ms=model_ms,
                    total_ms=total_ms,
                )
            )

        if self.jsonl_logger is not None:
            self.jsonl_logger.log_success(
                window_id=window_id,
                result=result,
            )

        return result

    def run(
        self,
        acquirer: AbstractAcquirer,
        *,
        max_windows: int | None = None,
        callback: Callable[[DecodeResult, np.ndarray], None] | None = None,
        stop_event: StopEvent | None = None,
    ) -> Iterator[DecodeResult]:
        self.reset()
        if self.run_stats is not None:
            self.run_stats.start()
        acquirer.start_stream()
        emitted = 0
        try:
            while max_windows is None or emitted < max_windows:
                if stop_event is not None and stop_event.is_set():
                    break
                samples, _ = acquirer.get_new_samples()
                if stop_event is not None and stop_event.is_set():
                    break
                if samples.shape[1] == 0:
                    break
                if stop_event is not None and stop_event.is_set():
                    break
                result = self.push(
                    samples,
                    trial_id=getattr(acquirer, "current_trial_id", None),
                    expected_class_id=getattr(acquirer, "current_label", None),
                )
                if result is None:
                    continue
                emitted += 1
                if callback is not None:
                    callback(result, samples)
                if stop_event is not None and stop_event.is_set():
                    break
                yield result
        finally:
            acquirer.stop_stream()

