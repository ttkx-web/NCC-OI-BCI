from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from bci_dayloop.acquisition.base import AbstractAcquirer
from bci_dayloop.control.commands import command_for_prediction
from bci_dayloop.inference.observability import JsonlWindowLogger, LatencyBreakdown, PipelineRunStats
from bci_dayloop.models.base import BaseModelAdapter, ModelPreprocessor, add_batch_dimension


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

    def __post_init__(self) -> None:
        if self.total_latency_ms == 0.0 and self.latency_ms != 0.0:
            object.__setattr__(self, "total_latency_ms", float(self.latency_ms))
        if self.latency_ms != self.total_latency_ms:
            raise ValueError("latency_ms must equal total_latency_ms")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SlidingWindowDecoder:
    def __init__(
        self,
        model: BaseModelAdapter,
        preprocessor: ModelPreprocessor,
        class_names: list[str],
        *,
        sample_rate: float,
        input_unit: str,
        window_sec: float = 4.0,
        step_sec: float = 0.5,
        confidence_threshold: float = 0.55,
        command_map: dict[str, str] | None = None,
        run_stats: PipelineRunStats | None = None,
        jsonl_logger: JsonlWindowLogger | None = None,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.class_names = list(class_names)
        self.sample_rate = float(sample_rate)
        self.input_unit = input_unit
        self.window_samples = round(window_sec * sample_rate)
        self.step_samples = round(step_sec * sample_rate)
        self.confidence_threshold = float(confidence_threshold)
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
        chunk = np.asarray(samples, dtype=np.float32)
        if chunk.ndim != 2:
            raise ValueError(f"Expected samples [C,T], got {chunk.shape}")
        if self.run_stats is not None:
            self.run_stats.record_chunk()
        self._buffer = chunk.copy() if self._buffer is None else np.concatenate((self._buffer, chunk), axis=1)
        self._buffer = self._buffer[:, -self.window_samples :]
        self._new_since_decode += chunk.shape[1]
        if self._buffer.shape[1] < self.window_samples or self._new_since_decode < self.step_samples:
            return None
        self._new_since_decode %= self.step_samples
        self._window_id += 1
        window_id = self._window_id
        total_started = time.perf_counter()
        try:
            preprocessing_started = time.perf_counter()
            model_input = self.preprocessor.transform(
                self._buffer,
                self.sample_rate,
                self.input_unit,
                reshape=True,
            )
            preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000.0
            model_started = time.perf_counter()
            probabilities = self.model.predict_proba(add_batch_dimension(model_input))[0]
            model_ms = (time.perf_counter() - model_started) * 1000.0
        except Exception as error:
            if self.run_stats is not None:
                self.run_stats.record_failure()
            if self.jsonl_logger is not None:
                try:
                    self.jsonl_logger.log_error(window_id=window_id, error=error)
                except Exception as logger_error:
                    raise error from logger_error
            raise
        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])
        prediction = self.class_names[class_id]
        command = command_for_prediction(prediction, confidence, self.confidence_threshold, self.command_map)
        total_ms = (time.perf_counter() - total_started) * 1000.0
        result = DecodeResult(
            prediction,
            confidence,
            total_ms,
            command,
            class_id,
            probabilities.tolist(),
            trial_id,
            expected_class_id,
            preprocessing_ms,
            model_ms,
            total_ms,
        )
        if self.run_stats is not None:
            self.run_stats.record_success(
                LatencyBreakdown(preprocessing_ms, model_ms, total_ms)
            )
        if self.jsonl_logger is not None:
            self.jsonl_logger.log_success(window_id=window_id, result=result)
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

