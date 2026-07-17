from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass

import numpy as np

from bci_dayloop.acquisition.base import AbstractAcquirer
from bci_dayloop.control.commands import command_for_prediction
from bci_dayloop.data.preprocessing import EEGPreprocessor
from bci_dayloop.models.base import BaseModelAdapter


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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SlidingWindowDecoder:
    def __init__(
        self,
        model: BaseModelAdapter,
        preprocessor: EEGPreprocessor,
        class_names: list[str],
        *,
        sample_rate: float,
        input_unit: str,
        window_sec: float = 4.0,
        step_sec: float = 0.5,
        confidence_threshold: float = 0.55,
        command_map: dict[str, str] | None = None,
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
        self._buffer: np.ndarray | None = None
        self._new_since_decode = 0

    def reset(self) -> None:
        self._buffer = None
        self._new_since_decode = 0

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
        self._buffer = chunk.copy() if self._buffer is None else np.concatenate((self._buffer, chunk), axis=1)
        self._buffer = self._buffer[:, -self.window_samples :]
        self._new_since_decode += chunk.shape[1]
        if self._buffer.shape[1] < self.window_samples or self._new_since_decode < self.step_samples:
            return None
        self._new_since_decode %= self.step_samples
        started = time.perf_counter()
        model_input = self.preprocessor.transform(
            self._buffer,
            self.sample_rate,
            self.input_unit,
            reshape=True,
        )
        probabilities = self.model.predict_proba(model_input[None, ...])[0]
        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])
        prediction = self.class_names[class_id]
        command = command_for_prediction(prediction, confidence, self.confidence_threshold, self.command_map)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return DecodeResult(
            prediction,
            confidence,
            latency_ms,
            command,
            class_id,
            probabilities.tolist(),
            trial_id,
            expected_class_id,
        )

    def run(
        self,
        acquirer: AbstractAcquirer,
        *,
        max_windows: int | None = None,
        callback: Callable[[DecodeResult, np.ndarray], None] | None = None,
    ) -> Iterator[DecodeResult]:
        self.reset()
        acquirer.start_stream()
        emitted = 0
        try:
            while max_windows is None or emitted < max_windows:
                samples, _ = acquirer.get_new_samples()
                if samples.shape[1] == 0:
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
                yield result
        finally:
            acquirer.stop_stream()

