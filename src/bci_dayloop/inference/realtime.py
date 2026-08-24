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

from bci_dayloop.runtime.adaptation_types import (
    OnlineObservation,
    OnlineUpdateResult,
)

from bci_dayloop.runtime.model import (
    RuntimeModel,
)
from bci_dayloop.runtime.types import (
    RawEEGWindow,
)

from bci_dayloop.inference.predictor import (
    PreparedPredictor,
    RawWindowPredictor,
)
from bci_dayloop.applications.three_mental_states.contract import ThreeMentalStatePrediction

OnlineObservationHandler = Callable[
    [OnlineObservation, int | None],
    OnlineUpdateResult | None,
]

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

    def to_dict(self) -> dict[str, object]:
        """
        Return a plain dictionary suitable for JSONL logs,
        UI events, pandas DataFrames, and session state.
        """

        return asdict(self)


@dataclass(frozen=True, slots=True)
class MultiHeadDecodeResult:
    """Decoder metadata paired with one shared-feature multi-head prediction."""

    prediction: ThreeMentalStatePrediction
    latency_ms: float

    trial_id: int | None = None
    expected_class_id: int | None = None

    preprocessing_latency_ms: float = 0.0
    model_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    preprocessing_trace: tuple[str, ...] = ()
    preprocessing_diagnostics: dict[str, object] = field(default_factory=dict)
    model_diagnostics: dict[str, object] = field(default_factory=dict)

    model_revision: str = "base"
    online_update_step: int = 0
    online_update_applied: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DecoderResult = DecodeResult | MultiHeadDecodeResult


class SlidingWindowDecoder:
    def __init__(
        self,
        runtime_model: RuntimeModel | None = None,
        class_names: Sequence[str] = (),
        channel_names: Sequence[str] = (),
        *,
        sample_rate: float,
        input_unit: str,
        predictor: PreparedPredictor | RawWindowPredictor | None = None,
        online_observation_handler: (
                OnlineObservationHandler | None
        ) = None,
        window_sec: float | None = None,
        step_sec: float = 0.5,
        confidence_threshold: float = 0.55,
        command_map: dict[str, str] | None = None,
        run_stats: PipelineRunStats | None = None,
        jsonl_logger: JsonlWindowLogger | None = None,
    ) -> None:
        self.runtime_model = runtime_model

        if predictor is None:
            if runtime_model is None:
                raise ValueError(
                    "runtime_model is required when predictor is not provided."
                )
            # 默认保持原来的静态 Runtime 路径。
            resolved_predictor: PreparedPredictor | RawWindowPredictor = runtime_model
            self._predictor_mode = "prepared"
        else:
            if isinstance(predictor, PreparedPredictor):
                if runtime_model is None:
                    raise ValueError(
                        "runtime_model is required for a PreparedPredictor."
                    )
                resolved_predictor = predictor
                self._predictor_mode = "prepared"
            elif isinstance(predictor, RawWindowPredictor):
                resolved_predictor = predictor
                self._predictor_mode = "raw_window"
            else:
                raise TypeError(
                    "predictor must implement PreparedPredictor or "
                    "RawWindowPredictor, got "
                    f"{type(predictor).__name__}."
                )

        self.predictor = (
            resolved_predictor
        )

        if (
            self._predictor_mode == "raw_window"
            and online_observation_handler is not None
        ):
            raise ValueError(
                "online_observation_handler is not supported for a "
                "RawWindowPredictor."
            )

        # 普通模式为 None；
        # NeuroOnline 模式为预测后的 observation/feedback/update 回调。
        self.online_observation_handler = (
            online_observation_handler
        )

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

        if self._predictor_mode == "prepared":
            assert runtime_model is not None
            contract_window_sec = float(runtime_model.input_contract.window_sec)
        else:
            contract_window_sec = float(
                getattr(resolved_predictor, "window_seconds")
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
                    "the predictor input contract: "
                    f"decoder={resolved_window_sec}, "
                    f"predictor={contract_window_sec}."
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

    def _decode_multi_head_window(
        self,
        *,
        window_id: int,
        total_started: float,
        trial_id: int | None,
        expected_class_id: int | None,
    ) -> MultiHeadDecodeResult:
        """Delegate raw-window preprocessing and inference to one predictor."""
        assert self._buffer is not None
        try:
            raw_window = RawEEGWindow(
                data=self._buffer.copy(),
                channel_names=list(self.channel_names),
                sample_rate=self.sample_rate,
                unit=self.input_unit,
                layout="CT",
                trial_id=str(trial_id) if trial_id is not None else None,
                window_id=str(window_id),
                label=expected_class_id,
                metadata={"source": "sliding_window_decoder"},
            )
            model_started = time.perf_counter()
            prediction = self.predictor.predict(raw_window)  # type: ignore[union-attr]
            total_ms = (time.perf_counter() - total_started) * 1000.0
            wall_model_ms = (time.perf_counter() - model_started) * 1000.0
            diagnostics = getattr(self.predictor, "last_diagnostics", None)
            preprocessing_ms = float(
                getattr(diagnostics, "preprocessing_latency_ms", 0.0)
            )
            model_ms = float(
                getattr(diagnostics, "backbone_latency_ms", wall_model_ms)
            ) + float(getattr(diagnostics, "heads_latency_ms", 0.0))
            if model_ms <= 0.0:
                model_ms = wall_model_ms
            model_diagnostics: dict[str, object] = {
                "predictor": type(self.predictor).__name__,
                "prediction_type": "MultiHeadPrediction",
            }
            if diagnostics is not None:
                model_diagnostics.update(
                    {
                        "preprocessing_calls": diagnostics.preprocessing_calls,
                        "backbone_forwards": diagnostics.backbone_forwards,
                        "head_forwards": dict(diagnostics.head_forwards),
                        "shared_feature_shape": list(diagnostics.shared_feature_shape),
                    }
                )
            result = MultiHeadDecodeResult(
                prediction=prediction,
                latency_ms=total_ms,
                trial_id=trial_id,
                expected_class_id=expected_class_id,
                preprocessing_latency_ms=preprocessing_ms,
                model_latency_ms=model_ms,
                total_latency_ms=total_ms,
                preprocessing_trace=("multi_head:Model50MPreprocessor",),
                preprocessing_diagnostics={
                    "owner": "MultiHeadPredictor",
                    "source_sample_rate": self.sample_rate,
                    "source_channel_count": len(self.channel_names),
                },
                model_diagnostics=model_diagnostics,
                model_revision=str(getattr(self.predictor, "model_revision", "base")),
            )
        except Exception as error:
            if self.run_stats is not None:
                self.run_stats.record_failure()
            if self.jsonl_logger is not None:
                try:
                    self.jsonl_logger.log_error(window_id=window_id, error=error)
                except Exception as logger_error:
                    raise error from logger_error
            raise

        if self.run_stats is not None:
            self.run_stats.record_success(
                LatencyBreakdown(
                    preprocessing_ms=result.preprocessing_latency_ms,
                    model_ms=result.model_latency_ms,
                    total_ms=result.total_latency_ms,
                )
            )
        if self.jsonl_logger is not None:
            self.jsonl_logger.log_success(window_id=window_id, result=result)
        return result

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
    ) -> DecoderResult | None:
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

        if self._predictor_mode == "raw_window":
            return self._decode_multi_head_window(
                window_id=window_id,
                total_started=total_started,
                trial_id=trial_id,
                expected_class_id=expected_class_id,
            )

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
                self.predictor.predict_prepared(
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

            model_diagnostics.setdefault(
                "predictor",
                type(self.predictor).__name__,
            )

            model_ms = (
                time.perf_counter()
                - model_started
            ) * 1000.0

            # 记录“本次预测真正使用”的模型版本。
            # 必须在在线更新之前读取。
            prediction_model_revision = str(
                getattr(
                    self.predictor,
                    "model_revision",
                    "base",
                )
            )

            prediction_update_step = int(
                getattr(
                    self.predictor,
                    "update_step",
                    0,
                )
            )

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

            # 到这里，本窗口的预测已经完全完成。
            # 后续标签只能更新下一窗口使用的参数，
            # 不会反过来改变当前窗口的 prediction。
            prediction_total_ms = (
                                          time.perf_counter()
                                          - total_started
                                  ) * 1000.0

            update_result: OnlineUpdateResult | None = None

            if self.online_observation_handler is not None:
                observation = OnlineObservation(
                    observation_id=(
                        f"decoder-window-{window_id}"
                    ),
                    prepared_input=prepared,
                    output=output,
                    timestamp_sec=time.time(),
                    metadata={
                        "trial_id": trial_id,
                        "window_id": window_id,
                    },
                )

                update_result = (
                    self.online_observation_handler(
                        observation,
                        expected_class_id,
                    )
                )

                if update_result is not None:
                    # asdict 已经在 realtime.py 顶部导入。
                    model_diagnostics[
                        "online_update"
                    ] = asdict(update_result)

            total_ms = prediction_total_ms

            online_update_applied = (
                    update_result is not None
                    and update_result.applied
            )

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

                # 这两个字段表示产生当前预测时使用的版本。
                model_revision=(
                    prediction_model_revision
                ),
                online_update_step=(
                    prediction_update_step
                ),

                # 表示当前预测结束后是否触发了一次更新。
                online_update_applied=(
                    online_update_applied
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
        callback: Callable[[DecoderResult, np.ndarray], None] | None = None,
        stop_event: StopEvent | None = None,
    ) -> Iterator[DecoderResult]:
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
