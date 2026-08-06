from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from bci_dayloop.data.hdf5_dataset import EEGHDF5
from bci_dayloop.runtime.model import RuntimeModel
from bci_dayloop.runtime.types import RawEEGWindow
from bci_dayloop.utils.config import dump_json


ShortTrialPolicy = Literal["error", "skip"]


@dataclass(frozen=True, slots=True)
class WindowEvaluationRecord:
    """
    一个 trial 内的一个窗口预测记录。

    confusion matrix 和最终指标不会直接从 JSONL 推断，
    而是从这些结构化记录统一聚合。
    """

    trial_index: int
    trial_id: str
    subject_id: int | None
    session_id: str | None

    window_index: int
    start_sample: int
    stop_sample: int
    start_sec: float
    stop_sec: float

    true_label: int
    predicted_label: int
    confidence: float
    probabilities: tuple[float, ...]

    preprocessing_latency_ms: float
    model_latency_ms: float
    total_latency_ms: float

    preprocessing_trace: tuple[str, ...] = ()
    preprocessing_diagnostics: dict[str, Any] = field(
        default_factory=dict
    )
    model_diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class TrialEvaluationRecord:
    """
    trial 级预测。

    一个 trial 产生多个窗口时，对所有窗口概率取平均，
    再对平均概率执行 argmax。
    """

    trial_index: int
    trial_id: str
    subject_id: int | None
    session_id: str | None

    true_label: int
    predicted_label: int
    confidence: float
    mean_probabilities: tuple[float, ...]
    num_windows: int


@dataclass(frozen=True, slots=True)
class SkippedTrial:
    trial_index: int
    trial_id: str
    reason: str
    available_samples: int
    required_samples: int


@dataclass(frozen=True, slots=True)
class RuntimeEvaluationResult:
    """
    一次 trial-aligned Runtime 评估的完整结果。
    """

    dataset_name: str | None
    session: str | None

    class_names: tuple[str, ...]
    window_sec: float
    step_sec: float
    source_sample_rate: float
    source_unit: str

    num_input_trials: int
    num_evaluated_trials: int
    num_skipped_trials: int
    num_windows: int

    window_metrics: dict[str, Any]
    trial_metrics: dict[str, Any]
    latency: dict[str, float | None]

    window_records: tuple[WindowEvaluationRecord, ...]
    trial_records: tuple[TrialEvaluationRecord, ...]
    skipped_trials: tuple[SkippedTrial, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(
        self,
        path: str | Path,
    ) -> Path:
        target = Path(path)
        dump_json(self.to_dict(), target)
        return target


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """
    按固定类别顺序计算分类指标。

    confusion matrix：
        行 = true label
        列 = predicted label

    balanced accuracy：
        对真实数据中有样本的类别计算 recall，
        再取宏平均。
    """

    true = np.asarray(
        y_true,
        dtype=np.int64,
    ).reshape(-1)

    predicted = np.asarray(
        y_pred,
        dtype=np.int64,
    ).reshape(-1)

    if true.size == 0:
        raise ValueError(
            "Cannot compute classification metrics "
            "from an empty target array."
        )

    if true.shape != predicted.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape: "
            f"{true.shape} != {predicted.shape}."
        )

    normalized_class_names = tuple(
        str(name)
        for name in class_names
    )

    if not normalized_class_names:
        raise ValueError(
            "class_names cannot be empty."
        )

    labels = np.arange(
        len(normalized_class_names),
        dtype=np.int64,
    )

    for source_name, values in (
        ("y_true", true),
        ("y_pred", predicted),
    ):
        invalid = values[
            (values < 0)
            | (values >= len(labels))
        ]

        if invalid.size > 0:
            raise ValueError(
                f"{source_name} contains labels outside "
                f"[0, {len(labels) - 1}]: "
                f"{np.unique(invalid).tolist()}."
            )

    matrix = confusion_matrix(
        true,
        predicted,
        labels=labels,
    ).astype(np.int64, copy=False)

    support = matrix.sum(axis=1)
    true_positive = np.diag(matrix)

    per_class_recall = np.divide(
        true_positive,
        support,
        out=np.full(
            len(labels),
            np.nan,
            dtype=np.float64,
        ),
        where=support > 0,
    )

    supported_mask = support > 0

    if not np.any(supported_mask):
        raise ValueError(
            "No supported classes were found."
        )

    balanced_accuracy = float(
        np.mean(
            per_class_recall[
                supported_mask
            ]
        )
    )

    missing_classes = [
        normalized_class_names[index]
        for index, count
        in enumerate(support.tolist())
        if count == 0
    ]

    per_class = []

    for index, class_name in enumerate(
        normalized_class_names
    ):
        recall_value = (
            None
            if support[index] == 0
            else float(
                per_class_recall[index]
            )
        )

        per_class.append(
            {
                "label": int(index),
                "class_name": class_name,
                "support": int(
                    support[index]
                ),
                "recall": recall_value,
            }
        )

    return {
        "accuracy": float(
            accuracy_score(
                true,
                predicted,
            )
        ),
        "balanced_accuracy": (
            balanced_accuracy
        ),
        "macro_f1": float(
            f1_score(
                true,
                predicted,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": (
            matrix.tolist()
        ),
        "confusion_matrix_labels": (
            list(normalized_class_names)
        ),
        "support": support.tolist(),
        "per_class": per_class,
        "missing_true_classes": (
            missing_classes
        ),
        "num_samples": int(true.size),
    }


def _latency_summary(
    records: Sequence[
        WindowEvaluationRecord
    ],
) -> dict[str, float | None]:
    if not records:
        return {
            "preprocessing_average_ms": None,
            "model_average_ms": None,
            "total_average_ms": None,
            "total_p50_ms": None,
            "total_p95_ms": None,
        }

    preprocessing = np.asarray(
        [
            record.preprocessing_latency_ms
            for record in records
        ],
        dtype=np.float64,
    )

    model = np.asarray(
        [
            record.model_latency_ms
            for record in records
        ],
        dtype=np.float64,
    )

    total = np.asarray(
        [
            record.total_latency_ms
            for record in records
        ],
        dtype=np.float64,
    )

    return {
        "preprocessing_average_ms": float(
            np.mean(preprocessing)
        ),
        "model_average_ms": float(
            np.mean(model)
        ),
        "total_average_ms": float(
            np.mean(total)
        ),
        "total_p50_ms": float(
            np.percentile(total, 50)
        ),
        "total_p95_ms": float(
            np.percentile(total, 95)
        ),
    }


class RuntimeEvaluator:
    """
    Trial-aligned Runtime 模型评估器。

    与连续 Replay 的区别：

    - 每个 trial 单独切窗；
    - trial 之间不会共享 buffer；
    - 窗口绝不会跨 trial；
    - 直接根据模型输出计算分类指标；
    - confidence threshold 和 command map 不参与准确率计算。
    """

    def __init__(
        self,
        *,
        runtime_model: RuntimeModel,
        class_names: Sequence[str],
        step_sec: float,
        window_sec: float | None = None,
        short_trial_policy: ShortTrialPolicy = (
            "error"
        ),
        require_all_classes: bool = True,
        include_preprocessing_details: bool = False,
    ) -> None:
        self.runtime_model = runtime_model

        self.class_names = tuple(
            str(name)
            for name in class_names
        )

        if not self.class_names:
            raise ValueError(
                "class_names cannot be empty."
            )

        backend_num_classes = int(
            runtime_model.backend.num_classes
        )

        if (
            len(self.class_names)
            != backend_num_classes
        ):
            raise ValueError(
                "class_names length does not match "
                "Runtime backend num_classes: "
                f"{len(self.class_names)} != "
                f"{backend_num_classes}."
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
                    "Evaluation window_sec does not "
                    "match RuntimeModel input contract: "
                    f"evaluation="
                    f"{resolved_window_sec}, "
                    f"contract="
                    f"{contract_window_sec}."
                )

        if step_sec <= 0:
            raise ValueError(
                "step_sec must be positive."
            )

        if step_sec > resolved_window_sec:
            raise ValueError(
                "step_sec must not exceed "
                "window_sec."
            )

        if short_trial_policy not in {
            "error",
            "skip",
        }:
            raise ValueError(
                "short_trial_policy must be "
                "'error' or 'skip'."
            )

        self.window_sec = (
            resolved_window_sec
        )
        self.step_sec = float(step_sec)
        self.short_trial_policy = (
            short_trial_policy
        )
        self.require_all_classes = bool(
            require_all_classes
        )
        self.include_preprocessing_details = (
            bool(
                include_preprocessing_details
            )
        )

    def evaluate_hdf5(
        self,
        data_path: str | Path,
        *,
        session: str,
        max_trials: int | None = None,
    ) -> RuntimeEvaluationResult:
        dataset = EEGHDF5(data_path)
        metadata = dataset.metadata

        dataset_class_names = tuple(
            str(name)
            for name in metadata.class_names
        )

        if (
            dataset_class_names
            != self.class_names
        ):
            raise ValueError(
                "Dataset class order does not "
                "match Runtime class order: "
                f"dataset={dataset_class_names}, "
                f"runtime={self.class_names}."
            )

        payload = dataset.load(session)

        return self.evaluate_arrays(
            data=payload["data"],
            labels=payload["labels"],
            channel_names=metadata.channel_names,
            sample_rate=metadata.sample_rate,
            input_unit=metadata.unit,
            subject_ids=payload[
                "subject_ids"
            ],
            session_ids=payload[
                "session_ids"
            ],
            trial_ids=payload[
                "trial_ids"
            ],
            dataset_name=(
                metadata.dataset_name
            ),
            session=session,
            max_trials=max_trials,
        )

    def evaluate_arrays(
        self,
        *,
        data: np.ndarray,
        labels: np.ndarray,
        channel_names: Sequence[str],
        sample_rate: float,
        input_unit: str,
        subject_ids: np.ndarray | None = None,
        session_ids: np.ndarray | None = None,
        trial_ids: np.ndarray | None = None,
        dataset_name: str | None = None,
        session: str | None = None,
        max_trials: int | None = None,
    ) -> RuntimeEvaluationResult:
        trials = np.asarray(
            data,
            dtype=np.float32,
        )

        targets = np.asarray(
            labels,
            dtype=np.int64,
        ).reshape(-1)

        if trials.ndim != 3:
            raise ValueError(
                "data must have shape [N,C,T], "
                f"got {trials.shape}."
            )

        num_trials = int(
            trials.shape[0]
        )

        if targets.shape != (
            num_trials,
        ):
            raise ValueError(
                "labels length does not match "
                "trial count: "
                f"{targets.shape} != "
                f"({num_trials},)."
            )

        normalized_channel_names = tuple(
            str(name)
            for name in channel_names
        )

        if (
            trials.shape[1]
            != len(normalized_channel_names)
        ):
            raise ValueError(
                "EEG channel dimension does not "
                "match channel_names: "
                f"{trials.shape[1]} != "
                f"{len(normalized_channel_names)}."
            )

        source_sample_rate = float(
            sample_rate
        )

        if (
            not np.isfinite(
                source_sample_rate
            )
            or source_sample_rate <= 0
        ):
            raise ValueError(
                "sample_rate must be finite "
                "and positive."
            )

        if max_trials is not None:
            if max_trials <= 0:
                raise ValueError(
                    "max_trials must be positive."
                )

            evaluation_trial_count = min(
                num_trials,
                int(max_trials),
            )
        else:
            evaluation_trial_count = (
                num_trials
            )

        resolved_subject_ids = (
            self._optional_int_metadata(
                subject_ids,
                num_trials=num_trials,
                name="subject_ids",
            )
        )

        resolved_session_ids = (
            self._optional_str_metadata(
                session_ids,
                num_trials=num_trials,
                name="session_ids",
            )
        )

        resolved_trial_ids = (
            self._trial_identifiers(
                trial_ids,
                num_trials=num_trials,
            )
        )

        window_samples = int(
            round(
                self.window_sec
                * source_sample_rate
            )
        )

        step_samples = int(
            round(
                self.step_sec
                * source_sample_rate
            )
        )

        if window_samples <= 0:
            raise ValueError(
                "Resolved window_samples "
                "must be positive."
            )

        if step_samples <= 0:
            raise ValueError(
                "Resolved step_samples "
                "must be positive."
            )

        window_records: list[
            WindowEvaluationRecord
        ] = []

        trial_records: list[
            TrialEvaluationRecord
        ] = []

        skipped_trials: list[
            SkippedTrial
        ] = []

        for trial_index in range(
            evaluation_trial_count
        ):
            trial = trials[trial_index]
            true_label = int(
                targets[trial_index]
            )

            if not 0 <= true_label < len(
                self.class_names
            ):
                raise ValueError(
                    "Trial label is outside class "
                    "range: "
                    f"trial_index={trial_index}, "
                    f"label={true_label}."
                )

            trial_id = (
                resolved_trial_ids[
                    trial_index
                ]
            )

            subject_id = (
                resolved_subject_ids[
                    trial_index
                ]
            )

            session_id = (
                resolved_session_ids[
                    trial_index
                ]
            )

            available_samples = int(
                trial.shape[-1]
            )

            if (
                available_samples
                < window_samples
            ):
                skipped = SkippedTrial(
                    trial_index=trial_index,
                    trial_id=trial_id,
                    reason="trial_too_short",
                    available_samples=(
                        available_samples
                    ),
                    required_samples=(
                        window_samples
                    ),
                )

                if (
                    self.short_trial_policy
                    == "error"
                ):
                    raise ValueError(
                        "Trial is shorter than the "
                        "model window: "
                        f"trial_index="
                        f"{trial_index}, "
                        f"trial_id={trial_id}, "
                        f"available="
                        f"{available_samples}, "
                        f"required="
                        f"{window_samples}."
                    )

                skipped_trials.append(
                    skipped
                )
                continue

            starts = range(
                0,
                (
                    available_samples
                    - window_samples
                    + 1
                ),
                step_samples,
            )

            current_trial_windows: list[
                WindowEvaluationRecord
            ] = []

            for window_index, start_sample in enumerate(
                starts
            ):
                stop_sample = (
                    start_sample
                    + window_samples
                )

                raw_data = np.ascontiguousarray(
                    trial[
                        :,
                        start_sample:stop_sample,
                    ],
                    dtype=np.float32,
                )

                raw_window = RawEEGWindow(
                    data=raw_data,
                    channel_names=list(
                        normalized_channel_names
                    ),
                    sample_rate=(
                        source_sample_rate
                    ),
                    unit=str(input_unit),
                    layout="CT",
                    start_time_sec=(
                        start_sample
                        / source_sample_rate
                    ),
                    trial_id=trial_id,
                    window_id=(
                        f"{trial_id}:"
                        f"{window_index}"
                    ),
                    label=true_label,
                    metadata={
                        "dataset_name": (
                            dataset_name
                        ),
                        "session": session,
                        "session_id": (
                            session_id
                        ),
                        "subject_id": (
                            subject_id
                        ),
                        "trial_index": (
                            trial_index
                        ),
                    },
                )

                total_started = (
                    time.perf_counter()
                )

                preprocessing_started = (
                    time.perf_counter()
                )

                prepared = (
                    self.runtime_model.prepare(
                        raw_window
                    )
                )

                preprocessing_ms = (
                    time.perf_counter()
                    - preprocessing_started
                ) * 1000.0

                model_started = (
                    time.perf_counter()
                )

                output = (
                    self.runtime_model
                    .predict_prepared(
                        prepared,
                        return_features=False,
                    )
                )

                model_ms = (
                    time.perf_counter()
                    - model_started
                ) * 1000.0

                total_ms = (
                    time.perf_counter()
                    - total_started
                ) * 1000.0

                probabilities = (
                    self._probability_vector(
                        output.probabilities
                    )
                )

                predicted_label = int(
                    output.predicted_class
                )

                argmax_label = int(
                    np.argmax(
                        probabilities
                    )
                )

                if (
                    predicted_label
                    != argmax_label
                ):
                    raise RuntimeError(
                        "ModelOutput.predicted_class "
                        "does not match probability "
                        "argmax: "
                        f"predicted_class="
                        f"{predicted_label}, "
                        f"argmax={argmax_label}."
                    )

                confidence = float(
                    output.confidence
                )

                expected_confidence = float(
                    probabilities[
                        predicted_label
                    ]
                )

                if not np.isclose(
                    confidence,
                    expected_confidence,
                    rtol=1e-5,
                    atol=1e-6,
                ):
                    raise RuntimeError(
                        "ModelOutput.confidence does "
                        "not match predicted-class "
                        "probability: "
                        f"confidence={confidence}, "
                        f"probability="
                        f"{expected_confidence}."
                    )

                if (
                    self
                    .include_preprocessing_details
                ):
                    preprocessing_trace = tuple(
                        str(step)
                        for step
                        in prepared
                        .preprocessing_trace
                    )

                    preprocessing_diagnostics = dict(
                        prepared.diagnostics
                    )

                    model_diagnostics = dict(
                        output.diagnostics
                    )
                else:
                    preprocessing_trace = ()
                    preprocessing_diagnostics = {}
                    model_diagnostics = {}

                record = WindowEvaluationRecord(
                    trial_index=trial_index,
                    trial_id=trial_id,
                    subject_id=subject_id,
                    session_id=session_id,
                    window_index=window_index,
                    start_sample=int(
                        start_sample
                    ),
                    stop_sample=int(
                        stop_sample
                    ),
                    start_sec=float(
                        start_sample
                        / source_sample_rate
                    ),
                    stop_sec=float(
                        stop_sample
                        / source_sample_rate
                    ),
                    true_label=true_label,
                    predicted_label=(
                        predicted_label
                    ),
                    confidence=confidence,
                    probabilities=tuple(
                        float(value)
                        for value
                        in probabilities.tolist()
                    ),
                    preprocessing_latency_ms=(
                        preprocessing_ms
                    ),
                    model_latency_ms=(
                        model_ms
                    ),
                    total_latency_ms=(
                        total_ms
                    ),
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

                window_records.append(record)
                current_trial_windows.append(
                    record
                )

            if not current_trial_windows:
                raise RuntimeError(
                    "No windows were emitted for "
                    f"trial {trial_id}."
                )

            trial_probabilities = np.mean(
                np.asarray(
                    [
                        record.probabilities
                        for record
                        in current_trial_windows
                    ],
                    dtype=np.float64,
                ),
                axis=0,
            )

            trial_prediction = int(
                np.argmax(
                    trial_probabilities
                )
            )

            trial_confidence = float(
                trial_probabilities[
                    trial_prediction
                ]
            )

            trial_records.append(
                TrialEvaluationRecord(
                    trial_index=trial_index,
                    trial_id=trial_id,
                    subject_id=subject_id,
                    session_id=session_id,
                    true_label=true_label,
                    predicted_label=(
                        trial_prediction
                    ),
                    confidence=(
                        trial_confidence
                    ),
                    mean_probabilities=tuple(
                        float(value)
                        for value
                        in trial_probabilities
                        .tolist()
                    ),
                    num_windows=len(
                        current_trial_windows
                    ),
                )
            )

        if not window_records:
            raise ValueError(
                "Evaluation produced no windows."
            )

        if not trial_records:
            raise ValueError(
                "Evaluation produced no trial "
                "predictions."
            )

        window_true = np.asarray(
            [
                record.true_label
                for record in window_records
            ],
            dtype=np.int64,
        )

        window_predicted = np.asarray(
            [
                record.predicted_label
                for record in window_records
            ],
            dtype=np.int64,
        )

        trial_true = np.asarray(
            [
                record.true_label
                for record in trial_records
            ],
            dtype=np.int64,
        )

        trial_predicted = np.asarray(
            [
                record.predicted_label
                for record in trial_records
            ],
            dtype=np.int64,
        )

        trial_metrics = (
            _classification_metrics(
                trial_true,
                trial_predicted,
                class_names=self.class_names,
            )
        )

        window_metrics = (
            _classification_metrics(
                window_true,
                window_predicted,
                class_names=self.class_names,
            )
        )

        if self.require_all_classes:
            missing = trial_metrics[
                "missing_true_classes"
            ]

            if missing:
                raise ValueError(
                    "Formal evaluation does not "
                    "contain every configured class. "
                    f"Missing classes: {missing}. "
                    "Use require_all_classes=False "
                    "only for smoke tests."
                )

        return RuntimeEvaluationResult(
            dataset_name=dataset_name,
            session=session,
            class_names=self.class_names,
            window_sec=self.window_sec,
            step_sec=self.step_sec,
            source_sample_rate=(
                source_sample_rate
            ),
            source_unit=str(input_unit),
            num_input_trials=(
                evaluation_trial_count
            ),
            num_evaluated_trials=len(
                trial_records
            ),
            num_skipped_trials=len(
                skipped_trials
            ),
            num_windows=len(
                window_records
            ),
            window_metrics=window_metrics,
            trial_metrics=trial_metrics,
            latency=_latency_summary(
                window_records
            ),
            window_records=tuple(
                window_records
            ),
            trial_records=tuple(
                trial_records
            ),
            skipped_trials=tuple(
                skipped_trials
            ),
        )

    def _probability_vector(
        self,
        probabilities: Any,
    ) -> np.ndarray:
        try:
            tensor = (
                probabilities
                .detach()
                .cpu()
            )
        except AttributeError as error:
            raise TypeError(
                "ModelOutput.probabilities must "
                "be a torch.Tensor."
            ) from error

        if tensor.ndim == 2:
            if tensor.shape[0] != 1:
                raise RuntimeError(
                    "RuntimeEvaluator expects "
                    "one window per prediction, "
                    f"got batch_size="
                    f"{tensor.shape[0]}."
                )

            tensor = tensor[0]

        if tensor.ndim != 1:
            raise RuntimeError(
                "Expected probabilities with "
                "shape [classes] or [1,classes], "
                f"got {tuple(tensor.shape)}."
            )

        values = (
            tensor.numpy()
            .astype(
                np.float64,
                copy=False,
            )
        )

        if values.shape != (
            len(self.class_names),
        ):
            raise RuntimeError(
                "Probability vector length does "
                "not match class_names: "
                f"{values.shape} != "
                f"({len(self.class_names)},)."
            )

        if not np.isfinite(values).all():
            raise RuntimeError(
                "Probability vector contains "
                "NaN or Inf."
            )

        if np.any(values < 0):
            raise RuntimeError(
                "Probability vector contains "
                "negative values."
            )

        probability_sum = float(
            np.sum(values)
        )

        if not np.isclose(
            probability_sum,
            1.0,
            rtol=1e-5,
            atol=1e-5,
        ):
            raise RuntimeError(
                "Model probabilities do not "
                "sum to 1: "
                f"{probability_sum}."
            )

        return values

    @staticmethod
    def _optional_int_metadata(
        values: np.ndarray | None,
        *,
        num_trials: int,
        name: str,
    ) -> list[int | None]:
        if values is None:
            return [
                None
                for _ in range(num_trials)
            ]

        array = np.asarray(
            values
        ).reshape(-1)

        if array.shape != (
            num_trials,
        ):
            raise ValueError(
                f"{name} length does not "
                "match trial count."
            )

        return [
            int(value)
            for value in array.tolist()
        ]

    @staticmethod
    def _optional_str_metadata(
        values: np.ndarray | None,
        *,
        num_trials: int,
        name: str,
    ) -> list[str | None]:
        if values is None:
            return [
                None
                for _ in range(num_trials)
            ]

        array = np.asarray(
            values
        ).reshape(-1)

        if array.shape != (
            num_trials,
        ):
            raise ValueError(
                f"{name} length does not "
                "match trial count."
            )

        return [
            str(value)
            for value in array.tolist()
        ]

    @staticmethod
    def _trial_identifiers(
        values: np.ndarray | None,
        *,
        num_trials: int,
    ) -> list[str]:
        if values is None:
            return [
                str(index)
                for index in range(num_trials)
            ]

        array = np.asarray(
            values
        ).reshape(-1)

        if array.shape != (
            num_trials,
        ):
            raise ValueError(
                "trial_ids length does not "
                "match trial count."
            )

        return [
            str(value)
            for value in array.tolist()
        ]