from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

import numpy as np
import torch

from _bootstrap import ROOT  # noqa: F401

from bci_dayloop.data.hdf5_dataset import EEGHDF5, HDF5Metadata
from bci_dayloop.inference.neuroonline_strategy import (
    NeuroOnlineConfig,
    NeuroOnlineStrategy,
)
from bci_dayloop.runtime.adaptation_types import (
    AdaptationContext,
    FeedbackEvent,
    OnlineObservation,
    OnlineUpdateResult,
)
from bci_dayloop.runtime.model import RuntimeModel
from bci_dayloop.runtime.types import ModelOutput, RawEEGWindow
from bci_dayloop.utils.config import load_yaml, resolve_path

if TYPE_CHECKING:
    from bci_dayloop.packages.loader import LoadedRuntimePackage
else:
    LoadedRuntimePackage = Any


FIXED_WINDOW_SEC = 4.0
FIXED_STEP_SEC = 4.0
VALID_STRATEGIES = ("none", "neuroonline", "both")


@dataclass(frozen=True, slots=True)
class SequentialSettings:
    data_path: Path
    model_package: Path
    session: str
    device: str
    online_strategy: Literal["none", "neuroonline", "both"]
    max_trials: int | None
    block_size: int
    rolling_window: int
    print_every: int
    output_dir: Path
    subject_id: str | None
    neuroonline_config: NeuroOnlineConfig


@dataclass(frozen=True, slots=True)
class SequentialDataset:
    metadata: HDF5Metadata
    data: np.ndarray
    labels: np.ndarray
    subject_ids: np.ndarray
    session_ids: np.ndarray
    trial_ids: np.ndarray

    @property
    def num_trials(self) -> int:
        return int(self.data.shape[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate static RuntimeModel and NeuroOnline in a strict "
            "4-second trial-by-trial causal protocol."
        )
    )
    parser.add_argument("--config", default="configs/stage0/day1_bnci_s01.yaml")
    parser.add_argument("--data")
    parser.add_argument("--model-package")
    parser.add_argument("--session")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument(
        "--online-strategy",
        choices=VALID_STRATEGIES,
        default=None,
    )
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--rolling-window", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=None)
    parser.add_argument("--output-dir")
    return parser


def _first_defined(command_line: Any, yaml_value: Any, default: Any) -> Any:
    if command_line is not None:
        return command_line
    if yaml_value is not None:
        return yaml_value
    return default


def _mapping(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping.")
    return dict(value)


def resolve_neuroonline_config(value: object) -> NeuroOnlineConfig:
    payload = _mapping(value, "online.neuroonline")
    allowed_fields = {item.name for item in dataclass_fields(NeuroOnlineConfig)}
    unknown_fields = set(payload) - allowed_fields
    if unknown_fields:
        raise ValueError(
            "Unknown NeuroOnline settings: "
            f"{sorted(unknown_fields)}."
        )
    return NeuroOnlineConfig(**payload)


def resolve_settings(
    args: argparse.Namespace,
    config: dict[str, Any],
    *,
    now: datetime | None = None,
) -> SequentialSettings:
    project = _mapping(config.get("project"), "project")
    data_config = _mapping(config.get("data"), "data")
    replay = _mapping(config.get("replay"), "replay")
    artifacts = _mapping(config.get("artifacts"), "artifacts")
    model = _mapping(config.get("model"), "model")
    online = _mapping(config.get("online"), "online")

    run_dir = resolve_path(project.get("run_dir", "runs"))
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S%fZ")
    default_output_dir = run_dir / "neuroonline_sequential" / timestamp

    data_value = _first_defined(
        args.data,
        data_config.get("output_hdf5"),
        "data/processed/bnci2014_001/subject_01.h5",
    )
    session_value = _first_defined(
        args.session,
        _first_defined(replay.get("session"), data_config.get("test_session"), None),
        "1test",
    )
    package_value = _first_defined(
        args.model_package,
        _first_defined(
            replay.get("model_package"),
            artifacts.get("model_package_dir"),
            None,
        ),
        run_dir / "model_package",
    )
    device = str(_first_defined(args.device, model.get("device"), "cpu"))
    online_strategy = str(
        _first_defined(args.online_strategy, online.get("strategy"), "none")
    ).strip().lower()

    if online_strategy not in VALID_STRATEGIES:
        raise ValueError(
            "online_strategy must be one of "
            f"{VALID_STRATEGIES}, got {online_strategy!r}."
        )

    max_trials = None if args.max_trials is None else int(args.max_trials)
    if max_trials is not None and max_trials <= 0:
        raise ValueError("max_trials must be positive.")

    block_size = int(_first_defined(args.block_size, None, 32))
    rolling_window = int(_first_defined(args.rolling_window, None, 32))
    print_every = int(_first_defined(args.print_every, None, 32))
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if rolling_window <= 0:
        raise ValueError("rolling_window must be positive.")
    if print_every <= 0:
        raise ValueError("print_every must be positive.")

    output_dir_value = _first_defined(args.output_dir, None, default_output_dir)
    subject_value = data_config.get("subject")
    subject_id = None if subject_value is None else str(subject_value)

    return SequentialSettings(
        data_path=resolve_path(data_value),
        model_package=resolve_path(package_value),
        session=str(session_value),
        device=device,
        online_strategy=online_strategy,  # type: ignore[arg-type]
        max_trials=max_trials,
        block_size=block_size,
        rolling_window=rolling_window,
        print_every=print_every,
        output_dir=resolve_path(output_dir_value),
        subject_id=subject_id,
        neuroonline_config=resolve_neuroonline_config(online.get("neuroonline")),
    )


def _trial_ids(values: np.ndarray, *, expected: int) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (expected,):
        raise ValueError(
            "trial_ids length does not match trial count: "
            f"{array.shape} != ({expected},)."
        )
    return array


def load_sequential_dataset(settings: SequentialSettings) -> SequentialDataset:
    dataset = EEGHDF5(settings.data_path)
    metadata = dataset.metadata
    payload = dataset.load(settings.session)

    data = np.asarray(payload["data"], dtype=np.float32)
    labels = np.asarray(payload["labels"], dtype=np.int64).reshape(-1)
    subject_ids = np.asarray(payload["subject_ids"], dtype=np.int64).reshape(-1)
    session_ids = np.asarray(payload["session_ids"]).reshape(-1)
    trial_ids = _trial_ids(np.asarray(payload["trial_ids"]), expected=data.shape[0])

    validate_trial_payload(
        data=data,
        labels=labels,
        subject_ids=subject_ids,
        trial_ids=trial_ids,
        metadata=metadata,
    )

    if settings.max_trials is not None:
        limit = min(settings.max_trials, data.shape[0])
        data = data[:limit]
        labels = labels[:limit]
        subject_ids = subject_ids[:limit]
        session_ids = session_ids[:limit]
        trial_ids = trial_ids[:limit]

    return SequentialDataset(
        metadata=metadata,
        data=data,
        labels=labels,
        subject_ids=subject_ids,
        session_ids=session_ids,
        trial_ids=trial_ids,
    )


def validate_trial_payload(
    *,
    data: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    trial_ids: np.ndarray,
    metadata: HDF5Metadata,
) -> None:
    if data.ndim != 3:
        raise ValueError(f"HDF5 data must have shape [N,C,T], got {data.shape}.")

    num_trials, num_channels, num_samples = data.shape
    if num_channels != len(metadata.channel_names):
        raise ValueError(
            "HDF5 channel dimension does not match metadata channel_names: "
            f"{num_channels} != {len(metadata.channel_names)}."
        )

    expected_shape = (num_trials,)
    for name, values in (
        ("labels", labels),
        ("subject_ids", subject_ids),
        ("trial_ids", trial_ids),
    ):
        if np.asarray(values).shape != expected_shape:
            raise ValueError(
                f"{name} length does not match trial count: "
                f"{np.asarray(values).shape} != {expected_shape}."
            )

    sample_rate = float(metadata.sample_rate)
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("HDF5 sample_rate must be finite and positive.")

    expected_samples = int(round(FIXED_WINDOW_SEC * sample_rate))
    if num_samples != expected_samples:
        duration = num_samples / sample_rate
        raise ValueError(
            "Every source trial must be exactly 4 seconds: "
            f"samples={num_samples}, sample_rate={sample_rate}, "
            f"duration={duration:.9f}, expected_samples={expected_samples}."
        )

    if not np.all(np.isfinite(data)):
        raise ValueError("HDF5 signal contains NaN or Inf.")


def validate_package_contract(
    loaded: LoadedRuntimePackage,
    *,
    metadata: HDF5Metadata,
) -> None:
    if not np.isclose(
        loaded.window_sec,
        FIXED_WINDOW_SEC,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError(
            "Runtime Package input window must be exactly 4 seconds: "
            f"{loaded.window_sec}."
        )

    dataset_classes = tuple(str(name) for name in metadata.class_names)
    if dataset_classes != loaded.class_names:
        raise ValueError(
            "Dataset class order does not match Runtime Package: "
            f"dataset={dataset_classes}, package={loaded.class_names}."
        )


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _probability_vector(output: ModelOutput, *, num_classes: int) -> list[float]:
    probabilities = output.probabilities.detach().cpu().numpy()
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim == 2 and probabilities.shape[0] == 1:
        probabilities = probabilities[0]
    if probabilities.shape != (num_classes,):
        raise ValueError(
            "ModelOutput.probabilities must be a single class vector: "
            f"{probabilities.shape} != ({num_classes},)."
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("ModelOutput.probabilities contains NaN or Inf.")

    predicted = int(output.predicted_class)
    if not 0 <= predicted < num_classes:
        raise ValueError(
            f"predicted_class is outside class range: {predicted}."
        )
    argmax = int(np.argmax(probabilities))
    if predicted != argmax:
        raise RuntimeError(
            "ModelOutput.predicted_class does not match probability argmax: "
            f"predicted_class={predicted}, argmax={argmax}."
        )
    confidence = float(output.confidence)
    if not np.isclose(
        confidence,
        float(probabilities[predicted]),
        atol=1e-6,
        rtol=1e-5,
    ):
        raise RuntimeError(
            "ModelOutput.confidence does not match predicted probability."
        )
    return [float(item) for item in probabilities.tolist()]


def _classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    class_names: Sequence[str],
) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    predicted = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    labels = np.arange(len(class_names), dtype=np.int64)

    if true.shape != predicted.shape:
        raise ValueError(
            f"y_true and y_pred must have identical shape: {true.shape} != {predicted.shape}."
        )
    if true.size == 0:
        return {
            "num_samples": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "confusion_matrix": np.zeros((len(labels), len(labels)), dtype=np.int64).tolist(),
            "confusion_matrix_labels": list(class_names),
            "per_class": [
                {
                    "label": int(index),
                    "class_name": str(name),
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "support": 0,
                }
                for index, name in enumerate(class_names)
            ],
            "present_class_ids": [],
            "present_class_names": [],
        }

    for source_name, values in (("y_true", true), ("y_pred", predicted)):
        invalid = values[(values < 0) | (values >= len(labels))]
        if invalid.size:
            raise ValueError(
                f"{source_name} contains labels outside "
                f"[0, {len(labels) - 1}]: {np.unique(invalid).tolist()}."
            )

    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true_label, predicted_label in zip(true, predicted, strict=True):
        matrix[int(true_label), int(predicted_label)] += 1

    support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    recall = np.divide(
        true_positive,
        support,
        out=np.full(len(labels), np.nan, dtype=np.float64),
        where=support > 0,
    )
    supported_mask = support > 0
    balanced_accuracy = float(np.mean(recall[supported_mask]))

    precision = np.divide(
        true_positive,
        predicted_support,
        out=np.zeros(len(labels), dtype=np.float64),
        where=predicted_support > 0,
    )
    recall_zero = np.divide(
        true_positive,
        support,
        out=np.zeros(len(labels), dtype=np.float64),
        where=support > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall_zero,
        precision + recall_zero,
        out=np.zeros(len(labels), dtype=np.float64),
        where=(precision + recall_zero) > 0,
    )

    per_class = []
    for index, name in enumerate(class_names):
        has_support = int(support[index]) > 0
        per_class.append(
            {
                "label": int(index),
                "class_name": str(name),
                "precision": float(precision[index]),
                "recall": None if not has_support else float(recall_zero[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
        )

    present_ids = [int(index) for index in np.flatnonzero(support > 0)]
    return {
        "num_samples": int(true.size),
        "accuracy": float(np.mean(true == predicted)),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": float(np.mean(f1)),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": list(class_names),
        "per_class": per_class,
        "present_class_ids": present_ids,
        "present_class_names": [str(class_names[index]) for index in present_ids],
    }


def compute_metrics_for_records(
    records: Sequence[dict[str, Any]],
    *,
    class_names: Sequence[str],
    warmup_feedback: int,
    block_size: int,
) -> dict[str, Any]:
    def metrics_for(selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return _classification_metrics(
            [int(item["true_class_id"]) for item in selected],
            [int(item["predicted_class_id"]) for item in selected],
            class_names=class_names,
        )

    warmup = [
        item
        for item in records
        if int(item["trial_ordinal"]) <= int(warmup_feedback)
    ]
    post_warmup = [
        item
        for item in records
        if int(item["trial_ordinal"]) > int(warmup_feedback)
    ]
    after_first_update = [
        item
        for item in records
        if int(item["update_step_used"]) > 0
    ]

    segments = []
    for start in range(0, len(records), block_size):
        segment = records[start : start + block_size]
        if not segment:
            continue
        segments.append(
            {
                "start_trial_ordinal": int(segment[0]["trial_ordinal"]),
                "end_trial_ordinal": int(segment[-1]["trial_ordinal"]),
                "num_trials": len(segment),
                "metrics": metrics_for(segment),
            }
        )

    return {
        "overall": metrics_for(records),
        "warmup_predictions": metrics_for(warmup),
        "post_warmup": metrics_for(post_warmup),
        "after_first_update": metrics_for(after_first_update),
        "segments": segments,
    }


def _latency_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean_ms": None, "p50_ms": None, "p95_ms": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(array)),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
    }


def summarize_updates(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    updates = [item for item in records if bool(item["update_applied"])]
    latencies = [float(item["update_latency_ms"]) for item in updates]
    return {
        "num_updates": len(updates),
        "trials_after_which_update_applied": [
            int(item["trial_ordinal"]) for item in updates
        ],
        "latency": _latency_summary(latencies),
        "updates": [
            {
                "trial_ordinal": int(item["trial_ordinal"]),
                "update_step_after_prediction": int(item["update_step_after_prediction"]),
                "model_revision_after_prediction": item[
                    "model_revision_after_prediction"
                ],
                "samples_used": int(item["update_samples_used"]),
                "latency_ms": float(item["update_latency_ms"]),
                "reason": item["update_reason"],
                "metrics": item["update_metrics"],
            }
            for item in updates
        ],
    }


def _raw_window_for_trial(
    *,
    trial: np.ndarray,
    metadata: HDF5Metadata,
    trial_id: str,
    subject_id: int,
    session: str,
    trial_ordinal: int,
) -> RawEEGWindow:
    return RawEEGWindow(
        data=np.ascontiguousarray(trial, dtype=np.float32),
        channel_names=list(metadata.channel_names),
        sample_rate=float(metadata.sample_rate),
        unit=str(metadata.unit),
        layout="CT",
        start_time_sec=0.0,
        trial_id=trial_id,
        window_id=f"{trial_id}:trial4s",
        label=None,
        metadata={
            "session": session,
            "subject_id": int(subject_id),
            "trial_ordinal": int(trial_ordinal),
            "protocol": "neuroonline_sequential_4s",
        },
    )


def _make_not_applied_result(*, mode: str, update_step: int, revision: str) -> OnlineUpdateResult:
    return OnlineUpdateResult(
        strategy_name=mode,
        applied=False,
        update_step=update_step,
        model_revision=revision,
        samples_used=0,
        latency_ms=0.0,
        reason="online_strategy_none",
        metrics={},
    )


def _progress_line(
    *,
    mode: str,
    record: dict[str, Any],
    total: int,
) -> str:
    return (
        f"[{mode}] trial {record['trial_ordinal']}/{total} "
        f"pred={record['predicted_class_name']} "
        f"true={record['true_class_name']} "
        f"acc={record['correct']} "
        f"update_step_used={record['update_step_used']}"
    )


def evaluate_mode(
    *,
    mode: Literal["none", "neuroonline"],
    loaded: LoadedRuntimePackage,
    dataset: SequentialDataset,
    settings: SequentialSettings,
    strategy_factory: Callable[[NeuroOnlineConfig], NeuroOnlineStrategy] = NeuroOnlineStrategy,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runtime_model: RuntimeModel = loaded.runtime_model
    class_names = tuple(loaded.class_names)
    strategy: NeuroOnlineStrategy | None = None

    if mode == "neuroonline":
        strategy = strategy_factory(settings.neuroonline_config)
        strategy.initialize(
            runtime_model=runtime_model,
            context=AdaptationContext(
                run_id=(
                    "neuroonline-sequential-"
                    f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
                ),
                subject_id=settings.subject_id,
                session=settings.session,
                metadata={
                    "source": "evaluate_neuroonline_sequential",
                    "data_path": str(settings.data_path),
                    "model_package": str(settings.model_package),
                },
            ),
        )

    records: list[dict[str, Any]] = []
    true_so_far: list[int] = []
    pred_so_far: list[int] = []

    for index in range(dataset.num_trials):
        trial_ordinal = index + 1
        true_label = int(dataset.labels[index])
        if not 0 <= true_label < len(class_names):
            raise ValueError(
                "Trial label is outside class range: "
                f"trial_ordinal={trial_ordinal}, label={true_label}."
            )

        trial_id = str(dataset.trial_ids[index])
        subject_id = int(dataset.subject_ids[index])
        raw_window = _raw_window_for_trial(
            trial=dataset.data[index],
            metadata=dataset.metadata,
            trial_id=trial_id,
            subject_id=subject_id,
            session=settings.session,
            trial_ordinal=trial_ordinal,
        )

        if mode == "none":
            update_step_used = 0
            revision_used = "static-0"
            predictor = runtime_model
        else:
            if strategy is None:
                raise RuntimeError("NeuroOnline strategy is missing.")
            update_step_used = int(strategy.update_step)
            revision_used = str(strategy.model_revision)
            predictor = strategy

        total_started = time.perf_counter()
        preprocessing_started = time.perf_counter()
        prepared = runtime_model.prepare(raw_window)
        preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000.0

        prediction_started = time.perf_counter()
        output = predictor.predict_prepared(prepared, return_features=False)
        prediction_ms = (time.perf_counter() - prediction_started) * 1000.0
        total_ms = (time.perf_counter() - total_started) * 1000.0

        probabilities = _probability_vector(output, num_classes=len(class_names))
        predicted_label = int(output.predicted_class)
        confidence = float(output.confidence)

        true_so_far.append(true_label)
        pred_so_far.append(predicted_label)
        rolling_start = max(0, len(true_so_far) - settings.rolling_window)
        rolling_metrics = _classification_metrics(
            true_so_far[rolling_start:],
            pred_so_far[rolling_start:],
            class_names=class_names,
        )

        if mode == "none":
            update_result = _make_not_applied_result(
                mode=mode,
                update_step=update_step_used,
                revision=revision_used,
            )
        else:
            if strategy is None:
                raise RuntimeError("NeuroOnline strategy is missing.")
            observation_id = f"{settings.session}:{trial_id}:{trial_ordinal}"
            observation = OnlineObservation(
                observation_id=observation_id,
                prepared_input=prepared,
                output=output,
                timestamp_sec=time.time(),
                metadata={
                    "trial_ordinal": trial_ordinal,
                    "trial_id": trial_id,
                    "subject_id": subject_id,
                    "label_revealed": False,
                },
            )

            # Causal order: prediction has already completed and been
            # materialized above. Only now is the true label revealed.
            strategy.observe(observation)
            strategy.submit_feedback(
                FeedbackEvent(
                    observation_id=observation_id,
                    label=true_label,
                    reward=None,
                    timestamp_sec=time.time(),
                    metadata={
                        "source": "offline_ground_truth",
                        "trial_ordinal": trial_ordinal,
                    },
                )
            )
            update_result = strategy.maybe_update(runtime_model=runtime_model)

        record = {
            "mode": mode,
            "trial_ordinal": trial_ordinal,
            "source_trial_id": trial_id,
            "subject_id": subject_id,
            "true_class_id": true_label,
            "true_class_name": class_names[true_label],
            "predicted_class_id": predicted_label,
            "predicted_class_name": class_names[predicted_label],
            "correct": predicted_label == true_label,
            "confidence": confidence,
            "probabilities": probabilities,
            "model_revision_used": revision_used,
            "update_step_used": update_step_used,
            "model_revision_after_prediction": str(update_result.model_revision),
            "update_step_after_prediction": int(update_result.update_step),
            "update_applied": bool(update_result.applied),
            "update_samples_used": int(update_result.samples_used),
            "update_latency_ms": float(update_result.latency_ms),
            "update_reason": update_result.reason,
            "update_metrics": _to_jsonable(update_result.metrics),
            "preprocessing_latency_ms": float(preprocessing_ms),
            "prediction_latency_ms": float(prediction_ms),
            "preprocess_predict_latency_ms": float(total_ms),
            "rolling_balanced_accuracy": rolling_metrics["balanced_accuracy"],
            "rolling_window_start_trial": rolling_start + 1,
            "rolling_window_sample_count": len(true_so_far[rolling_start:]),
            "rolling_present_class_ids": rolling_metrics["present_class_ids"],
            "rolling_present_class_names": rolling_metrics["present_class_names"],
            "model_output_diagnostics": _to_jsonable(output.diagnostics),
        }
        records.append(record)

        if (
            settings.print_every > 0
            and (trial_ordinal % settings.print_every == 0 or trial_ordinal == dataset.num_trials)
        ):
            print(_progress_line(mode=mode, record=record, total=dataset.num_trials))

    mode_summary = {
        "mode": mode,
        "num_trials": len(records),
        "metrics": compute_metrics_for_records(
            records,
            class_names=class_names,
            warmup_feedback=settings.neuroonline_config.warmup_feedback,
            block_size=settings.block_size,
        ),
        "updates": summarize_updates(records),
    }
    return records, mode_summary


def compare_identity_before_first_update(
    *,
    static_records: Sequence[dict[str, Any]],
    neuroonline_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    static_by_trial = {
        int(item["trial_ordinal"]): item
        for item in static_records
    }
    comparisons = []
    for online in neuroonline_records:
        if int(online["update_step_used"]) != 0:
            continue
        ordinal = int(online["trial_ordinal"])
        static = static_by_trial.get(ordinal)
        if static is None:
            raise ValueError(
                "Static and NeuroOnline records are not paired for "
                f"trial_ordinal={ordinal}."
            )
        static_prob = np.asarray(static["probabilities"], dtype=np.float64)
        online_prob = np.asarray(online["probabilities"], dtype=np.float64)
        comparisons.append(
            {
                "trial_ordinal": ordinal,
                "agreed": int(static["predicted_class_id"])
                == int(online["predicted_class_id"]),
                "max_probability_abs_diff": float(
                    np.max(np.abs(static_prob - online_prob))
                ),
            }
        )

    count = len(comparisons)
    agreement_rate = (
        None
        if count == 0
        else float(np.mean([item["agreed"] for item in comparisons]))
    )
    max_diff = (
        None
        if count == 0
        else float(max(item["max_probability_abs_diff"] for item in comparisons))
    )
    equivalent = bool(count > 0 and agreement_rate == 1.0 and (max_diff or 0.0) <= 1e-6)
    return {
        "comparison_trial_count": count,
        "prediction_agreement_rate": agreement_rate,
        "maximum_probability_absolute_difference": max_diff,
        "equivalent": equivalent,
        "warning": None
        if equivalent
        else "Static and NeuroOnline predictions differ before the first parameter update.",
    }


def _metric_gain(
    online_metrics: dict[str, Any],
    static_metrics: dict[str, Any],
    name: str,
) -> float | None:
    online_value = online_metrics.get(name)
    static_value = static_metrics.get(name)
    if online_value is None or static_value is None:
        return None
    return float(online_value) - float(static_value)


def paired_gains(
    *,
    static_summary: dict[str, Any],
    neuroonline_summary: dict[str, Any],
    static_records: Sequence[dict[str, Any]] | None = None,
    neuroonline_records: Sequence[dict[str, Any]] | None = None,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for section in ("overall", "post_warmup", "after_first_update"):
        static_metrics = static_summary["metrics"][section]
        online_metrics = neuroonline_summary["metrics"][section]
        if (
            section == "after_first_update"
            and static_records is not None
            and neuroonline_records is not None
            and class_names is not None
            and online_metrics.get("num_samples", 0) > 0
            and static_metrics.get("num_samples", 0) == 0
        ):
            post_update_ordinals = {
                int(item["trial_ordinal"])
                for item in neuroonline_records
                if int(item["update_step_used"]) > 0
            }
            paired_static = [
                item
                for item in static_records
                if int(item["trial_ordinal"]) in post_update_ordinals
            ]
            static_metrics = _classification_metrics(
                [int(item["true_class_id"]) for item in paired_static],
                [int(item["predicted_class_id"]) for item in paired_static],
                class_names=class_names,
            )
        result[section] = {
            "accuracy_gain": _metric_gain(online_metrics, static_metrics, "accuracy"),
            "balanced_accuracy_gain": _metric_gain(
                online_metrics,
                static_metrics,
                "balanced_accuracy",
            ),
            "macro_f1_gain": _metric_gain(online_metrics, static_metrics, "macro_f1"),
        }
    return result


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
    )


def _atomic_write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    lines = [
        json.dumps(_to_jsonable(record), ensure_ascii=False, separators=(",", ":"))
        for record in records
    ]
    _atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def _atomic_write_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "mode",
        "trial_ordinal",
        "source_trial_id",
        "subject_id",
        "true_class_id",
        "true_class_name",
        "predicted_class_id",
        "predicted_class_name",
        "correct",
        "confidence",
        "probabilities",
        "model_revision_used",
        "update_step_used",
        "model_revision_after_prediction",
        "update_step_after_prediction",
        "update_applied",
        "update_samples_used",
        "update_latency_ms",
        "update_reason",
        "update_metrics",
        "preprocessing_latency_ms",
        "prediction_latency_ms",
        "preprocess_predict_latency_ms",
        "rolling_balanced_accuracy",
        "rolling_window_start_trial",
        "rolling_window_sample_count",
        "rolling_present_class_ids",
        "rolling_present_class_names",
        "model_output_diagnostics",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {}
            for key in fieldnames:
                value = record.get(key)
                if isinstance(value, (dict, list, tuple)):
                    row[key] = json.dumps(_to_jsonable(value), ensure_ascii=False)
                else:
                    row[key] = value
            writer.writerow(row)
    tmp.replace(path)


def _selected_modes(strategy: str) -> list[Literal["none", "neuroonline"]]:
    if strategy == "both":
        return ["none", "neuroonline"]
    if strategy == "none":
        return ["none"]
    if strategy == "neuroonline":
        return ["neuroonline"]
    raise ValueError(f"Unsupported online strategy: {strategy!r}.")


def _build_warnings(settings: SequentialSettings) -> list[str]:
    messages: list[str] = []
    if (
        settings.online_strategy in {"neuroonline", "both"}
        and settings.max_trials is not None
        and settings.max_trials <= settings.neuroonline_config.warmup_feedback
    ):
        message = (
            "max_trials <= warmup_feedback; no prediction will use updated "
            "NeuroOnline parameters, so this run cannot judge accuracy gain."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        print(f"WARNING: {message}", file=sys.stderr)
        messages.append(message)
    return messages


def run_evaluation(
    settings: SequentialSettings,
    *,
    package_loader: Callable[..., LoadedRuntimePackage] | None = None,
    strategy_factory: Callable[[NeuroOnlineConfig], NeuroOnlineStrategy] = NeuroOnlineStrategy,
) -> dict[str, Any]:
    if package_loader is None:
        from bci_dayloop.packages.loader import load_runtime_package

        package_loader = load_runtime_package

    warnings_list = _build_warnings(settings)
    dataset = load_sequential_dataset(settings)

    mode_records: dict[str, list[dict[str, Any]]] = {}
    mode_summaries: dict[str, dict[str, Any]] = {}
    package_info: dict[str, Any] | None = None

    for mode in _selected_modes(settings.online_strategy):
        loaded = package_loader(
            settings.model_package,
            device=settings.device,
            verify_hashes=True,
        )
        validate_package_contract(loaded, metadata=dataset.metadata)
        if package_info is None:
            package_info = {
                "path": str(loaded.package_path),
                "model_type": loaded.model_type,
                "model_name": loaded.model_name,
                "package_step_sec": loaded.step_sec,
                "window_sec": loaded.window_sec,
                "class_names": list(loaded.class_names),
                "is_test_head": loaded.is_test_head,
                "warning_message": loaded.warning_message,
            }

        print(f"Running mode: {mode}")
        records, summary = evaluate_mode(
            mode=mode,
            loaded=loaded,
            dataset=dataset,
            settings=settings,
            strategy_factory=strategy_factory,
        )
        mode_records[mode] = records
        mode_summaries[mode] = summary

    all_records = [
        record
        for mode in _selected_modes(settings.online_strategy)
        for record in mode_records[mode]
    ]
    identity_check = None
    gains = None
    if settings.online_strategy == "both":
        identity_check = compare_identity_before_first_update(
            static_records=mode_records["none"],
            neuroonline_records=mode_records["neuroonline"],
        )
        if identity_check["warning"] is not None:
            warnings_list.append(str(identity_check["warning"]))
            print(f"WARNING: {identity_check['warning']}", file=sys.stderr)
        gains = paired_gains(
            static_summary=mode_summaries["none"],
            neuroonline_summary=mode_summaries["neuroonline"],
            static_records=mode_records["none"],
            neuroonline_records=mode_records["neuroonline"],
            class_names=dataset.metadata.class_names,
        )

    output_dir = settings.output_dir
    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "trial_predictions.csv"
    jsonl_path = output_dir / "trial_predictions.jsonl"

    summary_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "neuroonline_sequential_trial_4s",
            "window_sec": FIXED_WINDOW_SEC,
            "step_sec": FIXED_STEP_SEC,
            "one_prediction_per_source_trial": True,
            "uses_replay_acquirer": False,
            "uses_sliding_window_decoder": False,
            "continuous_stream_concatenation": False,
            "shuffle": False,
            "label_available_during_prediction": False,
            "package_step_sec_ignored_for_trial_sequence": True,
        },
        "data": {
            "path": str(settings.data_path),
            "session": settings.session,
            "dataset_name": dataset.metadata.dataset_name,
            "sample_rate": dataset.metadata.sample_rate,
            "unit": dataset.metadata.unit,
            "channel_names": dataset.metadata.channel_names,
            "class_names": dataset.metadata.class_names,
            "num_trials": dataset.num_trials,
            "source_shape": list(dataset.data.shape),
            "trial_ids_preserved_in_hdf5_order": [
                str(item) for item in dataset.trial_ids.tolist()
            ],
        },
        "runtime_package": package_info,
        "online_strategy": settings.online_strategy,
        "neuroonline_config": asdict(settings.neuroonline_config),
        "static": mode_summaries.get("none"),
        "neuroonline": mode_summaries.get("neuroonline"),
        "gains": gains,
        "identity_initialization_check": identity_check,
        "warnings": warnings_list,
        "output_files": {
            "summary_json": str(summary_path),
            "trial_predictions_csv": str(csv_path),
            "trial_predictions_jsonl": str(jsonl_path),
        },
    }

    _atomic_write_csv(csv_path, all_records)
    _atomic_write_jsonl(jsonl_path, all_records)
    _atomic_write_json(summary_path, summary_payload)
    return summary_payload


def _print_metric_line(title: str, metrics: dict[str, Any] | None) -> None:
    if metrics is None:
        return
    overall = metrics["metrics"]["overall"]
    print(
        f"{title}: accuracy={overall['accuracy']:.4f}, "
        f"bACC={overall['balanced_accuracy']:.4f}, "
        f"macro-F1={overall['macro_f1']:.4f}"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_yaml(resolve_path(args.config))
    settings = resolve_settings(args, config)
    summary = run_evaluation(settings)

    print()
    print("NeuroOnline sequential evaluation completed.")
    print(f"Dataset:     {settings.data_path}")
    print(f"Session:     {settings.session}")
    print(f"Package:     {settings.model_package}")
    print(f"Output dir:  {settings.output_dir}")
    _print_metric_line("Static", summary.get("static"))
    _print_metric_line("NeuroOnline", summary.get("neuroonline"))
    if summary.get("gains") is not None:
        print("Gains (NeuroOnline - Static):")
        print(json.dumps(summary["gains"], ensure_ascii=False, indent=2))
    if summary.get("identity_initialization_check") is not None:
        print("Identity initialization check:")
        print(
            json.dumps(
                summary["identity_initialization_check"],
                ensure_ascii=False,
                indent=2,
            )
        )
    print(f"Summary:     {summary['output_files']['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
