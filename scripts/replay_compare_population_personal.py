from __future__ import annotations

"""Compare population and personal 50M Runtime Model Packages on identical EEG windows.

The recommended Stage-1 use is target-subject ``1test`` with a 4-second
Runtime Model Package. In that case each source trial is one label-pure model
window. For a 10-second package backed by 4-second source trials, the optional
``same_label_concat`` mode concatenates trials only within the same class.

Per-window outputs include ground truth, both predictions/confidences,
agreement, correctness, latency, source Trial IDs, and completion state.
"""

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from _bootstrap import ROOT  # noqa: F401
from bci_dayloop.data.hdf5_dataset import EEGHDF5, HDF5Metadata
from bci_dayloop.models.factory import ModelFactory
from bci_dayloop.models.runtime_package import ModelRuntimePackage
from bci_dayloop.utils.config import load_yaml, resolve_path


@dataclass(frozen=True, slots=True)
class ComparisonWindow:
    window_id: int
    signal: np.ndarray
    class_id: int
    class_name: str
    source_trial_ids: tuple[int, ...]
    construction: str


@dataclass(frozen=True, slots=True)
class Prediction:
    class_id: int
    class_name: str
    confidence: float
    probabilities: tuple[float, ...]
    model_latency_ms: float
    total_latency_ms: float
    adapter_timing: dict[str, float] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WindowResult:
    window_id: int
    status: str
    construction: str
    source_trial_ids: tuple[int, ...]
    ground_truth_id: int
    ground_truth_name: str
    preprocessing_latency_ms: float | None
    population: Prediction | None
    personal: Prediction | None
    models_agree: bool | None
    population_correct: bool | None
    personal_correct: bool | None
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "status": self.status,
            "construction": self.construction,
            "source_trial_ids": list(self.source_trial_ids),
            "ground_truth_id": self.ground_truth_id,
            "ground_truth_name": self.ground_truth_name,
            "preprocessing_latency_ms": self.preprocessing_latency_ms,
            "population": self.population.to_dict() if self.population else None,
            "personal": self.personal.to_dict() if self.personal else None,
            "models_agree": self.models_agree,
            "population_correct": self.population_correct,
            "personal_correct": self.personal_correct,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare population and personal model packages on the same target EEG windows."
    )
    parser.add_argument("--data", default="data/processed/bnci2014_001/subject_01.h5")
    parser.add_argument("--session", default="1test")
    parser.add_argument("--population-package", required=True)
    parser.add_argument("--personal-package", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--window-mode",
        choices=("auto", "direct_trial", "same_label_concat"),
        default="auto",
    )
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--start-window", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--alternate-model-order", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--allow-contract-mismatch", action="store_true")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--jsonl-output")
    parser.add_argument("--csv-output")
    parser.add_argument("--summary-output")
    return parser


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, default=_json_default) + "\n")


def _default_outputs(data_path: Path, session: str) -> tuple[Path, Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = ROOT / "runs" / "stage1" / "comparisons" / data_path.stem / session / timestamp
    return root / "windows.jsonl", root / "windows.csv", root / "summary.json"


def _contract(package: Path) -> dict[str, Any]:
    model = load_yaml(package / "model.yaml")
    preprocessing = load_yaml(package / "preprocessing.yaml")
    model_keys = (
        "name", "num_classes", "aggregation", "output_layer_idx",
        "window_seconds", "target_sample_rate", "patch_seconds",
        "patch_stride_seconds", "n_channels", "d_model",
        "model_n_time_patches", "class_names",
    )
    preprocessing_keys = (
        "filter_enabled", "filter_low_hz", "filter_high_hz", "filter_order",
        "reference_mode", "zscore_enabled", "zscore_eps",
        "missing_channel_fill_value", "strict_window_duration",
        "window_tolerance_seconds",
    )
    return {
        "model": {key: model[key] for key in model_keys if key in model},
        "preprocessing": {
            key: preprocessing[key] for key in preprocessing_keys if key in preprocessing
        },
    }


def validate_packages(
    population: ModelRuntimePackage,
    personal: ModelRuntimePackage,
    population_path: Path,
    personal_path: Path,
    allow_mismatch: bool,
) -> dict[str, Any]:
    required = {
        "model_name": (population.model_name, personal.model_name),
        "class_names": (tuple(population.class_names), tuple(personal.class_names)),
        "window_sec": (float(population.window_sec), float(personal.window_sec)),
        "target_sample_rate": (
            float(population.target_sample_rate), float(personal.target_sample_rate)
        ),
    }
    errors = [
        f"{name}: population={left!r}, personal={right!r}"
        for name, (left, right) in required.items()
        if left != right
    ]
    population_contract = _contract(population_path)
    personal_contract = _contract(personal_path)
    warnings: list[str] = []
    if population_contract != personal_contract:
        message = "Population and personal model/preprocessing contracts are not identical."
        (warnings if allow_mismatch else errors).append(message)
    if errors:
        raise ValueError(
            "Package compatibility check failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    return {
        "population_contract": population_contract,
        "personal_contract": personal_contract,
        "warnings": warnings,
    }


def _validate_session(data: Mapping[str, np.ndarray], metadata: HDF5Metadata) -> None:
    required = {"data", "labels", "subject_ids", "session_ids", "trial_ids"}
    missing = required - set(data)
    if missing:
        raise KeyError(f"HDF5 session is missing fields: {sorted(missing)}")
    trials = np.asarray(data["data"])
    labels = np.asarray(data["labels"])
    if trials.ndim != 3 or len(trials) == 0:
        raise ValueError(f"Expected non-empty [N,C,T] trials, got {trials.shape}")
    if trials.shape[1] != len(metadata.channel_names):
        raise ValueError("Channel dimension does not match HDF5 metadata")
    if len(labels) != len(trials) or len(data["trial_ids"]) != len(trials):
        raise ValueError("Trial-level array lengths differ")
    if not np.isfinite(trials).all():
        raise ValueError("HDF5 trials contain NaN or Inf")
    if labels.min() < 0 or labels.max() >= len(metadata.class_names):
        raise ValueError("HDF5 labels are outside the class range")


def _direct_windows(
    data: Mapping[str, np.ndarray], metadata: HDF5Metadata, expected_samples: int
) -> Iterator[ComparisonWindow]:
    trials = np.asarray(data["data"], dtype=np.float32)
    labels = np.asarray(data["labels"], dtype=np.int64)
    trial_ids = np.asarray(data["trial_ids"], dtype=np.int64)
    if trials.shape[-1] != expected_samples:
        raise ValueError(
            "direct_trial requires source trial duration to equal the model window: "
            f"source={trials.shape[-1] / metadata.sample_rate:.3f}s, "
            f"model={expected_samples / metadata.sample_rate:.3f}s"
        )
    for index, trial in enumerate(trials):
        class_id = int(labels[index])
        yield ComparisonWindow(
            window_id=index + 1,
            signal=np.ascontiguousarray(trial),
            class_id=class_id,
            class_name=str(metadata.class_names[class_id]),
            source_trial_ids=(int(trial_ids[index]),),
            construction="direct_trial",
        )


def _same_label_windows(
    data: Mapping[str, np.ndarray], metadata: HDF5Metadata, expected_samples: int
) -> Iterator[ComparisonWindow]:
    trials = np.asarray(data["data"], dtype=np.float32)
    labels = np.asarray(data["labels"], dtype=np.int64)
    trial_ids = np.asarray(data["trial_ids"], dtype=np.int64)
    source_trial_samples = int(trials.shape[-1])
    window_id = 0
    for class_id, class_name in enumerate(metadata.class_names):
        indices = np.flatnonzero(labels == class_id)
        if len(indices) == 0:
            continue
        signal = np.concatenate([trials[int(i)] for i in indices], axis=-1)
        ids = trial_ids[indices]
        for start in range(0, signal.shape[-1] - expected_samples + 1, expected_samples):
            stop = start + expected_samples
            first = start // source_trial_samples
            last = (stop - 1) // source_trial_samples
            source_ids = tuple(int(v) for v in ids[first : last + 1].tolist())
            window_id += 1
            yield ComparisonWindow(
                window_id=window_id,
                signal=np.ascontiguousarray(signal[:, start:stop]),
                class_id=int(class_id),
                class_name=str(class_name),
                source_trial_ids=source_ids,
                construction="same_label_concat",
            )


def build_windows(
    data: Mapping[str, np.ndarray],
    metadata: HDF5Metadata,
    window_sec: float,
    mode: str,
) -> tuple[str, list[ComparisonWindow]]:
    value = window_sec * metadata.sample_rate
    expected_samples = int(round(value))
    if not math.isclose(value, expected_samples, abs_tol=1e-6):
        raise ValueError(f"window_sec * sample_rate is not integral: {value}")
    source_samples = int(np.asarray(data["data"]).shape[-1])
    selected = mode
    if selected == "auto":
        selected = "direct_trial" if source_samples == expected_samples else "same_label_concat"
    iterator = (
        _direct_windows(data, metadata, expected_samples)
        if selected == "direct_trial"
        else _same_label_windows(data, metadata, expected_samples)
    )
    windows = list(iterator)
    if not windows:
        raise ValueError(f"No windows generated with mode={selected}")
    return selected, windows


def _batch(model_input: Mapping[str, np.ndarray] | np.ndarray) -> Any:
    if isinstance(model_input, Mapping):
        return {str(key): np.asarray(value)[None, ...] for key, value in model_input.items()}
    return np.asarray(model_input)[None, ...]


def _adapter_timing(model: Any) -> dict[str, float] | None:
    timing = getattr(model, "last_timing", None)
    if timing is None:
        return None
    to_dict = getattr(timing, "to_dict", None)
    if not callable(to_dict):
        return None
    return {str(key): float(value) for key, value in to_dict().items()}


def predict(
    runtime: ModelRuntimePackage,
    model_input: Any,
    preprocessing_ms: float,
) -> Prediction:
    started = time.perf_counter()
    batch_probabilities = np.asarray(runtime.model.predict_proba(model_input), dtype=np.float32)
    latency_ms = (time.perf_counter() - started) * 1000.0
    expected = (1, len(runtime.class_names))
    if batch_probabilities.shape != expected:
        raise RuntimeError(f"Expected probability shape {expected}, got {batch_probabilities.shape}")
    probabilities = batch_probabilities[0]
    if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1.0, atol=1e-4):
        raise RuntimeError("Invalid model probability vector")
    class_id = int(np.argmax(probabilities))
    return Prediction(
        class_id=class_id,
        class_name=str(runtime.class_names[class_id]),
        confidence=float(probabilities[class_id]),
        probabilities=tuple(float(v) for v in probabilities.tolist()),
        model_latency_ms=float(latency_ms),
        total_latency_ms=float(preprocessing_ms + latency_ms),
        adapter_timing=_adapter_timing(runtime.model),
    )


def metrics(labels: Sequence[int], predictions: Sequence[int], class_names: Sequence[str]) -> dict[str, Any]:
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, prediction in zip(labels, predictions):
        confusion[int(label), int(prediction)] += 1
    accuracy = float(np.trace(confusion) / max(int(confusion.sum()), 1))
    recalls: list[float] = []
    f1s: list[float] = []
    per_class: list[dict[str, Any]] = []
    for index, name in enumerate(class_names):
        tp = int(confusion[index, index])
        support = int(confusion[index, :].sum())
        predicted = int(confusion[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            recalls.append(recall)
            f1s.append(f1)
        per_class.append({
            "class_id": index,
            "class_name": str(name),
            "support": support,
            "predicted": predicted,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"),
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def latency_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=np.float64)
    if array.size == 0:
        return {key: None for key in ("current_ms", "mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms")} | {"count": 0}
    return {
        "count": int(array.size),
        "current_ms": float(array[-1]),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def print_result(result: WindowResult, threshold: float) -> None:
    print(f"\nCurrent Window #{result.window_id}")
    print(f"Ground Truth:       {result.ground_truth_name}")
    if result.status != "success":
        print(f"Status:             FAILED ({result.error_type}: {result.error_message})")
        return
    assert result.population is not None and result.personal is not None
    pop_flag = "LOW" if result.population.confidence < threshold else "OK"
    per_flag = "LOW" if result.personal.confidence < threshold else "OK"
    print(
        f"Population Model:   {result.population.class_name:<14} "
        f"{result.population.confidence:.4f} [{pop_flag}] "
        f"{result.population.model_latency_ms:.1f} ms"
    )
    print(
        f"Personal Model:     {result.personal.class_name:<14} "
        f"{result.personal.confidence:.4f} [{per_flag}] "
        f"{result.personal.model_latency_ms:.1f} ms"
    )
    print(f"Models Agree:       {'YES' if result.models_agree else 'NO'}")
    print(
        f"Correct:            population={result.population_correct}, "
        f"personal={result.personal_correct}"
    )
    print(f"Shared Preprocess:  {result.preprocessing_latency_ms:.1f} ms")
    print(f"Window Source:      {result.construction}, trials={list(result.source_trial_ids)}")


def save_csv(results: Sequence[WindowResult], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        pop, per = result.population, result.personal
        rows.append({
            "window_id": result.window_id,
            "status": result.status,
            "construction": result.construction,
            "source_trial_ids": json.dumps(list(result.source_trial_ids)),
            "ground_truth_id": result.ground_truth_id,
            "ground_truth_name": result.ground_truth_name,
            "preprocessing_latency_ms": result.preprocessing_latency_ms,
            "population_class_id": pop.class_id if pop else None,
            "population_class_name": pop.class_name if pop else None,
            "population_confidence": pop.confidence if pop else None,
            "population_model_latency_ms": pop.model_latency_ms if pop else None,
            "population_total_latency_ms": pop.total_latency_ms if pop else None,
            "population_probabilities": json.dumps(list(pop.probabilities)) if pop else None,
            "personal_class_id": per.class_id if per else None,
            "personal_class_name": per.class_name if per else None,
            "personal_confidence": per.confidence if per else None,
            "personal_model_latency_ms": per.model_latency_ms if per else None,
            "personal_total_latency_ms": per.total_latency_ms if per else None,
            "personal_probabilities": json.dumps(list(per.probabilities)) if per else None,
            "models_agree": result.models_agree,
            "population_correct": result.population_correct,
            "personal_correct": result.personal_correct,
            "error_type": result.error_type,
            "error_message": result.error_message,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_windows is not None and args.max_windows < 0:
        raise ValueError("--max-windows must be non-negative")
    if args.start_window < 0 or args.print_every < 0:
        raise ValueError("--start-window and --print-every must be non-negative")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be in [0,1]")

    data_path = resolve_path(args.data)
    population_path = resolve_path(args.population_package)
    personal_path = resolve_path(args.personal_package)
    default_jsonl, default_csv, default_summary = _default_outputs(data_path, args.session)
    jsonl_path = resolve_path(args.jsonl_output) if args.jsonl_output else default_jsonl
    csv_path = resolve_path(args.csv_output) if args.csv_output else default_csv
    summary_path = resolve_path(args.summary_output) if args.summary_output else default_summary
    if jsonl_path.exists():
        jsonl_path.unlink()

    dataset = EEGHDF5(data_path)
    metadata = dataset.metadata
    session_data = dataset.load(args.session)
    _validate_session(session_data, metadata)

    print("Loading population package...")
    population_runtime = ModelFactory.load_runtime_package(population_path, metadata, device=args.device)
    print("Loading personal package...")
    personal_runtime = ModelFactory.load_runtime_package(personal_path, metadata, device=args.device)
    compatibility = validate_packages(
        population_runtime,
        personal_runtime,
        population_path,
        personal_path,
        args.allow_contract_mismatch,
    )

    selected_mode, available_windows = build_windows(
        session_data, metadata, population_runtime.window_sec, args.window_mode
    )
    selected_windows = available_windows[args.start_window :]
    if args.max_windows is not None:
        selected_windows = selected_windows[: args.max_windows]
    if not selected_windows:
        raise ValueError("No windows remain after --start-window/--max-windows")

    print("\n" + "=" * 88)
    print("Population vs Personal Replay Comparison")
    print("=" * 88)
    print("data:", data_path)
    print("session:", args.session)
    print("population package:", population_path)
    print("personal package:", personal_path)
    print("device:", args.device)
    print("window sec:", population_runtime.window_sec)
    print("window construction:", selected_mode)
    print("available / selected windows:", len(available_windows), "/", len(selected_windows))
    for warning in compatibility["warnings"]:
        print("WARNING:", warning, file=sys.stderr)

    results: list[WindowResult] = []
    preprocess_latencies: list[float] = []
    pop_latencies: list[float] = []
    pop_total_latencies: list[float] = []
    per_latencies: list[float] = []
    per_total_latencies: list[float] = []
    run_started = time.perf_counter()

    for ordinal, window in enumerate(selected_windows, start=1):
        try:
            started = time.perf_counter()
            model_input = population_runtime.preprocessor.transform(
                window.signal, metadata.sample_rate, metadata.unit, reshape=True
            )
            preprocessing_ms = (time.perf_counter() - started) * 1000.0
            batched = _batch(model_input)
            personal_first = args.alternate_model_order and ordinal % 2 == 0
            if personal_first:
                personal_prediction = predict(personal_runtime, batched, preprocessing_ms)
                population_prediction = predict(population_runtime, batched, preprocessing_ms)
            else:
                population_prediction = predict(population_runtime, batched, preprocessing_ms)
                personal_prediction = predict(personal_runtime, batched, preprocessing_ms)
            result = WindowResult(
                window_id=window.window_id,
                status="success",
                construction=window.construction,
                source_trial_ids=window.source_trial_ids,
                ground_truth_id=window.class_id,
                ground_truth_name=window.class_name,
                preprocessing_latency_ms=preprocessing_ms,
                population=population_prediction,
                personal=personal_prediction,
                models_agree=population_prediction.class_id == personal_prediction.class_id,
                population_correct=population_prediction.class_id == window.class_id,
                personal_correct=personal_prediction.class_id == window.class_id,
            )
            preprocess_latencies.append(preprocessing_ms)
            pop_latencies.append(population_prediction.model_latency_ms)
            pop_total_latencies.append(population_prediction.total_latency_ms)
            per_latencies.append(personal_prediction.model_latency_ms)
            per_total_latencies.append(personal_prediction.total_latency_ms)
        except Exception as error:  # noqa: BLE001
            result = WindowResult(
                window_id=window.window_id,
                status="failed",
                construction=window.construction,
                source_trial_ids=window.source_trial_ids,
                ground_truth_id=window.class_id,
                ground_truth_name=window.class_name,
                preprocessing_latency_ms=None,
                population=None,
                personal=None,
                models_agree=None,
                population_correct=None,
                personal_correct=None,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        results.append(result)
        _append_jsonl(jsonl_path, result.to_dict())
        if args.print_every and (ordinal == 1 or ordinal % args.print_every == 0):
            print_result(result, args.confidence_threshold)
        if result.status == "failed" and not args.continue_on_error:
            save_csv(results, csv_path)
            print(f"Window #{result.window_id} failed; partial outputs saved.", file=sys.stderr)
            return 1

    run_seconds = time.perf_counter() - run_started
    successful = [result for result in results if result.status == "success"]
    failed = [result for result in results if result.status == "failed"]
    if not successful:
        save_csv(results, csv_path)
        _atomic_json(summary_path, {
            "status": "failed",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "attempted_windows": len(results),
            "failed_windows": len(failed),
        })
        return 1

    labels = [result.ground_truth_id for result in successful]
    pop_predictions = [result.population.class_id for result in successful if result.population]
    per_predictions = [result.personal.class_id for result in successful if result.personal]
    pop_metrics = metrics(labels, pop_predictions, population_runtime.class_names)
    per_metrics = metrics(labels, per_predictions, population_runtime.class_names)
    agreement_count = sum(int(bool(result.models_agree)) for result in successful)
    completion_rate = len(successful) / len(selected_windows)

    summary = {
        "status": "completed" if not failed else "completed_with_errors",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "path": str(data_path),
            "dataset_name": metadata.dataset_name,
            "session": args.session,
            "sample_rate": metadata.sample_rate,
            "input_unit": metadata.unit,
            "class_names": list(metadata.class_names),
            "source_trial_count": int(len(session_data["data"])),
            "source_trial_duration_sec": float(session_data["data"].shape[-1] / metadata.sample_rate),
        },
        "packages": {
            "population": str(population_path),
            "personal": str(personal_path),
            "model_name": population_runtime.model_name,
            "window_sec": population_runtime.window_sec,
            "step_sec": population_runtime.step_sec,
            "target_sample_rate": population_runtime.target_sample_rate,
            "compatibility": compatibility,
        },
        "windowing": {
            "requested_mode": args.window_mode,
            "selected_mode": selected_mode,
            "available_windows": len(available_windows),
            "selected_windows": len(selected_windows),
            "successful_windows": len(successful),
            "failed_windows": len(failed),
            "window_completion_rate": completion_rate,
        },
        "comparison": {
            "agreement_count": agreement_count,
            "agreement_rate": agreement_count / len(successful),
            "population": pop_metrics,
            "personal": per_metrics,
            "gain": {
                "accuracy": per_metrics["accuracy"] - pop_metrics["accuracy"],
                "balanced_accuracy": per_metrics["balanced_accuracy"] - pop_metrics["balanced_accuracy"],
                "macro_f1": per_metrics["macro_f1"] - pop_metrics["macro_f1"],
            },
        },
        "latency_ms": {
            "shared_preprocessing": latency_summary(preprocess_latencies),
            "population_model": latency_summary(pop_latencies),
            "population_total": latency_summary(pop_total_latencies),
            "personal_model": latency_summary(per_latencies),
            "personal_total": latency_summary(per_total_latencies),
        },
        "runtime_seconds": run_seconds,
        "outputs": {"jsonl": str(jsonl_path), "csv": str(csv_path), "summary": str(summary_path)},
        "errors": [
            {"window_id": item.window_id, "error_type": item.error_type, "error_message": item.error_message}
            for item in failed
        ],
    }
    save_csv(results, csv_path)
    _atomic_json(summary_path, summary)

    print("\n" + "=" * 88)
    print("Comparison Summary")
    print("=" * 88)
    print(f"Windows successful / selected: {len(successful)} / {len(selected_windows)} ({completion_rate:.2%})")
    print(f"Agreement: {agreement_count} / {len(successful)} ({agreement_count / len(successful):.2%})")
    print(
        "Population Acc / BAcc / Macro-F1: "
        f"{pop_metrics['accuracy']:.4f} / {pop_metrics['balanced_accuracy']:.4f} / {pop_metrics['macro_f1']:.4f}"
    )
    print(
        "Personal Acc / BAcc / Macro-F1:   "
        f"{per_metrics['accuracy']:.4f} / {per_metrics['balanced_accuracy']:.4f} / {per_metrics['macro_f1']:.4f}"
    )
    gain = summary["comparison"]["gain"]
    print(
        "Gain Acc / BAcc / Macro-F1:       "
        f"{gain['accuracy']:+.4f} / {gain['balanced_accuracy']:+.4f} / {gain['macro_f1']:+.4f}"
    )
    print(
        "Population / Personal model P95:  "
        f"{summary['latency_ms']['population_model']['p95_ms']:.1f} ms / "
        f"{summary['latency_ms']['personal_model']['p95_ms']:.1f} ms"
    )
    print("JSONL:", jsonl_path)
    print("CSV:", csv_path)
    print("Summary:", summary_path)
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
