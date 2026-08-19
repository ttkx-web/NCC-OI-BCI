from __future__ import annotations

import argparse
import csv
import json
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from _bootstrap import ROOT

from bci_dayloop.data.hdf5_dataset import EEGHDF5
from bci_dayloop.data.splits import stratified_source_trial_split
from bci_dayloop.models.model_50m.backbone import Model50MBackbone
from bci_dayloop.models.model_50m.classifier import (
    Model50MClassifier,
    save_classifier_checkpoint,
)
from bci_dayloop.models.model_50m.config import (
    STANDARD_64_CHANNELS,
    Model50MConfig,
)
from bci_dayloop.models.model_50m.preprocessing import Model50MPreprocessor
from bci_dayloop.models.model_50m.tokenization import (
    Model50MTokenizer,
    stack_model50m_tokens,
)


@dataclass(frozen=True, slots=True)
class WindowSet:
    windows: np.ndarray  # [N,C,T]
    labels: np.ndarray  # [N]
    source_trial_ids: tuple[tuple[int, ...], ...]
    construction: str

    def __post_init__(self) -> None:
        if self.windows.ndim != 3:
            raise ValueError(
                f"windows must have shape [N,C,T], got {self.windows.shape}."
            )
        if self.labels.shape != (len(self.windows),):
            raise ValueError(
                "labels length does not match windows: "
                f"{self.labels.shape} vs {len(self.windows)}."
            )
        if len(self.source_trial_ids) != len(self.windows):
            raise ValueError("source_trial_ids length does not match windows.")
        if len(self.windows) == 0:
            raise ValueError("WindowSet is empty.")


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    loss: float
    accuracy: float
    balanced_accuracy: float
    confusion_matrix: list[list[int]]
    per_class_recall: list[float | None]


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable."
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def class_counts(labels: np.ndarray, num_classes: int) -> dict[int, int]:
    labels = np.asarray(labels, dtype=np.int64)
    return {
        class_index: int(np.sum(labels == class_index))
        for class_index in range(num_classes)
    }


def validate_labels(
    labels: np.ndarray,
    *,
    num_classes: int,
    split_name: str,
) -> None:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError(
            f"{split_name} labels must be one-dimensional, got {labels.shape}."
        )
    if len(labels) == 0:
        raise ValueError(f"{split_name} labels are empty.")
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(
            f"{split_name} labels must be in [0,{num_classes - 1}], "
            f"got {np.unique(labels).tolist()}."
        )
    missing = sorted(set(range(num_classes)) - set(labels.tolist()))
    if missing:
        raise ValueError(f"{split_name} is missing class(es): {missing}.")


def build_same_label_concat_windows(
    *,
    trials: np.ndarray,
    labels: np.ndarray,
    trial_ids: np.ndarray,
    sample_rate: float,
    window_seconds: float,
    stride_seconds: float,
    num_classes: int,
    seed: int,
    shuffle_trials_within_class: bool,
    split_name: str,
) -> WindowSet:
    """
    Construct real 10-second single-label windows from 4-second trials.

    Trials are grouped by class first, concatenated inside each class, then
    windowed. A window may cross trial boundaries, but never mixes labels.
    """
    trials = np.asarray(trials, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    trial_ids = np.asarray(trial_ids, dtype=np.int64)

    if trials.ndim != 3:
        raise ValueError(
            f"{split_name}: trials must have shape [N,C,T], got {trials.shape}."
        )
    if labels.shape != (len(trials),):
        raise ValueError(f"{split_name}: labels shape mismatch: {labels.shape}.")
    if trial_ids.shape != (len(trials),):
        raise ValueError(
            f"{split_name}: trial_ids shape mismatch: {trial_ids.shape}."
        )
    if not np.isfinite(trials).all():
        raise ValueError(f"{split_name}: trials contain NaN or Inf.")
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample_rate={sample_rate}.")
    if window_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("window_seconds and stride_seconds must be positive.")

    window_samples = int(round(window_seconds * sample_rate))
    stride_samples = int(round(stride_seconds * sample_rate))
    rng = np.random.default_rng(seed)

    output_windows: list[np.ndarray] = []
    output_labels: list[int] = []
    output_sources: list[tuple[int, ...]] = []

    for class_index in range(num_classes):
        class_indices = np.flatnonzero(labels == class_index)
        if len(class_indices) == 0:
            raise ValueError(
                f"{split_name}: no source trials for class {class_index}."
            )

        class_indices = class_indices.copy()
        if shuffle_trials_within_class:
            rng.shuffle(class_indices)

        class_trials = trials[class_indices]
        class_trial_ids = trial_ids[class_indices]
        samples_per_trial = int(class_trials.shape[-1])

        class_stream = (
            class_trials.transpose(1, 0, 2)
            .reshape(class_trials.shape[1], -1)
            .astype(np.float32, copy=False)
        )
        source_id_stream = np.repeat(class_trial_ids, samples_per_trial)

        if class_stream.shape[-1] < window_samples:
            raise ValueError(
                f"{split_name}: class {class_index} has only "
                f"{class_stream.shape[-1] / sample_rate:.2f}s of data, "
                f"less than one {window_seconds:.2f}s window."
            )

        for start in range(
            0,
            class_stream.shape[-1] - window_samples + 1,
            stride_samples,
        ):
            end = start + window_samples
            output_windows.append(class_stream[:, start:end].copy())
            output_labels.append(class_index)
            output_sources.append(
                tuple(
                    int(x)
                    for x in np.unique(source_id_stream[start:end]).tolist()
                )
            )

    windows = np.stack(output_windows, axis=0).astype(np.float32, copy=False)
    output_labels_array = np.asarray(output_labels, dtype=np.int64)

    permutation = rng.permutation(len(windows))
    windows = windows[permutation]
    output_labels_array = output_labels_array[permutation]
    output_sources = [output_sources[int(i)] for i in permutation]

    validate_labels(
        output_labels_array,
        num_classes=num_classes,
        split_name=f"{split_name} derived windows",
    )

    return WindowSet(
        windows=windows,
        labels=output_labels_array,
        source_trial_ids=tuple(output_sources),
        construction="same_label_trial_concatenation",
    )


def limit_windows_per_class(
    window_set: WindowSet,
    *,
    max_per_class: int | None,
    num_classes: int,
    seed: int,
) -> WindowSet:
    if max_per_class is None:
        return window_set
    if max_per_class <= 0:
        raise ValueError("max_windows_per_class must be positive.")

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_index in range(num_classes):
        indices = np.flatnonzero(window_set.labels == class_index)
        rng.shuffle(indices)
        selected.extend(
            int(x) for x in indices[: min(max_per_class, len(indices))]
        )

    rng.shuffle(selected)
    selected_array = np.asarray(selected, dtype=np.int64)
    return WindowSet(
        windows=window_set.windows[selected_array],
        labels=window_set.labels[selected_array],
        source_trial_ids=tuple(
            window_set.source_trial_ids[int(i)] for i in selected_array
        ),
        construction=window_set.construction,
    )


def feature_cache_dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported feature cache dtype: {name!r}.")
    return mapping[name]


@torch.no_grad()
def extract_frozen_features(
    *,
    window_set: WindowSet,
    metadata: Any,
    config: Model50MConfig,
    classifier: Model50MClassifier,
    preprocess_batch_size: int,
    cache_dtype: torch.dtype,
    split_name: str,
    log_every: int,
) -> TensorDataset:
    """Preprocess, tokenize and cache frozen 50M features on CPU."""
    if preprocess_batch_size <= 0:
        raise ValueError("preprocess_batch_size must be positive.")

    preprocessor = Model50MPreprocessor(config)
    tokenizer = Model50MTokenizer(config)
    feature_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    split_start = time.perf_counter()
    mapped_counts: set[int] = set()
    missing_counts: set[int] = set()
    classifier.eval()

    for batch_start in range(0, len(window_set.windows), preprocess_batch_size):
        batch_end = min(
            batch_start + preprocess_batch_size,
            len(window_set.windows),
        )

        preprocess_start = time.perf_counter()
        tokenized_samples = []
        for raw_window in window_set.windows[batch_start:batch_end]:
            result = preprocessor(
                signal=raw_window,
                channel_names=metadata.channel_names,
                original_sample_rate=metadata.sample_rate,
                input_unit=metadata.unit,
            )
            mapped_counts.add(int(result.mapped_channel_count))
            missing_counts.add(int(result.missing_channel_count))
            tokenized_samples.append(tokenizer(result))

        model_batch = stack_model50m_tokens(
            tokenized_samples,
            device=classifier.device,
        )
        preprocess_seconds = time.perf_counter() - preprocess_start

        compute_start = time.perf_counter()
        features = classifier.extract_features(model_batch)
        if classifier.device.type == "cuda":
            torch.cuda.synchronize(classifier.device)
        elif classifier.device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()
        compute_seconds = time.perf_counter() - compute_start

        features_cpu = (
            features.detach()
            .to(device="cpu", dtype=cache_dtype)
            .contiguous()
        )
        labels_cpu = torch.from_numpy(
            window_set.labels[batch_start:batch_end].copy()
        ).long()
        feature_chunks.append(features_cpu)
        label_chunks.append(labels_cpu)

        batch_number = batch_start // preprocess_batch_size + 1
        if (
            batch_number == 1
            or batch_end == len(window_set.windows)
            or batch_number % log_every == 0
        ):
            print(
                f"[FeatureCache] split={split_name} "
                f"batch={batch_number} "
                f"samples={batch_end}/{len(window_set.windows)} "
                f"preprocess={preprocess_seconds:.2f}s "
                f"backbone={compute_seconds:.2f}s",
                flush=True,
            )

    features_all = torch.cat(feature_chunks, dim=0).contiguous()
    labels_all = torch.cat(label_chunks, dim=0).contiguous()

    expected_shape = (
        len(window_set.windows),
        config.classifier_input_dim,
    )
    if features_all.shape != expected_shape:
        raise RuntimeError(
            f"{split_name}: unexpected feature shape "
            f"{tuple(features_all.shape)}, expected {expected_shape}."
        )
    if not torch.isfinite(features_all.float()).all():
        raise RuntimeError(f"{split_name}: feature cache contains NaN or Inf.")

    size_mib = (
        features_all.numel()
        * features_all.element_size()
        / 1024**2
    )
    print(
        f"[FeatureCache] completed split={split_name} "
        f"shape={tuple(features_all.shape)} "
        f"dtype={features_all.dtype} "
        f"size={size_mib:.1f} MiB "
        f"mapped_channels={sorted(mapped_counts)} "
        f"missing_channels={sorted(missing_counts)} "
        f"time={time.perf_counter() - split_start:.1f}s",
        flush=True,
    )
    return TensorDataset(features_all, labels_all)


def save_feature_cache(
    dataset: TensorDataset,
    path: Path,
    *,
    split_name: str,
) -> None:
    features, labels = dataset.tensors
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features,
            "labels": labels,
            "split": split_name,
        },
        path,
    )


def confusion_to_metrics(
    confusion: torch.Tensor,
) -> tuple[float, list[float | None]]:
    support = confusion.sum(dim=1)
    correct = confusion.diag()
    recall = correct / support.clamp_min(1)
    valid = support > 0
    balanced_accuracy = (
        float(recall[valid].mean().item())
        if valid.any()
        else float("nan")
    )
    per_class_recall: list[float | None] = []
    for class_index in range(confusion.shape[0]):
        if support[class_index].item() > 0:
            per_class_recall.append(float(recall[class_index].item()))
        else:
            per_class_recall.append(None)
    return balanced_accuracy, per_class_recall


def run_head_epoch(
    *,
    head: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    optimizer: torch.optim.Optimizer | None,
) -> EpochMetrics:
    is_train = optimizer is not None
    head.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    confusion = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.long,
    )

    for features, labels in loader:
        features = features.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        labels = labels.to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = head(features)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        predictions = logits.argmax(dim=-1)
        batch_size = int(labels.numel())
        total_loss += float(loss.item()) * batch_size
        total_correct += int((predictions == labels).sum().item())
        total_count += batch_size

        flat_indices = (
            labels.detach().cpu() * num_classes
            + predictions.detach().cpu()
        )
        confusion += torch.bincount(
            flat_indices,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)

    if total_count <= 0:
        raise RuntimeError("Empty feature loader.")

    balanced_accuracy, per_class_recall = confusion_to_metrics(confusion)
    return EpochMetrics(
        loss=total_loss / total_count,
        accuracy=total_correct / total_count,
        balanced_accuracy=balanced_accuracy,
        confusion_matrix=confusion.tolist(),
        per_class_recall=per_class_recall,
    )
def build_direct_trial_windows(
    *,
    trials: np.ndarray,
    labels: np.ndarray,
    trial_ids: np.ndarray,
    sample_rate: float,
    window_seconds: float,
    num_classes: int,
    seed: int,
    split_name: str,
) -> WindowSet:
    """
    将每个原始 Trial 直接作为一个模型窗口。

    适用于：
        BNCI 4 秒 Trial
        + 4 秒模型输入

    不执行拼接、补零或跨 Trial 切片。
    """
    trials = np.asarray(trials, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    trial_ids = np.asarray(trial_ids, dtype=np.int64)

    if trials.ndim != 3:
        raise ValueError(
            f"{split_name}: expected trials [N,C,T], "
            f"got {trials.shape}."
        )

    if labels.shape != (len(trials),):
        raise ValueError(
            f"{split_name}: labels shape mismatch: {labels.shape}."
        )

    if trial_ids.shape != (len(trials),):
        raise ValueError(
            f"{split_name}: trial_ids shape mismatch: "
            f"{trial_ids.shape}."
        )

    if not np.isfinite(trials).all():
        raise ValueError(
            f"{split_name}: trials contain NaN or Inf."
        )

    if sample_rate <= 0:
        raise ValueError(
            f"{split_name}: invalid sample_rate={sample_rate}."
        )

    expected_samples = int(
        round(window_seconds * sample_rate)
    )
    actual_samples = int(trials.shape[-1])

    if actual_samples != expected_samples:
        raise ValueError(
            f"{split_name}: direct-trial mode requires exactly "
            f"{window_seconds:.3f}s per source trial. "
            f"Expected {expected_samples} samples at "
            f"{sample_rate:.3f} Hz, got {actual_samples} "
            f"({actual_samples / sample_rate:.3f}s)."
        )

    validate_labels(
        labels,
        num_classes=num_classes,
        split_name=split_name,
    )

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(trials))

    return WindowSet(
        windows=trials[permutation].copy(),
        labels=labels[permutation].copy(),
        source_trial_ids=tuple(
            (int(trial_ids[index]),)
            for index in permutation
        ),
        construction="direct_source_trial",
    )

def metric_is_better(
    *,
    metric_name: str,
    current: EpochMetrics,
    best_value: float,
) -> tuple[bool, float]:
    if metric_name == "val_bacc":
        value = current.balanced_accuracy
        return value > best_value, value
    if metric_name == "val_acc":
        value = current.accuracy
        return value > best_value, value
    if metric_name == "val_loss":
        value = current.loss
        return value < best_value, value
    raise ValueError(f"Unsupported metric_for_best={metric_name!r}.")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a frozen-50M linear classification head on "
            "bnci2014_001_s01.h5."
        )
    )
    parser.add_argument(
        "--data",
        default="data/processed/bnci2014_001/subject_01.h5",
    )
    parser.add_argument("--train-session", default="0train")
    parser.add_argument("--test-session", default="1test")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/model_deploy.pt",
        help="Dependency-free 50M backbone checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/heads/stage05/bnci2014_001/subject_01/10s_flatten/head.pt",
        help="Output linear-head checkpoint.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Metrics directory. Default: "
            "runs/stage05_50m/linear_probe_<timestamp>."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "mps", "auto"),
    )

    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--window-stride-sec", type=float, default=10.0)
    parser.add_argument("--target-sample-rate", type=float, default=100.0)
    parser.add_argument("--patch-sec", type=float, default=1.0)
    parser.add_argument("--patch-stride-sec", type=float, default=1.0)
    parser.add_argument("--output-layer-idx", type=int, default=8)
    parser.add_argument(
        "--aggregation",
        choices=("flatten", "mean"),
        default="flatten",
    )

    parser.add_argument("--filter-low-hz", type=float, default=0.1)
    parser.add_argument("--filter-high-hz", type=float, default=75.0)
    parser.add_argument(
        "--reference-mode",
        choices=("none", "average"),
        default="none",
    )

    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--max-windows-per-class",
        type=int,
        default=None,
        help="Fast-debug limit applied independently to train/val/test.",
    )

    parser.add_argument(
        "--feature-batch-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--feature-cache-dtype",
        choices=("float16", "float32", "bfloat16"),
        default="float16",
    )
    parser.add_argument("--save-feature-cache", action="store_true")
    parser.add_argument("--feature-log-every", type=int, default=10)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--head-batch-size", type=int, default=32)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--metric-for-best",
        choices=("val_bacc", "val_acc", "val_loss"),
        default="val_bacc",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    if abs(args.window_sec - 10.0) > 1e-6:
        raise ValueError(
            "This stage-0.5 script currently expects the original 50M "
            "10-second configuration. Use --window-sec 10.0."
        )
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.head_batch_size <= 0 or args.feature_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if args.patience < 0:
        raise ValueError("--patience must be >= 0.")
    if args.feature_log_every <= 0:
        raise ValueError("--feature-log-every must be positive.")

    if len(STANDARD_64_CHANNELS) != 64:
        raise RuntimeError(
            "STANDARD_64_CHANNELS must contain exactly 64 names, but "
            f"config.py currently contains {len(STANDARD_64_CHANNELS)}. "
            "Fix the channel template before training; do not change "
            "n_channels away from 64 to bypass this check."
        )

    set_seed(args.seed)
    data_path = resolve_repo_path(args.data)
    checkpoint_path = resolve_repo_path(args.checkpoint)
    output_path = resolve_repo_path(args.output)

    if args.run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = ROOT / "runs" / "stage05_50m" / f"linear_probe_{timestamp}"
    else:
        run_dir = resolve_repo_path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    for name, path in (
        ("HDF5 data", data_path),
        ("50M backbone checkpoint", checkpoint_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} was not found: {path}")

    dataset = EEGHDF5(data_path)
    metadata = dataset.metadata
    train_session = dataset.load(args.train_session)
    test_session = dataset.load(args.test_session)
    num_classes = len(metadata.class_names)

    validate_labels(
        train_session["labels"],
        num_classes=num_classes,
        split_name=args.train_session,
    )
    validate_labels(
        test_session["labels"],
        num_classes=num_classes,
        split_name=args.test_session,
    )

    print("=" * 78)
    print("50M frozen-backbone linear probe")
    print("=" * 78)
    print("data:", data_path)
    print("dataset:", metadata.dataset_name)
    print("train session:", args.train_session)
    print("test session:", args.test_session)
    print("class names:", metadata.class_names)
    print("raw train shape:", train_session["data"].shape)
    print("raw test shape:", test_session["data"].shape)
    print("raw sample rate:", metadata.sample_rate)
    print("raw unit:", metadata.unit)
    print()
    print(
        "IMPORTANT: the HDF5 contains 4-second trials. This script splits "
        "source trials first, then concatenates trials within the same class "
        "to construct temporary real 10-second single-label windows."
    )
    print()

    train_source_indices, val_source_indices = stratified_source_trial_split(
        train_session["labels"],
        val_fraction=args.val_fraction,
        seed=args.split_seed,
        num_classes=num_classes,
    )

    def select(
        session: dict[str, np.ndarray],
        indices: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return {key: value[indices] for key, value in session.items()}

    source_train = select(train_session, train_source_indices)
    source_val = select(train_session, val_source_indices)

    train_windows = build_same_label_concat_windows(
        trials=source_train["data"],
        labels=source_train["labels"],
        trial_ids=source_train["trial_ids"],
        sample_rate=metadata.sample_rate,
        window_seconds=args.window_sec,
        stride_seconds=args.window_stride_sec,
        num_classes=num_classes,
        seed=args.split_seed + 1,
        shuffle_trials_within_class=True,
        split_name="train",
    )
    val_windows = build_same_label_concat_windows(
        trials=source_val["data"],
        labels=source_val["labels"],
        trial_ids=source_val["trial_ids"],
        sample_rate=metadata.sample_rate,
        window_seconds=args.window_sec,
        stride_seconds=args.window_stride_sec,
        num_classes=num_classes,
        seed=args.split_seed + 2,
        shuffle_trials_within_class=False,
        split_name="val",
    )
    test_windows = build_same_label_concat_windows(
        trials=test_session["data"],
        labels=test_session["labels"],
        trial_ids=test_session["trial_ids"],
        sample_rate=metadata.sample_rate,
        window_seconds=args.window_sec,
        stride_seconds=args.window_stride_sec,
        num_classes=num_classes,
        seed=args.split_seed + 3,
        shuffle_trials_within_class=False,
        split_name="test",
    )

    train_windows = limit_windows_per_class(
        train_windows,
        max_per_class=args.max_windows_per_class,
        num_classes=num_classes,
        seed=args.seed + 10,
    )
    val_windows = limit_windows_per_class(
        val_windows,
        max_per_class=args.max_windows_per_class,
        num_classes=num_classes,
        seed=args.seed + 11,
    )
    test_windows = limit_windows_per_class(
        test_windows,
        max_per_class=args.max_windows_per_class,
        num_classes=num_classes,
        seed=args.seed + 12,
    )

    print("source trial counts:")
    print("  train:", class_counts(source_train["labels"], num_classes))
    print("  val:  ", class_counts(source_val["labels"], num_classes))
    print("  test: ", class_counts(test_session["labels"], num_classes))
    print("derived 10-second window counts:")
    print("  train:", class_counts(train_windows.labels, num_classes))
    print("  val:  ", class_counts(val_windows.labels, num_classes))
    print("  test: ", class_counts(test_windows.labels, num_classes))
    print()

    config = Model50MConfig(
        checkpoint_path=checkpoint_path,
        classifier_path=None,
        device=args.device,
        target_sample_rate=args.target_sample_rate,
        window_seconds=args.window_sec,
        patch_seconds=args.patch_sec,
        patch_stride_seconds=args.patch_stride_sec,
        filter_enabled=True,
        filter_low_hz=args.filter_low_hz,
        filter_high_hz=args.filter_high_hz,
        reference_mode=args.reference_mode,
        strict_window_duration=True,
        output_layer_idx=args.output_layer_idx,
        aggregation=args.aggregation,
        num_classes=num_classes,
    )

    print("50M config:")
    print("  target shape:", (config.n_channels, config.target_num_points))
    print("  token shape:", (config.num_tokens, config.patch_num_points))
    print("  output layer idx:", config.output_layer_idx)
    print("  aggregation:", config.aggregation)
    print("  classifier input dim:", config.classifier_input_dim)
    print()

    load_start = time.perf_counter()
    backbone = Model50MBackbone(
        config=config,
        load_checkpoint=True,
        freeze=True,
    )
    classifier = Model50MClassifier(
        config=config,
        backbone=backbone,
    )
    classifier.eval()
    load_seconds = time.perf_counter() - load_start
    print(
        f"Backbone loaded on {classifier.device} in {load_seconds:.2f}s; "
        f"trainable backbone params={backbone.trainable_parameters}."
    )

    cache_dtype = feature_cache_dtype_from_name(args.feature_cache_dtype)
    train_features = extract_frozen_features(
        window_set=train_windows,
        metadata=metadata,
        config=config,
        classifier=classifier,
        preprocess_batch_size=args.feature_batch_size,
        cache_dtype=cache_dtype,
        split_name="train",
        log_every=args.feature_log_every,
    )
    val_features = extract_frozen_features(
        window_set=val_windows,
        metadata=metadata,
        config=config,
        classifier=classifier,
        preprocess_batch_size=args.feature_batch_size,
        cache_dtype=cache_dtype,
        split_name="val",
        log_every=args.feature_log_every,
    )
    test_features = extract_frozen_features(
        window_set=test_windows,
        metadata=metadata,
        config=config,
        classifier=classifier,
        preprocess_batch_size=args.feature_batch_size,
        cache_dtype=cache_dtype,
        split_name="test",
        log_every=args.feature_log_every,
    )

    if args.save_feature_cache:
        save_feature_cache(
            train_features,
            run_dir / "features_train.pt",
            split_name="train",
        )
        save_feature_cache(
            val_features,
            run_dir / "features_val.pt",
            split_name="val",
        )
        save_feature_cache(
            test_features,
            run_dir / "features_test.pt",
            split_name="test",
        )

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_features,
        batch_size=min(args.head_batch_size, len(train_features)),
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=classifier.device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_features,
        batch_size=min(args.head_batch_size, len(val_features)),
        shuffle=False,
        num_workers=0,
        pin_memory=classifier.device.type == "cuda",
        drop_last=False,
    )
    test_loader = DataLoader(
        test_features,
        batch_size=min(args.head_batch_size, len(test_features)),
        shuffle=False,
        num_workers=0,
        pin_memory=classifier.device.type == "cuda",
        drop_last=False,
    )

    for parameter in classifier.backbone.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.SGD(
        classifier.head.parameters(),
        lr=args.head_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_value = (
        float("inf")
        if args.metric_for_best == "val_loss"
        else -float("inf")
    )
    best_epoch = -1
    best_head_state: dict[str, torch.Tensor] | None = None
    best_val_metrics: EpochMetrics | None = None
    epochs_without_improvement = 0
    epoch_rows: list[dict[str, Any]] = []

    print()
    print("Training linear head with SGD")
    print(
        f"epochs={args.epochs}, lr={args.head_lr}, momentum={args.momentum}, "
        f"weight_decay={args.weight_decay}, best_metric={args.metric_for_best}"
    )

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_head_epoch(
            head=classifier.head,
            loader=train_loader,
            criterion=criterion,
            device=classifier.device,
            num_classes=num_classes,
            optimizer=optimizer,
        )
        with torch.no_grad():
            val_metrics = run_head_epoch(
                head=classifier.head,
                loader=val_loader,
                criterion=criterion,
                device=classifier.device,
                num_classes=num_classes,
                optimizer=None,
            )

        improved, current_value = metric_is_better(
            metric_name=args.metric_for_best,
            current=val_metrics,
            best_value=best_value,
        )
        if improved:
            best_value = current_value
            best_epoch = epoch
            best_head_state = deepcopy(
                {
                    key: value.detach().cpu()
                    for key, value in classifier.head.state_dict().items()
                }
            )
            best_val_metrics = val_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_acc": train_metrics.accuracy,
            "train_bacc": train_metrics.balanced_accuracy,
            "val_loss": val_metrics.loss,
            "val_acc": val_metrics.accuracy,
            "val_bacc": val_metrics.balanced_accuracy,
            "is_best": improved,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        epoch_rows.append(row)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics.loss:.4f} "
            f"train_acc={train_metrics.accuracy:.4f} "
            f"train_bacc={train_metrics.balanced_accuracy:.4f} "
            f"val_loss={val_metrics.loss:.4f} "
            f"val_acc={val_metrics.accuracy:.4f} "
            f"val_bacc={val_metrics.balanced_accuracy:.4f} "
            f"{'*' if improved else ''}",
            flush=True,
        )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                f"Early stopping: no improvement for {args.patience} epoch(s)."
            )
            break

    if best_head_state is None or best_val_metrics is None:
        raise RuntimeError("No best linear-head state was recorded.")

    classifier.head.load_state_dict(best_head_state, strict=True)
    classifier.eval()

    with torch.no_grad():
        test_metrics = run_head_epoch(
            head=classifier.head,
            loader=test_loader,
            criterion=criterion,
            device=classifier.device,
            num_classes=num_classes,
            optimizer=None,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path = save_classifier_checkpoint(
        classifier=classifier,
        checkpoint_path=output_path,
        extra_metadata={
            "task": "BNCI2014_001_motor_imagery",
            "dataset": metadata.dataset_name,
            "data_path": str(data_path),
            "train_session": args.train_session,
            "test_session": args.test_session,
            "class_names": list(metadata.class_names),
            "mode": "linear_probe",
            "optimizer": "SGD",
            "head_lr": float(args.head_lr),
            "momentum": float(args.momentum),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "split_seed": int(args.split_seed),
            "val_fraction": float(args.val_fraction),
            "metric_for_best": args.metric_for_best,
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_metrics.loss),
            "best_val_acc": float(best_val_metrics.accuracy),
            "best_val_bacc": float(best_val_metrics.balanced_accuracy),
            "test_loss": float(test_metrics.loss),
            "test_acc": float(test_metrics.accuracy),
            "test_bacc": float(test_metrics.balanced_accuracy),
            "window_construction": "same_label_trial_concatenation",
            "window_stride_seconds": float(args.window_stride_sec),
            "warning": (
                "Temporary stage-0.5 head: 10-second samples were built by "
                "concatenating 4-second source trials within the same class."
            ),
        },
    )

    metrics_csv = run_dir / "epoch_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(epoch_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(epoch_rows)

    report = {
        "status": "completed",
        "warning": (
            "Temporary 10-second baseline: each derived window contains one "
            "class, but may cross original 4-second trial boundaries."
        ),
        "files": {
            "data": data_path,
            "backbone_checkpoint": checkpoint_path,
            "classifier_checkpoint": saved_path,
            "run_dir": run_dir,
        },
        "dataset": {
            "name": metadata.dataset_name,
            "sample_rate": metadata.sample_rate,
            "unit": metadata.unit,
            "channel_names": metadata.channel_names,
            "class_names": metadata.class_names,
        },
        "source_trials": {
            "train": class_counts(source_train["labels"], num_classes),
            "val": class_counts(source_val["labels"], num_classes),
            "test": class_counts(test_session["labels"], num_classes),
        },
        "derived_windows": {
            "train": class_counts(train_windows.labels, num_classes),
            "val": class_counts(val_windows.labels, num_classes),
            "test": class_counts(test_windows.labels, num_classes),
            "window_seconds": args.window_sec,
            "stride_seconds": args.window_stride_sec,
            "construction": train_windows.construction,
        },
        "model": {
            "device": str(classifier.device),
            "load_seconds": load_seconds,
            "target_sample_rate": config.target_sample_rate,
            "target_num_points": config.target_num_points,
            "num_tokens": config.num_tokens,
            "patch_num_points": config.patch_num_points,
            "output_layer_idx": config.output_layer_idx,
            "aggregation": config.aggregation,
            "classifier_input_dim": config.classifier_input_dim,
            "feature_cache_dtype": args.feature_cache_dtype,
        },
        "training": {
            "optimizer": "SGD",
            "epochs_requested": args.epochs,
            "epochs_completed": len(epoch_rows),
            "best_epoch": best_epoch,
            "metric_for_best": args.metric_for_best,
            "head_lr": args.head_lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
            "split_seed": args.split_seed,
        },
        "best_validation": {
            "loss": best_val_metrics.loss,
            "accuracy": best_val_metrics.accuracy,
            "balanced_accuracy": best_val_metrics.balanced_accuracy,
            "confusion_matrix": best_val_metrics.confusion_matrix,
            "per_class_recall": best_val_metrics.per_class_recall,
        },
        "test": {
            "loss": test_metrics.loss,
            "accuracy": test_metrics.accuracy,
            "balanced_accuracy": test_metrics.balanced_accuracy,
            "confusion_matrix": test_metrics.confusion_matrix,
            "per_class_recall": test_metrics.per_class_recall,
        },
    }

    report_path = run_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            report,
            handle,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )

    print()
    print("=" * 78)
    print("Linear probe completed")
    print("=" * 78)
    print("best epoch:", best_epoch)
    print(
        "best val: "
        f"loss={best_val_metrics.loss:.4f}, "
        f"acc={best_val_metrics.accuracy:.4f}, "
        f"bacc={best_val_metrics.balanced_accuracy:.4f}"
    )
    print(
        "test: "
        f"loss={test_metrics.loss:.4f}, "
        f"acc={test_metrics.accuracy:.4f}, "
        f"bacc={test_metrics.balanced_accuracy:.4f}"
    )
    print("classifier saved to:", saved_path)
    print("metrics CSV:", metrics_csv)
    print("report JSON:", report_path)


if __name__ == "__main__":
    main()
