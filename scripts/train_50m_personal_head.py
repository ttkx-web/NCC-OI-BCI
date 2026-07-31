from __future__ import annotations

"""
Train a Stage-1 few-shot personal linear head for one BNCI2014_001 subject.

Protocol
--------
For the target subject:

- Personalization source:
    target subject / 0train
- Personalization training:
    N source trials per class from a personalization pool
- Personal validation:
    a fixed, disjoint set of source trials per class from 0train
- Final test:
    target subject / 1test

The target subject's 1test session is not opened until personal-head model
selection has finished.

The 50M backbone is frozen. By default, the personal head is initialized from
the LOSO population head trained by scripts/train_50m_population_head.py.

Important
---------
BNCI2014_001 HDF5 files contain 4-second source trials. To preserve the
Stage-0.5 50M input contract, this script constructs temporary 10-second
single-label windows by concatenating trials only within the same:

    subject + session + class + split

Training, validation, and final-test source-trial IDs are checked for leakage.
"""

import argparse
import csv
import json
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from _bootstrap import ROOT
from bci_dayloop.data.hdf5_dataset import EEGHDF5, HDF5Metadata
from bci_dayloop.models.model_50m.backbone import Model50MBackbone
from bci_dayloop.models.model_50m.classifier import (
    Model50MClassifier,
    load_classifier_checkpoint,
    save_classifier_checkpoint,
)
from bci_dayloop.models.model_50m.config import (
    STANDARD_64_CHANNELS,
    Model50MConfig,
)

# Reuse Stage-0.5 helpers that already implement:
# - trial-safe 10-second window construction
# - 50M preprocessing/tokenization
# - frozen-feature extraction
# - linear-head epoch training and checkpoint compatibility
from train_50m_linear_head import (
    EpochMetrics,
    WindowSet,
    build_same_label_concat_windows,
    class_counts,
    extract_frozen_features,
    feature_cache_dtype_from_name,
    json_default,
    metric_is_better,
    resolve_repo_path,
    run_head_epoch,
    set_seed,
    validate_labels,
)

# Reuse Stage-1 population helpers for:
# - stable file/hash/report utilities
# - subject-file resolution
# - metadata/session checks
# - extended Macro-F1 metrics
# - collision-free source-trial IDs
from train_50m_population_head import (
    ExtendedMetrics,
    atomic_write_json,
    class_name_counts,
    current_git_commit,
    encode_trial_ids,
    extend_metrics,
    resolve_subject_file,
    sha256_file,
    stable_json_hash,
    validate_loaded_session,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PersonalTrialSplit:
    """Source-trial split performed before 10-second window construction."""

    train_indices: np.ndarray
    validation_indices: np.ndarray
    pool_indices: np.ndarray
    train_trial_ids_by_class: dict[str, list[int]]
    validation_trial_ids_by_class: dict[str, list[int]]
    pool_trial_ids_by_class: dict[str, list[int]]

    def __post_init__(self) -> None:
        for name, values in (
            ("train_indices", self.train_indices),
            ("validation_indices", self.validation_indices),
            ("pool_indices", self.pool_indices),
        ):
            if values.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional.")

        train_set = set(self.train_indices.tolist())
        validation_set = set(self.validation_indices.tolist())
        pool_set = set(self.pool_indices.tolist())

        if train_set & validation_set:
            raise RuntimeError(
                "Personal train and validation source indices overlap."
            )
        if not train_set.issubset(pool_set):
            raise RuntimeError(
                "Selected personal train indices are not a subset of the "
                "personalization pool."
            )
        if validation_set & pool_set:
            raise RuntimeError(
                "Personal validation indices overlap the personalization pool."
            )


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    metrics: ExtendedMetrics
    labels: list[int]
    predictions: list[int]
    confidences: list[float]
    probabilities: list[list[float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.metrics.to_dict(),
            "labels": self.labels,
            "predictions": self.predictions,
            "confidences": self.confidences,
            "probabilities": self.probabilities,
        }


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def safe_load_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if not isinstance(payload, Mapping):
        raise TypeError(
            f"Expected a mapping checkpoint at {path}, "
            f"got {type(payload)!r}."
        )
    return payload


def load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    payload = safe_load_mapping(path)
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError(
            f"Population checkpoint metadata must be a mapping: {path}"
        )
    return dict(metadata)


def resolve_head_device(
    requested: str,
    *,
    feature_device: torch.device,
    classifier_input_dim: int,
) -> torch.device:
    """
    Keep the very wide Flatten head off MPS by default.

    The Stage-0.5 Flatten feature has 327680 dimensions. On some Apple
    MPS/ANE paths, training Linear(327680, 4) can terminate the Python
    process with a native bus error rather than raising a Python exception.
    """
    requested = str(requested).lower()

    if requested == "auto":
        if feature_device.type == "cuda":
            device = feature_device
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--head-device cuda was requested, but CUDA is unavailable."
        )
    if device.type == "mps":
        if not hasattr(torch.backends, "mps"):
            raise RuntimeError(
                "--head-device mps was requested, but this PyTorch build "
                "does not expose the MPS backend."
            )
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "--head-device mps was requested, but MPS is unavailable."
            )
        if classifier_input_dim > 16_384:
            raise ValueError(
                "Refusing to train the wide linear head on MPS. "
                f"feature_dim={classifier_input_dim} exceeds the observed "
                "safe MPS/ANE width. Use --head-device cpu. The 50M backbone "
                "can still use --device mps for feature extraction."
            )

    return device


def reset_module_parameters(module: nn.Module) -> None:
    reset_count = 0
    for child in module.modules():
        if child is module:
            continue
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()
            reset_count += 1
    if reset_count == 0:
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()
            reset_count += 1
    if reset_count == 0:
        raise RuntimeError(
            "Could not find reset_parameters() in the personal head."
        )


def source_trial_set(window_set: WindowSet) -> set[int]:
    return {
        int(trial_id)
        for source_ids in window_set.source_trial_ids
        for trial_id in source_ids
    }


def assert_no_window_source_leakage(
    left: WindowSet,
    right: WindowSet,
    *,
    left_name: str,
    right_name: str,
) -> None:
    overlap = source_trial_set(left) & source_trial_set(right)
    if overlap:
        raise RuntimeError(
            f"Source-trial leakage between {left_name} and {right_name}. "
            f"Examples: {sorted(overlap)[:10]}"
        )


def select_session_rows(
    session: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value)[indices]
        for key, value in session.items()
    }


def class_index_counts(
    labels: np.ndarray,
    num_classes: int,
) -> dict[int, int]:
    return {
        class_index: int(np.sum(labels == class_index))
        for class_index in range(num_classes)
    }


# ---------------------------------------------------------------------------
# Trial split
# ---------------------------------------------------------------------------


def build_personal_trial_split(
    *,
    labels: np.ndarray,
    trial_ids: np.ndarray,
    class_names: Sequence[str],
    trials_per_class: int,
    validation_trials_per_class: int,
    validation_seed: int,
    personalization_seed: int,
) -> PersonalTrialSplit:
    """
    Build a fixed validation set and a seed-specific nested train subset.

    Validation:
        controlled only by validation_seed

    Personalization pool order:
        controlled by personalization_seed

    Therefore, running 5/10/20/40 trials per class with the same seeds gives:

        train_5 ⊂ train_10 ⊂ train_20 ⊂ train_40
    """
    labels = np.asarray(labels, dtype=np.int64)
    trial_ids = np.asarray(trial_ids, dtype=np.int64)

    if labels.ndim != 1 or trial_ids.shape != labels.shape:
        raise ValueError(
            f"Invalid labels/trial_ids shapes: {labels.shape}, "
            f"{trial_ids.shape}."
        )
    if trials_per_class <= 0:
        raise ValueError("--trials-per-class must be positive.")
    if validation_trials_per_class <= 0:
        raise ValueError(
            "--validation-trials-per-class must be positive."
        )

    num_classes = len(class_names)
    validate_labels(
        labels,
        num_classes=num_classes,
        split_name="target personalization source session",
    )

    validation_rng = np.random.default_rng(validation_seed)
    personalization_rng = np.random.default_rng(personalization_seed)

    train_indices: list[int] = []
    validation_indices: list[int] = []
    pool_indices: list[int] = []

    train_ids_by_class: dict[str, list[int]] = {}
    validation_ids_by_class: dict[str, list[int]] = {}
    pool_ids_by_class: dict[str, list[int]] = {}

    for class_index, class_name in enumerate(class_names):
        indices = np.flatnonzero(labels == class_index).astype(
            np.int64,
            copy=False,
        )
        required = validation_trials_per_class + trials_per_class
        if len(indices) < required:
            raise ValueError(
                f"Class {class_index} ({class_name}) contains "
                f"{len(indices)} source trials, but the requested split "
                f"requires at least {required}: "
                f"{validation_trials_per_class} validation + "
                f"{trials_per_class} personalization."
            )

        # Validation is fixed independently of the personalization seed.
        validation_order = indices.copy()
        validation_rng.shuffle(validation_order)
        class_validation = validation_order[
            :validation_trials_per_class
        ]

        validation_set = set(class_validation.tolist())
        class_pool = np.asarray(
            [
                int(index)
                for index in indices
                if int(index) not in validation_set
            ],
            dtype=np.int64,
        )

        # A single seed-specific permutation defines all nested budgets.
        personalization_order = class_pool.copy()
        personalization_rng.shuffle(personalization_order)
        class_train = personalization_order[:trials_per_class]

        train_indices.extend(int(index) for index in class_train)
        validation_indices.extend(
            int(index) for index in class_validation
        )
        pool_indices.extend(int(index) for index in class_pool)

        train_ids_by_class[str(class_name)] = [
            int(trial_ids[index]) for index in class_train
        ]
        validation_ids_by_class[str(class_name)] = [
            int(trial_ids[index]) for index in class_validation
        ]
        pool_ids_by_class[str(class_name)] = [
            int(trial_ids[index]) for index in personalization_order
        ]

    train_array = np.asarray(train_indices, dtype=np.int64)
    validation_array = np.asarray(
        validation_indices,
        dtype=np.int64,
    )
    pool_array = np.asarray(pool_indices, dtype=np.int64)

    # Shuffle only the row order; membership remains fixed.
    row_rng = np.random.default_rng(personalization_seed + 100_003)
    row_rng.shuffle(train_array)
    validation_row_rng = np.random.default_rng(validation_seed + 100_003)
    validation_row_rng.shuffle(validation_array)

    return PersonalTrialSplit(
        train_indices=train_array,
        validation_indices=validation_array,
        pool_indices=pool_array,
        train_trial_ids_by_class=train_ids_by_class,
        validation_trial_ids_by_class=validation_ids_by_class,
        pool_trial_ids_by_class=pool_ids_by_class,
    )


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------


def build_windows_from_selected_trials(
    *,
    selected: Mapping[str, np.ndarray],
    subject_id: int,
    metadata: HDF5Metadata,
    split_name: str,
    window_seconds: float,
    stride_seconds: float,
    seed: int,
    shuffle_trials_within_class: bool,
    max_windows_per_class: int | None,
) -> WindowSet:
    raw_trial_ids = np.asarray(selected["trial_ids"], dtype=np.int64)
    encoded_ids = encode_trial_ids(subject_id, raw_trial_ids)

    window_set = build_same_label_concat_windows(
        trials=np.asarray(selected["data"], dtype=np.float32),
        labels=np.asarray(selected["labels"], dtype=np.int64),
        trial_ids=encoded_ids,
        sample_rate=metadata.sample_rate,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        num_classes=len(metadata.class_names),
        seed=seed,
        shuffle_trials_within_class=shuffle_trials_within_class,
        split_name=split_name,
    )

    if max_windows_per_class is None:
        return window_set

    if max_windows_per_class <= 0:
        raise ValueError("--max-windows-per-class must be positive.")

    rng = np.random.default_rng(seed + 55_001)
    selected_window_indices: list[int] = []
    for class_index in range(len(metadata.class_names)):
        indices = np.flatnonzero(
            window_set.labels == class_index
        ).astype(np.int64, copy=False)
        rng.shuffle(indices)
        selected_window_indices.extend(
            int(index)
            for index in indices[:max_windows_per_class]
        )

    rng.shuffle(selected_window_indices)
    index_array = np.asarray(selected_window_indices, dtype=np.int64)
    limited = WindowSet(
        windows=window_set.windows[index_array],
        labels=window_set.labels[index_array],
        source_trial_ids=tuple(
            window_set.source_trial_ids[int(index)]
            for index in index_array
        ),
        construction=window_set.construction,
    )
    validate_labels(
        limited.labels,
        num_classes=len(metadata.class_names),
        split_name=f"{split_name} limited windows",
    )
    return limited


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_head(
    *,
    head: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Sequence[str],
) -> EvaluationResult:
    head.eval()
    num_classes = len(class_names)

    total_loss = 0.0
    total_count = 0
    labels_all: list[int] = []
    predictions_all: list[int] = []
    confidences_all: list[float] = []
    probabilities_all: list[list[float]] = []

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

        logits = head(features)
        loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=-1)
        confidences, predictions = probabilities.max(dim=-1)

        batch_size = int(labels.numel())
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

        labels_cpu = labels.detach().cpu()
        predictions_cpu = predictions.detach().cpu()
        probabilities_cpu = probabilities.detach().cpu()
        confidences_cpu = confidences.detach().cpu()

        labels_all.extend(int(value) for value in labels_cpu.tolist())
        predictions_all.extend(
            int(value) for value in predictions_cpu.tolist()
        )
        confidences_all.extend(
            float(value) for value in confidences_cpu.tolist()
        )
        probabilities_all.extend(
            [
                [float(value) for value in row]
                for row in probabilities_cpu.tolist()
            ]
        )

        flat_indices = (
            labels_cpu * num_classes + predictions_cpu
        )
        confusion += torch.bincount(
            flat_indices,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)

    if total_count <= 0:
        raise RuntimeError("Cannot evaluate an empty loader.")

    total_correct = int(confusion.diag().sum().item())
    support = confusion.sum(dim=1)
    recall = confusion.diag() / support.clamp_min(1)
    valid = support > 0
    balanced_accuracy = float(
        recall[valid].mean().item()
    ) if valid.any() else float("nan")

    raw_metrics = EpochMetrics(
        loss=total_loss / total_count,
        accuracy=total_correct / total_count,
        balanced_accuracy=balanced_accuracy,
        confusion_matrix=confusion.tolist(),
        per_class_recall=[
            (
                float(recall[index].item())
                if support[index].item() > 0
                else None
            )
            for index in range(num_classes)
        ],
    )

    return EvaluationResult(
        metrics=extend_metrics(
            raw_metrics,
            class_names=class_names,
        ),
        labels=labels_all,
        predictions=predictions_all,
        confidences=confidences_all,
        probabilities=probabilities_all,
    )


# ---------------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------------


def save_personal_feature_cache(
    *,
    dataset: TensorDataset,
    window_set: WindowSet,
    path: Path,
    split_name: str,
    target_subject: int,
    class_names: Sequence[str],
    source_trial_ids_by_class: Mapping[str, Sequence[int]] | None,
    backbone_sha256: str,
    preprocessing_hash: str,
) -> None:
    features, labels = dataset.tensors
    if len(features) != len(window_set.windows):
        raise ValueError(
            f"{split_name}: feature count {len(features)} does not match "
            f"window count {len(window_set.windows)}."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    torch.save(
        {
            "format_version": 1,
            "split": split_name,
            "target_subject": int(target_subject),
            "features": features,
            "labels": labels,
            "window_subject_ids": torch.full(
                (len(features),),
                fill_value=int(target_subject),
                dtype=torch.long,
            ),
            "source_trial_ids": [
                list(source_ids)
                for source_ids in window_set.source_trial_ids
            ],
            "selected_raw_trial_ids_by_class": (
                {
                    str(key): [int(value) for value in values]
                    for key, values in source_trial_ids_by_class.items()
                }
                if source_trial_ids_by_class is not None
                else None
            ),
            "class_names": [str(name) for name in class_names],
            "class_counts": class_name_counts(
                window_set.labels,
                class_names,
            ),
            "window_construction": window_set.construction,
            "source_trial_encoding": (
                "(subject_id << 32) | file_local_trial_id"
            ),
            "backbone_sha256": backbone_sha256,
            "preprocessing_hash": preprocessing_hash,
        },
        temporary,
    )
    temporary.replace(path)


# ---------------------------------------------------------------------------
# Optimizer and scheduler
# ---------------------------------------------------------------------------


def build_optimizer(
    *,
    name: str,
    head: nn.Module,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if name == "sgd":
        return torch.optim.SGD(
            head.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            head.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {name!r}")


def build_scheduler(
    *,
    name: str,
    optimizer: torch.optim.Optimizer,
    metric_for_best: str,
    factor: float,
    patience: int,
    min_lr: float,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if name == "none":
        return None
    if name != "plateau":
        raise ValueError(f"Unsupported scheduler: {name!r}")

    mode = "min" if metric_for_best == "val_loss" else "max"
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a Stage-1 few-shot personal 50M linear head for one "
            "BNCI2014_001 target subject."
        )
    )

    parser.add_argument(
        "--data-root",
        default="data/processed/bnci2014_001",
        help="Directory containing one HDF5 file per subject.",
    )
    parser.add_argument(
        "--data-pattern",
        default="subject_{subject:02d}.h5",
        help=(
            "Subject filename pattern using {subject}; "
            "default: subject_{subject:02d}.h5."
        ),
    )
    parser.add_argument(
        "--target-subject",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--personalization-session",
        default="0train",
    )
    parser.add_argument(
        "--final-test-session",
        default="1test",
    )

    parser.add_argument(
        "--population-head",
        default=None,
        help=(
            "LOSO population-head checkpoint. Default: "
            "checkpoints/stage1/subject_XX/population_head.pt."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/50m/model_deploy.pt",
        help="Dependency-free 50M backbone checkpoint.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output personal-head checkpoint. Default: "
            "checkpoints/stage1/subject_XX/personal/"
            "trials_NN_seed_S/personal_head.pt."
        ),
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Run directory. Default: runs/stage1/subject_XX/personal/"
            "trials_NN_seed_S_<timestamp>."
        ),
    )

    parser.add_argument(
        "--trials-per-class",
        type=int,
        default=20,
        help="Number of target 0train source trials per class for training.",
    )
    parser.add_argument(
        "--validation-trials-per-class",
        type=int,
        default=16,
        help=(
            "Fixed number of target 0train source trials per class reserved "
            "for personal validation."
        ),
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=2026,
        help=(
            "Controls the fixed personal validation split. Keep this value "
            "unchanged across data budgets and personalization seeds."
        ),
    )
    parser.add_argument(
        "--personalization-seed",
        type=int,
        default=42,
        help=(
            "Controls the nested permutation of the personalization pool. "
            "Using the same seed makes 5/10/20/40-trial subsets nested."
        ),
    )
    parser.add_argument(
        "--head-init",
        choices=("population", "random"),
        default="population",
        help=(
            "Initialize the personal head from the LOSO population head "
            "(formal Stage-1 method) or randomly (optional control)."
        ),
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "mps", "auto"),
        help="Device used by the frozen 50M backbone for feature extraction.",
    )
    parser.add_argument(
        "--head-device",
        default="auto",
        choices=("cpu", "cuda", "mps", "auto"),
        help=(
            "Device used to train the linear head. auto uses CUDA when the "
            "backbone is on CUDA, otherwise CPU. This avoids the known wide "
            "Flatten-head MPS crash."
        ),
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

    parser.add_argument(
        "--max-windows-per-class",
        type=int,
        default=None,
        help=(
            "Optional debug-only window limit applied independently to "
            "personal train, personal validation, and final test."
        ),
    )
    parser.add_argument("--feature-batch-size", type=int, default=1)
    parser.add_argument(
        "--feature-cache-dtype",
        choices=("float16", "float32", "bfloat16"),
        default="float16",
    )
    parser.add_argument("--save-feature-cache", action="store_true")
    parser.add_argument("--feature-log-every", type=int, default=10)

    parser.add_argument(
        "--optimizer",
        choices=("sgd", "adamw"),
        default="sgd",
    )
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

    parser.add_argument(
        "--scheduler",
        choices=("none", "plateau"),
        default="none",
    )
    parser.add_argument("--scheduler-factor", type=float, default=0.3)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-5)

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Controls classifier initialization/training and DataLoader "
            "order. Usually keep equal to --personalization-seed."
        ),
    )
    parser.add_argument(
        "--window-seed",
        type=int,
        default=42,
        help="Controls derived-window ordering after source-trial splitting.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = build_argument_parser().parse_args()

    if abs(args.window_sec - 10.0) > 1e-6:
        raise ValueError(
            "Stage 1 currently reuses the Stage-0.5 10-second input "
            "contract. Use --window-sec 10.0."
        )
    if args.target_subject <= 0:
        raise ValueError("--target-subject must be positive.")
    if args.trials_per_class <= 0:
        raise ValueError("--trials-per-class must be positive.")
    if args.validation_trials_per_class <= 0:
        raise ValueError(
            "--validation-trials-per-class must be positive."
        )
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.head_batch_size <= 0 or args.feature_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if args.patience < 0:
        raise ValueError("--patience must be >= 0.")
    if args.feature_log_every <= 0:
        raise ValueError("--feature-log-every must be positive.")
    if not 0.0 < args.scheduler_factor < 1.0:
        raise ValueError("--scheduler-factor must be in (0,1).")
    if args.scheduler_patience < 0:
        raise ValueError("--scheduler-patience must be >= 0.")
    if args.scheduler_min_lr < 0:
        raise ValueError("--scheduler-min-lr must be >= 0.")
    if args.max_windows_per_class is not None:
        if args.max_windows_per_class <= 0:
            raise ValueError(
                "--max-windows-per-class must be positive."
            )
    if len(STANDARD_64_CHANNELS) != 64:
        raise RuntimeError(
            "STANDARD_64_CHANNELS must contain exactly 64 channel names."
        )

    set_seed(args.seed)
    random.seed(args.seed)

    target_subject = int(args.target_subject)
    target_tag = f"subject_{target_subject:02d}"
    budget_tag = f"trials_{args.trials_per_class:02d}"
    seed_tag = f"seed_{args.personalization_seed}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data_root = resolve_repo_path(args.data_root)
    data_path = resolve_subject_file(
        data_root=data_root,
        pattern=args.data_pattern,
        subject_id=target_subject,
    )
    backbone_path = resolve_repo_path(args.checkpoint)

    if args.population_head is None:
        population_head_path = (
            ROOT
            / "checkpoints"
            / "stage1"
            / target_tag
            / "population_head.pt"
        ).resolve()
    else:
        population_head_path = resolve_repo_path(args.population_head)

    if args.output is None:
        output_path = (
            ROOT
            / "checkpoints"
            / "stage1"
            / target_tag
            / "personal"
            / f"{budget_tag}_{seed_tag}"
            / "personal_head.pt"
        ).resolve()
    else:
        output_path = resolve_repo_path(args.output)

    if args.run_dir is None:
        run_dir = (
            ROOT
            / "runs"
            / "stage1"
            / target_tag
            / "personal"
            / f"{budget_tag}_{seed_tag}_{timestamp}"
        ).resolve()
    else:
        run_dir = resolve_repo_path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    for name, path in (
        ("target HDF5", data_path),
        ("50M backbone", backbone_path),
        ("population head", population_head_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} was not found: {path}")

    git_commit = current_git_commit()
    backbone_sha256 = sha256_file(backbone_path)
    population_head_sha256 = sha256_file(population_head_path)
    population_metadata = load_checkpoint_metadata(
        population_head_path
    )

    # Protocol-level population-head checks.
    population_target = population_metadata.get("target_subject")
    if population_target is not None:
        if int(population_target) != target_subject:
            raise ValueError(
                "Population head target_subject does not match the requested "
                f"personal subject: checkpoint={population_target}, "
                f"requested={target_subject}."
            )

    excluded = {
        int(value)
        for value in population_metadata.get("excluded_subjects", [])
    }
    if excluded and target_subject not in excluded:
        raise ValueError(
            f"Population head metadata does not exclude target subject "
            f"{target_subject}: excluded_subjects={sorted(excluded)}."
        )

    population_training_subjects = {
        int(value)
        for value in population_metadata.get(
            "population_training_subjects",
            [],
        )
    }
    if target_subject in population_training_subjects:
        raise ValueError(
            f"Population head metadata shows target subject {target_subject} "
            "inside population training subjects."
        )

    saved_backbone_hash = population_metadata.get("backbone_sha256")
    if (
        saved_backbone_hash is not None
        and str(saved_backbone_hash) != backbone_sha256
    ):
        raise ValueError(
            "Population head was trained with a different 50M backbone. "
            f"checkpoint metadata={saved_backbone_hash}, "
            f"current={backbone_sha256}."
        )

    print("=" * 88)
    print("Stage 1: few-shot personal 50M linear-head training")
    print("=" * 88)
    print("target subject:", target_subject)
    print("data:", data_path)
    print("personalization session:", args.personalization_session)
    print("final test session:", args.final_test_session)
    print("trials per class:", args.trials_per_class)
    print(
        "validation trials per class:",
        args.validation_trials_per_class,
    )
    print("validation seed:", args.validation_seed)
    print("personalization seed:", args.personalization_seed)
    print("population head:", population_head_path)
    print("personal head init:", args.head_init)
    print("backbone:", backbone_path)
    print("output:", output_path)
    print("run dir:", run_dir)
    print()
    print(
        "IMPORTANT: target 1test is not loaded until personal model "
        "selection has completed."
    )
    print(
        "IMPORTANT: source trials are split before 10-second windows are "
        "constructed."
    )
    print()

    initial_run_config = {
        "status": "started",
        "timestamp": timestamp,
        "git_commit": git_commit,
        "target_subject": target_subject,
        "data_path": str(data_path),
        "backbone_path": str(backbone_path),
        "backbone_sha256": backbone_sha256,
        "population_head_path": str(population_head_path),
        "population_head_sha256": population_head_sha256,
        "output_path": str(output_path),
        "arguments": vars(args),
    }
    atomic_write_json(run_dir / "run_config.json", initial_run_config)

    # ------------------------------------------------------------------
    # Load only target 0train and perform source-trial split.
    # ------------------------------------------------------------------

    dataset = EEGHDF5(data_path)
    metadata = dataset.metadata
    num_classes = len(metadata.class_names)
    class_names = list(metadata.class_names)

    source_session = dataset.load(args.personalization_session)
    validate_loaded_session(
        source_session,
        expected_subject=target_subject,
        expected_session=args.personalization_session,
        num_classes=num_classes,
        path=data_path,
    )

    source_labels = np.asarray(
        source_session["labels"],
        dtype=np.int64,
    )
    source_trial_ids = np.asarray(
        source_session["trial_ids"],
        dtype=np.int64,
    )

    personal_split = build_personal_trial_split(
        labels=source_labels,
        trial_ids=source_trial_ids,
        class_names=class_names,
        trials_per_class=args.trials_per_class,
        validation_trials_per_class=(
            args.validation_trials_per_class
        ),
        validation_seed=args.validation_seed,
        personalization_seed=args.personalization_seed,
    )

    personal_train_source = select_session_rows(
        source_session,
        personal_split.train_indices,
    )
    personal_validation_source = select_session_rows(
        source_session,
        personal_split.validation_indices,
    )

    train_trial_id_set = set(
        np.asarray(
            personal_train_source["trial_ids"],
            dtype=np.int64,
        ).tolist()
    )
    validation_trial_id_set = set(
        np.asarray(
            personal_validation_source["trial_ids"],
            dtype=np.int64,
        ).tolist()
    )
    if train_trial_id_set & validation_trial_id_set:
        raise RuntimeError(
            "Personal train and validation raw trial IDs overlap."
        )

    expected_train_count = args.trials_per_class
    expected_validation_count = args.validation_trials_per_class
    train_source_counts = class_index_counts(
        np.asarray(personal_train_source["labels"], dtype=np.int64),
        num_classes,
    )
    validation_source_counts = class_index_counts(
        np.asarray(
            personal_validation_source["labels"],
            dtype=np.int64,
        ),
        num_classes,
    )
    if any(
        count != expected_train_count
        for count in train_source_counts.values()
    ):
        raise RuntimeError(
            f"Unexpected personal train class counts: {train_source_counts}."
        )
    if any(
        count != expected_validation_count
        for count in validation_source_counts.values()
    ):
        raise RuntimeError(
            "Unexpected personal validation class counts: "
            f"{validation_source_counts}."
        )

    print("Source-trial split:")
    print(
        "  personalization train:",
        class_name_counts(
            np.asarray(
                personal_train_source["labels"],
                dtype=np.int64,
            ),
            class_names,
        ),
    )
    print(
        "  personal validation:",
        class_name_counts(
            np.asarray(
                personal_validation_source["labels"],
                dtype=np.int64,
            ),
            class_names,
        ),
    )
    print(
        "  remaining personalization pool per class:",
        {
            class_name: len(
                personal_split.pool_trial_ids_by_class[class_name]
            )
            for class_name in class_names
        },
    )
    print()

    # Construct windows only after source-trial membership is frozen.
    personal_train_windows = build_windows_from_selected_trials(
        selected=personal_train_source,
        subject_id=target_subject,
        metadata=metadata,
        split_name="personal_train",
        window_seconds=args.window_sec,
        stride_seconds=args.window_stride_sec,
        seed=args.window_seed + 1_000,
        shuffle_trials_within_class=False,
        max_windows_per_class=args.max_windows_per_class,
    )
    personal_validation_windows = build_windows_from_selected_trials(
        selected=personal_validation_source,
        subject_id=target_subject,
        metadata=metadata,
        split_name="personal_validation",
        window_seconds=args.window_sec,
        stride_seconds=args.window_stride_sec,
        seed=args.window_seed + 2_000,
        shuffle_trials_within_class=False,
        max_windows_per_class=args.max_windows_per_class,
    )

    assert_no_window_source_leakage(
        personal_train_windows,
        personal_validation_windows,
        left_name="personal train",
        right_name="personal validation",
    )

    print("Derived 10-second windows:")
    print(
        "  personal train:",
        len(personal_train_windows.windows),
        class_name_counts(
            personal_train_windows.labels,
            class_names,
        ),
    )
    print(
        "  personal validation:",
        len(personal_validation_windows.windows),
        class_name_counts(
            personal_validation_windows.labels,
            class_names,
        ),
    )
    print()

    # ------------------------------------------------------------------
    # Construct Stage-0.5-compatible 50M runtime and load population head.
    # ------------------------------------------------------------------

    config = Model50MConfig(
        checkpoint_path=backbone_path,
        classifier_path=population_head_path,
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

    preprocessing_contract = {
        "target_sample_rate": float(config.target_sample_rate),
        "window_seconds": float(config.window_seconds),
        "target_num_points": int(config.target_num_points),
        "patch_seconds": float(config.patch_seconds),
        "patch_stride_seconds": float(config.patch_stride_seconds),
        "patch_num_points": int(config.patch_num_points),
        "num_tokens": int(config.num_tokens),
        "standard_channels": list(config.standard_channels),
        "filter_enabled": bool(config.filter_enabled),
        "filter_low_hz": float(config.filter_low_hz),
        "filter_high_hz": float(config.filter_high_hz),
        "reference_mode": str(config.reference_mode),
        "strict_window_duration": bool(config.strict_window_duration),
        "output_layer_idx": int(config.output_layer_idx),
        "aggregation": str(config.aggregation),
        "num_classes": int(config.num_classes),
    }
    preprocessing_hash = stable_json_hash(preprocessing_contract)

    saved_preprocessing_hash = population_metadata.get(
        "preprocessing_hash"
    )
    if (
        saved_preprocessing_hash is not None
        and str(saved_preprocessing_hash) != preprocessing_hash
    ):
        raise ValueError(
            "Current preprocessing settings do not match the population "
            "head metadata. "
            f"population={saved_preprocessing_hash}, "
            f"current={preprocessing_hash}. Use the same Stage-1 population "
            "configuration for personal training."
        )

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
    load_report = load_classifier_checkpoint(
        classifier=classifier,
        checkpoint_path=population_head_path,
        strict_metadata=True,
    )
    model_load_seconds = time.perf_counter() - load_start

    if backbone.trainable_parameters != 0:
        raise RuntimeError(
            "The 50M backbone is not fully frozen: "
            f"{backbone.trainable_parameters} trainable parameters."
        )

    feature_device = classifier.device
    head_device = resolve_head_device(
        args.head_device,
        feature_device=feature_device,
        classifier_input_dim=config.classifier_input_dim,
    )

    # Preserve the population baseline before any personal optimization.
    population_head = deepcopy(classifier.head).to(head_device)
    population_head.eval()
    for parameter in population_head.parameters():
        parameter.requires_grad = False

    classifier.head.to(head_device)
    if args.head_init == "random":
        set_seed(args.seed)
        reset_module_parameters(classifier.head)
    classifier.head.train()

    trainable_backbone_parameters = sum(
        parameter.numel()
        for parameter in classifier.backbone.parameters()
        if parameter.requires_grad
    )
    if trainable_backbone_parameters != 0:
        raise RuntimeError(
            "Backbone parameters became trainable unexpectedly."
        )

    print("50M runtime:")
    print("  feature extraction device:", feature_device)
    print("  linear-head training device:", head_device)
    print("  population head load seconds:", load_report.load_seconds)
    print("  total model load seconds:", model_load_seconds)
    print("  aggregation:", config.aggregation)
    print("  feature dim:", config.classifier_input_dim)
    print("  preprocessing hash:", preprocessing_hash)
    print("  trainable backbone parameters:", 0)
    print()

    # ------------------------------------------------------------------
    # Extract target 0train train/validation features.
    # ------------------------------------------------------------------

    cache_dtype = feature_cache_dtype_from_name(
        args.feature_cache_dtype
    )

    personal_train_features = extract_frozen_features(
        window_set=personal_train_windows,
        metadata=metadata,
        config=config,
        classifier=classifier,
        preprocess_batch_size=args.feature_batch_size,
        cache_dtype=cache_dtype,
        split_name="personal_train",
        log_every=args.feature_log_every,
    )
    personal_validation_features = extract_frozen_features(
        window_set=personal_validation_windows,
        metadata=metadata,
        config=config,
        classifier=classifier,
        preprocess_batch_size=args.feature_batch_size,
        cache_dtype=cache_dtype,
        split_name="personal_validation",
        log_every=args.feature_log_every,
    )

    if args.save_feature_cache:
        save_personal_feature_cache(
            dataset=personal_train_features,
            window_set=personal_train_windows,
            path=run_dir / "features_personal_train.pt",
            split_name="personal_train",
            target_subject=target_subject,
            class_names=class_names,
            source_trial_ids_by_class=(
                personal_split.train_trial_ids_by_class
            ),
            backbone_sha256=backbone_sha256,
            preprocessing_hash=preprocessing_hash,
        )
        save_personal_feature_cache(
            dataset=personal_validation_features,
            window_set=personal_validation_windows,
            path=run_dir / "features_personal_validation.pt",
            split_name="personal_validation",
            target_subject=target_subject,
            class_names=class_names,
            source_trial_ids_by_class=(
                personal_split.validation_trial_ids_by_class
            ),
            backbone_sha256=backbone_sha256,
            preprocessing_hash=preprocessing_hash,
        )

    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        personal_train_features,
        batch_size=min(
            args.head_batch_size,
            len(personal_train_features),
        ),
        shuffle=True,
        generator=train_generator,
        num_workers=0,
        pin_memory=head_device.type == "cuda",
        drop_last=False,
    )
    validation_loader = DataLoader(
        personal_validation_features,
        batch_size=min(
            args.head_batch_size,
            len(personal_validation_features),
        ),
        shuffle=False,
        num_workers=0,
        pin_memory=head_device.type == "cuda",
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # Train only the personal linear head.
    # ------------------------------------------------------------------

    optimizer = build_optimizer(
        name=args.optimizer,
        head=classifier.head,
        learning_rate=args.head_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        name=args.scheduler,
        optimizer=optimizer,
        metric_for_best=args.metric_for_best,
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )
    criterion = nn.CrossEntropyLoss()

    best_value = (
        float("inf")
        if args.metric_for_best == "val_loss"
        else -float("inf")
    )
    best_epoch = -1
    best_head_state: dict[str, torch.Tensor] | None = None
    best_validation_raw: EpochMetrics | None = None
    epochs_without_improvement = 0
    epoch_rows: list[dict[str, Any]] = []

    training_start = time.perf_counter()

    print("Training personal linear head")
    print(
        f"init={args.head_init}, optimizer={args.optimizer}, "
        f"epochs={args.epochs}, lr={args.head_lr}, "
        f"momentum={args.momentum}, "
        f"weight_decay={args.weight_decay}, "
        f"scheduler={args.scheduler}, "
        f"best_metric={args.metric_for_best}"
    )

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()

        train_metrics_raw = run_head_epoch(
            head=classifier.head,
            loader=train_loader,
            criterion=criterion,
            device=head_device,
            num_classes=num_classes,
            optimizer=optimizer,
        )
        with torch.no_grad():
            validation_metrics_raw = run_head_epoch(
                head=classifier.head,
                loader=validation_loader,
                criterion=criterion,
                device=head_device,
                num_classes=num_classes,
                optimizer=None,
            )

        train_metrics = extend_metrics(
            train_metrics_raw,
            class_names=class_names,
        )
        validation_metrics = extend_metrics(
            validation_metrics_raw,
            class_names=class_names,
        )

        improved, current_value = metric_is_better(
            metric_name=args.metric_for_best,
            current=validation_metrics_raw,
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
            best_validation_raw = validation_metrics_raw
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler_value = (
                validation_metrics.loss
                if args.metric_for_best == "val_loss"
                else (
                    validation_metrics.accuracy
                    if args.metric_for_best == "val_acc"
                    else validation_metrics.balanced_accuracy
                )
            )
            scheduler.step(scheduler_value)

        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_metrics.loss,
            "train_acc": train_metrics.accuracy,
            "train_bacc": train_metrics.balanced_accuracy,
            "train_macro_f1": train_metrics.macro_f1,
            "val_loss": validation_metrics.loss,
            "val_acc": validation_metrics.accuracy,
            "val_bacc": validation_metrics.balanced_accuracy,
            "val_macro_f1": validation_metrics.macro_f1,
            "is_best": improved,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        epoch_rows.append(row)

        print(
            f"epoch={epoch:03d} "
            f"lr={current_lr:.6g} "
            f"train_loss={train_metrics.loss:.4f} "
            f"train_acc={train_metrics.accuracy:.4f} "
            f"train_bacc={train_metrics.balanced_accuracy:.4f} "
            f"train_f1={train_metrics.macro_f1:.4f} "
            f"val_loss={validation_metrics.loss:.4f} "
            f"val_acc={validation_metrics.accuracy:.4f} "
            f"val_bacc={validation_metrics.balanced_accuracy:.4f} "
            f"val_f1={validation_metrics.macro_f1:.4f} "
            f"{'*' if improved else ''}",
            flush=True,
        )

        if (
            args.patience > 0
            and epochs_without_improvement >= args.patience
        ):
            print(
                "Early stopping: no validation improvement for "
                f"{args.patience} epoch(s)."
            )
            break

    personal_training_seconds = time.perf_counter() - training_start

    if best_head_state is None or best_validation_raw is None:
        raise RuntimeError(
            "No best personal-head checkpoint was selected."
        )

    classifier.head.load_state_dict(best_head_state, strict=True)
    classifier.head.to(head_device)
    classifier.head.eval()

    selected_validation = evaluate_head(
        head=classifier.head,
        loader=validation_loader,
        criterion=criterion,
        device=head_device,
        class_names=class_names,
    )

    # ------------------------------------------------------------------
    # Only now open target 1test for final population/personal comparison.
    # ------------------------------------------------------------------

    print()
    print(
        "Personal model selection is complete. "
        "Opening target-subject final test session..."
    )

    final_test_session = dataset.load(args.final_test_session)
    validate_loaded_session(
        final_test_session,
        expected_subject=target_subject,
        expected_session=args.final_test_session,
        num_classes=num_classes,
        path=data_path,
    )

    final_test_raw_ids = np.asarray(
        final_test_session["trial_ids"],
        dtype=np.int64,
    )
    if train_trial_id_set & set(final_test_raw_ids.tolist()):
        raise RuntimeError(
            "Raw trial IDs overlap between personal train and final test."
        )
    if validation_trial_id_set & set(final_test_raw_ids.tolist()):
        raise RuntimeError(
            "Raw trial IDs overlap between personal validation and final test."
        )

    final_test_windows = build_windows_from_selected_trials(
        selected=final_test_session,
        subject_id=target_subject,
        metadata=metadata,
        split_name="target_final_test",
        window_seconds=args.window_sec,
        stride_seconds=args.window_stride_sec,
        seed=args.window_seed + 3_000,
        shuffle_trials_within_class=False,
        max_windows_per_class=args.max_windows_per_class,
    )

    assert_no_window_source_leakage(
        personal_train_windows,
        final_test_windows,
        left_name="personal train",
        right_name="target final test",
    )
    assert_no_window_source_leakage(
        personal_validation_windows,
        final_test_windows,
        left_name="personal validation",
        right_name="target final test",
    )

    final_test_features = extract_frozen_features(
        window_set=final_test_windows,
        metadata=metadata,
        config=config,
        classifier=classifier,
        preprocess_batch_size=args.feature_batch_size,
        cache_dtype=cache_dtype,
        split_name="target_final_test",
        log_every=args.feature_log_every,
    )

    if args.save_feature_cache:
        save_personal_feature_cache(
            dataset=final_test_features,
            window_set=final_test_windows,
            path=run_dir / "features_target_final_test.pt",
            split_name="target_final_test",
            target_subject=target_subject,
            class_names=class_names,
            source_trial_ids_by_class=None,
            backbone_sha256=backbone_sha256,
            preprocessing_hash=preprocessing_hash,
        )

    final_test_loader = DataLoader(
        final_test_features,
        batch_size=min(
            args.head_batch_size,
            len(final_test_features),
        ),
        shuffle=False,
        num_workers=0,
        pin_memory=head_device.type == "cuda",
        drop_last=False,
    )

    population_final = evaluate_head(
        head=population_head,
        loader=final_test_loader,
        criterion=criterion,
        device=head_device,
        class_names=class_names,
    )
    personal_final = evaluate_head(
        head=classifier.head,
        loader=final_test_loader,
        criterion=criterion,
        device=head_device,
        class_names=class_names,
    )

    if population_final.labels != personal_final.labels:
        raise RuntimeError(
            "Population and personal evaluations used different labels."
        )

    accuracy_gain = (
        personal_final.metrics.accuracy
        - population_final.metrics.accuracy
    )
    bacc_gain = (
        personal_final.metrics.balanced_accuracy
        - population_final.metrics.balanced_accuracy
    )
    macro_f1_gain = (
        personal_final.metrics.macro_f1
        - population_final.metrics.macro_f1
    )

    # ------------------------------------------------------------------
    # Save personal head and reports.
    # ------------------------------------------------------------------

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_start = time.perf_counter()
    saved_path = save_classifier_checkpoint(
        classifier=classifier,
        checkpoint_path=output_path,
        extra_metadata={
            "task": "BNCI2014_001_motor_imagery",
            "dataset": metadata.dataset_name,
            "mode": "personal_few_shot_linear_head",
            "stage": "stage1",
            "model_type": "50m_personal_head",
            "user_id": target_tag,
            "target_subject": target_subject,
            "personalization_session": args.personalization_session,
            "final_test_session": args.final_test_session,
            "trials_per_class": int(args.trials_per_class),
            "validation_trials_per_class": int(
                args.validation_trials_per_class
            ),
            "validation_seed": int(args.validation_seed),
            "personalization_seed": int(
                args.personalization_seed
            ),
            "training_seed": int(args.seed),
            "window_seed": int(args.window_seed),
            "head_initialization": args.head_init,
            "base_population_model": str(population_head_path),
            "base_population_sha256": population_head_sha256,
            "population_training_subjects": sorted(
                population_training_subjects
            ),
            "backbone_sha256": backbone_sha256,
            "preprocessing_hash": preprocessing_hash,
            "class_names": class_names,
            "source_train_trial_ids_by_class": (
                personal_split.train_trial_ids_by_class
            ),
            "source_validation_trial_ids_by_class": (
                personal_split.validation_trial_ids_by_class
            ),
            "freeze_backbone": True,
            "trainable_backbone_parameters": 0,
            "feature_extraction_device": str(feature_device),
            "head_training_device": str(head_device),
            "optimizer": args.optimizer,
            "head_lr": float(args.head_lr),
            "momentum": float(args.momentum),
            "weight_decay": float(args.weight_decay),
            "scheduler": args.scheduler,
            "best_epoch": int(best_epoch),
            "best_personal_validation": (
                selected_validation.metrics.to_dict()
            ),
            "population_final_test": (
                population_final.metrics.to_dict()
            ),
            "personal_final_test": (
                personal_final.metrics.to_dict()
            ),
            "final_test_gain": {
                "accuracy": float(accuracy_gain),
                "balanced_accuracy": float(bacc_gain),
                "macro_f1": float(macro_f1_gain),
            },
            "personal_training_seconds": float(
                personal_training_seconds
            ),
            "window_construction": (
                "same_label_trial_concatenation_within_split"
            ),
            "warning": (
                "Temporary Stage-1 baseline: 10-second samples are built "
                "from 4-second source trials, but never across train/val/test "
                "splits, sessions, subjects, or labels."
            ),
            "git_commit": git_commit,
        },
    )
    save_seconds = time.perf_counter() - save_start

    # Verify that the saved personal head can be loaded in a fresh classifier.
    reload_start = time.perf_counter()
    reload_backbone = Model50MBackbone(
        config=config,
        load_checkpoint=True,
        freeze=True,
    )
    reload_classifier = Model50MClassifier(
        config=config,
        backbone=reload_backbone,
    )
    reload_report = load_classifier_checkpoint(
        classifier=reload_classifier,
        checkpoint_path=saved_path,
        strict_metadata=True,
    )
    reloaded_head = deepcopy(reload_classifier.head).to(head_device)
    reloaded_final = evaluate_head(
        head=reloaded_head,
        loader=final_test_loader,
        criterion=criterion,
        device=head_device,
        class_names=class_names,
    )
    reload_total_seconds = time.perf_counter() - reload_start

    if reloaded_final.predictions != personal_final.predictions:
        raise RuntimeError(
            "Reloaded personal head predictions differ from the original."
        )
    if not np.allclose(
        np.asarray(reloaded_final.probabilities),
        np.asarray(personal_final.probabilities),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise RuntimeError(
            "Reloaded personal head probabilities differ from the original."
        )

    metrics_csv = run_dir / "epoch_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(epoch_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(epoch_rows)

    predictions_csv = run_dir / "final_test_predictions.csv"
    with predictions_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = [
            "window_index",
            "label",
            "label_name",
            "population_prediction",
            "population_prediction_name",
            "population_confidence",
            "personal_prediction",
            "personal_prediction_name",
            "personal_confidence",
            "population_correct",
            "personal_correct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for index, label in enumerate(personal_final.labels):
            population_prediction = (
                population_final.predictions[index]
            )
            personal_prediction = personal_final.predictions[index]
            writer.writerow(
                {
                    "window_index": index,
                    "label": label,
                    "label_name": class_names[label],
                    "population_prediction": population_prediction,
                    "population_prediction_name": class_names[
                        population_prediction
                    ],
                    "population_confidence": (
                        population_final.confidences[index]
                    ),
                    "personal_prediction": personal_prediction,
                    "personal_prediction_name": class_names[
                        personal_prediction
                    ],
                    "personal_confidence": (
                        personal_final.confidences[index]
                    ),
                    "population_correct": (
                        population_prediction == label
                    ),
                    "personal_correct": (
                        personal_prediction == label
                    ),
                }
            )

    report = {
        "status": "completed",
        "stage": "stage1",
        "experiment": "few_shot_personal_linear_head",
        "warning": (
            "Temporary 10-second baseline: derived windows may cross "
            "original 4-second trial boundaries, but never cross source "
            "split, session, subject, or label boundaries."
        ),
        "files": {
            "target_data": str(data_path),
            "backbone_checkpoint": str(backbone_path),
            "population_head": str(population_head_path),
            "personal_head": str(saved_path),
            "run_dir": str(run_dir),
            "epoch_metrics_csv": str(metrics_csv),
            "final_predictions_csv": str(predictions_csv),
        },
        "protocol": {
            "target_subject": target_subject,
            "personalization_session": args.personalization_session,
            "final_test_session": args.final_test_session,
            "trials_per_class": int(args.trials_per_class),
            "validation_trials_per_class": int(
                args.validation_trials_per_class
            ),
            "validation_seed": int(args.validation_seed),
            "personalization_seed": int(
                args.personalization_seed
            ),
            "head_initialization": args.head_init,
            "target_final_test_opened_after_model_selection": True,
            "target_final_test_used_for_training": False,
            "target_final_test_used_for_validation": False,
        },
        "dataset": {
            "name": metadata.dataset_name,
            "sample_rate": metadata.sample_rate,
            "unit": metadata.unit,
            "channel_names": metadata.channel_names,
            "class_names": class_names,
        },
        "source_trial_split": {
            "train_counts": class_name_counts(
                np.asarray(
                    personal_train_source["labels"],
                    dtype=np.int64,
                ),
                class_names,
            ),
            "validation_counts": class_name_counts(
                np.asarray(
                    personal_validation_source["labels"],
                    dtype=np.int64,
                ),
                class_names,
            ),
            "train_trial_ids_by_class": (
                personal_split.train_trial_ids_by_class
            ),
            "validation_trial_ids_by_class": (
                personal_split.validation_trial_ids_by_class
            ),
            "personalization_pool_order_by_class": (
                personal_split.pool_trial_ids_by_class
            ),
        },
        "derived_windows": {
            "personal_train_total": int(
                len(personal_train_windows.windows)
            ),
            "personal_train_per_class": class_name_counts(
                personal_train_windows.labels,
                class_names,
            ),
            "personal_validation_total": int(
                len(personal_validation_windows.windows)
            ),
            "personal_validation_per_class": class_name_counts(
                personal_validation_windows.labels,
                class_names,
            ),
            "final_test_total": int(
                len(final_test_windows.windows)
            ),
            "final_test_per_class": class_name_counts(
                final_test_windows.labels,
                class_names,
            ),
            "window_seconds": float(args.window_sec),
            "stride_seconds": float(args.window_stride_sec),
        },
        "model": {
            "feature_extraction_device": str(feature_device),
            "head_training_device": str(head_device),
            "model_load_seconds": model_load_seconds,
            "backbone_sha256": backbone_sha256,
            "population_head_sha256": population_head_sha256,
            "trainable_backbone_parameters": 0,
            "aggregation": config.aggregation,
            "feature_dim": config.classifier_input_dim,
            "output_layer_idx": config.output_layer_idx,
            "preprocessing_contract": preprocessing_contract,
            "preprocessing_hash": preprocessing_hash,
            "feature_cache_dtype": args.feature_cache_dtype,
        },
        "training": {
            "optimizer": args.optimizer,
            "epochs_requested": int(args.epochs),
            "epochs_completed": len(epoch_rows),
            "head_lr": float(args.head_lr),
            "momentum": float(args.momentum),
            "weight_decay": float(args.weight_decay),
            "scheduler": args.scheduler,
            "scheduler_factor": float(args.scheduler_factor),
            "scheduler_patience": int(
                args.scheduler_patience
            ),
            "scheduler_min_lr": float(args.scheduler_min_lr),
            "patience": int(args.patience),
            "metric_for_best": args.metric_for_best,
            "best_epoch": int(best_epoch),
            "training_seconds": float(personal_training_seconds),
            "training_seed": int(args.seed),
            "window_seed": int(args.window_seed),
        },
        "best_personal_validation": (
            selected_validation.to_dict()
        ),
        "final_test": {
            "population": population_final.to_dict(),
            "personal": personal_final.to_dict(),
            "gain": {
                "accuracy": float(accuracy_gain),
                "balanced_accuracy": float(bacc_gain),
                "macro_f1": float(macro_f1_gain),
            },
        },
        "save_reload": {
            "save_seconds": float(save_seconds),
            "classifier_load_seconds": float(
                reload_report.load_seconds
            ),
            "reload_total_seconds": float(
                reload_total_seconds
            ),
            "predictions_identical": True,
            "probabilities_allclose": True,
        },
        "reproducibility": {
            "git_commit": git_commit,
            "backbone_sha256": backbone_sha256,
            "population_head_sha256": population_head_sha256,
            "preprocessing_hash": preprocessing_hash,
            "source_trial_encoding": (
                "(subject_id << 32) | file_local_trial_id"
            ),
        },
    }

    report_path = run_dir / "personal_training_report.json"
    atomic_write_json(report_path, report)

    summary = {
        "status": "completed",
        "target_subject": target_subject,
        "trials_per_class": int(args.trials_per_class),
        "validation_trials_per_class": int(
            args.validation_trials_per_class
        ),
        "personalization_seed": int(
            args.personalization_seed
        ),
        "head_initialization": args.head_init,
        "best_epoch": int(best_epoch),
        "personal_validation": (
            selected_validation.metrics.to_dict()
        ),
        "population_final_test": (
            population_final.metrics.to_dict()
        ),
        "personal_final_test": (
            personal_final.metrics.to_dict()
        ),
        "gain": {
            "accuracy": float(accuracy_gain),
            "balanced_accuracy": float(bacc_gain),
            "macro_f1": float(macro_f1_gain),
        },
        "personal_head": str(saved_path),
        "report": str(report_path),
    }
    atomic_write_json(run_dir / "summary.json", summary)

    print()
    print("=" * 88)
    print("Personal-head training completed")
    print("=" * 88)
    print("best epoch:", best_epoch)
    print(
        "personal validation:",
        f"acc={selected_validation.metrics.accuracy:.4f}, "
        f"bacc={selected_validation.metrics.balanced_accuracy:.4f}, "
        f"macro_f1={selected_validation.metrics.macro_f1:.4f}",
    )
    print(
        "population final test:",
        f"acc={population_final.metrics.accuracy:.4f}, "
        f"bacc={population_final.metrics.balanced_accuracy:.4f}, "
        f"macro_f1={population_final.metrics.macro_f1:.4f}",
    )
    print(
        "personal final test:",
        f"acc={personal_final.metrics.accuracy:.4f}, "
        f"bacc={personal_final.metrics.balanced_accuracy:.4f}, "
        f"macro_f1={personal_final.metrics.macro_f1:.4f}",
    )
    print(
        "gain:",
        f"accuracy={accuracy_gain:+.4f}, "
        f"bacc={bacc_gain:+.4f}, "
        f"macro_f1={macro_f1_gain:+.4f}",
    )
    print("saved personal head:", saved_path)
    print("save/reload verification: passed")
    print("report:", report_path)
    print()


if __name__ == "__main__":
    main()
