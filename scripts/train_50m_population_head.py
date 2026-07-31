from __future__ import annotations

"""
Train a Stage-1 LOSO population linear head on BNCI2014_001.

Protocol
--------
For one target subject:

- Population training:
    all non-target subjects / 0train
- Population validation:
    all non-target subjects / 1test
- Final unseen-subject test:
    target subject / 1test

The target subject is never used to train or select the population model.

Important
---------
The BNCI HDF5 files contain 4-second source trials. To preserve the Stage-0.5
50M input contract, this script constructs temporary 10-second single-label
windows by concatenating trials only within the same:

    subject + session + class

It never concatenates trials across subjects, sessions, or labels.

The 50M backbone is frozen. Only the linear classification head is trained.
"""

import argparse
import csv
import hashlib
import json
import random
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from _bootstrap import ROOT
from bci_dayloop.data.hdf5_dataset import EEGHDF5, HDF5Metadata
from bci_dayloop.models.model_50m.backbone import Model50MBackbone
from bci_dayloop.models.model_50m.classifier import (
    Model50MClassifier,
    save_classifier_checkpoint,
)
from bci_dayloop.models.model_50m.config import (
    STANDARD_64_CHANNELS,
    Model50MConfig,
)

# Reuse the already validated Stage-0.5 preprocessing, window construction,
# frozen-feature extraction, and linear-head training helpers.
from train_50m_linear_head import (
    EpochMetrics,
    WindowSet,
    build_same_label_concat_windows,
    class_counts,
    extract_frozen_features,
    feature_cache_dtype_from_name,
    json_default,
    limit_windows_per_class,
    metric_is_better,
    resolve_repo_path,
    run_head_epoch,
    set_seed,
    validate_labels,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WindowBundle:
    """A WindowSet plus the subject identity of every derived window."""

    window_set: WindowSet
    window_subject_ids: np.ndarray  # [N]

    def __post_init__(self) -> None:
        expected = (len(self.window_set.windows),)
        if self.window_subject_ids.shape != expected:
            raise ValueError(
                "window_subject_ids shape mismatch: "
                f"expected {expected}, got {self.window_subject_ids.shape}."
            )


@dataclass(frozen=True, slots=True)
class SplitBuildResult:
    bundle: WindowBundle
    source_trial_summary: dict[str, Any]
    subject_paths: dict[int, Path]
    metadata: HDF5Metadata


@dataclass(frozen=True, slots=True)
class ExtendedMetrics:
    loss: float
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    confusion_matrix: list[list[int]]
    per_class: list[dict[str, float | int | None]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "confusion_matrix": self.confusion_matrix,
            "per_class": self.per_class,
        }


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def normalize_subjects(values: Iterable[int]) -> list[int]:
    subjects = sorted(set(int(value) for value in values))
    if not subjects:
        raise ValueError("At least one subject is required.")
    invalid = [subject for subject in subjects if subject <= 0]
    if invalid:
        raise ValueError(f"Subject IDs must be positive, got {invalid}.")
    return subjects


def class_name_counts(
    labels: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, int]:
    numeric = class_counts(labels, len(class_names))
    return {
        str(class_names[index]): int(numeric[index])
        for index in range(len(class_names))
    }


def encoded_trial_id(subject_id: int, trial_id: int) -> int:
    """
    Encode subject and file-local trial ID into one collision-free int64 key.

    High 32 bits: subject ID
    Low 32 bits: non-negative trial ID
    """
    subject_id = int(subject_id)
    trial_id = int(trial_id)
    if subject_id <= 0:
        raise ValueError(f"subject_id must be positive, got {subject_id}.")
    if trial_id < 0 or trial_id >= 2**32:
        raise ValueError(
            "trial_id must be in [0, 2**32), "
            f"got {trial_id} for subject {subject_id}."
        )
    return (subject_id << 32) | trial_id


def encode_trial_ids(
    subject_id: int,
    trial_ids: np.ndarray,
) -> np.ndarray:
    trial_ids = np.asarray(trial_ids, dtype=np.int64)
    return np.asarray(
        [encoded_trial_id(subject_id, value) for value in trial_ids],
        dtype=np.int64,
    )


def source_id_set(window_set: WindowSet) -> set[int]:
    return {
        int(source_id)
        for source_ids in window_set.source_trial_ids
        for source_id in source_ids
    }


def validate_no_source_leakage(
    left: WindowSet,
    right: WindowSet,
    *,
    left_name: str,
    right_name: str,
) -> None:
    overlap = source_id_set(left) & source_id_set(right)
    if overlap:
        examples = sorted(overlap)[:10]
        raise RuntimeError(
            f"Source-trial leakage between {left_name} and {right_name}. "
            f"Example encoded trial IDs: {examples}."
        )


def validate_metadata_compatibility(
    reference: HDF5Metadata,
    candidate: HDF5Metadata,
    *,
    subject_id: int,
    path: Path,
) -> None:
    mismatches: list[str] = []

    if not np.isclose(reference.sample_rate, candidate.sample_rate):
        mismatches.append(
            "sample_rate "
            f"{candidate.sample_rate} != {reference.sample_rate}"
        )
    if list(reference.channel_names) != list(candidate.channel_names):
        mismatches.append("channel_names differ")
    if list(reference.class_names) != list(candidate.class_names):
        mismatches.append("class_names differ")
    if str(reference.unit) != str(candidate.unit):
        mismatches.append(f"unit {candidate.unit!r} != {reference.unit!r}")
    if str(reference.dataset_name) != str(candidate.dataset_name):
        mismatches.append(
            "dataset_name "
            f"{candidate.dataset_name!r} != {reference.dataset_name!r}"
        )

    if mismatches:
        raise ValueError(
            f"Metadata mismatch for subject {subject_id} at {path}: "
            + "; ".join(mismatches)
        )


def resolve_subject_file(
    *,
    data_root: Path,
    pattern: str,
    subject_id: int,
) -> Path:
    try:
        relative_name = pattern.format(subject=subject_id)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "--data-pattern must be a valid Python format string using "
            "{subject}, for example 'subject_{subject:02d}.h5'."
        ) from exc

    candidates = [
        data_root / relative_name,
        data_root / f"subject_{subject_id:02d}.h5",
        data_root / f"bnci2014_001_s{subject_id:02d}.h5",
        ROOT / "data" / "processed" / f"bnci2014_001_s{subject_id:02d}.h5",
    ]

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        if not resolved.is_absolute():
            resolved = ROOT / resolved
        resolved = resolved.resolve()
        if resolved not in seen:
            unique_candidates.append(resolved)
            seen.add(resolved)

    for candidate in unique_candidates:
        if candidate.is_file():
            return candidate

    formatted = "\n".join(f"  - {path}" for path in unique_candidates)
    raise FileNotFoundError(
        f"Could not find HDF5 data for subject {subject_id}. Tried:\n"
        f"{formatted}"
    )


def validate_loaded_session(
    session_data: Mapping[str, np.ndarray],
    *,
    expected_subject: int,
    expected_session: str,
    num_classes: int,
    path: Path,
) -> None:
    required = {
        "data",
        "labels",
        "subject_ids",
        "session_ids",
        "trial_ids",
    }
    missing = required - set(session_data)
    if missing:
        raise KeyError(
            f"{path}: loaded session is missing keys {sorted(missing)}."
        )

    n_trials = len(session_data["data"])
    if n_trials <= 0:
        raise ValueError(
            f"{path}: session {expected_session!r} contains no trials."
        )

    for key in ("labels", "subject_ids", "session_ids", "trial_ids"):
        if len(session_data[key]) != n_trials:
            raise ValueError(
                f"{path}: {key} length {len(session_data[key])} "
                f"does not match data length {n_trials}."
            )

    subject_values = sorted(
        set(np.asarray(session_data["subject_ids"], dtype=np.int64).tolist())
    )
    if subject_values != [expected_subject]:
        raise ValueError(
            f"{path}: expected only subject {expected_subject}, "
            f"found {subject_values}."
        )

    session_values = sorted(
        set(np.asarray(session_data["session_ids"]).astype(str).tolist())
    )
    if session_values != [expected_session]:
        raise ValueError(
            f"{path}: expected only session {expected_session!r}, "
            f"found {session_values}."
        )

    trial_ids = np.asarray(session_data["trial_ids"], dtype=np.int64)
    if len(np.unique(trial_ids)) != len(trial_ids):
        raise ValueError(
            f"{path}: duplicate trial_ids found in session "
            f"{expected_session!r}."
        )

    validate_labels(
        np.asarray(session_data["labels"], dtype=np.int64),
        num_classes=num_classes,
        split_name=f"subject_{expected_subject:02d}/{expected_session}",
    )

    signal = np.asarray(session_data["data"])
    if signal.ndim != 3:
        raise ValueError(
            f"{path}: EEG data must have shape [N,C,T], got {signal.shape}."
        )
    if not np.isfinite(signal).all():
        raise ValueError(
            f"{path}: session {expected_session!r} contains NaN or Inf."
        )


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------


def build_subject_window_bundle(
    *,
    subject_id: int,
    path: Path,
    session_name: str,
    reference_metadata: HDF5Metadata | None,
    window_seconds: float,
    stride_seconds: float,
    seed: int,
    shuffle_trials_within_class: bool,
    max_windows_per_class: int | None,
) -> tuple[WindowBundle, HDF5Metadata, dict[str, Any]]:
    dataset = EEGHDF5(path)
    metadata = dataset.metadata

    if reference_metadata is not None:
        validate_metadata_compatibility(
            reference_metadata,
            metadata,
            subject_id=subject_id,
            path=path,
        )

    num_classes = len(metadata.class_names)
    session_data = dataset.load(session_name)
    validate_loaded_session(
        session_data,
        expected_subject=subject_id,
        expected_session=session_name,
        num_classes=num_classes,
        path=path,
    )

    raw_trial_ids = np.asarray(session_data["trial_ids"], dtype=np.int64)
    global_trial_ids = encode_trial_ids(subject_id, raw_trial_ids)

    window_set = build_same_label_concat_windows(
        trials=np.asarray(session_data["data"], dtype=np.float32),
        labels=np.asarray(session_data["labels"], dtype=np.int64),
        trial_ids=global_trial_ids,
        sample_rate=metadata.sample_rate,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        num_classes=num_classes,
        seed=seed,
        shuffle_trials_within_class=shuffle_trials_within_class,
        split_name=f"subject_{subject_id:02d}/{session_name}",
    )

    window_set = limit_windows_per_class(
        window_set,
        max_per_class=max_windows_per_class,
        num_classes=num_classes,
        seed=seed + 1,
    )

    bundle = WindowBundle(
        window_set=window_set,
        window_subject_ids=np.full(
            len(window_set.windows),
            subject_id,
            dtype=np.int64,
        ),
    )

    summary = {
        "subject_id": subject_id,
        "path": str(path),
        "session": session_name,
        "raw_shape": list(np.asarray(session_data["data"]).shape),
        "source_trials_total": int(len(session_data["labels"])),
        "source_trials_per_class": class_name_counts(
            np.asarray(session_data["labels"], dtype=np.int64),
            metadata.class_names,
        ),
        "derived_windows_total": int(len(window_set.windows)),
        "derived_windows_per_class": class_name_counts(
            window_set.labels,
            metadata.class_names,
        ),
        "unique_source_trials_used": int(len(source_id_set(window_set))),
        "window_seconds": float(window_seconds),
        "stride_seconds": float(stride_seconds),
        "construction": window_set.construction,
    }
    return bundle, metadata, summary


def combine_window_bundles(
    bundles: Sequence[WindowBundle],
    *,
    seed: int,
    construction: str,
) -> WindowBundle:
    if not bundles:
        raise ValueError("No window bundles were provided.")

    windows = np.concatenate(
        [bundle.window_set.windows for bundle in bundles],
        axis=0,
    ).astype(np.float32, copy=False)
    labels = np.concatenate(
        [bundle.window_set.labels for bundle in bundles],
        axis=0,
    ).astype(np.int64, copy=False)
    window_subject_ids = np.concatenate(
        [bundle.window_subject_ids for bundle in bundles],
        axis=0,
    ).astype(np.int64, copy=False)

    source_trial_ids: list[tuple[int, ...]] = []
    for bundle in bundles:
        source_trial_ids.extend(bundle.window_set.source_trial_ids)

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(windows))

    combined_set = WindowSet(
        windows=windows[permutation],
        labels=labels[permutation],
        source_trial_ids=tuple(
            source_trial_ids[int(index)] for index in permutation
        ),
        construction=construction,
    )
    return WindowBundle(
        window_set=combined_set,
        window_subject_ids=window_subject_ids[permutation],
    )


def build_population_split(
    *,
    subjects: Sequence[int],
    data_root: Path,
    data_pattern: str,
    session_name: str,
    window_seconds: float,
    stride_seconds: float,
    base_seed: int,
    shuffle_trials_within_class: bool,
    max_windows_per_class_per_subject: int | None,
    reference_metadata: HDF5Metadata | None = None,
) -> SplitBuildResult:
    bundles: list[WindowBundle] = []
    summaries: dict[str, Any] = {}
    paths: dict[int, Path] = {}
    common_metadata = reference_metadata

    for offset, subject_id in enumerate(subjects):
        path = resolve_subject_file(
            data_root=data_root,
            pattern=data_pattern,
            subject_id=subject_id,
        )
        paths[subject_id] = path

        bundle, metadata, summary = build_subject_window_bundle(
            subject_id=subject_id,
            path=path,
            session_name=session_name,
            reference_metadata=common_metadata,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            seed=base_seed + offset * 100,
            shuffle_trials_within_class=shuffle_trials_within_class,
            max_windows_per_class=max_windows_per_class_per_subject,
        )
        if common_metadata is None:
            common_metadata = metadata

        bundles.append(bundle)
        summaries[f"subject_{subject_id:02d}"] = summary

    assert common_metadata is not None
    combined = combine_window_bundles(
        bundles,
        seed=base_seed + 99_999,
        construction=(
            "same_label_trial_concatenation_per_subject_and_session"
        ),
    )

    return SplitBuildResult(
        bundle=combined,
        source_trial_summary=summaries,
        subject_paths=paths,
        metadata=common_metadata,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def extend_metrics(
    metrics: EpochMetrics,
    *,
    class_names: Sequence[str],
) -> ExtendedMetrics:
    confusion = np.asarray(metrics.confusion_matrix, dtype=np.int64)
    if confusion.shape != (len(class_names), len(class_names)):
        raise ValueError(
            "Confusion matrix shape does not match class names: "
            f"{confusion.shape} vs {len(class_names)}."
        )

    true_support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    true_positive = np.diag(confusion)

    per_class: list[dict[str, float | int | None]] = []
    f1_values: list[float] = []

    for index, class_name in enumerate(class_names):
        tp = int(true_positive[index])
        support = int(true_support[index])
        predicted = int(predicted_support[index])

        precision = (tp / predicted) if predicted > 0 else 0.0
        recall = (tp / support) if support > 0 else 0.0
        denominator = precision + recall
        f1 = (
            2.0 * precision * recall / denominator
            if denominator > 0
            else 0.0
        )
        f1_values.append(float(f1))
        per_class.append(
            {
                "class_index": index,
                "class_name": str(class_name),
                "support": support,
                "predicted": predicted,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    return ExtendedMetrics(
        loss=float(metrics.loss),
        accuracy=float(metrics.accuracy),
        balanced_accuracy=float(metrics.balanced_accuracy),
        macro_f1=float(np.mean(f1_values)),
        confusion_matrix=metrics.confusion_matrix,
        per_class=per_class,
    )


# ---------------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------------


def save_population_feature_cache(
    *,
    dataset: TensorDataset,
    bundle: WindowBundle,
    path: Path,
    split_name: str,
    class_names: Sequence[str],
    subject_ids: Sequence[int],
    backbone_sha256: str,
    preprocessing_hash: str,
) -> None:
    features, labels = dataset.tensors
    if len(features) != len(bundle.window_set.windows):
        raise ValueError(
            f"{split_name}: feature count {len(features)} does not match "
            f"window count {len(bundle.window_set.windows)}."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    torch.save(
        {
            "format_version": 1,
            "split": split_name,
            "features": features,
            "labels": labels,
            "window_subject_ids": torch.from_numpy(
                bundle.window_subject_ids.astype(np.int64, copy=False)
            ),
            "source_trial_ids": [
                list(source_ids)
                for source_ids in bundle.window_set.source_trial_ids
            ],
            "subjects": [int(subject) for subject in subject_ids],
            "class_names": [str(name) for name in class_names],
            "class_counts": class_name_counts(
                bundle.window_set.labels,
                class_names,
            ),
            "window_construction": bundle.window_set.construction,
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
# CLI
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a frozen-50M LOSO population linear head on "
            "multi-subject BNCI2014_001 HDF5 files."
        )
    )

    parser.add_argument(
        "--data-root",
        default="data/processed/bnci2014_001",
        help=(
            "Directory containing one HDF5 file per subject. "
            "Relative paths are resolved from the repository root."
        ),
    )
    parser.add_argument(
        "--data-pattern",
        default="subject_{subject:02d}.h5",
        help=(
            "Subject filename pattern. It must contain {subject}; "
            "default: subject_{subject:02d}.h5."
        ),
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=list(range(1, 10)),
        help="All available subject IDs. Default: 1 2 ... 9.",
    )
    parser.add_argument(
        "--target-subject",
        type=int,
        default=1,
        help=(
            "LOSO target subject excluded from population training and "
            "validation. Its 1test session is used only for final testing."
        ),
    )

    parser.add_argument("--train-session", default="0train")
    parser.add_argument("--validation-session", default="1test")
    parser.add_argument("--final-test-session", default="1test")

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/50m/model_deploy.pt",
        help="Dependency-free 50M backbone checkpoint.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output population-head checkpoint. Default: "
            "checkpoints/stage1/subject_XX/population_head.pt."
        ),
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Run directory. Default: "
            "runs/stage1/subject_XX/population_<timestamp>."
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

    parser.add_argument(
        "--max-windows-per-class-per-subject",
        type=int,
        default=None,
        help=(
            "Optional debug limit, independently applied to every subject "
            "and split. Do not use for formal experiments."
        ),
    )
    parser.add_argument("--feature-batch-size", type=int, default=2)
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

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed and DataLoader seed.",
    )
    parser.add_argument(
        "--window-seed",
        type=int,
        default=42,
        help="Seed used for per-subject trial order and window construction.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = build_argument_parser().parse_args()

    if abs(args.window_sec - 10.0) > 1e-6:
        raise ValueError(
            "The current Stage-1 baseline reuses the original 50M "
            "10-second input contract. Use --window-sec 10.0."
        )
    if args.window_stride_sec <= 0:
        raise ValueError("--window-stride-sec must be positive.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.head_batch_size <= 0 or args.feature_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if args.patience < 0:
        raise ValueError("--patience must be >= 0.")
    if args.feature_log_every <= 0:
        raise ValueError("--feature-log-every must be positive.")
    if (
        args.max_windows_per_class_per_subject is not None
        and args.max_windows_per_class_per_subject <= 0
    ):
        raise ValueError(
            "--max-windows-per-class-per-subject must be positive."
        )
    if len(STANDARD_64_CHANNELS) != 64:
        raise RuntimeError(
            "STANDARD_64_CHANNELS must contain exactly 64 names, but "
            f"config.py contains {len(STANDARD_64_CHANNELS)}."
        )

    subjects = normalize_subjects(args.subjects)
    target_subject = int(args.target_subject)
    if target_subject not in subjects:
        raise ValueError(
            f"Target subject {target_subject} is not in --subjects {subjects}."
        )

    population_subjects = [
        subject for subject in subjects if subject != target_subject
    ]
    if not population_subjects:
        raise ValueError(
            "LOSO population training requires at least one non-target subject."
        )

    set_seed(args.seed)

    data_root = resolve_repo_path(args.data_root)
    checkpoint_path = resolve_repo_path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"50M backbone checkpoint was not found: {checkpoint_path}"
        )

    target_tag = f"subject_{target_subject:02d}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output is None:
        output_path = (
            ROOT
            / "checkpoints"
            / "stage1"
            / target_tag
            / "population_head.pt"
        ).resolve()
    else:
        output_path = resolve_repo_path(args.output)

    if args.run_dir is None:
        run_dir = (
            ROOT
            / "runs"
            / "stage1"
            / target_tag
            / f"population_{timestamp}"
        ).resolve()
    else:
        run_dir = resolve_repo_path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    git_commit = current_git_commit()
    backbone_sha256 = sha256_file(checkpoint_path)

    print("=" * 88)
    print("Stage 1: frozen-50M LOSO population-head training")
    print("=" * 88)
    print("target subject:", target_subject)
    print("population subjects:", population_subjects)
    print("population train session:", args.train_session)
    print("population validation session:", args.validation_session)
    print("final unseen-subject test:", f"{target_tag}/{args.final_test_session}")
    print("data root:", data_root)
    print("data pattern:", args.data_pattern)
    print("backbone:", checkpoint_path)
    print("output:", output_path)
    print("run dir:", run_dir)
    print()
    print(
        "IMPORTANT: windows are built separately for every subject and "
        "session. Trials are never concatenated across subjects, sessions, "
        "or labels."
    )
    print(
        "IMPORTANT: the target subject is not loaded for final evaluation "
        "until population training and model selection are complete."
    )
    print()

    # Check all subject file paths before a long run starts.
    all_subject_paths = {
        subject: resolve_subject_file(
            data_root=data_root,
            pattern=args.data_pattern,
            subject_id=subject,
        )
        for subject in subjects
    }

    initial_run_config = {
        "status": "started",
        "timestamp": timestamp,
        "git_commit": git_commit,
        "target_subject": target_subject,
        "population_subjects": population_subjects,
        "sessions": {
            "population_train": args.train_session,
            "population_validation": args.validation_session,
            "final_target_test": args.final_test_session,
        },
        "data_root": str(data_root),
        "data_pattern": args.data_pattern,
        "subject_paths": {
            str(subject): str(path)
            for subject, path in all_subject_paths.items()
        },
        "backbone_checkpoint": str(checkpoint_path),
        "backbone_sha256": backbone_sha256,
        "output": str(output_path),
        "arguments": vars(args),
    }
    atomic_write_json(run_dir / "run_config.json", initial_run_config)

    # ------------------------------------------------------------------
    # Build population train and validation windows.
    # ------------------------------------------------------------------

    print("Building population training windows...")
    train_build = build_population_split(
        subjects=population_subjects,
        data_root=data_root,
        data_pattern=args.data_pattern,
        session_name=args.train_session,
        window_seconds=args.window_sec,
        stride_seconds=args.window_stride_sec,
        base_seed=args.window_seed + 1_000,
        shuffle_trials_within_class=True,
        max_windows_per_class_per_subject=(
            args.max_windows_per_class_per_subject
        ),
    )

    print("Building population validation windows...")
    val_build = build_population_split(
        subjects=population_subjects,
        data_root=data_root,
        data_pattern=args.data_pattern,
        session_name=args.validation_session,
        window_seconds=args.window_sec,
        stride_seconds=args.window_stride_sec,
        base_seed=args.window_seed + 2_000,
        shuffle_trials_within_class=False,
        max_windows_per_class_per_subject=(
            args.max_windows_per_class_per_subject
        ),
        reference_metadata=train_build.metadata,
    )

    metadata = train_build.metadata
    class_names = list(metadata.class_names)
    num_classes = len(class_names)

    validate_labels(
        train_build.bundle.window_set.labels,
        num_classes=num_classes,
        split_name="population train windows",
    )
    validate_labels(
        val_build.bundle.window_set.labels,
        num_classes=num_classes,
        split_name="population validation windows",
    )

    validate_no_source_leakage(
        train_build.bundle.window_set,
        val_build.bundle.window_set,
        left_name="population train",
        right_name="population validation",
    )

    train_window_subjects = set(
        train_build.bundle.window_subject_ids.tolist()
    )
    val_window_subjects = set(val_build.bundle.window_subject_ids.tolist())
    if target_subject in train_window_subjects:
        raise RuntimeError("Target subject leaked into population training.")
    if target_subject in val_window_subjects:
        raise RuntimeError("Target subject leaked into population validation.")
    if train_window_subjects != set(population_subjects):
        raise RuntimeError(
            "Population train window subjects do not match the requested "
            f"population subjects: {sorted(train_window_subjects)} vs "
            f"{population_subjects}."
        )
    if val_window_subjects != set(population_subjects):
        raise RuntimeError(
            "Population validation window subjects do not match the requested "
            f"population subjects: {sorted(val_window_subjects)} vs "
            f"{population_subjects}."
        )

    print("Population source and window summary:")
    print(
        "  train windows:",
        len(train_build.bundle.window_set.windows),
        class_name_counts(
            train_build.bundle.window_set.labels,
            class_names,
        ),
    )
    print(
        "  val windows:  ",
        len(val_build.bundle.window_set.windows),
        class_name_counts(
            val_build.bundle.window_set.labels,
            class_names,
        ),
    )
    print()

    # ------------------------------------------------------------------
    # Build exactly the same Stage-0.5 50M preprocessing/runtime config.
    # ------------------------------------------------------------------

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

    print("50M config:")
    print("  raw metadata sample rate:", metadata.sample_rate)
    print("  raw unit:", metadata.unit)
    print("  target shape:", (config.n_channels, config.target_num_points))
    print("  token shape:", (config.num_tokens, config.patch_num_points))
    print("  output layer idx:", config.output_layer_idx)
    print("  aggregation:", config.aggregation)
    print("  classifier input dim:", config.classifier_input_dim)
    print("  preprocessing hash:", preprocessing_hash)
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
    model_load_seconds = time.perf_counter() - load_start

    for parameter in classifier.backbone.parameters():
        parameter.requires_grad = False
    if backbone.trainable_parameters != 0:
        raise RuntimeError(
            "Backbone is not fully frozen: "
            f"trainable parameters={backbone.trainable_parameters}."
        )

    print(
        f"Backbone loaded on {classifier.device} in "
        f"{model_load_seconds:.2f}s."
    )
    print("Trainable backbone parameters:", backbone.trainable_parameters)
    print("Trainable classifier parameters:", classifier.trainable_parameters)
    print()

    # ------------------------------------------------------------------
    # Extract population features. Target data is still unopened here.
    # ------------------------------------------------------------------

    cache_dtype = feature_cache_dtype_from_name(
        args.feature_cache_dtype
    )

    train_features = extract_frozen_features(
        window_set=train_build.bundle.window_set,
        metadata=metadata,
        config=config,
        classifier=classifier,
        preprocess_batch_size=args.feature_batch_size,
        cache_dtype=cache_dtype,
        split_name="population_train",
        log_every=args.feature_log_every,
    )
    val_features = extract_frozen_features(
        window_set=val_build.bundle.window_set,
        metadata=metadata,
        config=config,
        classifier=classifier,
        preprocess_batch_size=args.feature_batch_size,
        cache_dtype=cache_dtype,
        split_name="population_validation",
        log_every=args.feature_log_every,
    )

    if args.save_feature_cache:
        save_population_feature_cache(
            dataset=train_features,
            bundle=train_build.bundle,
            path=run_dir / "features_population_train.pt",
            split_name="population_train",
            class_names=class_names,
            subject_ids=population_subjects,
            backbone_sha256=backbone_sha256,
            preprocessing_hash=preprocessing_hash,
        )
        save_population_feature_cache(
            dataset=val_features,
            bundle=val_build.bundle,
            path=run_dir / "features_population_validation.pt",
            split_name="population_validation",
            class_names=class_names,
            subject_ids=population_subjects,
            backbone_sha256=backbone_sha256,
            preprocessing_hash=preprocessing_hash,
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

    # ------------------------------------------------------------------
    # Train only the linear head.
    # ------------------------------------------------------------------

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

    training_start = time.perf_counter()

    print("Training population linear head with SGD")
    print(
        f"epochs={args.epochs}, lr={args.head_lr}, "
        f"momentum={args.momentum}, "
        f"weight_decay={args.weight_decay}, "
        f"best_metric={args.metric_for_best}"
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

        train_extended = extend_metrics(
            train_metrics,
            class_names=class_names,
        )
        val_extended = extend_metrics(
            val_metrics,
            class_names=class_names,
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
            "train_loss": train_extended.loss,
            "train_acc": train_extended.accuracy,
            "train_bacc": train_extended.balanced_accuracy,
            "train_macro_f1": train_extended.macro_f1,
            "val_loss": val_extended.loss,
            "val_acc": val_extended.accuracy,
            "val_bacc": val_extended.balanced_accuracy,
            "val_macro_f1": val_extended.macro_f1,
            "is_best": improved,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        epoch_rows.append(row)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_extended.loss:.4f} "
            f"train_acc={train_extended.accuracy:.4f} "
            f"train_bacc={train_extended.balanced_accuracy:.4f} "
            f"train_f1={train_extended.macro_f1:.4f} "
            f"val_loss={val_extended.loss:.4f} "
            f"val_acc={val_extended.accuracy:.4f} "
            f"val_bacc={val_extended.balanced_accuracy:.4f} "
            f"val_f1={val_extended.macro_f1:.4f} "
            f"{'*' if improved else ''}",
            flush=True,
        )

        if (
            args.patience > 0
            and epochs_without_improvement >= args.patience
        ):
            print(
                "Early stopping: no improvement for "
                f"{args.patience} epoch(s)."
            )
            break

    training_seconds = time.perf_counter() - training_start

    if best_head_state is None or best_val_metrics is None:
        raise RuntimeError("No best population-head state was recorded.")

    classifier.head.load_state_dict(best_head_state, strict=True)
    classifier.eval()

    with torch.no_grad():
        selected_val_metrics_raw = run_head_epoch(
            head=classifier.head,
            loader=val_loader,
            criterion=criterion,
            device=classifier.device,
            num_classes=num_classes,
            optimizer=None,
        )
    selected_val_metrics = extend_metrics(
        selected_val_metrics_raw,
        class_names=class_names,
    )

    # ------------------------------------------------------------------
    # Only now open the unseen target subject final test set.
    # ------------------------------------------------------------------

    print()
    print(
        "Population model selection is complete. "
        "Opening the unseen target-subject final test set..."
    )

    target_build = build_population_split(
        subjects=[target_subject],
        data_root=data_root,
        data_pattern=args.data_pattern,
        session_name=args.final_test_session,
        window_seconds=args.window_sec,
        stride_seconds=args.window_stride_sec,
        base_seed=args.window_seed + 3_000,
        shuffle_trials_within_class=False,
        max_windows_per_class_per_subject=(
            args.max_windows_per_class_per_subject
        ),
        reference_metadata=metadata,
    )

    target_subject_values = set(
        target_build.bundle.window_subject_ids.tolist()
    )
    if target_subject_values != {target_subject}:
        raise RuntimeError(
            "Final test windows contain unexpected subjects: "
            f"{sorted(target_subject_values)}."
        )

    validate_no_source_leakage(
        train_build.bundle.window_set,
        target_build.bundle.window_set,
        left_name="population train",
        right_name="target final test",
    )
    validate_no_source_leakage(
        val_build.bundle.window_set,
        target_build.bundle.window_set,
        left_name="population validation",
        right_name="target final test",
    )

    target_features = extract_frozen_features(
        window_set=target_build.bundle.window_set,
        metadata=metadata,
        config=config,
        classifier=classifier,
        preprocess_batch_size=args.feature_batch_size,
        cache_dtype=cache_dtype,
        split_name="target_final_test",
        log_every=args.feature_log_every,
    )

    if args.save_feature_cache:
        save_population_feature_cache(
            dataset=target_features,
            bundle=target_build.bundle,
            path=run_dir / "features_target_final_test.pt",
            split_name="target_final_test",
            class_names=class_names,
            subject_ids=[target_subject],
            backbone_sha256=backbone_sha256,
            preprocessing_hash=preprocessing_hash,
        )

    target_loader = DataLoader(
        target_features,
        batch_size=min(args.head_batch_size, len(target_features)),
        shuffle=False,
        num_workers=0,
        pin_memory=classifier.device.type == "cuda",
        drop_last=False,
    )

    with torch.no_grad():
        target_metrics_raw = run_head_epoch(
            head=classifier.head,
            loader=target_loader,
            criterion=criterion,
            device=classifier.device,
            num_classes=num_classes,
            optimizer=None,
        )
    target_metrics = extend_metrics(
        target_metrics_raw,
        class_names=class_names,
    )

    # ------------------------------------------------------------------
    # Save model and reports.
    # ------------------------------------------------------------------

    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path = save_classifier_checkpoint(
        classifier=classifier,
        checkpoint_path=output_path,
        extra_metadata={
            "task": "BNCI2014_001_motor_imagery",
            "dataset": metadata.dataset_name,
            "mode": "population_loso_linear_probe",
            "stage": "stage1",
            "target_subject": target_subject,
            "excluded_subjects": [target_subject],
            "population_training_subjects": population_subjects,
            "population_validation_subjects": population_subjects,
            "population_train_session": args.train_session,
            "population_validation_session": args.validation_session,
            "final_test_subject": target_subject,
            "final_test_session": args.final_test_session,
            "subject_data_paths": {
                str(subject): str(path)
                for subject, path in all_subject_paths.items()
            },
            "class_names": class_names,
            "backbone_sha256": backbone_sha256,
            "preprocessing_hash": preprocessing_hash,
            "freeze_backbone": True,
            "trainable_backbone_parameters": 0,
            "optimizer": "SGD",
            "head_lr": float(args.head_lr),
            "momentum": float(args.momentum),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "window_seed": int(args.window_seed),
            "metric_for_best": args.metric_for_best,
            "best_epoch": int(best_epoch),
            "best_validation": selected_val_metrics.to_dict(),
            "target_final_test": target_metrics.to_dict(),
            "window_construction": (
                "same_label_trial_concatenation_per_subject_and_session"
            ),
            "window_stride_seconds": float(args.window_stride_sec),
            "warning": (
                "Temporary Stage-1 baseline: 10-second samples are built "
                "from 4-second source trials, but never across subjects, "
                "sessions, or labels."
            ),
            "git_commit": git_commit,
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
        "stage": "stage1",
        "experiment": "population_loso_linear_probe",
        "warning": (
            "Temporary 10-second baseline: each derived window contains one "
            "subject, one session, and one class, but may cross original "
            "4-second trial boundaries."
        ),
        "files": {
            "backbone_checkpoint": str(checkpoint_path),
            "classifier_checkpoint": str(saved_path),
            "run_dir": str(run_dir),
            "epoch_metrics_csv": str(metrics_csv),
            "subject_data_paths": {
                str(subject): str(path)
                for subject, path in all_subject_paths.items()
            },
        },
        "protocol": {
            "all_subjects": subjects,
            "target_subject": target_subject,
            "population_subjects": population_subjects,
            "population_train_session": args.train_session,
            "population_validation_session": args.validation_session,
            "final_target_test_session": args.final_test_session,
            "target_subject_used_for_training": False,
            "target_subject_used_for_validation": False,
            "final_test_opened_after_model_selection": True,
        },
        "dataset": {
            "name": metadata.dataset_name,
            "sample_rate": metadata.sample_rate,
            "unit": metadata.unit,
            "channel_names": metadata.channel_names,
            "class_names": class_names,
        },
        "source_trials": {
            "population_train": train_build.source_trial_summary,
            "population_validation": val_build.source_trial_summary,
            "target_final_test": target_build.source_trial_summary,
        },
        "derived_windows": {
            "population_train_total": int(
                len(train_build.bundle.window_set.windows)
            ),
            "population_train_per_class": class_name_counts(
                train_build.bundle.window_set.labels,
                class_names,
            ),
            "population_validation_total": int(
                len(val_build.bundle.window_set.windows)
            ),
            "population_validation_per_class": class_name_counts(
                val_build.bundle.window_set.labels,
                class_names,
            ),
            "target_final_test_total": int(
                len(target_build.bundle.window_set.windows)
            ),
            "target_final_test_per_class": class_name_counts(
                target_build.bundle.window_set.labels,
                class_names,
            ),
            "window_seconds": float(args.window_sec),
            "stride_seconds": float(args.window_stride_sec),
            "construction": (
                train_build.bundle.window_set.construction
            ),
        },
        "model": {
            "device": str(classifier.device),
            "load_seconds": model_load_seconds,
            "backbone_sha256": backbone_sha256,
            "trainable_backbone_parameters": (
                backbone.trainable_parameters
            ),
            "target_sample_rate": config.target_sample_rate,
            "target_num_points": config.target_num_points,
            "num_tokens": config.num_tokens,
            "patch_num_points": config.patch_num_points,
            "output_layer_idx": config.output_layer_idx,
            "aggregation": config.aggregation,
            "classifier_input_dim": config.classifier_input_dim,
            "feature_cache_dtype": args.feature_cache_dtype,
            "preprocessing_contract": preprocessing_contract,
            "preprocessing_hash": preprocessing_hash,
        },
        "training": {
            "optimizer": "SGD",
            "epochs_requested": args.epochs,
            "epochs_completed": len(epoch_rows),
            "training_seconds": training_seconds,
            "best_epoch": best_epoch,
            "metric_for_best": args.metric_for_best,
            "head_lr": args.head_lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
            "window_seed": args.window_seed,
        },
        "best_population_validation": selected_val_metrics.to_dict(),
        "unseen_target_final_test": target_metrics.to_dict(),
        "reproducibility": {
            "git_commit": git_commit,
            "backbone_sha256": backbone_sha256,
            "preprocessing_hash": preprocessing_hash,
            "source_trial_encoding": (
                "(subject_id << 32) | file_local_trial_id"
            ),
        },
    }

    report_path = run_dir / "population_training_report.json"
    atomic_write_json(report_path, report)

    summary = {
        "status": "completed",
        "target_subject": target_subject,
        "population_subjects": population_subjects,
        "best_epoch": best_epoch,
        "population_validation": selected_val_metrics.to_dict(),
        "unseen_target_final_test": target_metrics.to_dict(),
        "classifier_checkpoint": str(saved_path),
        "report": str(report_path),
    }
    atomic_write_json(run_dir / "summary.json", summary)

    print()
    print("=" * 88)
    print("Population training completed")
    print("=" * 88)
    print("best epoch:", best_epoch)
    print(
        "population validation:",
        f"acc={selected_val_metrics.accuracy:.4f}, "
        f"bacc={selected_val_metrics.balanced_accuracy:.4f}, "
        f"macro_f1={selected_val_metrics.macro_f1:.4f}",
    )
    print(
        f"unseen target subject {target_subject}:",
        f"acc={target_metrics.accuracy:.4f}, "
        f"bacc={target_metrics.balanced_accuracy:.4f}, "
        f"macro_f1={target_metrics.macro_f1:.4f}",
    )
    print("saved classifier:", saved_path)
    print("report:", report_path)
    print()


if __name__ == "__main__":
    main()
