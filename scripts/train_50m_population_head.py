from __future__ import annotations

"""
Train a Stage-1 50M classification head with LOSO or within-subject splits.

Protocol
--------
``--split-mode loso`` preserves the Stage-1 BNCI2014_001 protocol. For one
target subject:

- Population training:
    all non-target subjects / 0train
- Population validation:
    all non-target subjects / 1test
- Final unseen-subject test:
    target subject / 1test

The target subject is never used to train or select the population model.

``--split-mode within-subject`` uses one subject's source session for a
trial-level stratified train/validation split and a distinct session only for
final testing.

Important
---------
The BNCI HDF5 files contain 4-second source trials. When the legacy
same-label-concatenation window mode is selected, the script constructs
temporary single-label windows only within the same:

    subject + session + class

It never concatenates trials across subjects, sessions, or labels. Direct-trial
mode continues to use each source trial as its own window, including 2-second
self-collected trials.

The 50M backbone is frozen. Only the configured classification head is trained.
"""

import argparse
import csv
import hashlib
import json
import random
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from _bootstrap import ROOT
from bci_dayloop.data.hdf5_dataset import EEGHDF5, HDF5Metadata
from bci_dayloop.data.splits import (
    WithinSubjectTrialSplit,
    resolve_within_subject_trial_split,
)
from bci_dayloop.models.model_50m.backbone import Model50MBackbone
from bci_dayloop.models.model_50m.classifier import (
    Model50MClassifier,
    save_classifier_checkpoint,
)
from bci_dayloop.models.model_50m.config import (
    STANDARD_64_CHANNELS,
    Model50MConfig,
)
from bci_dayloop.models.model_50m.finetuning import (
    resolve_backbone_adaptation,
    resolve_embedding_layer,
    resolve_trainable_block_indices,
    uses_frozen_feature_cache,
)
from bci_dayloop.models.model_50m.lora import (
    inject_lora_adapters,
    lora_state_dict,
    load_lora_state_dict,
    normalize_lora_target_modules,
)
from bci_dayloop.models.model_50m.preprocessing import Model50MPreprocessor
from bci_dayloop.models.model_50m.tokenization import (
    Model50MBatchedInput,
    Model50MTokenizer,
    stack_model50m_tokens,
)

# Reuse the already validated Stage-0.5 preprocessing, window construction,
# frozen-feature extraction, and classification-head training helpers.
from train_50m_linear_head import (
    EpochMetrics,
    WindowSet,
    build_same_label_concat_windows,
    class_counts,
    confusion_to_metrics,
    extract_frozen_features,
    feature_cache_dtype_from_name,
    json_default,
    limit_windows_per_class,
    metric_is_better,
    resolve_repo_path,
    run_head_epoch,
    set_seed,
    validate_labels,
    build_direct_trial_windows,
)

from bci_dayloop.utils.paths import (
    population_head_path,
    population_run_dir,
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


def tokenize_windows_for_finetuning(
    *,
    window_set: WindowSet,
    metadata: HDF5Metadata,
    config: Model50MConfig,
    preprocess_batch_size: int,
    split_name: str,
    log_every: int,
) -> TensorDataset:
    """Stage token inputs on CPU while keeping every backbone forward live.

    Preprocessing/tokenization are fixed input transformations. Unlike the
    frozen feature cache, this dataset stores no backbone embeddings, so each
    training batch still executes the selected backbone layer with autograd.
    """
    if preprocess_batch_size <= 0:
        raise ValueError("preprocess_batch_size must be positive.")

    preprocessor = Model50MPreprocessor(config)
    tokenizer = Model50MTokenizer(config)
    token_input_chunks: list[torch.Tensor] = []
    channel_index_chunks: list[torch.Tensor] = []
    time_index_chunks: list[torch.Tensor] = []
    token_mask_chunks: list[torch.Tensor] = []
    channel_mask_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    split_start = time.perf_counter()

    for batch_start in range(0, len(window_set.windows), preprocess_batch_size):
        batch_end = min(batch_start + preprocess_batch_size, len(window_set.windows))
        tokenized_samples = []
        for raw_window in window_set.windows[batch_start:batch_end]:
            processed = preprocessor(
                signal=raw_window,
                channel_names=metadata.channel_names,
                original_sample_rate=metadata.sample_rate,
                input_unit=metadata.unit,
            )
            tokenized_samples.append(tokenizer(processed))

        batch = stack_model50m_tokens(tokenized_samples)
        token_input_chunks.append(batch.token_inputs.contiguous())
        channel_index_chunks.append(batch.token_channel_indices.contiguous())
        time_index_chunks.append(batch.token_time_indices.contiguous())
        token_mask_chunks.append(batch.token_valid_mask.contiguous())
        channel_mask_chunks.append(batch.channel_valid_mask.contiguous())
        label_chunks.append(
            torch.from_numpy(window_set.labels[batch_start:batch_end].copy()).long()
        )

        batch_number = batch_start // preprocess_batch_size + 1
        if (
            batch_number == 1
            or batch_end == len(window_set.windows)
            or batch_number % log_every == 0
        ):
            print(
                f"[TokenInputs] split={split_name} batch={batch_number} "
                f"samples={batch_end}/{len(window_set.windows)}",
                flush=True,
            )

    dataset = TensorDataset(
        torch.cat(token_input_chunks, dim=0).contiguous(),
        torch.cat(channel_index_chunks, dim=0).contiguous(),
        torch.cat(time_index_chunks, dim=0).contiguous(),
        torch.cat(token_mask_chunks, dim=0).contiguous(),
        torch.cat(channel_mask_chunks, dim=0).contiguous(),
        torch.cat(label_chunks, dim=0).contiguous(),
    )
    if len(dataset) != len(window_set.windows):
        raise RuntimeError(
            f"{split_name}: token input count {len(dataset)} does not match "
            f"window count {len(window_set.windows)}."
        )
    print(
        f"[TokenInputs] completed split={split_name} samples={len(dataset)} "
        f"time={time.perf_counter() - split_start:.1f}s",
        flush=True,
    )
    return dataset


def run_finetune_epoch(
    *,
    classifier: Model50MClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    num_classes: int,
    optimizer: torch.optim.Optimizer | None,
) -> EpochMetrics:
    """Run an epoch that recomputes backbone features from token inputs."""
    is_train = optimizer is not None
    classifier.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for (
        token_inputs,
        token_channel_indices,
        token_time_indices,
        token_valid_mask,
        channel_valid_mask,
        labels,
    ) in loader:
        batch = Model50MBatchedInput(
            token_inputs=token_inputs,
            token_channel_indices=token_channel_indices,
            token_time_indices=token_time_indices,
            token_valid_mask=token_valid_mask,
            channel_valid_mask=channel_valid_mask,
        ).to(classifier.device, non_blocking=True)
        labels = labels.to(
            device=classifier.device,
            dtype=torch.long,
            non_blocking=True,
        )
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = classifier(batch)
        loss = criterion(logits, labels)
        if is_train:
            loss.backward()
            optimizer.step()

        predictions = logits.argmax(dim=-1)
        batch_size = int(labels.numel())
        total_loss += float(loss.item()) * batch_size
        total_correct += int((predictions == labels).sum().item())
        total_count += batch_size
        flat_indices = labels.detach().cpu() * num_classes + predictions.detach().cpu()
        confusion += torch.bincount(
            flat_indices,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)

    if total_count <= 0:
        raise RuntimeError("Empty token-input loader.")

    balanced_accuracy, per_class_recall = confusion_to_metrics(confusion)
    return EpochMetrics(
        loss=total_loss / total_count,
        accuracy=total_correct / total_count,
        balanced_accuracy=balanced_accuracy,
        confusion_matrix=confusion.tolist(),
        per_class_recall=per_class_recall,
    )


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
            "--data-pattern must be a valid Python format string. It may "
            "use {subject}, for example 'subject_{subject:02d}.h5'."
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
def select_direct_trial_segment(
    *,
    trials: np.ndarray,
    sample_rate: float,
    window_seconds: float,
    anchor: str,
    split_name: str,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    """
    从每个源 trial 中显式选择一个连续窗口。

    不补零、不拼接、不跨 trial；每个输出窗口仍只对应一个源 trial。
    """
    trials = np.asarray(trials, dtype=np.float32)

    if trials.ndim != 3:
        raise ValueError(
            f"{split_name}: trials must have shape [N,C,T], "
            f"got {trials.shape}."
        )
    if sample_rate <= 0:
        raise ValueError(
            f"{split_name}: sample_rate must be positive, "
            f"got {sample_rate}."
        )
    if window_seconds <= 0:
        raise ValueError(
            f"{split_name}: window_seconds must be positive, "
            f"got {window_seconds}."
        )
    if anchor not in {"start", "center", "end"}:
        raise ValueError(
            f"{split_name}: unsupported direct-trial anchor "
            f"{anchor!r}; expected start, center, or end."
        )

    source_samples = int(trials.shape[-1])
    target_samples = int(round(window_seconds * sample_rate))

    if target_samples <= 0:
        raise ValueError(
            f"{split_name}: target window has no samples."
        )

    if target_samples > source_samples:
        raise ValueError(
            f"{split_name}: source trials are only "
            f"{source_samples / sample_rate:.3f}s, but "
            f"--window-sec={window_seconds:.3f}s requires "
            f"{target_samples} samples. Direct-trial mode does not "
            "pad, concatenate, or cross source-trial boundaries."
        )

    if anchor == "start":
        start_sample = 0
    elif anchor == "center":
        start_sample = (source_samples - target_samples) // 2
    else:  # anchor == "end"
        start_sample = source_samples - target_samples

    end_sample = start_sample + target_samples

    return (
        trials[..., start_sample:end_sample],
        {
            "policy": "one_contiguous_window_per_source_trial",
            "anchor": anchor,
            "source_samples": source_samples,
            "source_seconds": source_samples / sample_rate,
            "selected_start_sample": start_sample,
            "selected_end_sample_exclusive": end_sample,
            "selected_start_seconds": start_sample / sample_rate,
            "selected_end_seconds": end_sample / sample_rate,
            "selected_samples": target_samples,
            "selected_seconds": target_samples / sample_rate,
        },
    )

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
    window_construction: str,
    direct_trial_anchor: str,
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

    session_data = dataset.load(session_name)
    return build_window_bundle_from_session_data(
        subject_id=subject_id,
        path=path,
        session_name=session_name,
        metadata=metadata,
        session_data=session_data,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        seed=seed,
        shuffle_trials_within_class=shuffle_trials_within_class,
        max_windows_per_class=max_windows_per_class,
        window_construction=window_construction,
        direct_trial_anchor=direct_trial_anchor,
    )


def build_window_bundle_from_session_data(
    *,
    subject_id: int,
    path: Path,
    session_name: str,
    metadata: HDF5Metadata,
    session_data: Mapping[str, np.ndarray],
    class_names: Sequence[str] | None = None,
    window_seconds: float,
    stride_seconds: float,
    seed: int,
    shuffle_trials_within_class: bool,
    max_windows_per_class: int | None,
    window_construction: str,
    direct_trial_anchor: str,
) -> tuple[WindowBundle, HDF5Metadata, dict[str, Any]]:
    """Build windows from an already selected source-trial subset."""
    effective_class_names = list(class_names or metadata.class_names)
    num_classes = len(effective_class_names)
    validate_loaded_session(
        session_data,
        expected_subject=subject_id,
        expected_session=session_name,
        num_classes=num_classes,
        path=path,
    )

    raw_trial_ids = np.asarray(session_data["trial_ids"], dtype=np.int64)
    global_trial_ids = encode_trial_ids(subject_id, raw_trial_ids)

    trials = np.asarray(
        session_data["data"],
        dtype=np.float32,
    )
    labels = np.asarray(
        session_data["labels"],
        dtype=np.int64,
    )

    if window_construction == "direct_trial":
        direct_trials, direct_trial_selection = (
            select_direct_trial_segment(
                trials=trials,
                sample_rate=metadata.sample_rate,
                window_seconds=window_seconds,
                anchor=direct_trial_anchor,
                split_name=(
                    f"subject_{subject_id:02d}/{session_name}"
                ),
            )
        )

        window_set = build_direct_trial_windows(
            trials=direct_trials,
            labels=labels,
            trial_ids=global_trial_ids,
            sample_rate=metadata.sample_rate,
            window_seconds=window_seconds,
            num_classes=num_classes,
            seed=seed,
            split_name=(
                f"subject_{subject_id:02d}/{session_name}"
            ),
        )

    elif window_construction == "same_label_concat":
        direct_trial_selection = None
        window_set = build_same_label_concat_windows(
            trials=trials,
            labels=labels,
            trial_ids=global_trial_ids,
            sample_rate=metadata.sample_rate,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            num_classes=num_classes,
            seed=seed,
            shuffle_trials_within_class=(
                shuffle_trials_within_class
            ),
            split_name=(
                f"subject_{subject_id:02d}/{session_name}"
            ),
        )

    else:
        raise ValueError(
            f"Unsupported window_construction="
            f"{window_construction!r}."
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
            effective_class_names,
        ),
        "direct_trial_selection": direct_trial_selection,
        "derived_windows_total": int(len(window_set.windows)),
        "derived_windows_per_class": class_name_counts(
            window_set.labels,
            effective_class_names,
        ),
        "unique_source_trials_used": int(len(source_id_set(window_set))),
        "window_seconds": float(window_seconds),
        "stride_seconds": float(stride_seconds),
        "construction": window_set.construction,
    }
    return bundle, metadata, summary


def select_trial_rows(
    trial_data: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return a trial-level subset without changing the underlying HDF5 data."""
    indices = np.asarray(indices, dtype=np.int64)
    return {
        key: np.asarray(values)[indices]
        for key, values in trial_data.items()
    }


def select_trial_ids(
    trial_data: Mapping[str, np.ndarray],
    trial_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Select source trials by ID and fail rather than silently dropping one."""
    requested = np.asarray(trial_ids, dtype=np.int64)
    available = np.asarray(trial_data["trial_ids"], dtype=np.int64)
    mask = np.isin(available, requested)
    selected = select_trial_rows(trial_data, np.flatnonzero(mask))
    selected_ids = np.asarray(selected["trial_ids"], dtype=np.int64)
    if set(selected_ids.tolist()) != set(requested.tolist()):
        missing = sorted(set(requested.tolist()) - set(selected_ids.tolist()))
        raise RuntimeError(
            "Selected source trials are missing from the loaded session: "
            f"{missing[:10]}."
        )
    return selected


def resolve_class_names(
    *,
    metadata: HDF5Metadata,
    explicit_class_names: Sequence[str] | None,
) -> list[str]:
    """Resolve logit semantics without assuming every 4-class task is BNCI."""
    num_classes = len(metadata.class_names)
    if num_classes <= 0:
        raise ValueError("HDF5 metadata class_names must not be empty.")

    if explicit_class_names is not None:
        class_names = [str(name).strip() for name in explicit_class_names]
        source = "--class-names"
    else:
        metadata_names = [str(name).strip() for name in metadata.class_names]
        if len(metadata_names) == num_classes and all(metadata_names):
            class_names = metadata_names
            source = "HDF5 metadata"
        else:
            class_names = [f"class_{index}" for index in range(num_classes)]
            source = "numeric fallback"

    if len(class_names) != num_classes:
        raise ValueError(
            "class_names length must match the HDF5 class count: "
            f"{len(class_names)} != {num_classes}."
        )
    if not all(class_names):
        raise ValueError("class_names must not contain empty values.")
    if len(set(class_names)) != len(class_names):
        raise ValueError(f"class_names must be unique, got {class_names}.")
    print(f"Class semantics source: {source}")
    print("Class semantics:", {index: name for index, name in enumerate(class_names)})
    return class_names


def build_within_subject_splits(
    *,
    subject_id: int,
    path: Path,
    train_session: str,
    test_session: str,
    validation_ratio: float,
    seed: int,
    window_seconds: float,
    stride_seconds: float,
    max_windows_per_class: int | None,
    window_construction: str,
    direct_trial_anchor: str,
    explicit_class_names: Sequence[str] | None,
) -> tuple[
    SplitBuildResult,
    SplitBuildResult,
    HDF5Metadata,
    list[str],
    WithinSubjectTrialSplit,
    dict[str, np.ndarray],
]:
    """Resolve a held-out cross-session split before any window construction."""
    dataset = EEGHDF5(path)
    metadata = dataset.metadata
    class_names = resolve_class_names(
        metadata=metadata,
        explicit_class_names=explicit_class_names,
    )
    all_trial_metadata = dataset.trial_metadata()
    split = resolve_within_subject_trial_split(
        subject_ids=all_trial_metadata["subject_ids"],
        session_ids=all_trial_metadata["session_ids"],
        labels=all_trial_metadata["labels"],
        subject_id=subject_id,
        train_session=train_session,
        test_session=test_session,
        validation_ratio=validation_ratio,
        seed=seed,
        num_classes=len(class_names),
    )

    train_session_data = dataset.load(train_session)
    train_data = select_trial_ids(
        train_session_data,
        all_trial_metadata["trial_ids"][split.train_indices],
    )
    validation_data = select_trial_ids(
        train_session_data,
        all_trial_metadata["trial_ids"][split.validation_indices],
    )
    train_bundle, _, train_summary = build_window_bundle_from_session_data(
        subject_id=subject_id,
        path=path,
        session_name=train_session,
        metadata=metadata,
        session_data=train_data,
        class_names=class_names,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        seed=seed + 1_000,
        shuffle_trials_within_class=True,
        max_windows_per_class=max_windows_per_class,
        window_construction=window_construction,
        direct_trial_anchor=direct_trial_anchor,
    )
    validation_bundle, _, validation_summary = build_window_bundle_from_session_data(
        subject_id=subject_id,
        path=path,
        session_name=train_session,
        metadata=metadata,
        session_data=validation_data,
        class_names=class_names,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        seed=seed + 2_000,
        shuffle_trials_within_class=False,
        max_windows_per_class=max_windows_per_class,
        window_construction=window_construction,
        direct_trial_anchor=direct_trial_anchor,
    )

    common_paths = {subject_id: path}
    train_summary["selected_trial_ids"] = train_data["trial_ids"].tolist()
    validation_summary["selected_trial_ids"] = validation_data["trial_ids"].tolist()
    return (
        SplitBuildResult(
            bundle=train_bundle,
            source_trial_summary={f"subject_{subject_id:02d}": train_summary},
            subject_paths=common_paths,
            metadata=metadata,
        ),
        SplitBuildResult(
            bundle=validation_bundle,
            source_trial_summary={f"subject_{subject_id:02d}": validation_summary},
            subject_paths=common_paths,
            metadata=metadata,
        ),
        metadata,
        class_names,
        split,
        all_trial_metadata,
    )


def build_within_subject_test_split(
    *,
    subject_id: int,
    path: Path,
    metadata: HDF5Metadata,
    class_names: Sequence[str],
    split: WithinSubjectTrialSplit,
    all_trial_metadata: Mapping[str, np.ndarray],
    window_seconds: float,
    stride_seconds: float,
    max_windows_per_class: int | None,
    window_construction: str,
    direct_trial_anchor: str,
) -> SplitBuildResult:
    """Build the held-out test windows only after model selection completes."""
    dataset = EEGHDF5(path)
    test_session_data = dataset.load(split.test_session)
    test_data = select_trial_ids(
        test_session_data,
        np.asarray(all_trial_metadata["trial_ids"], dtype=np.int64)[
            split.test_indices
        ],
    )
    test_bundle, _, test_summary = build_window_bundle_from_session_data(
        subject_id=subject_id,
        path=path,
        session_name=split.test_session,
        metadata=metadata,
        session_data=test_data,
        class_names=class_names,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        seed=0,
        shuffle_trials_within_class=False,
        max_windows_per_class=max_windows_per_class,
        window_construction=window_construction,
        direct_trial_anchor=direct_trial_anchor,
    )
    test_summary["selected_trial_ids"] = test_data["trial_ids"].tolist()
    return SplitBuildResult(
        bundle=test_bundle,
        source_trial_summary={f"subject_{subject_id:02d}": test_summary},
        subject_paths={subject_id: path},
        metadata=metadata,
    )


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
    window_construction: str,
    direct_trial_anchor: str,
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
            window_construction=window_construction,
            direct_trial_anchor=direct_trial_anchor,
        )
        if common_metadata is None:
            common_metadata = metadata

        bundles.append(bundle)
        summaries[f"subject_{subject_id:02d}"] = summary

    assert common_metadata is not None
    combined = combine_window_bundles(
        bundles,
        seed=base_seed + 99_999,
        construction=window_construction,
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
            "Train a frozen or adapted 50M classification head with either "
            "LOSO population or within-subject HDF5 splits."
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
            "Subject filename pattern. It may use {subject}; a static filename "
            "is supported for a single-subject HDF5. "
            "Default: subject_{subject:02d}.h5."
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
    parser.add_argument(
        "--split-mode",
        choices=("loso", "within-subject"),
        default="loso",
        help=(
            "Data split protocol. 'loso' preserves the existing population "
            "training behavior; 'within-subject' uses --target-subject as "
            "the selected subject."
        ),
    )

    parser.add_argument("--train-session", default="0train")
    parser.add_argument("--validation-session", default="1test")
    parser.add_argument("--final-test-session", default="1test")
    parser.add_argument(
        "--test-session",
        default=None,
        help=(
            "Held-out final-test session for --split-mode within-subject. "
            "Must differ from --train-session."
        ),
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help=(
            "Trial-level validation fraction of --train-session for "
            "--split-mode within-subject (default: 0.2)."
        ),
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help=(
            "Optional logit semantics in label order, overriding HDF5 "
            "metadata; for example: left_hand right_hand both_hand rest."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/backbones/"
            "50m/model_deploy.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output classifier checkpoint. "
            "When omitted, a standard Stage-1 path "
            "is generated automatically."
        ),
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Optional run-directory override. "
            "When omitted, the standard Stage-1 path is generated: "
            "runs/stage1/<dataset>/<subject>/population/"
            "<contract>/<timestamp>/."
        ),
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "mps", "auto"),
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--window-stride-sec",
        type=float,
        default=4.0,
        help=(
            "Only used by same_label_concat mode. "
            "Direct-trial mode uses one source trial per sample."
        ),
    )

    parser.add_argument(
        "--window-construction",
        choices=("direct_trial", "same_label_concat"),
        default="direct_trial",
    )
    parser.add_argument(
        "--direct-trial-anchor",
        choices=("start", "center", "end"),
        default="end",
        help=(
            "When --window-construction=direct_trial and a source trial "
            "is longer than --window-sec, select one contiguous segment "
            "from the start, center, or end. Default: end."
        ),
    )

    parser.add_argument(
        "--model-n-time-patches",
        type=int,
        default=10,
        help=(
            "Number of time embeddings in the pretrained backbone. "
            "Keep 10 when using the 10-second pretrained checkpoint."
        ),
    )
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
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=1e-4,
        help=(
            "Learning rate for encoder blocks selected by "
            "--unfreeze-last-n-blocks."
        ),
    )
    parser.add_argument(
        "--backbone-adaptation",
        choices=("frozen", "partial", "lora"),
        default=None,
        help=(
            "Backbone adaptation regime. When omitted, legacy "
            "--unfreeze-last-n-blocks behavior is preserved."
        ),
    )
    parser.add_argument(
        "--unfreeze-last-n-blocks",
        type=int,
        default=0,
        help=(
            "Number of 1-based encoder blocks ending at the selected "
            "embedding layer to train. 0 keeps the frozen baseline."
        ),
    )
    parser.add_argument(
        "--embedding-layer",
        default="auto",
        help=(
            "Embedding layer used by the downstream head: 'auto' resolves "
            "the existing --output-layer-idx, or supply a 1-based block "
            "number such as 9."
        ),
    )
    parser.add_argument(
        "--lora-last-n-blocks",
        type=int,
        default=2,
        help="Effective blocks ending at the embedding layer for LoRA.",
    )
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=("q", "v"),
        help=(
            "Fused attention projection slices for LoRA. The 50M backbone "
            "supports q, k, and v; default: q v."
        ),
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-lr", type=float, default=5e-4)
    parser.add_argument(
        "--head-type",
        choices=("linear", "mlp"),
        default="linear",
        help="Classification head architecture. Default keeps the existing Linear Probe.",
    )
    parser.add_argument(
        "--head-hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension for --head-type mlp.",
    )
    parser.add_argument(
        "--head-dropout",
        type=float,
        default=0.0,
        help="Dropout probability for --head-type mlp.",
    )
    parser.add_argument(
        "--head-norm",
        choices=("none", "layernorm", "batchnorm"),
        default="none",
        help="Input normalization for --head-type mlp.",
    )
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

    if args.window_sec <= 0:
        raise ValueError("--window-sec must be positive.")

    if args.model_n_time_patches <= 0:
        raise ValueError(
            "--model-n-time-patches must be positive."
        )
    if args.window_stride_sec <= 0:
        raise ValueError("--window-stride-sec must be positive.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.head_batch_size <= 0 or args.feature_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if args.head_hidden_dim <= 0:
        raise ValueError("--head-hidden-dim must be positive.")
    if not 0.0 <= args.head_dropout < 1.0:
        raise ValueError("--head-dropout must be in [0, 1).")
    if args.head_type == "linear" and args.head_norm != "none":
        raise ValueError("--head-norm is only supported with --head-type mlp.")
    if args.head_type == "linear" and args.head_dropout != 0.0:
        raise ValueError("--head-dropout is only supported with --head-type mlp.")
    if args.backbone_lr <= 0:
        raise ValueError("--backbone-lr must be positive.")
    if args.unfreeze_last_n_blocks < 0:
        raise ValueError("--unfreeze-last-n-blocks must be >= 0.")
    if args.lora_last_n_blocks < 1:
        raise ValueError("--lora-last-n-blocks must be >= 1.")
    if args.lora_rank <= 0:
        raise ValueError("--lora-rank must be positive.")
    if args.lora_alpha <= 0:
        raise ValueError("--lora-alpha must be positive.")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("--lora-dropout must be in [0, 1).")
    if args.lora_lr <= 0:
        raise ValueError("--lora-lr must be positive.")
    normalize_lora_target_modules(args.lora_target_modules)
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

    target_subject = int(args.target_subject)
    if args.split_mode == "loso":
        subjects = normalize_subjects(args.subjects)
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
    else:
        if target_subject <= 0:
            raise ValueError("--target-subject must be positive.")
        if args.test_session is None:
            raise ValueError(
                "--test-session is required for --split-mode within-subject."
            )
        if not 0.0 < args.validation_ratio < 1.0:
            raise ValueError("--validation-ratio must be in (0,1).")
        subjects = [target_subject]
        population_subjects: list[int] = []
    train_validation_subjects = (
        population_subjects
        if args.split_mode == "loso"
        else [target_subject]
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
        output_path = population_head_path(
            stage="stage1",
            dataset=(
                "bnci2014_001"
                if args.split_mode == "loso"
                else "within_subject"
            ),
            subject_id=args.target_subject,
            window_seconds=args.window_sec,
            aggregation=args.aggregation,
        )
    else:
        output_path = resolve_repo_path(
            args.output
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.run_dir is None:
        run_dir = population_run_dir(
            stage="stage1",
            dataset=(
                "bnci2014_001"
                if args.split_mode == "loso"
                else "within_subject"
            ),
            subject_id=target_subject,
            window_seconds=args.window_sec,
            aggregation=args.aggregation,
            run_id=timestamp,
        )
    else:
        run_dir = resolve_repo_path(
            args.run_dir
        )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    git_commit = current_git_commit()
    backbone_sha256 = sha256_file(checkpoint_path)

    print("=" * 88)
    print(
        "Stage 1: 50M LOSO population-head training"
        if args.split_mode == "loso"
        else "Stage 1: 50M within-subject classification-head training"
    )
    print("=" * 88)
    print("split mode:", args.split_mode)
    print("target subject:", target_subject)
    if args.split_mode == "loso":
        print("population subjects:", population_subjects)
        print("population train session:", args.train_session)
        print("population validation session:", args.validation_session)
        print("final unseen-subject test:", f"{target_tag}/{args.final_test_session}")
    else:
        print("within-subject train source session:", args.train_session)
        print("within-subject final test session:", args.test_session)
        print("within-subject validation ratio:", args.validation_ratio)
        print("within-subject split seed:", args.seed)
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
    if args.split_mode == "loso":
        print(
            "IMPORTANT: the target subject is not loaded for final evaluation "
            "until population training and model selection are complete."
        )
    else:
        print(
            "IMPORTANT: train/validation are split at source-trial level; "
            "the held-out test session is only windowed after model selection."
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
        "split_mode": args.split_mode,
        "sessions": (
            {
                "population_train": args.train_session,
                "population_validation": args.validation_session,
                "final_target_test": args.final_test_session,
            }
            if args.split_mode == "loso"
            else {
                "within_subject_train": args.train_session,
                "within_subject_test": args.test_session,
                "validation_ratio": args.validation_ratio,
            }
        ),
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
    # Resolve train/validation source trials before any window construction.
    # ------------------------------------------------------------------

    within_subject_split: WithinSubjectTrialSplit | None = None
    within_subject_all_trials: dict[str, np.ndarray] | None = None
    if args.split_mode == "loso":
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
            window_construction=args.window_construction,
            direct_trial_anchor=args.direct_trial_anchor,
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
            window_construction=args.window_construction,
            direct_trial_anchor=args.direct_trial_anchor,
        )
        metadata = train_build.metadata
        class_names = resolve_class_names(
            metadata=metadata,
            explicit_class_names=args.class_names,
        )
    else:
        print("Building within-subject training and validation windows...")
        train_build, val_build, metadata, class_names, within_subject_split, within_subject_all_trials = (
            build_within_subject_splits(
                subject_id=target_subject,
                path=all_subject_paths[target_subject],
                train_session=args.train_session,
                test_session=str(args.test_session),
                validation_ratio=args.validation_ratio,
                seed=args.seed,
                window_seconds=args.window_sec,
                stride_seconds=args.window_stride_sec,
                max_windows_per_class=args.max_windows_per_class_per_subject,
                window_construction=args.window_construction,
                direct_trial_anchor=args.direct_trial_anchor,
                explicit_class_names=args.class_names,
            )
        )
        print(f"Available sessions for subject {target_subject}:")
        for session_name in within_subject_split.available_sessions:
            print(f"- {session_name}")

    num_classes = len(class_names)
    label_mapping = {
        str(index): str(name)
        for index, name in enumerate(class_names)
    }
    train_split_name = (
        "population train" if args.split_mode == "loso" else "within-subject train"
    )
    validation_split_name = (
        "population validation"
        if args.split_mode == "loso"
        else "within-subject validation"
    )
    validate_labels(
        train_build.bundle.window_set.labels,
        num_classes=num_classes,
        split_name=f"{train_split_name} windows",
    )
    validate_labels(
        val_build.bundle.window_set.labels,
        num_classes=num_classes,
        split_name=f"{validation_split_name} windows",
    )
    validate_no_source_leakage(
        train_build.bundle.window_set,
        val_build.bundle.window_set,
        left_name=train_split_name,
        right_name=validation_split_name,
    )

    train_window_subjects = set(train_build.bundle.window_subject_ids.tolist())
    val_window_subjects = set(val_build.bundle.window_subject_ids.tolist())
    if args.split_mode == "loso":
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
    elif train_window_subjects != {target_subject} or val_window_subjects != {target_subject}:
        raise RuntimeError("Within-subject train/validation windows contain another subject.")

    print(f"{train_split_name.title()} source and window summary:")
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
    if within_subject_split is not None:
        assert within_subject_all_trials is not None
        test_labels = within_subject_all_trials["labels"][
            within_subject_split.test_indices
        ]
        print(
            "  held-out test source trials:",
            len(within_subject_split.test_indices),
            class_name_counts(test_labels, class_names),
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
        model_n_time_patches=args.model_n_time_patches,
        head_type=args.head_type,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        head_norm=args.head_norm,
    )

    embedding_layer = resolve_embedding_layer(
        requested=str(args.embedding_layer),
        output_layer_idx=config.output_layer_idx,
        depth=config.depth,
    )
    # A manual CLI layer is user-facing 1-based; Model50MConfig and the
    # backbone remain 0-based internally.
    if int(config.output_layer_idx) != embedding_layer - 1:
        config = replace(
            config,
            output_layer_idx=embedding_layer - 1,
        )
    backbone_adaptation = resolve_backbone_adaptation(
        requested=args.backbone_adaptation,
        unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
    )
    partial_finetuning_enabled = backbone_adaptation == "partial"
    lora_enabled = backbone_adaptation == "lora"
    requires_live_backbone_forward = (
        partial_finetuning_enabled or lora_enabled
    )
    if partial_finetuning_enabled:
        trainable_block_indices = resolve_trainable_block_indices(
            embedding_layer=embedding_layer,
            unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
        )
    elif lora_enabled:
        trainable_block_indices = resolve_trainable_block_indices(
            embedding_layer=embedding_layer,
            unfreeze_last_n_blocks=args.lora_last_n_blocks,
        )
    else:
        trainable_block_indices = ()
    feature_cache_enabled = uses_frozen_feature_cache(
        unfreeze_last_n_blocks=(
            0 if backbone_adaptation == "frozen" else 1
        ),
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
        "model_n_time_patches": int(
            config.model_n_time_patches
        ),
        "window_construction": args.window_construction,
        "direct_trial_selection": (
            {
                "policy": "one_contiguous_window_per_source_trial",
                "anchor": args.direct_trial_anchor,
            }
            if args.window_construction == "direct_trial"
            else None
        ),
    }
    preprocessing_hash = stable_json_hash(preprocessing_contract)

    print("50M config:")
    print("  raw metadata sample rate:", metadata.sample_rate)
    print("  raw unit:", metadata.unit)
    print("  target shape:", (config.n_channels, config.target_num_points))
    print("  token shape:", (config.num_tokens, config.patch_num_points))
    print("  output layer idx:", config.output_layer_idx)
    print("  embedding layer requested:", args.embedding_layer)
    print("  embedding layer resolved (1-based):", embedding_layer)
    print("  embedding layer internal index:", config.output_layer_idx)
    print("  backbone adaptation:", backbone_adaptation)
    print("  unfreeze last N blocks:", args.unfreeze_last_n_blocks)
    if lora_enabled:
        print("  LoRA last N effective blocks:", args.lora_last_n_blocks)
        print("  LoRA target modules:", list(args.lora_target_modules))
        print("  LoRA rank:", args.lora_rank)
        print("  LoRA alpha:", args.lora_alpha)
        print("  LoRA scale:", args.lora_alpha / args.lora_rank)
        print("  LoRA dropout:", args.lora_dropout)
    print(
        "  trainable backbone blocks (1-based):",
        [index + 1 for index in trainable_block_indices],
    )
    print(
        "  frozen backbone blocks (1-based):",
        [
            index + 1
            for index in range(config.depth)
            if index not in trainable_block_indices
        ],
    )
    print("  aggregation:", config.aggregation)
    print("  classifier input dim:", config.classifier_input_dim)
    print("  backbone frozen:", not partial_finetuning_enabled)
    print("  head type:", config.head_type)
    if config.head_type == "mlp":
        print("  head hidden dim:", config.head_hidden_dim)
        print("  head norm:", config.head_norm)
        print("  head dropout:", config.head_dropout)
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
    # Model50MClassifier initializes as a frozen probe. Configure the
    # explicit adaptation scope only after constructing the classifier.
    lora_report = None
    if requires_live_backbone_forward:
        backbone.set_trainable_encoder_blocks(trainable_block_indices)
    elif lora_enabled:
        lora_report = inject_lora_adapters(
            backbone,
            block_indices=trainable_block_indices,
            target_modules=args.lora_target_modules,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
    else:
        backbone.set_trainable_encoder_blocks(())
    classifier.eval()
    model_load_seconds = time.perf_counter() - load_start

    selected_block_parameter_ids = {
        id(parameter)
        for block_index in trainable_block_indices
        for parameter in backbone.model.encoder.encoder.layers[block_index].parameters()
    }
    lora_parameters = (
        [
            parameter
            for adapter in lora_report.adapter_modules
            for parameter in adapter.parameters()
            if parameter.requires_grad
        ]
        if lora_report is not None
        else []
    )
    lora_parameter_ids = {id(parameter) for parameter in lora_parameters}
    if lora_enabled:
        # Keep this guard local to the training entrypoint as well as the
        # injector. Some PyTorch parametrization versions expose additional
        # original parameters after registration; they must never enter the
        # LoRA optimizer group.
        for parameter in backbone.parameters():
            if id(parameter) not in lora_parameter_ids:
                parameter.requires_grad = False
        trainable_backbone_parameters: list[torch.nn.Parameter] = []
    else:
        trainable_backbone_parameters = [
            parameter
            for parameter in backbone.parameters()
            if parameter.requires_grad and id(parameter) not in lora_parameter_ids
        ]
    original_backbone_parameter_count = sum(
        parameter.numel()
        for parameter in backbone.parameters()
        if id(parameter) not in lora_parameter_ids
    )
    trainable_lora_parameter_count = sum(
        parameter.numel() for parameter in lora_parameters
    )
    trainable_head_parameter_count = sum(
        parameter.numel() for parameter in classifier.head.parameters()
    )
    trainable_original_backbone_parameter_count = sum(
        parameter.numel() for parameter in trainable_backbone_parameters
    )
    total_model_parameter_count = sum(
        parameter.numel() for parameter in classifier.parameters()
    )
    total_trainable_parameter_count = (
        trainable_original_backbone_parameter_count
        + trainable_lora_parameter_count
        + trainable_head_parameter_count
    )
    if partial_finetuning_enabled and (
        {id(parameter) for parameter in trainable_backbone_parameters} != (
        selected_block_parameter_ids
        )
    ):
        raise RuntimeError(
            "Backbone trainable parameters do not exactly match the selected "
            "encoder blocks."
        )
    if partial_finetuning_enabled and (
        tuple(backbone.trainable_encoder_block_indices) != trainable_block_indices
    ):
        raise RuntimeError(
            "Backbone did not retain the requested trainable encoder blocks."
        )

    print(
        f"Backbone loaded on {classifier.device} in "
        f"{model_load_seconds:.2f}s."
    )
    print("Feature cache enabled:", feature_cache_enabled)
    if requires_live_backbone_forward:
        print(
            "Feature cache disabled because backbone fine-tuning is enabled."
        )
    if lora_enabled:
        print("Feature cache disabled because LoRA backbone adaptation is enabled.")
    print("Trainable backbone parameters:", backbone.trainable_parameters)
    print("Trainable classifier parameters:", classifier.trainable_parameters)
    print("Head trainable:", all(parameter.requires_grad for parameter in classifier.head.parameters()))
    print("Head LR:", args.head_lr)
    print("Backbone LR:", args.backbone_lr if partial_finetuning_enabled else None)
    print("LoRA LR:", args.lora_lr if lora_enabled else None)
    print("Total model parameters:", total_model_parameter_count)
    print("Original backbone parameters:", original_backbone_parameter_count)
    print("Original backbone trainable params:", trainable_original_backbone_parameter_count)
    print("LoRA trainable params:", trainable_lora_parameter_count)
    print("Head trainable params:", trainable_head_parameter_count)
    print("Total trainable params:", total_trainable_parameter_count)
    print(
        "Trainable percentage:",
        100.0 * total_trainable_parameter_count / total_model_parameter_count,
    )
    print("Weight decay:", args.weight_decay)
    print()

    # ------------------------------------------------------------------
    # Extract population features. Target data is still unopened here.
    # ------------------------------------------------------------------

    cache_dtype = feature_cache_dtype_from_name(
        args.feature_cache_dtype
    )

    if requires_live_backbone_forward:
        train_features = tokenize_windows_for_finetuning(
            window_set=train_build.bundle.window_set,
            metadata=metadata,
            config=config,
            preprocess_batch_size=args.feature_batch_size,
            split_name="population_train",
            log_every=args.feature_log_every,
        )
        val_features = tokenize_windows_for_finetuning(
            window_set=val_build.bundle.window_set,
            metadata=metadata,
            config=config,
            preprocess_batch_size=args.feature_batch_size,
            split_name="population_validation",
            log_every=args.feature_log_every,
        )
    else:
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

    if args.save_feature_cache and feature_cache_enabled:
        save_population_feature_cache(
            dataset=train_features,
            bundle=train_build.bundle,
            path=run_dir / "features_population_train.pt",
            split_name="population_train",
            class_names=class_names,
            subject_ids=train_validation_subjects,
            backbone_sha256=backbone_sha256,
            preprocessing_hash=preprocessing_hash,
        )

    if args.save_feature_cache and not feature_cache_enabled:
        print(
            "Skipping --save-feature-cache: backbone adaptation uses "
            "token inputs and cannot use detached backbone features."
        )
    if args.save_feature_cache and feature_cache_enabled:
        save_population_feature_cache(
            dataset=val_features,
            bundle=val_build.bundle,
            path=run_dir / "features_population_validation.pt",
            split_name="population_validation",
            class_names=class_names,
            subject_ids=train_validation_subjects,
            backbone_sha256=backbone_sha256,
            preprocessing_hash=preprocessing_hash,
        )

    head_train_batch_size = min(args.head_batch_size, len(train_features))
    if (
        config.head_type == "mlp"
        and config.head_norm == "batchnorm"
        and (
            head_train_batch_size < 2
            or len(train_features) % head_train_batch_size == 1
        )
    ):
        raise ValueError(
            "MLP BatchNorm requires every training batch to contain at least "
            "two samples. Adjust --head-batch-size or the training split."
        )

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_features,
        batch_size=head_train_batch_size,
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
    # Train the classification head, plus explicitly selected backbone blocks.
    # ------------------------------------------------------------------

    head_parameters = list(classifier.head.parameters())
    optimizer_groups: list[dict[str, Any]] = [
        {"params": head_parameters, "lr": args.head_lr},
    ]
    if trainable_backbone_parameters:
        optimizer_groups.append(
            {
                "params": trainable_backbone_parameters,
                "lr": args.backbone_lr,
            }
        )
    if lora_parameters:
        optimizer_groups.append(
            {
                "params": lora_parameters,
                "lr": args.lora_lr,
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=args.weight_decay,
    )
    backbone_parameter_ids = {id(parameter) for parameter in backbone.parameters()}
    head_parameter_ids = {id(parameter) for parameter in head_parameters}
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected_optimizer_parameter_ids = (
        head_parameter_ids
        | {id(parameter) for parameter in trainable_backbone_parameters}
        | lora_parameter_ids
    )
    if not head_parameter_ids or optimizer_parameter_ids != expected_optimizer_parameter_ids:
        raise RuntimeError(
            "Optimizer parameters must be exactly the classification head "
            "and selected trainable backbone blocks."
        )
    if not partial_finetuning_enabled and not lora_enabled and (
        backbone_parameter_ids & optimizer_parameter_ids
    ):
        raise RuntimeError("Frozen backbone parameters were added to the optimizer.")
    if partial_finetuning_enabled and (
        backbone_parameter_ids & optimizer_parameter_ids
        != {id(parameter) for parameter in trainable_backbone_parameters}
    ):
        raise RuntimeError("Optimizer backbone scope differs from selected blocks.")
    if lora_enabled:
        missing_lora_parameter_ids = (
            lora_parameter_ids - optimizer_parameter_ids
        )
        unexpected_original_backbone_parameter_ids = (
            (backbone_parameter_ids - lora_parameter_ids)
            & optimizer_parameter_ids
        )
        if (
            missing_lora_parameter_ids
            or unexpected_original_backbone_parameter_ids
        ):
            raise RuntimeError(
                "Optimizer LoRA scope differs from injected adapters: "
                f"missing_lora={len(missing_lora_parameter_ids)}, "
                "unexpected_original_backbone="
                f"{len(unexpected_original_backbone_parameter_ids)}."
            )
    if any(not parameter.requires_grad for parameter in head_parameters):
        raise RuntimeError("Classification head parameters must require gradients.")
    criterion = nn.CrossEntropyLoss()

    best_value = (
        float("inf")
        if args.metric_for_best == "val_loss"
        else -float("inf")
    )
    best_epoch = -1
    best_head_state: dict[str, torch.Tensor] | None = None
    best_backbone_state: dict[str, torch.Tensor] | None = None
    best_lora_state: dict[str, torch.Tensor] | None = None
    best_val_metrics: EpochMetrics | None = None
    epochs_without_improvement = 0
    epoch_rows: list[dict[str, Any]] = []

    training_start = time.perf_counter()

    print("Training population classification head with AdamW")
    print(
        f"epochs={args.epochs}, head_lr={args.head_lr}, "
        f"backbone_lr={args.backbone_lr if partial_finetuning_enabled else None}, "
        f"lora_lr={args.lora_lr if lora_enabled else None}, "
        f"momentum={args.momentum}, "
        f"weight_decay={args.weight_decay}, "
        f"best_metric={args.metric_for_best}"
    )
    print(
        f"head_type={config.head_type}, "
        f"head_hidden_dim={config.head_hidden_dim}, "
        f"head_norm={config.head_norm}, "
        f"head_dropout={config.head_dropout}, "
        f"head_trainable_parameters={trainable_head_parameter_count}, "
        f"backbone_trainable_parameters={trainable_original_backbone_parameter_count}, "
        f"lora_trainable_parameters={trainable_lora_parameter_count}"
    )

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()

        if requires_live_backbone_forward:
            train_metrics = run_finetune_epoch(
                classifier=classifier,
                loader=train_loader,
                criterion=criterion,
                num_classes=num_classes,
                optimizer=optimizer,
            )
        else:
            train_metrics = run_head_epoch(
                head=classifier.head,
                loader=train_loader,
                criterion=criterion,
                device=classifier.device,
                num_classes=num_classes,
                optimizer=optimizer,
            )
        with torch.no_grad():
            if requires_live_backbone_forward:
                val_metrics = run_finetune_epoch(
                    classifier=classifier,
                    loader=val_loader,
                    criterion=criterion,
                    num_classes=num_classes,
                    optimizer=None,
                )
            else:
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
            if partial_finetuning_enabled:
                best_backbone_state = deepcopy(
                    {
                        key: value.detach().cpu()
                        for key, value in classifier.backbone.model.state_dict().items()
                    }
                )
            if lora_enabled:
                best_lora_state = deepcopy(
                    lora_state_dict(classifier.backbone.model)
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
    if best_backbone_state is not None:
        classifier.backbone.model.load_state_dict(best_backbone_state, strict=True)
    if best_lora_state is not None:
        load_lora_state_dict(classifier.backbone.model, best_lora_state)
    classifier.eval()

    with torch.no_grad():
        if requires_live_backbone_forward:
            selected_val_metrics_raw = run_finetune_epoch(
                classifier=classifier,
                loader=val_loader,
                criterion=criterion,
                num_classes=num_classes,
                optimizer=None,
            )
        else:
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
    # Only now construct held-out final-test windows.
    # ------------------------------------------------------------------

    print()
    print(
        "Population model selection is complete. "
        "Opening the unseen target-subject final test set..."
        if args.split_mode == "loso"
        else "Within-subject model selection is complete. "
        "Constructing held-out final-test session windows..."
    )

    if args.split_mode == "loso":
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
            window_construction=args.window_construction,
            direct_trial_anchor=args.direct_trial_anchor,
        )
        final_test_session = args.final_test_session
    else:
        assert within_subject_split is not None
        assert within_subject_all_trials is not None
        target_build = build_within_subject_test_split(
            subject_id=target_subject,
            path=all_subject_paths[target_subject],
            metadata=metadata,
            class_names=class_names,
            split=within_subject_split,
            all_trial_metadata=within_subject_all_trials,
            window_seconds=args.window_sec,
            stride_seconds=args.window_stride_sec,
            max_windows_per_class=args.max_windows_per_class_per_subject,
            window_construction=args.window_construction,
            direct_trial_anchor=args.direct_trial_anchor,
        )
        final_test_session = within_subject_split.test_session

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
        left_name=train_split_name,
        right_name="target final test",
    )
    validate_no_source_leakage(
        val_build.bundle.window_set,
        target_build.bundle.window_set,
        left_name=validation_split_name,
        right_name="target final test",
    )

    if requires_live_backbone_forward:
        target_features = tokenize_windows_for_finetuning(
            window_set=target_build.bundle.window_set,
            metadata=metadata,
            config=config,
            preprocess_batch_size=args.feature_batch_size,
            split_name="target_final_test",
            log_every=args.feature_log_every,
        )
    else:
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

    if args.save_feature_cache and feature_cache_enabled:
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
        if requires_live_backbone_forward:
            target_metrics_raw = run_finetune_epoch(
                classifier=classifier,
                loader=target_loader,
                criterion=criterion,
                num_classes=num_classes,
                optimizer=None,
            )
        else:
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

    if within_subject_split is not None:
        assert within_subject_all_trials is not None
        within_subject_metadata: dict[str, Any] = {
            "subject": int(target_subject),
            "train_session": within_subject_split.train_session,
            "test_session": within_subject_split.test_session,
            "validation_ratio": float(args.validation_ratio),
            "split_seed": int(args.seed),
            "available_sessions": list(within_subject_split.available_sessions),
            "train_trial_ids": within_subject_all_trials["trial_ids"][
                within_subject_split.train_indices
            ].tolist(),
            "validation_trial_ids": within_subject_all_trials["trial_ids"][
                within_subject_split.validation_indices
            ].tolist(),
            "test_trial_ids": within_subject_all_trials["trial_ids"][
                within_subject_split.test_indices
            ].tolist(),
            "train_class_counts": class_name_counts(
                within_subject_all_trials["labels"][within_subject_split.train_indices],
                class_names,
            ),
            "validation_class_counts": class_name_counts(
                within_subject_all_trials["labels"][within_subject_split.validation_indices],
                class_names,
            ),
            "test_class_counts": class_name_counts(
                within_subject_all_trials["labels"][within_subject_split.test_indices],
                class_names,
            ),
        }
    else:
        within_subject_metadata = None

    experiment_prefix = (
        "population_loso" if args.split_mode == "loso" else "within_subject"
    )
    experiment_name = (
        f"{experiment_prefix}_lora"
        if lora_enabled
        else (
            f"{experiment_prefix}_partial_finetune"
            if partial_finetuning_enabled
            else (
                f"{experiment_prefix}_linear_probe"
                if config.head_type == "linear"
                else f"{experiment_prefix}_mlp_head"
            )
        )
    )

    # ------------------------------------------------------------------
    # Save model and reports.
    # ------------------------------------------------------------------

    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path = save_classifier_checkpoint(
        classifier=classifier,
        checkpoint_path=output_path,
        extra_metadata={
            "task": (
                "BNCI2014_001_motor_imagery"
                if args.split_mode == "loso"
                else f"{metadata.dataset_name}_classification"
            ),
            "dataset": metadata.dataset_name,
            "mode": experiment_name,
            "stage": "stage1",
            "split_mode": args.split_mode,
            "target_subject": target_subject,
            "excluded_subjects": (
                [target_subject] if args.split_mode == "loso" else []
            ),
            "population_training_subjects": population_subjects,
            "population_validation_subjects": population_subjects,
            "population_train_session": args.train_session,
            "population_validation_session": args.validation_session,
            "final_test_subject": target_subject,
            "final_test_session": final_test_session,
            "subject_data_paths": {
                str(subject): str(path)
                for subject, path in all_subject_paths.items()
            },
            "class_names": class_names,
            "label_mapping": label_mapping,
            "num_classes": int(num_classes),
            "within_subject_split": within_subject_metadata,
            "backbone_sha256": backbone_sha256,
            "preprocessing_hash": preprocessing_hash,
            "backbone_adaptation": backbone_adaptation,
            "freeze_backbone": not partial_finetuning_enabled,
            "trainable_backbone_parameters": sum(
                parameter.numel() for parameter in trainable_backbone_parameters
            ),
            "embedding_layer_requested": str(args.embedding_layer),
            "embedding_layer_resolved": int(embedding_layer),
            "embedding_layer_internal_index": int(config.output_layer_idx),
            "unfreeze_last_n_blocks": int(args.unfreeze_last_n_blocks),
            "unfrozen_block_indices": [
                int(index) for index in (
                    trainable_block_indices if partial_finetuning_enabled else ()
                )
            ],
            "lora_last_n_blocks": (
                int(args.lora_last_n_blocks) if lora_enabled else None
            ),
            "lora_block_indices": [
                int(index) for index in (
                    trainable_block_indices if lora_enabled else ()
                )
            ],
            "lora_target_modules": (
                list(normalize_lora_target_modules(args.lora_target_modules))
                if lora_enabled
                else []
            ),
            "lora_rank": int(args.lora_rank) if lora_enabled else None,
            "lora_alpha": float(args.lora_alpha) if lora_enabled else None,
            "lora_dropout": float(args.lora_dropout) if lora_enabled else None,
            "head_type": config.head_type,
            "head_hidden_dim": int(config.head_hidden_dim),
            "head_dropout": float(config.head_dropout),
            "head_norm": config.head_norm,
            "head_trainable_parameters": sum(
                parameter.numel() for parameter in classifier.head.parameters()
            ),
            "optimizer": "AdamW",
            "head_lr": float(args.head_lr),
            "backbone_lr": float(args.backbone_lr),
            "lora_lr": float(args.lora_lr) if lora_enabled else None,
            "trainable_lora_parameters": sum(
                parameter.numel() for parameter in lora_parameters
            ),
            "trainable_total_parameters": total_trainable_parameter_count,
            "momentum": float(args.momentum),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "window_seed": int(args.window_seed),
            "metric_for_best": args.metric_for_best,
            "best_epoch": int(best_epoch),
            "best_validation": selected_val_metrics.to_dict(),
            "target_final_test": target_metrics.to_dict(),
            "window_construction": (
                train_build.bundle.window_set.construction
            ),
            "model_n_time_patches": int(
                config.model_n_time_patches
            ),
            "warning": (
                None
                if args.window_construction == "direct_trial"
                else (
                    "Samples were constructed by concatenating "
                    "same-label source trials."
                )
            ),
            "git_commit": git_commit,
        },
        backbone_state_dict=(
            classifier.backbone.model.state_dict()
            if partial_finetuning_enabled
            else None
        ),
        lora_state_dict=(
            lora_state_dict(classifier.backbone.model)
            if lora_enabled
            else None
        ),
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
        "experiment": experiment_name,
        "split_mode": args.split_mode,
        "warning": (
            None
            if args.window_construction == "direct_trial"
            else (
                "Samples were constructed by concatenating same-label source "
                "trials."
            )
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
            "final_target_test_session": final_test_session,
            "target_subject_used_for_training": args.split_mode == "within-subject",
            "target_subject_used_for_validation": args.split_mode == "within-subject",
            "final_test_opened_after_model_selection": True,
            "within_subject": within_subject_metadata,
        },
        "dataset": {
            "name": metadata.dataset_name,
            "sample_rate": metadata.sample_rate,
            "unit": metadata.unit,
            "channel_names": metadata.channel_names,
            "class_names": class_names,
            "label_mapping": label_mapping,
            "num_classes": num_classes,
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
            "embedding_layer_requested": str(args.embedding_layer),
            "embedding_layer_resolved": int(embedding_layer),
            "embedding_layer_internal_index": int(config.output_layer_idx),
            "unfreeze_last_n_blocks": int(args.unfreeze_last_n_blocks),
            "unfrozen_blocks": [
                int(index + 1) for index in trainable_block_indices
            ],
            "feature_cache_enabled": feature_cache_enabled,
            "backbone_adaptation": backbone_adaptation,
            "lora_last_n_blocks": (
                int(args.lora_last_n_blocks) if lora_enabled else None
            ),
            "lora_blocks": [
                int(index + 1) for index in (
                    trainable_block_indices if lora_enabled else ()
                )
            ],
            "lora_target_modules": (
                list(normalize_lora_target_modules(args.lora_target_modules))
                if lora_enabled
                else []
            ),
            "lora_rank": int(args.lora_rank) if lora_enabled else None,
            "lora_alpha": float(args.lora_alpha) if lora_enabled else None,
            "lora_dropout": float(args.lora_dropout) if lora_enabled else None,
            "trainable_lora_params": sum(
                parameter.numel() for parameter in lora_parameters
            ),
            "trainable_total_params": total_trainable_parameter_count,
            "target_sample_rate": config.target_sample_rate,
            "target_num_points": config.target_num_points,
            "num_tokens": config.num_tokens,
            "patch_num_points": config.patch_num_points,
            "output_layer_idx": config.output_layer_idx,
            "aggregation": config.aggregation,
            "classifier_input_dim": config.classifier_input_dim,
            "head_type": config.head_type,
            "head_hidden_dim": config.head_hidden_dim,
            "head_dropout": config.head_dropout,
            "head_norm": config.head_norm,
            "feature_cache_dtype": args.feature_cache_dtype,
            "preprocessing_contract": preprocessing_contract,
            "preprocessing_hash": preprocessing_hash,
        },
        "training": {
            "optimizer": "AdamW",
            "epochs_requested": args.epochs,
            "epochs_completed": len(epoch_rows),
            "training_seconds": training_seconds,
            "best_epoch": best_epoch,
            "metric_for_best": args.metric_for_best,
            "head_lr": args.head_lr,
            "backbone_lr": args.backbone_lr,
            "lora_lr": args.lora_lr if lora_enabled else None,
            "backbone_adaptation": backbone_adaptation,
            "lora_last_n_blocks": (
                args.lora_last_n_blocks if lora_enabled else None
            ),
            "lora_blocks": [
                int(index + 1) for index in (
                    trainable_block_indices if lora_enabled else ()
                )
            ],
            "lora_target_modules": (
                list(normalize_lora_target_modules(args.lora_target_modules))
                if lora_enabled
                else []
            ),
            "lora_rank": args.lora_rank if lora_enabled else None,
            "lora_alpha": args.lora_alpha if lora_enabled else None,
            "lora_dropout": args.lora_dropout if lora_enabled else None,
            "trainable_lora_params": sum(
                parameter.numel() for parameter in lora_parameters
            ),
            "trainable_total_params": total_trainable_parameter_count,
            "unfreeze_last_n_blocks": args.unfreeze_last_n_blocks,
            "embedding_layer": embedding_layer,
            "embedding_layer_internal_index": config.output_layer_idx,
            "unfrozen_blocks": [
                int(index + 1) for index in trainable_block_indices
            ],
            "trainable_backbone_params": sum(
                parameter.numel() for parameter in trainable_backbone_parameters
            ),
            "trainable_head_params": sum(
                parameter.numel() for parameter in head_parameters
            ),
            "head_type": config.head_type,
            "head_hidden_dim": config.head_hidden_dim,
            "head_dropout": config.head_dropout,
            "head_norm": config.head_norm,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
            "window_seed": args.window_seed,
        },
        "best_population_validation": selected_val_metrics.to_dict(),
        "unseen_target_final_test": target_metrics.to_dict(),
        "best_validation": selected_val_metrics.to_dict(),
        "final_test": target_metrics.to_dict(),
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
        "split_mode": args.split_mode,
        "target_subject": target_subject,
        "population_subjects": population_subjects,
        "within_subject": within_subject_metadata,
        "train_session": args.train_session,
        "test_session": final_test_session,
        "validation_ratio": (
            float(args.validation_ratio)
            if args.split_mode == "within-subject"
            else None
        ),
        "num_classes": num_classes,
        "class_names": class_names,
        "label_mapping": label_mapping,
        "best_epoch": best_epoch,
        "population_validation": selected_val_metrics.to_dict(),
        "unseen_target_final_test": target_metrics.to_dict(),
        "classifier_checkpoint": str(saved_path),
        "report": str(report_path),
        "head_type": config.head_type,
        "head_hidden_dim": config.head_hidden_dim,
        "head_dropout": config.head_dropout,
        "head_norm": config.head_norm,
        "head_lr": args.head_lr,
        "backbone_adaptation": backbone_adaptation,
        "backbone_lr": args.backbone_lr,
        "embedding_layer": embedding_layer,
        "embedding_layer_internal_index": config.output_layer_idx,
        "unfreeze_last_n_blocks": args.unfreeze_last_n_blocks,
        "unfrozen_blocks": [
            int(index + 1) for index in trainable_block_indices
        ],
        "trainable_backbone_params": sum(
            parameter.numel() for parameter in trainable_backbone_parameters
        ),
        "trainable_head_params": sum(
            parameter.numel() for parameter in head_parameters
        ),
        "lora_last_n_blocks": args.lora_last_n_blocks if lora_enabled else None,
        "lora_blocks": [
            int(index + 1) for index in (
                trainable_block_indices if lora_enabled else ()
            )
        ],
        "lora_target_modules": (
            list(normalize_lora_target_modules(args.lora_target_modules))
            if lora_enabled
            else []
        ),
        "lora_rank": args.lora_rank if lora_enabled else None,
        "lora_alpha": args.lora_alpha if lora_enabled else None,
        "lora_dropout": args.lora_dropout if lora_enabled else None,
        "lora_lr": args.lora_lr if lora_enabled else None,
        "trainable_lora_params": sum(
            parameter.numel() for parameter in lora_parameters
        ),
        "trainable_total_params": total_trainable_parameter_count,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
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
