from __future__ import annotations

"""
Train a CBRaMod frozen-backbone population head with LOSO or within-subject
splits.

For one target subject:

    Population training:
        all non-target subjects / 0train

    Population validation:
        all non-target subjects / 1test

    Final independent test:
        target subject / 1test

The target subject is never used to train or select the population head.

With ``--split-mode within-subject``, one source session is split at the
source-trial level for training/validation and a distinct session is held out
for final testing.

The CBRaMod backbone is frozen. Only the downstream classification head is
updated. The default head is the official CBRaMod MLP head:

    Flatten(22 * 4 * 200)
    -> Linear(..., 800) -> ELU -> Dropout
    -> Linear(800, 200) -> ELU -> Dropout
    -> Linear(200, 4)
"""

import argparse
import csv
import hashlib
import json
import random
import subprocess
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

from _bootstrap import ROOT

from bci_dayloop.data.hdf5_dataset import (
    EEGHDF5,
    HDF5Metadata,
)
from bci_dayloop.data.splits import (
    WithinSubjectTrialSplit,
    resolve_within_subject_trial_split,
)
from bci_dayloop.models.cbramod.backbone import (
    CBraModBackbone,
)
from bci_dayloop.models.cbramod.classifier import (
    CBraModClassifier,
    build_cbramod_classifier,
)
from bci_dayloop.models.cbramod.config import (
    CBraModConfig,
)
from bci_dayloop.models.cbramod.preprocessing import (
    CBraModPipelinePreprocessor,
)
from bci_dayloop.models.cbramod.runtime import (
    save_cbramod_classifier_checkpoint,
)
from bci_dayloop.preprocessing.canonical import (
    SignalCanonicalizer,
)
from bci_dayloop.runtime.types import RawEEGWindow
from bci_dayloop.utils.config import dump_yaml
from bci_dayloop.data.trial_windows import (
    select_direct_trial_window,
)

# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeatureSplit:
    """
    冻结 CBRaMod backbone 后的特征集合。

    features:
        [N, 22, 4, 200]

    labels:
        [N]

    source_trial_ids:
        使用 subject_id 和原始 trial_id 编码后的全局唯一 ID。
    """

    features: torch.Tensor
    labels: torch.Tensor
    source_trial_ids: np.ndarray
    subject_ids: np.ndarray
    session_name: str

    def __post_init__(self) -> None:
        num_samples = int(self.features.shape[0])

        if self.features.ndim != 4:
            raise ValueError(
                "features must have shape [N, C, S, D], got "
                f"{tuple(self.features.shape)}."
            )

        if tuple(self.labels.shape) != (num_samples,):
            raise ValueError(
                "labels shape mismatch: expected "
                f"{(num_samples,)}, got "
                f"{tuple(self.labels.shape)}."
            )

        if self.source_trial_ids.shape != (num_samples,):
            raise ValueError(
                "source_trial_ids shape mismatch: expected "
                f"{(num_samples,)}, got "
                f"{self.source_trial_ids.shape}."
            )

        if self.subject_ids.shape != (num_samples,):
            raise ValueError(
                "subject_ids shape mismatch: expected "
                f"{(num_samples,)}, got "
                f"{self.subject_ids.shape}."
            )


@dataclass(frozen=True, slots=True)
class Metrics:
    loss: float
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "confusion_matrix": self.confusion_matrix,
        }


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def resolve_cbramod_time_segments(
    *,
    window_seconds: float,
    target_sample_rate: float,
    points_per_patch: int = 200,
) -> int:
    """
    官方 CBraMod 保持每个 patch 为 200 点。

    在 200 Hz 下，这等价于要求 window_seconds 是整数秒。
    """
    if window_seconds <= 0:
        raise ValueError(
            "--window-seconds must be positive."
        )

    if target_sample_rate <= 0:
        raise ValueError(
            "--target-sample-rate must be positive."
        )

    target_samples_float = (
        window_seconds * target_sample_rate
    )
    target_samples = int(round(target_samples_float))

    if not np.isclose(
        target_samples_float,
        target_samples,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "CBraMod window duration does not map to an "
            "integer number of samples: "
            f"{window_seconds} s × "
            f"{target_sample_rate} Hz = "
            f"{target_samples_float}."
        )

    if target_samples % points_per_patch != 0:
        raise ValueError(
            "CBraMod currently requires an integer number "
            "of 200-sample patches. At 200 Hz, "
            "--window-seconds must be an integer number "
            f"of seconds. Got {window_seconds}s -> "
            f"{target_samples} samples."
        )

    time_segments = target_samples // points_per_patch

    if time_segments <= 0:
        raise ValueError(
            "CBraMod time_segments must be positive."
        )

    return time_segments

def resolve_repo_path(
    value: str | Path,
) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def atomic_write_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(
        f"{path.suffix}.tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def sha256_file(
    path: Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def stable_json_hash(
    payload: Mapping[str, Any],
) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
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
    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return None

    value = result.stdout.strip()

    return value or None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def normalize_subjects(
    values: Sequence[int],
) -> list[int]:
    subjects = sorted(set(int(value) for value in values))

    if not subjects:
        raise ValueError(
            "At least one subject must be provided."
        )

    invalid = [
        subject
        for subject in subjects
        if subject <= 0
    ]

    if invalid:
        raise ValueError(
            f"Subject IDs must be positive, got {invalid}."
        )

    return subjects


def encode_source_trial_id(
    subject_id: int,
    trial_id: int,
) -> int:
    """
    将文件内 trial_id 编成全局唯一 int64：

    high 32 bits: subject_id
    low 32 bits : trial_id
    """

    subject_id = int(subject_id)
    trial_id = int(trial_id)

    if subject_id <= 0:
        raise ValueError(
            f"subject_id must be positive, got {subject_id}."
        )

    if not 0 <= trial_id < 2**32:
        raise ValueError(
            "trial_id must be in [0, 2**32), got "
            f"{trial_id}."
        )

    return (subject_id << 32) | trial_id


def class_counts(
    labels: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, int]:
    labels = np.asarray(labels, dtype=np.int64)

    return {
        str(class_name): int(
            np.sum(labels == class_index)
        )
        for class_index, class_name in enumerate(
            class_names
        )
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
            f"{split_name}: labels must be 1-D, got "
            f"{labels.shape}."
        )

    if len(labels) == 0:
        raise ValueError(
            f"{split_name}: labels are empty."
        )

    invalid = labels[
        (labels < 0)
        | (labels >= num_classes)
    ]

    if len(invalid) > 0:
        raise ValueError(
            f"{split_name}: labels outside "
            f"[0, {num_classes - 1}]: "
            f"{sorted(set(invalid.tolist()))}."
        )


def validate_no_trial_leakage(
    left: FeatureSplit,
    right: FeatureSplit,
    *,
    left_name: str,
    right_name: str,
) -> None:
    overlap = np.intersect1d(
        left.source_trial_ids,
        right.source_trial_ids,
    )

    if len(overlap) > 0:
        raise RuntimeError(
            f"Source-trial leakage between {left_name} and "
            f"{right_name}. Example encoded IDs: "
            f"{overlap[:10].tolist()}."
        )


def validate_metadata_compatibility(
    reference: HDF5Metadata,
    candidate: HDF5Metadata,
    *,
    subject_id: int,
    path: Path,
) -> None:
    mismatches: list[str] = []

    if reference.dataset_name != candidate.dataset_name:
        mismatches.append(
            "dataset_name "
            f"{candidate.dataset_name!r} != "
            f"{reference.dataset_name!r}"
        )

    if reference.class_names != candidate.class_names:
        mismatches.append(
            "class_names differ"
        )

    if mismatches:
        raise ValueError(
            f"Metadata mismatch for subject {subject_id} "
            f"at {path}: {'; '.join(mismatches)}."
        )


def resolve_subject_file(
    *,
    data_root: Path,
    data_pattern: str,
    subject_id: int,
) -> Path:
    try:
        formatted = data_pattern.format(
            subject=subject_id
        )
    except (KeyError, ValueError) as error:
        raise ValueError(
            "--data-pattern must be a valid Python format "
            "string containing {subject}; for example "
            "'subject_{subject:02d}.h5'."
        ) from error

    candidates = [
        data_root / formatted,
        data_root / f"subject_{subject_id:02d}.h5",
        data_root / f"bnci2014_001_s{subject_id:02d}.h5",
        ROOT
        / "data"
        / "processed"
        / f"bnci2014_001_s{subject_id:02d}.h5",
    ]

    unique_candidates: list[Path] = []
    seen: set[Path] = set()

    for candidate in candidates:
        resolved = resolve_repo_path(candidate)

        if resolved not in seen:
            unique_candidates.append(resolved)
            seen.add(resolved)

    for candidate in unique_candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not find HDF5 data for subject {subject_id}. "
        "Tried:\n"
        + "\n".join(
            f"  - {candidate}"
            for candidate in unique_candidates
        )
    )


# ---------------------------------------------------------------------
# Frozen feature extraction
# ---------------------------------------------------------------------


def prepare_cbramod_trials(
    *,
    data: np.ndarray,
    metadata: HDF5Metadata,
    trial_ids: np.ndarray,
    subject_id: int,
    session_name: str,
    canonicalizer: SignalCanonicalizer,
    preprocessor: CBraModPipelinePreprocessor,
) -> np.ndarray:
    """
    将 HDF5 的 [N, C, T] trial 转成 CBRaMod 的 [N, 22, 4, 200]。
    """

    values = np.asarray(data, dtype=np.float32)

    if values.ndim != 3:
        raise ValueError(
            "HDF5 trials must have shape [N, C, T], got "
            f"{values.shape}."
        )

    if len(trial_ids) != len(values):
        raise ValueError(
            "trial_ids length does not match number of trials."
        )

    prepared_trials: list[np.ndarray] = []

    for index, trial in enumerate(values):
        raw_window = RawEEGWindow(
            data=trial,
            channel_names=list(
                metadata.channel_names
            ),
            sample_rate=float(metadata.sample_rate),
            unit=str(metadata.unit),
            layout="CT",
            trial_id=str(int(trial_ids[index])),
            metadata={
                "subject_id": int(subject_id),
                "session": session_name,
                "source": "train_cbramod_population_head",
            },
        )

        canonical_window = canonicalizer.transform(
            raw_window
        )

        prepared = preprocessor.transform(
            canonical_window
        )

        signal = prepared.model_input["signal"]

        if not isinstance(signal, torch.Tensor):
            raise TypeError(
                "CBraMod preprocessor did not return a Tensor "
                "under model_input['signal']."
            )

        expected_shape = (
            1,
            preprocessor.config.n_channels,
            preprocessor.config.time_segments,
            preprocessor.config.points_per_patch,
        )

        if tuple(signal.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected CBRaMod prepared trial shape. "
                f"Expected {expected_shape}, got "
                f"{tuple(signal.shape)}."
            )

        prepared_trials.append(
            signal[0].detach().cpu().numpy()
        )

    return np.ascontiguousarray(
        np.stack(prepared_trials, axis=0),
        dtype=np.float32,
    )


@torch.no_grad()
def extract_frozen_features(
    *,
    backbone: CBraModBackbone,
    prepared_trials: np.ndarray,
    batch_size: int,
) -> torch.Tensor:
    """
    输入：
        prepared_trials: [N, 22, 4, 200]

    输出：
        features: [N, 22, 4, 200]，保存在 CPU。
    """

    if batch_size <= 0:
        raise ValueError(
            f"batch_size must be positive, got {batch_size}."
        )

    values = np.asarray(
        prepared_trials,
        dtype=np.float32,
    )

    expected_tail = backbone.config.expected_unbatched_shape

    if values.ndim != 4:
        raise ValueError(
            "prepared_trials must have shape [N, C, S, P], "
            f"got {values.shape}."
        )

    if tuple(values.shape[1:]) != expected_tail:
        raise ValueError(
            "prepared_trials shape mismatch. Expected "
            f"[N, {expected_tail[0]}, "
            f"{expected_tail[1]}, "
            f"{expected_tail[2]}], got "
            f"{values.shape}."
        )

    backbone.freeze()

    outputs: list[torch.Tensor] = []

    for start in range(0, len(values), batch_size):
        end = min(start + batch_size, len(values))

        batch = torch.from_numpy(
            values[start:end]
        ).to(
            backbone.device,
            dtype=torch.float32,
            non_blocking=(
                backbone.device.type == "cuda"
            ),
        )

        features = backbone.encode(batch)

        outputs.append(
            features.detach().cpu()
        )

    result = torch.cat(outputs, dim=0).contiguous()

    expected_shape = (
        len(values),
        backbone.config.n_channels,
        backbone.config.time_segments,
        backbone.config.backbone_output_dim,
    )

    if tuple(result.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected frozen CBRaMod feature shape. "
            f"Expected {expected_shape}, got "
            f"{tuple(result.shape)}."
        )

    return result


def build_subject_feature_split(
    *,
    subject_id: int,
    session_name: str,
    subject_path: Path,
    reference_metadata: HDF5Metadata | None,
    canonicalizer: SignalCanonicalizer,
    preprocessor: CBraModPipelinePreprocessor,
    backbone: CBraModBackbone,
    feature_batch_size: int,
    direct_trial_anchor: str,
) -> tuple[FeatureSplit, HDF5Metadata]:
    dataset = EEGHDF5(subject_path)
    metadata = dataset.metadata

    if reference_metadata is not None:
        validate_metadata_compatibility(
            reference_metadata,
            metadata,
            subject_id=subject_id,
            path=subject_path,
        )

    loaded = dataset.load(session_name)
    return build_feature_split_from_session_data(
        subject_id=subject_id,
        session_name=session_name,
        subject_path=subject_path,
        metadata=metadata,
        loaded=loaded,
        canonicalizer=canonicalizer,
        preprocessor=preprocessor,
        backbone=backbone,
        feature_batch_size=feature_batch_size,
        direct_trial_anchor=direct_trial_anchor,
    )


def build_feature_split_from_session_data(
    *,
    subject_id: int,
    session_name: str,
    subject_path: Path,
    metadata: HDF5Metadata,
    loaded: Mapping[str, np.ndarray],
    canonicalizer: SignalCanonicalizer,
    preprocessor: CBraModPipelinePreprocessor,
    backbone: CBraModBackbone,
    feature_batch_size: int,
    direct_trial_anchor: str,
) -> tuple[FeatureSplit, HDF5Metadata]:
    """Extract features from an already selected set of source trials."""

    data = np.asarray(
        loaded["data"],
        dtype=np.float32,
    )

    labels = np.asarray(
        loaded["labels"],
        dtype=np.int64,
    )

    trial_ids = np.asarray(
        loaded["trial_ids"],
        dtype=np.int64,
    )

    subject_ids = np.asarray(
        loaded["subject_ids"],
        dtype=np.int64,
    )

    if data.ndim != 3:
        raise ValueError(
            f"{subject_path}: expected [N, C, T] data, got "
            f"{data.shape}."
        )

    if len(data) == 0:
        raise ValueError(
            f"{subject_path}: session {session_name!r} is empty."
        )

    if len(labels) != len(data):
        raise ValueError(
            f"{subject_path}: labels length mismatch."
        )

    if not np.all(subject_ids == subject_id):
        actual_subjects = sorted(
            set(subject_ids.tolist())
        )

        raise ValueError(
            f"{subject_path}: expected subject {subject_id}, "
            f"found {actual_subjects}."
        )

    loaded_sessions = set(
        np.asarray(
            loaded["session_ids"]
        ).astype(str).tolist()
    )

    if loaded_sessions != {session_name}:
        raise ValueError(
            f"{subject_path}: expected only session "
            f"{session_name!r}, got "
            f"{sorted(loaded_sessions)}."
        )

    validate_labels(
        labels,
        num_classes=len(metadata.class_names),
        split_name=(
            f"subject_{subject_id:02d}/{session_name}"
        ),
    )

    data, _ = select_direct_trial_window(
        data,
        sample_rate=float(metadata.sample_rate),
        window_seconds=preprocessor.config.window_seconds,
        anchor=direct_trial_anchor,
        context=(
            f"subject_{subject_id:02d}/"
            f"{session_name}"
        ),
    )

    prepared_trials = prepare_cbramod_trials(
        data=data,
        metadata=metadata,
        trial_ids=trial_ids,
        subject_id=subject_id,
        session_name=session_name,
        canonicalizer=canonicalizer,
        preprocessor=preprocessor,
    )

    features = extract_frozen_features(
        backbone=backbone,
        prepared_trials=prepared_trials,
        batch_size=feature_batch_size,
    )

    global_trial_ids = np.asarray(
        [
            encode_source_trial_id(
                subject_id,
                int(trial_id),
            )
            for trial_id in trial_ids
        ],
        dtype=np.int64,
    )

    return (
        FeatureSplit(
            features=features,
            labels=torch.from_numpy(
                labels.astype(np.int64, copy=False)
            ),
            source_trial_ids=global_trial_ids,
            subject_ids=subject_ids,
            session_name=session_name,
        ),
        metadata,
    )


def select_trial_rows(
    trial_data: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Select source trials without modifying the HDF5-backed arrays."""
    indices = np.asarray(indices, dtype=np.int64)
    return {
        key: np.asarray(values)[indices]
        for key, values in trial_data.items()
    }


def select_trial_ids(
    trial_data: Mapping[str, np.ndarray],
    trial_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Select trial IDs and fail if the loaded session does not contain one."""
    requested = np.asarray(trial_ids, dtype=np.int64)
    available = np.asarray(
        trial_data["trial_ids"],
        dtype=np.int64,
    )
    selected = select_trial_rows(
        trial_data,
        np.flatnonzero(np.isin(available, requested)),
    )
    selected_ids = np.asarray(
        selected["trial_ids"],
        dtype=np.int64,
    )
    if set(selected_ids.tolist()) != set(requested.tolist()):
        missing = sorted(
            set(requested.tolist()) - set(selected_ids.tolist())
        )
        raise RuntimeError(
            "Selected source trials are missing from the loaded session: "
            f"{missing[:10]}."
        )
    return selected


def resolve_class_names(
    *,
    metadata: HDF5Metadata,
    explicit_class_names: Sequence[str] | None,
) -> tuple[str, ...]:
    """Resolve logit semantics without assuming every task is BNCI MI."""
    num_classes = len(metadata.class_names)
    if num_classes <= 0:
        raise ValueError("HDF5 metadata class_names must not be empty.")

    if explicit_class_names is not None:
        class_names = tuple(
            str(name).strip()
            for name in explicit_class_names
        )
        source = "--class-names"
    else:
        metadata_names = tuple(
            str(name).strip()
            for name in metadata.class_names
        )
        if len(metadata_names) == num_classes and all(metadata_names):
            class_names = metadata_names
            source = "HDF5 metadata"
        else:
            class_names = tuple(
                f"class_{index}"
                for index in range(num_classes)
            )
            source = "numeric fallback"

    if len(class_names) != num_classes:
        raise ValueError(
            "class_names length must match the HDF5 class count: "
            f"{len(class_names)} != {num_classes}."
        )
    if not all(class_names):
        raise ValueError("class_names must not contain empty values.")
    if len(set(class_names)) != len(class_names):
        raise ValueError(
            f"class_names must be unique, got {list(class_names)}."
        )
    print(f"Class semantics source: {source}")
    print(
        "Class semantics:",
        {index: name for index, name in enumerate(class_names)},
    )
    return class_names


def build_within_subject_train_validation_splits(
    *,
    subject_id: int,
    subject_path: Path,
    train_session: str,
    test_session: str,
    validation_ratio: float,
    seed: int,
    class_names: Sequence[str],
    canonicalizer: SignalCanonicalizer,
    preprocessor: CBraModPipelinePreprocessor,
    backbone: CBraModBackbone,
    feature_batch_size: int,
    direct_trial_anchor: str,
) -> tuple[
    FeatureSplit,
    FeatureSplit,
    HDF5Metadata,
    WithinSubjectTrialSplit,
    dict[str, np.ndarray],
]:
    """Split one source session before CBraMod preprocessing or encoding."""
    dataset = EEGHDF5(subject_path)
    metadata = dataset.metadata
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
    source_session_data = dataset.load(train_session)
    train_data = select_trial_ids(
        source_session_data,
        all_trial_metadata["trial_ids"][split.train_indices],
    )
    validation_data = select_trial_ids(
        source_session_data,
        all_trial_metadata["trial_ids"][split.validation_indices],
    )
    train_split, _ = build_feature_split_from_session_data(
        subject_id=subject_id,
        session_name=train_session,
        subject_path=subject_path,
        metadata=metadata,
        loaded=train_data,
        canonicalizer=canonicalizer,
        preprocessor=preprocessor,
        backbone=backbone,
        feature_batch_size=feature_batch_size,
        direct_trial_anchor=direct_trial_anchor,
    )
    validation_split, _ = build_feature_split_from_session_data(
        subject_id=subject_id,
        session_name=train_session,
        subject_path=subject_path,
        metadata=metadata,
        loaded=validation_data,
        canonicalizer=canonicalizer,
        preprocessor=preprocessor,
        backbone=backbone,
        feature_batch_size=feature_batch_size,
        direct_trial_anchor=direct_trial_anchor,
    )
    return (
        train_split,
        validation_split,
        metadata,
        split,
        all_trial_metadata,
    )


def build_within_subject_final_test_split(
    *,
    subject_id: int,
    subject_path: Path,
    metadata: HDF5Metadata,
    split: WithinSubjectTrialSplit,
    all_trial_metadata: Mapping[str, np.ndarray],
    canonicalizer: SignalCanonicalizer,
    preprocessor: CBraModPipelinePreprocessor,
    backbone: CBraModBackbone,
    feature_batch_size: int,
    direct_trial_anchor: str,
) -> FeatureSplit:
    """Load the held-out test session only after model selection."""
    dataset = EEGHDF5(subject_path)
    test_session_data = dataset.load(split.test_session)
    test_data = select_trial_ids(
        test_session_data,
        np.asarray(
            all_trial_metadata["trial_ids"][split.test_indices],
            dtype=np.int64,
        ),
    )
    test_split, _ = build_feature_split_from_session_data(
        subject_id=subject_id,
        session_name=split.test_session,
        subject_path=subject_path,
        metadata=metadata,
        loaded=test_data,
        canonicalizer=canonicalizer,
        preprocessor=preprocessor,
        backbone=backbone,
        feature_batch_size=feature_batch_size,
        direct_trial_anchor=direct_trial_anchor,
    )
    return test_split


def combine_feature_splits(
    splits: Sequence[FeatureSplit],
    *,
    session_name: str,
) -> FeatureSplit:
    if not splits:
        raise ValueError(
            "Cannot combine an empty feature split list."
        )

    return FeatureSplit(
        features=torch.cat(
            [split.features for split in splits],
            dim=0,
        ).contiguous(),
        labels=torch.cat(
            [split.labels for split in splits],
            dim=0,
        ).contiguous(),
        source_trial_ids=np.concatenate(
            [
                split.source_trial_ids
                for split in splits
            ],
            axis=0,
        ).astype(np.int64, copy=False),
        subject_ids=np.concatenate(
            [
                split.subject_ids
                for split in splits
            ],
            axis=0,
        ).astype(np.int64, copy=False),
        session_name=session_name,
    )


# ---------------------------------------------------------------------
# Optional feature cache
# ---------------------------------------------------------------------


def save_feature_cache(
    *,
    path: Path,
    split_name: str,
    split: FeatureSplit,
    manifest: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "split_name": split_name,
            "manifest": dict(manifest),
            "features": split.features.contiguous(),
            "labels": split.labels.contiguous(),
            "source_trial_ids": split.source_trial_ids,
            "subject_ids": split.subject_ids,
            "session_name": split.session_name,
        },
        path,
    )


def load_feature_cache(
    *,
    path: Path,
    split_name: str,
    expected_manifest: Mapping[str, Any],
) -> FeatureSplit:
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature cache was not found: {path}."
        )

    payload: Any = torch.load(
        path,
        map_location="cpu",
    )

    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Invalid feature cache payload: {path}."
        )

    if payload.get("split_name") != split_name:
        raise ValueError(
            "Feature cache split mismatch: "
            f"cache={payload.get('split_name')!r}, "
            f"expected={split_name!r}."
        )

    if payload.get("manifest") != dict(expected_manifest):
        raise ValueError(
            "Feature cache manifest mismatch. Do not reuse "
            "features created with a different backbone, "
            "preprocessing configuration, subjects or sessions."
        )

    required_keys = {
        "features",
        "labels",
        "source_trial_ids",
        "subject_ids",
        "session_name",
    }

    missing = required_keys - set(payload)

    if missing:
        raise KeyError(
            f"Feature cache is missing keys: {sorted(missing)}."
        )

    return FeatureSplit(
        features=torch.as_tensor(
            payload["features"],
            dtype=torch.float32,
        ).contiguous(),
        labels=torch.as_tensor(
            payload["labels"],
            dtype=torch.int64,
        ).contiguous(),
        source_trial_ids=np.asarray(
            payload["source_trial_ids"],
            dtype=np.int64,
        ),
        subject_ids=np.asarray(
            payload["subject_ids"],
            dtype=np.int64,
        ),
        session_name=str(payload["session_name"]),
    )


# ---------------------------------------------------------------------
# Head training and evaluation
# ---------------------------------------------------------------------


def evaluate_head(
    *,
    classifier: CBraModClassifier,
    split: FeatureSplit,
    device: torch.device,
    batch_size: int,
    num_classes: int,
) -> Metrics:
    classifier.eval()

    dataset = TensorDataset(
        split.features,
        split.labels,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss(
        reduction="sum"
    )

    total_loss = 0.0
    total_count = 0

    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    with torch.inference_mode():
        for features, labels in loader:
            features = features.to(
                device,
                dtype=torch.float32,
                non_blocking=(
                    device.type == "cuda"
                ),
            )

            labels = labels.to(
                device,
                dtype=torch.int64,
                non_blocking=(
                    device.type == "cuda"
                ),
            )

            logits = classifier(features)

            total_loss += float(
                criterion(logits, labels).item()
            )

            total_count += int(labels.shape[0])

            all_targets.append(
                labels.detach().cpu().numpy()
            )

            all_predictions.append(
                logits.argmax(dim=-1)
                .detach()
                .cpu()
                .numpy()
            )

    if total_count <= 0:
        raise ValueError(
            "Cannot evaluate an empty feature split."
        )

    targets = np.concatenate(all_targets)
    predictions = np.concatenate(all_predictions)

    return Metrics(
        loss=total_loss / total_count,
        accuracy=float(
            accuracy_score(targets, predictions)
        ),
        balanced_accuracy=float(
            balanced_accuracy_score(
                targets,
                predictions,
            )
        ),
        macro_f1=float(
            f1_score(
                targets,
                predictions,
                average="macro",
                labels=list(range(num_classes)),
                zero_division=0,
            )
        ),
        confusion_matrix=confusion_matrix(
            targets,
            predictions,
            labels=list(range(num_classes)),
        ).astype(int).tolist(),
    )


def train_head_epoch(
    *,
    classifier: CBraModClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    classifier.train()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_count = 0

    for features, labels in loader:
        features = features.to(
            device,
            dtype=torch.float32,
            non_blocking=(device.type == "cuda"),
        )

        labels = labels.to(
            device,
            dtype=torch.int64,
            non_blocking=(device.type == "cuda"),
        )

        optimizer.zero_grad(set_to_none=True)

        logits = classifier(features)

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = int(labels.shape[0])

        total_loss += float(loss.detach().item()) * batch_size
        total_count += batch_size

    if total_count <= 0:
        raise ValueError(
            "Population training split is empty."
        )

    return total_loss / total_count


def metric_is_better(
    *,
    candidate: Metrics,
    best: Metrics | None,
    metric_name: str,
) -> bool:
    if best is None:
        return True

    candidate_value = float(
        getattr(candidate, metric_name)
    )

    best_value = float(
        getattr(best, metric_name)
    )

    if metric_name == "loss":
        return candidate_value < best_value - 1e-12

    return candidate_value > best_value + 1e-12


def write_confusion_matrix_csv(
    *,
    path: Path,
    matrix: Sequence[Sequence[int]],
    class_names: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            ["true_label", *class_names]
        )

        for class_name, row in zip(
            class_names,
            matrix,
            strict=True,
        ):
            writer.writerow(
                [class_name, *row]
            )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a CBRaMod frozen-backbone classification head with "
            "LOSO population or within-subject splits."
        )
    )

    parser.add_argument(
        "--data-root",
        default="data/processed/bnci2014_001",
    )

    parser.add_argument(
        "--data-pattern",
        default="subject_{subject:02d}.h5",
        help=(
            "Subject HDF5 filename pattern. It may use {subject}; a static "
            "filename is supported for a single-subject HDF5."
        ),
    )

    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    parser.add_argument(
        "--target-subject",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--split-mode",
        choices=["loso", "within-subject"],
        default="loso",
        help=(
            "Data split protocol. 'loso' preserves the existing population "
            "behavior; 'within-subject' uses --target-subject as the "
            "selected subject."
        ),
    )

    parser.add_argument(
        "--train-session",
        default="0train",
    )

    parser.add_argument(
        "--validation-session",
        default="1test",
    )

    parser.add_argument(
        "--final-test-session",
        default="1test",
    )

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
            "checkpoints/backbones/cbramod/"
            "pretrained_weights.pth"
        ),
    )

    parser.add_argument(
        "--output-head",
        default=None,
        help=(
            "Default: checkpoints/heads/stage1/"
            "bnci2014_001/subject_XX/cbramod/"
            "4s_flatten/head.pt"
        ),
    )

    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Default: runs/stage1/bnci2014_001/"
            "subject_XX/cbramod/4s_flatten"
        ),
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--feature-batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--head-batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--head-lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--metric-for-best",
        choices=[
            "balanced_accuracy",
            "macro_f1",
            "accuracy",
            "loss",
        ],
        default="balanced_accuracy",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--target-sample-rate",
        type=float,
        default=200.0,
    )

    parser.add_argument(
        "--window-seconds",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--direct-trial-anchor",
        choices=["start", "center", "end"],
        default="end",
        help=(
            "When source trials are longer than "
            "--window-seconds, choose one contiguous "
            "direct-trial segment. Default: end."
        ),
    )

    parser.add_argument(
        "--input-unit",
        default="uV",
    )

    parser.add_argument(
        "--filter-enabled",
        action="store_true",
    )

    parser.add_argument(
        "--filter-low-hz",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--filter-high-hz",
        type=float,
        default=75.0,
    )

    parser.add_argument(
        "--filter-order",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--reference-mode",
        choices=["none", "average"],
        default="none",
    )

    parser.add_argument(
        "--normalization",
        choices=["none", "per_window_zscore"],
        default="none",
    )

    parser.add_argument(
        "--head-type",
        choices=["official_mlp", "linear"],
        default="official_mlp",
    )

    parser.add_argument(
        "--head-hidden-dim-1",
        type=int,
        default=800,
    )

    parser.add_argument(
        "--head-hidden-dim-2",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--head-dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--feature-cache-dir",
        default=None,
        help=(
            "Optional frozen feature cache directory. "
            "Feature cache is never reused unless "
            "--reuse-feature-cache is set."
        ),
    )

    parser.add_argument(
        "--reuse-feature-cache",
        action="store_true",
    )

    return parser


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = build_argument_parser().parse_args()

    time_segments = resolve_cbramod_time_segments(
        window_seconds=args.window_seconds,
        target_sample_rate=args.target_sample_rate,
        points_per_patch=200,
    )

    window_tag = (
        f"{args.window_seconds:g}s_flatten"
    )

    set_seed(args.seed)

    target_subject = int(args.target_subject)
    if args.split_mode == "loso":
        subjects = normalize_subjects(args.subjects)
        if target_subject not in subjects:
            raise ValueError(
                "--target-subject must be included in --subjects. "
                f"Got target={target_subject}, subjects={subjects}."
            )
        population_subjects = [
            subject
            for subject in subjects
            if subject != target_subject
        ]
        if not population_subjects:
            raise ValueError(
                "LOSO population training requires at least one "
                "non-target subject."
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

    data_root = resolve_repo_path(args.data_root)
    checkpoint_path = resolve_repo_path(args.checkpoint)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "CBraMod backbone checkpoint was not found: "
            f"{checkpoint_path}"
        )

    default_run_dir = (
        ROOT
        / "runs"
        / "stage1"
        / (
            "bnci2014_001"
            if args.split_mode == "loso"
            else "within_subject"
        )
        / f"subject_{target_subject:02d}"
        / "cbramod"
        / window_tag
    )

    run_dir = (
        resolve_repo_path(args.run_dir)
        if args.run_dir is not None
        else default_run_dir
    )

    default_head_path = (
        ROOT
        / "checkpoints"
        / "heads"
        / "stage1"
        / (
            "bnci2014_001"
            if args.split_mode == "loso"
            else "within_subject"
        )
        / f"subject_{target_subject:02d}"
        / "cbramod"
        / window_tag
        / "head.pt"
    )

    output_head_path = (
        resolve_repo_path(args.output_head)
        if args.output_head is not None
        else default_head_path
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_names: tuple[str, ...] | None = None
    reference_metadata: HDF5Metadata | None = None
    subject_paths: dict[int, Path] = {}

    for subject_id in subjects:
        subject_path = resolve_subject_file(
            data_root=data_root,
            data_pattern=args.data_pattern,
            subject_id=subject_id,
        )

        subject_paths[subject_id] = subject_path

        metadata = EEGHDF5(subject_path).metadata

        if reference_metadata is None:
            reference_metadata = metadata
        else:
            validate_metadata_compatibility(
                reference_metadata,
                metadata,
                subject_id=subject_id,
                path=subject_path,
            )

        if class_names is None:
            class_names = resolve_class_names(
                metadata=metadata,
                explicit_class_names=args.class_names,
            )

    if reference_metadata is None or class_names is None:
        raise RuntimeError(
            "Could not resolve dataset metadata."
        )

    if len(class_names) != 4:
        raise ValueError(
            "CBRaMod classification requires four classes. Got "
            f"class_names={list(class_names)}."
        )

    label_mapping = {
        str(index): str(name)
        for index, name in enumerate(class_names)
    }

    config = CBraModConfig(
        checkpoint_path=checkpoint_path,
        classifier_path=output_head_path,
        device=args.device,

        target_sample_rate=args.target_sample_rate,
        window_seconds=args.window_seconds,
        n_channels=22,
        time_segments=time_segments,
        points_per_patch=200,
        input_unit=args.input_unit,

        strict_window_duration=True,
        filter_enabled=bool(args.filter_enabled),
        filter_low_hz=args.filter_low_hz,
        filter_high_hz=args.filter_high_hz,
        filter_order=args.filter_order,
        reference_mode=args.reference_mode,
        normalization=args.normalization,

        num_classes=len(class_names),
        head_type=args.head_type,
        head_hidden_dim_1=args.head_hidden_dim_1,
        head_hidden_dim_2=args.head_hidden_dim_2,
        head_dropout=args.head_dropout,
    )

    backbone_sha256 = sha256_file(checkpoint_path)

    preprocessing_manifest = {
        "standard_channels": list(
            config.standard_channels
        ),
        "target_sample_rate": (
            config.target_sample_rate
        ),
        "window_seconds": config.window_seconds,
        "time_segments": config.time_segments,
        "points_per_patch": (
            config.points_per_patch
        ),
        "input_unit": config.input_unit,
        "filter_enabled": config.filter_enabled,
        "filter_low_hz": config.filter_low_hz,
        "filter_high_hz": config.filter_high_hz,
        "filter_order": config.filter_order,
        "reference_mode": config.reference_mode,
        "normalization": config.normalization,
        "zscore_eps": config.zscore_eps,
        "strict_window_duration": (
            config.strict_window_duration
        ),
        "window_tolerance_seconds": (
            config.window_tolerance_seconds
        ),
        "training_source_trial_selection": {
            "policy": (
                "one_contiguous_window_per_source_trial"
            ),
            "anchor": args.direct_trial_anchor,
            "padding": False,
            "cross_trial_concatenation": False,
        },
        "missing_channel_policy": (
            config.missing_channel_policy
        ),
        "min_observed_channels": (
            config.min_observed_channels
        ),
        "spline_alpha": config.spline_alpha,
    }

    preprocessing_hash = stable_json_hash(
        preprocessing_manifest
    )

    run_config = {
        "model_name": "cbramod-frozen-head",
        "split_mode": args.split_mode,
        "target_subject": target_subject,
        "population_subjects": population_subjects,
        "train_session": args.train_session,
        "validation_session": args.validation_session,
        "final_test_session": (
            args.final_test_session
            if args.split_mode == "loso"
            else args.test_session
        ),
        "validation_ratio": (
            float(args.validation_ratio)
            if args.split_mode == "within-subject"
            else None
        ),
        "class_names": list(class_names),
        "label_mapping": label_mapping,
        "backbone_checkpoint": str(checkpoint_path),
        "backbone_sha256": backbone_sha256,
        "output_head": str(output_head_path),
        "preprocessing": preprocessing_manifest,
        "preprocessing_hash": preprocessing_hash,
        "head_type": args.head_type,
        "feature_batch_size": args.feature_batch_size,
        "head_batch_size": args.head_batch_size,
        "epochs": args.epochs,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "metric_for_best": args.metric_for_best,
        "seed": args.seed,
        "git_commit": current_git_commit(),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    dump_yaml(
        run_config,
        run_dir / "config.yaml",
    )

    atomic_write_json(
        run_dir / "preprocessing.json",
        preprocessing_manifest,
    )

    print("target subject:", target_subject)
    print("population subjects:", population_subjects)
    print("backbone:", checkpoint_path)
    print("device:", args.device)
    print("head type:", args.head_type)
    print("preprocessing hash:", preprocessing_hash)

    backbone = CBraModBackbone(config)

    # 明确断言群体基线中 backbone 完全冻结。
    frozen_parameters = [
        name
        for name, parameter in backbone.named_parameters()
        if parameter.requires_grad
    ]

    if frozen_parameters:
        raise RuntimeError(
            "CBraMod backbone must be frozen, but these "
            "parameters are trainable: "
            f"{frozen_parameters[:10]}."
        )

    canonicalizer = SignalCanonicalizer(
        target_unit=config.input_unit
    )

    preprocessor = CBraModPipelinePreprocessor(
        config
    )

    feature_cache_dir = (
        resolve_repo_path(args.feature_cache_dir)
        if args.feature_cache_dir is not None
        else None
    )

    def build_or_load_split(
        *,
        split_name: str,
        split_subjects: Sequence[int],
        session_name: str,
    ) -> FeatureSplit:
        cache_manifest = {
            "cache_format_version": 1,
            "split_name": split_name,
            "subjects": list(split_subjects),
            "session_name": session_name,
            "backbone_sha256": backbone_sha256,
            "preprocessing_hash": preprocessing_hash,
            "class_names": list(class_names),
        }

        cache_path = (
            feature_cache_dir
            / (
                f"{split_name}_"
                f"{window_tag}_"
                f"target_{target_subject:02d}_"
                f"seed_{args.seed}.pt"
            )
            if feature_cache_dir is not None
            else None
        )

        if (
            args.reuse_feature_cache
            and cache_path is not None
            and cache_path.is_file()
        ):
            print(
                f"loading feature cache: {cache_path}"
            )

            return load_feature_cache(
                path=cache_path,
                split_name=split_name,
                expected_manifest=cache_manifest,
            )

        subject_splits: list[FeatureSplit] = []

        for subject_id in split_subjects:
            print(
                f"extracting {split_name}: "
                f"subject_{subject_id:02d}/"
                f"{session_name}"
            )

            split, loaded_metadata = (
                build_subject_feature_split(
                    subject_id=subject_id,
                    session_name=session_name,
                    subject_path=subject_paths[
                        subject_id
                    ],
                    reference_metadata=reference_metadata,
                    canonicalizer=canonicalizer,
                    preprocessor=preprocessor,
                    backbone=backbone,
                    feature_batch_size=(
                        args.feature_batch_size
                    ),
                    direct_trial_anchor=(
                        args.direct_trial_anchor
                    ),
                )
            )

            validate_metadata_compatibility(
                reference_metadata,
                loaded_metadata,
                subject_id=subject_id,
                path=subject_paths[subject_id],
            )

            subject_splits.append(split)

        combined = combine_feature_splits(
            subject_splits,
            session_name=session_name,
        )

        if cache_path is not None:
            save_feature_cache(
                path=cache_path,
                split_name=split_name,
                split=combined,
                manifest=cache_manifest,
            )

            print(
                f"saved feature cache: {cache_path}"
            )

        return combined

    within_subject_split: WithinSubjectTrialSplit | None = None
    within_subject_all_trials: dict[str, np.ndarray] | None = None
    if args.split_mode == "loso":
        population_train = build_or_load_split(
            split_name="population_train",
            split_subjects=population_subjects,
            session_name=args.train_session,
        )
        population_validation = build_or_load_split(
            split_name="population_validation",
            split_subjects=population_subjects,
            session_name=args.validation_session,
        )
        final_test: FeatureSplit | None = build_or_load_split(
            split_name="target_final_test",
            split_subjects=[target_subject],
            session_name=args.final_test_session,
        )
        final_test_session = args.final_test_session
    else:
        print("Building within-subject training and validation features...")
        (
            population_train,
            population_validation,
            within_metadata,
            within_subject_split,
            within_subject_all_trials,
        ) = build_within_subject_train_validation_splits(
            subject_id=target_subject,
            subject_path=subject_paths[target_subject],
            train_session=args.train_session,
            test_session=str(args.test_session),
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            class_names=class_names,
            canonicalizer=canonicalizer,
            preprocessor=preprocessor,
            backbone=backbone,
            feature_batch_size=args.feature_batch_size,
            direct_trial_anchor=args.direct_trial_anchor,
        )
        validate_metadata_compatibility(
            reference_metadata,
            within_metadata,
            subject_id=target_subject,
            path=subject_paths[target_subject],
        )
        final_test = None
        final_test_session = within_subject_split.test_session
        print(f"Available sessions for subject {target_subject}:")
        for session_name in within_subject_split.available_sessions:
            print(f"- {session_name}")
        print(
            "within-subject train:",
            len(population_train.labels),
            class_counts(population_train.labels.numpy(), class_names),
        )
        print(
            "within-subject validation:",
            len(population_validation.labels),
            class_counts(population_validation.labels.numpy(), class_names),
        )
        test_labels = within_subject_all_trials["labels"][
            within_subject_split.test_indices
        ]
        print(
            "within-subject held-out test:",
            len(within_subject_split.test_indices),
            class_counts(test_labels, class_names),
        )

    validate_no_trial_leakage(
        population_train,
        population_validation,
        left_name="population_train",
        right_name="population_validation",
    )

    if final_test is not None:
        validate_no_trial_leakage(
            population_train,
            final_test,
            left_name="population_train",
            right_name="target_final_test",
        )
        validate_no_trial_leakage(
            population_validation,
            final_test,
            left_name="population_validation",
            right_name="target_final_test",
        )

    classifier = build_cbramod_classifier(config).to(
        backbone.device
    )

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.head_lr,
        weight_decay=args.weight_decay,
    )

    train_dataset = TensorDataset(
        population_train.features,
        population_train.labels,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.head_batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(
            args.seed
        ),
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_validation_metrics: Metrics | None = None
    best_epoch: int | None = None
    stale_epochs = 0

    training_history: list[dict[str, Any]] = []

    training_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_head_epoch(
            classifier=classifier,
            loader=train_loader,
            optimizer=optimizer,
            device=backbone.device,
        )

        validation_metrics = evaluate_head(
            classifier=classifier,
            split=population_validation,
            device=backbone.device,
            batch_size=args.head_batch_size,
            num_classes=config.num_classes,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "population_validation": (
                validation_metrics.to_dict()
            ),
        }

        training_history.append(epoch_record)

        current_metric = getattr(
            validation_metrics,
            args.metric_for_best,
        )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"val_bacc="
            f"{validation_metrics.balanced_accuracy:.4f} "
            f"val_macro_f1="
            f"{validation_metrics.macro_f1:.4f} "
            f"selected_{args.metric_for_best}="
            f"{current_metric:.6f}"
        )

        if metric_is_better(
            candidate=validation_metrics,
            best=best_validation_metrics,
            metric_name=args.metric_for_best,
        ):
            best_state = deepcopy(
                classifier.state_dict()
            )

            best_validation_metrics = validation_metrics
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

            if stale_epochs >= args.patience:
                print(
                    "early stopping: no validation improvement "
                    f"for {args.patience} epochs."
                )
                break

    training_seconds = (
        time.perf_counter() - training_started
    )

    if best_state is None or best_epoch is None:
        raise RuntimeError(
            "No best classifier checkpoint was selected."
        )

    classifier.load_state_dict(best_state)
    classifier.eval()

    if final_test is None:
        assert within_subject_split is not None
        assert within_subject_all_trials is not None
        print(
            "Within-subject model selection is complete. "
            "Extracting held-out final-test features..."
        )
        final_test = build_within_subject_final_test_split(
            subject_id=target_subject,
            subject_path=subject_paths[target_subject],
            metadata=reference_metadata,
            split=within_subject_split,
            all_trial_metadata=within_subject_all_trials,
            canonicalizer=canonicalizer,
            preprocessor=preprocessor,
            backbone=backbone,
            feature_batch_size=args.feature_batch_size,
            direct_trial_anchor=args.direct_trial_anchor,
        )
        validate_no_trial_leakage(
            population_train,
            final_test,
            left_name="population_train",
            right_name="target_final_test",
        )
        validate_no_trial_leakage(
            population_validation,
            final_test,
            left_name="population_validation",
            right_name="target_final_test",
        )

    final_test_metrics = evaluate_head(
        classifier=classifier,
        split=final_test,
        device=backbone.device,
        batch_size=args.head_batch_size,
        num_classes=config.num_classes,
    )

    if within_subject_split is None:
        within_subject_metadata: dict[str, Any] | None = None
    else:
        assert within_subject_all_trials is not None
        within_subject_metadata = {
            "subject": target_subject,
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
            "train_class_counts": class_counts(
                within_subject_all_trials["labels"][
                    within_subject_split.train_indices
                ],
                class_names,
            ),
            "validation_class_counts": class_counts(
                within_subject_all_trials["labels"][
                    within_subject_split.validation_indices
                ],
                class_names,
            ),
            "test_class_counts": class_counts(
                within_subject_all_trials["labels"][
                    within_subject_split.test_indices
                ],
                class_names,
            ),
        }

    saved_head_path = save_cbramod_classifier_checkpoint(
        classifier,
        output_head_path,
        config=config,
        class_names=class_names,
        extra_metadata={
            "split_mode": args.split_mode,
            "target_subject": target_subject,
            "population_training_subjects": (
                population_subjects
            ),
            "population_training_session": (
                args.train_session
            ),
            "population_validation_subjects": (
                population_subjects
            ),
            "population_validation_session": (
                args.validation_session
            ),
            "final_test_subject": target_subject,
            "final_test_session": (
                final_test_session
            ),
            "label_mapping": label_mapping,
            "within_subject_split": within_subject_metadata,
            "best_epoch": best_epoch,
            "best_metric_name": (
                args.metric_for_best
            ),
            "best_validation_metric": float(
                getattr(
                    best_validation_metrics,
                    args.metric_for_best,
                )
            ),
            "backbone_checkpoint": str(
                checkpoint_path
            ),
            "backbone_sha256": backbone_sha256,
            "preprocessing_hash": preprocessing_hash,
            "seed": args.seed,
        },
    )

    write_confusion_matrix_csv(
        path=run_dir / "final_confusion_matrix.csv",
        matrix=final_test_metrics.confusion_matrix,
        class_names=class_names,
    )

    report = {
        "model_name": "cbramod-frozen-head",
        "protocol": (
            "LOSO population head"
            if args.split_mode == "loso"
            else "within-subject cross-session head"
        ),
        "split_mode": args.split_mode,
        "target_subject": target_subject,
        "population_training_subjects": population_subjects,
        "population_training_session": (
            args.train_session
        ),
        "population_validation_subjects": (
            population_subjects
        ),
        "population_validation_session": (
            args.validation_session
        ),
        "final_test_subject": target_subject,
        "final_test_session": final_test_session,
        "class_names": list(class_names),
        "label_mapping": label_mapping,
        "num_classes": int(config.num_classes),
        "within_subject": within_subject_metadata,
        "seed": args.seed,
        "backbone_checkpoint": str(checkpoint_path),
        "backbone_sha256": backbone_sha256,
        "classifier_checkpoint": str(saved_head_path),
        "preprocessing": preprocessing_manifest,
        "preprocessing_hash": preprocessing_hash,
        "best_epoch": best_epoch,
        "training_seconds": training_seconds,
        "population_train": {
            "num_samples": int(
                population_train.features.shape[0]
            ),
            "per_class": class_counts(
                population_train.labels.numpy(),
                class_names,
            ),
            "subjects": sorted(
                set(
                    population_train.subject_ids.tolist()
                )
            ),
        },
        "population_validation": {
            "num_samples": int(
                population_validation.features.shape[0]
            ),
            "per_class": class_counts(
                population_validation.labels.numpy(),
                class_names,
            ),
            "subjects": sorted(
                set(
                    population_validation.subject_ids.tolist()
                )
            ),
            "best_metrics": (
                best_validation_metrics.to_dict()
            ),
        },
        "target_final_test": {
            "num_samples": int(
                final_test.features.shape[0]
            ),
            "per_class": class_counts(
                final_test.labels.numpy(),
                class_names,
            ),
            "subjects": sorted(
                set(final_test.subject_ids.tolist()
                )
            ),
            "metrics": final_test_metrics.to_dict(),
        },
        "training_history": training_history,
        "git_commit": current_git_commit(),
        "finished_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    atomic_write_json(
        run_dir / "training_report.json",
        report,
    )

    atomic_write_json(
        run_dir / "final_metrics.json",
        {
            "model_name": "cbramod-frozen-head",
            "split_mode": args.split_mode,
            "target_subject": target_subject,
            "train_session": args.train_session,
            "test_session": final_test_session,
            "validation_ratio": (
                float(args.validation_ratio)
                if args.split_mode == "within-subject"
                else None
            ),
            "within_subject": within_subject_metadata,
            "num_classes": int(config.num_classes),
            "class_names": list(class_names),
            "label_mapping": label_mapping,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "population_validation": (
                best_validation_metrics.to_dict()
            ),
            "target_final_test": (
                final_test_metrics.to_dict()
            ),
            "classifier_checkpoint": str(
                saved_head_path
            ),
            "backbone_sha256": backbone_sha256,
            "preprocessing_hash": preprocessing_hash,
        },
    )

    print()
    print("Training complete.")
    print("best epoch:", best_epoch)
    print(
        "best validation bACC:",
        f"{best_validation_metrics.balanced_accuracy:.4f}",
    )
    print(
        "final target-test bACC:",
        f"{final_test_metrics.balanced_accuracy:.4f}",
    )
    print(
        "final target-test macro-F1:",
        f"{final_test_metrics.macro_f1:.4f}",
    )
    print("saved head:", saved_head_path)
    print("run report:", run_dir / "training_report.json")


if __name__ == "__main__":
    main()
