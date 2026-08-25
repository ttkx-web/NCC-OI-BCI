"""Frozen 50M feature extraction and feature-cache artifacts.

This module deliberately handles only detached frozen-backbone features. Live
partial-finetune and LoRA forwards remain in the training runner until their
adaptation layer is extracted.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import TensorDataset

from bci_dayloop.data.trial_reader import DataReaderName
from bci_dayloop.models.model_50m.classifier import Model50MClassifier
from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.models.model_50m.preprocessing import Model50MPreprocessor
from bci_dayloop.models.model_50m.tokenization import Model50MTokenizer, stack_model50m_tokens


def _class_name_counts(labels: np.ndarray, class_names: Sequence[str]) -> dict[str, int]:
    return {
        str(class_name): int(np.sum(labels == index))
        for index, class_name in enumerate(class_names)
    }


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


def build_feature_cache_split_identity(
    *,
    split_mode: str,
    train_sessions: Sequence[str],
    test_session: str | None,
    validation_session: str | None,
    validation_ratio: float | None,
    split_seed: int | None,
) -> dict[str, Any]:
    """Return deterministic split provenance for a frozen feature artifact."""
    return {
        "split_mode": str(split_mode),
        "train_sessions": [str(session) for session in train_sessions],
        "test_session": None if test_session is None else str(test_session),
        "validation_session": (
            None if validation_session is None else str(validation_session)
        ),
        "validation_ratio": (
            None if validation_ratio is None else float(validation_ratio)
        ),
        "split_seed": None if split_seed is None else int(split_seed),
    }


def population_feature_cache_path(run_dir: Path, *, split_name: str) -> Path:
    """Return the established artifact name for a population feature cache."""
    names = {
        "population_train": "features_population_train.pt",
        "population_validation": "features_population_validation.pt",
        "target_final_test": "features_target_final_test.pt",
    }
    try:
        return run_dir / names[split_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported population feature-cache split: {split_name!r}.") from exc


def save_population_feature_cache(
    *,
    dataset: TensorDataset,
    bundle: WindowBundle,
    path: Path,
    split_name: str,
    class_names: Sequence[str],
    subject_ids: Sequence[int],
    data_reader: DataReaderName,
    subject_identities: Mapping[str, Mapping[str, int | str]],
    backbone_sha256: str,
    preprocessing_hash: str,
    split_identity: Mapping[str, Any],
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
            "data_reader": data_reader,
            "subject_identities": {
                str(subject): dict(subject_identities[str(subject)])
                for subject in subject_ids
            },
            "class_names": [str(name) for name in class_names],
            "class_counts": _class_name_counts(
                bundle.window_set.labels,
                class_names,
            ),
            "window_construction": bundle.window_set.construction,
            "source_trial_encoding": (
                "(subject_id << 32) | file_local_trial_id; Workload "
                "file_local_trial_id=(S<n> << 20) | trial_ordinal"
            ),
            "backbone_sha256": backbone_sha256,
            "preprocessing_hash": preprocessing_hash,
            "split_identity": dict(split_identity),
        },
        temporary,
    )
    temporary.replace(path)
