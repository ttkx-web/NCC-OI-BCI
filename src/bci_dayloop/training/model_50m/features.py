"""Frozen 50M feature extraction and feature-cache artifacts.

This module deliberately handles only detached frozen-backbone features. Live
partial-finetune and LoRA forwards are owned by ``adaptation.py``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class LoadedFeatureCache:
    """Strictly validated detached features and their persisted provenance."""

    dataset: TensorDataset
    window_subject_ids: torch.Tensor
    metadata: dict[str, Any]


def feature_cache_contract_hash(contract: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible cache contract with deterministic ordering."""

    encoded = json.dumps(
        dict(contract),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_feature_cache_contract(
    *, actual: Mapping[str, Any], expected: Mapping[str, Any], path: Path
) -> None:
    """Fail fast if any cache identity field differs from the requested run."""

    actual_hash = feature_cache_contract_hash(actual)
    expected_hash = feature_cache_contract_hash(expected)
    if actual_hash != expected_hash:
        differing = sorted(
            key
            for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        )
        raise ValueError(
            f"Feature cache contract mismatch at {path}; differing fields: "
            f"{differing}. Refusing to reuse stale or incompatible features."
        )


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
    cache_contract: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> None:
    features, labels = dataset.tensors
    if len(features) != len(bundle.window_set.windows):
        raise ValueError(
            f"{split_name}: feature count {len(features)} does not match "
            f"window count {len(bundle.window_set.windows)}."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    payload: dict[str, Any] = {
        "format_version": 2 if cache_contract is not None else 1,
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
    }
    if cache_contract is not None:
        payload["cache_contract"] = dict(cache_contract)
        payload["cache_contract_hash"] = feature_cache_contract_hash(cache_contract)
    if extra_metadata is not None:
        payload["extra_metadata"] = dict(extra_metadata)
    torch.save(payload, temporary)
    temporary.replace(path)


def _safe_load_cache(path: Path, *, mmap: bool) -> Any:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(path, weights_only=True, **kwargs)
    except TypeError:
        kwargs.pop("mmap", None)
        return torch.load(path, **kwargs)


def load_population_feature_cache(
    path: str | Path,
    *,
    expected_contract: Mapping[str, Any],
    expected_split: str,
    expected_subject: int | None = None,
    mmap: bool = True,
) -> LoadedFeatureCache:
    """Load one cache shard only after validating its complete identity."""

    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Feature cache shard not found: {target}")
    payload = _safe_load_cache(target, mmap=mmap)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Feature cache must be a mapping: {target}")
    if int(payload.get("format_version", 0)) < 2:
        raise ValueError(
            f"Feature cache {target} predates the strict cache contract; "
            "refusing to reuse it."
        )
    if str(payload.get("split")) != expected_split:
        raise ValueError(
            f"Feature cache split mismatch at {target}: "
            f"{payload.get('split')!r} != {expected_split!r}."
        )
    actual_contract = payload.get("cache_contract")
    if not isinstance(actual_contract, Mapping):
        raise ValueError(f"Feature cache has no cache_contract: {target}")
    validate_feature_cache_contract(
        actual=actual_contract,
        expected=expected_contract,
        path=target,
    )
    stored_hash = str(payload.get("cache_contract_hash", ""))
    actual_hash = feature_cache_contract_hash(actual_contract)
    if stored_hash != actual_hash:
        raise ValueError(f"Feature cache contract hash is corrupt at {target}.")

    features = payload.get("features")
    labels = payload.get("labels")
    subject_ids = payload.get("window_subject_ids")
    if not all(
        isinstance(value, torch.Tensor)
        for value in (features, labels, subject_ids)
    ):
        raise TypeError(f"Feature cache tensors are missing at {target}.")
    assert isinstance(features, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert isinstance(subject_ids, torch.Tensor)
    expected_dim = int(expected_contract["feature_dim"])
    if features.ndim != 2 or features.shape[1] != expected_dim:
        raise ValueError(
            f"Feature shape mismatch at {target}: {tuple(features.shape)}, "
            f"expected [N,{expected_dim}]."
        )
    if labels.shape != (len(features),) or subject_ids.shape != (len(features),):
        raise ValueError(f"Feature cache vector lengths differ at {target}.")
    if expected_subject is not None:
        observed = set(subject_ids.to(torch.int64).tolist())
        if observed != {int(expected_subject)}:
            raise ValueError(
                f"Feature cache subject mismatch at {target}: "
                f"{sorted(observed)} != {[int(expected_subject)]}."
            )
    if not torch.isfinite(features.float()).all():
        raise ValueError(f"Feature cache contains NaN or Inf at {target}.")
    metadata = {
        str(key): value
        for key, value in payload.items()
        if key not in {"features", "labels"}
    }
    return LoadedFeatureCache(
        dataset=TensorDataset(features, labels.to(torch.int64)),
        window_subject_ids=subject_ids.to(torch.int64),
        metadata=metadata,
    )
