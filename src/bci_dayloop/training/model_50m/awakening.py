"""Awakening orchestration built from the shared 50M training components.

This module owns dataset-contract checks and phase orchestration only.  Signal
preprocessing, tokenization, backbone execution, feature aggregation, head
training, checkpoint IO, and the established metrics remain in their existing
shared modules.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from bci_dayloop.data.dataset_adapter_registry import (
    DEFAULT_DATASET_ADAPTER_REGISTRY,
    LegacyHDF5Adapter,
    inspect_hdf5_dataset,
)
from bci_dayloop.data.dataset_registry import AWAKENING_2S, DatasetSpec
from bci_dayloop.data.hdf5_dataset import HDF5Metadata
from bci_dayloop.models.model_50m.backbone import Model50MBackbone
from bci_dayloop.models.model_50m.classifier import (
    Model50MClassifier,
    load_classifier_checkpoint,
    save_classifier_checkpoint,
)
from bci_dayloop.models.model_50m.config import Model50MConfig, STANDARD_64_CHANNELS
from bci_dayloop.training.model_50m.artifacts import (
    atomic_write_json,
    sha256_file,
    stable_json_hash,
)
from bci_dayloop.training.model_50m.data import build_subject_window_bundle
from bci_dayloop.training.model_50m.engine import (
    build_optimizer,
    fit_with_early_stopping,
)
from bci_dayloop.training.model_50m.evaluation import (
    evaluate_binary_feature_dataset,
    extend_metrics,
)
from bci_dayloop.training.model_50m.features import (
    extract_frozen_features,
    feature_cache_contract_hash,
    feature_cache_dtype_from_name,
    load_population_feature_cache,
    save_population_feature_cache,
    validate_feature_cache_contract,
)
from bci_dayloop.training.model_50m.linear_head import resolve_repo_path, set_seed


PREPROCESSING_VERSION = "model50m_preprocessing_v1"
CACHE_MANIFEST_NAME = "cache_manifest.json"
DATA_MANIFEST_FILES = (
    "subject_mapping.json",
    "split_manifest.json",
    "conversion_report.json",
)
REQUIRED_DATASETS = (
    "data",
    "labels",
    "subject_ids",
    "session_ids",
    "trial_ids",
    "source_lance_row_ids",
    "source_subwindow_indices",
    "source_sample_ids",
)


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    data_root: Path
    subject_paths: dict[int, Path]
    window_count: int
    counts_by_session: dict[str, int]
    counts_by_class: dict[str, int]
    counts_by_subject: dict[str, dict[str, Any]]
    data_manifest_hashes: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_root": str(self.data_root),
            "subject_paths": {
                str(subject): str(path)
                for subject, path in self.subject_paths.items()
            },
            "window_count": self.window_count,
            "counts_by_session": self.counts_by_session,
            "counts_by_class": self.counts_by_class,
            "counts_by_subject": self.counts_by_subject,
            "data_manifest_hashes": self.data_manifest_hashes,
        }


@dataclass(frozen=True, slots=True)
class CacheBundle:
    manifest: dict[str, Any]
    contract: dict[str, Any]
    dataset: ConcatDataset
    subject_ids: torch.Tensor
    labels: torch.Tensor


def awakening_model_config(
    *, checkpoint: str | Path, device: str = "cpu"
) -> Model50MConfig:
    """Build the exact two-second frozen-50M contract."""

    return Model50MConfig(
        checkpoint_path=resolve_repo_path(checkpoint),
        classifier_path=None,
        device=device,
        target_sample_rate=100.0,
        window_seconds=2.0,
        patch_seconds=1.0,
        patch_stride_seconds=1.0,
        filter_enabled=True,
        filter_low_hz=0.1,
        filter_high_hz=75.0,
        reference_mode="none",
        strict_window_duration=True,
        output_layer_idx=8,
        aggregation="flatten",
        num_classes=2,
        model_n_time_patches=10,
        head_type="linear",
        head_dropout=0.0,
        head_norm="none",
    )


def channel_profile_contract(spec: DatasetSpec = AWAKENING_2S) -> dict[str, Any]:
    mapped = tuple(name for name in spec.source_channels if name in STANDARD_64_CHANNELS)
    ignored = tuple(name for name in spec.source_channels if name not in STANDARD_64_CHANNELS)
    missing = tuple(name for name in STANDARD_64_CHANNELS if name not in spec.source_channels)
    if len(mapped) != 60 or set(ignored) != set(spec.ignored_source_channels):
        raise RuntimeError(
            "Awakening source-to-model channel profile drifted: "
            f"mapped={len(mapped)}, ignored={ignored}."
        )
    if missing != spec.missing_model_channels:
        raise RuntimeError(
            "Awakening missing-model-channel profile drifted: "
            f"{missing} != {spec.missing_model_channels}."
        )
    return {
        "id": spec.channel_profile,
        "source_channels": list(spec.source_channels),
        "target_channels": list(STANDARD_64_CHANNELS),
        "mapped_channel_count": len(mapped),
        "ignored_source_channels": list(spec.ignored_source_channels),
        "missing_model_channels": list(missing),
        "missing_channel_strategy": "zero_with_valid_mask",
    }


def preprocessing_contract(config: Model50MConfig) -> dict[str, Any]:
    return {
        "version": PREPROCESSING_VERSION,
        "source_sample_rate": AWAKENING_2S.sample_rate,
        "target_sample_rate": float(config.target_sample_rate),
        "window_seconds": float(config.window_seconds),
        "source_shape": [len(AWAKENING_2S.source_channels), 400],
        "target_shape": [config.n_channels, config.target_num_points],
        "input_unit": AWAKENING_2S.unit,
        "unit_scaling": "uV_to_uV_once",
        "channel_profile": channel_profile_contract(),
        "filter_enabled": bool(config.filter_enabled),
        "filter_low_hz": float(config.filter_low_hz),
        "filter_high_hz": float(config.filter_high_hz),
        "reference_mode": config.reference_mode,
        "resampler": "scipy.signal.resample_poly",
        "expected_resample_count": 1,
        "temporal_padding_allowed": False,
        "temporal_cropping_allowed": False,
        "patch_seconds": float(config.patch_seconds),
        "patch_stride_seconds": float(config.patch_stride_seconds),
        "num_tokens": int(config.num_tokens),
        "aggregation": config.aggregation,
        "feature_dim": int(config.classifier_input_dim),
    }


def _json_attr(attrs: h5py.AttributeManager, name: str) -> Any:
    if name not in attrs:
        raise ValueError(f"Missing HDF5 attribute: {name}.")
    value = attrs[name]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def _manifest_hashes(data_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in DATA_MANIFEST_FILES:
        path = data_root / name
        if not path.is_file():
            raise FileNotFoundError(f"Awakening data manifest is missing: {path}")
        hashes[name] = sha256_file(path)
    return hashes


def validate_awakening_dataset(
    data_root: str | Path,
    *,
    subjects: Sequence[int] = AWAKENING_2S.canonical_subjects,
    require_manifests: bool = True,
) -> DatasetAudit:
    """Validate schema, metadata, classes, and parent-group split isolation.

    EEG values are not read.  Only HDF5 headers and trial-level vectors are
    inspected, so this preflight remains lightweight for the full dataset.
    """

    root = resolve_repo_path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Awakening HDF5 directory not found: {root}")
    normalized_subjects = tuple(sorted(set(int(value) for value in subjects)))
    if not normalized_subjects:
        raise ValueError("At least one Awakening subject is required.")
    unexpected = sorted(set(normalized_subjects) - set(AWAKENING_2S.canonical_subjects))
    if unexpected:
        raise ValueError(f"Unregistered Awakening subject IDs: {unexpected}.")

    total = 0
    counts_by_session = {"S1": 0, "S2": 0}
    counts_by_class = {name: 0 for name in AWAKENING_2S.class_names}
    counts_by_subject: dict[str, dict[str, Any]] = {}
    subject_paths: dict[int, Path] = {}
    parent_assignments: dict[int, tuple[int, str, int]] = {}
    parent_subwindows: dict[int, set[int]] = {}

    for subject in normalized_subjects:
        path = root / AWAKENING_2S.data_pattern.format(subject=subject)
        if not path.is_file():
            raise FileNotFoundError(f"Awakening subject HDF5 not found: {path}")
        subject_paths[subject] = path
        descriptor = inspect_hdf5_dataset(path)
        adapter = DEFAULT_DATASET_ADAPTER_REGISTRY.resolve(descriptor)
        if not isinstance(adapter, LegacyHDF5Adapter):
            raise ValueError(f"{path}: Awakening must resolve to LegacyHDF5Adapter.")

        with h5py.File(path, "r") as handle:
            missing = [name for name in REQUIRED_DATASETS if name not in handle]
            if missing:
                raise ValueError(f"{path}: missing datasets {missing}.")
            data = handle["data"]
            n = int(data.shape[0])
            expected_shapes = {
                "data": (n, 62, 400),
                "labels": (n,),
                "subject_ids": (n,),
                "session_ids": (n,),
                "trial_ids": (n,),
                "source_lance_row_ids": (n,),
                "source_subwindow_indices": (n,),
                "source_sample_ids": (n,),
            }
            for name, expected in expected_shapes.items():
                if handle[name].shape != expected:
                    raise ValueError(
                        f"{path}: /{name} shape {handle[name].shape} != {expected}."
                    )
            if data.dtype != np.dtype("float32"):
                raise ValueError(f"{path}: /data must be float32, got {data.dtype}.")
            for name in ("labels", "subject_ids", "trial_ids", "source_lance_row_ids", "source_subwindow_indices"):
                if handle[name].dtype != np.dtype("int64"):
                    raise ValueError(f"{path}: /{name} must be int64.")

            attrs = handle.attrs
            if str(attrs.get("dataset_name", "")) != AWAKENING_2S.name:
                raise ValueError(f"{path}: dataset_name must be 'awakening'.")
            if not np.isclose(float(attrs.get("sample_rate", -1)), AWAKENING_2S.sample_rate):
                raise ValueError(f"{path}: sample_rate must be 200 Hz.")
            if not np.isclose(float(attrs.get("window_seconds", -1)), 2.0):
                raise ValueError(f"{path}: window_seconds must be 2 seconds.")
            if int(attrs.get("samples_per_window", -1)) != 400:
                raise ValueError(f"{path}: samples_per_window must be 400.")
            if str(attrs.get("unit", "")) != AWAKENING_2S.unit:
                raise ValueError(f"{path}: unit must be 'uV'.")
            if not bool(attrs.get("no_temporal_padding", False)):
                raise ValueError(f"{path}: no_temporal_padding must be true.")
            if str(attrs.get("session_semantics", "")) != AWAKENING_2S.session_semantics:
                raise ValueError(f"{path}: unexpected logical-session semantics.")
            if tuple(_json_attr(attrs, "channel_names")) != AWAKENING_2S.source_channels:
                raise ValueError(f"{path}: source channel order mismatch.")
            if tuple(_json_attr(attrs, "class_names")) != AWAKENING_2S.class_names:
                raise ValueError(f"{path}: class order mismatch.")
            if _json_attr(attrs, "label_mapping") != {"0": "non_awakening", "1": "awakening"}:
                raise ValueError(f"{path}: label_mapping mismatch.")

            labels = handle["labels"][:].astype(np.int64, copy=False)
            persisted_subjects = handle["subject_ids"][:].astype(np.int64, copy=False)
            sessions = handle["session_ids"].asstr()[:]
            parents = handle["source_lance_row_ids"][:].astype(np.int64, copy=False)
            subwindows = handle["source_subwindow_indices"][:].astype(np.int64, copy=False)
            trial_ids = handle["trial_ids"][:].astype(np.int64, copy=False)
            if set(persisted_subjects.tolist()) != {subject}:
                raise ValueError(f"{path}: canonical subject_ids mismatch.")
            if not set(labels.tolist()).issubset({0, 1}):
                raise ValueError(f"{path}: labels must be 0 or 1.")
            if set(sessions.tolist()) != {"S1", "S2"}:
                raise ValueError(f"{path}: both logical splits S1 and S2 are required.")
            if np.any((subwindows < 0) | (subwindows > 4)):
                raise ValueError(f"{path}: source_subwindow_indices must be in [0,4].")
            if len(np.unique(trial_ids)) != n:
                raise ValueError(f"{path}: trial_ids must be unique within the file.")
            for parent, session, label, subwindow in zip(
                parents.tolist(),
                sessions.tolist(),
                labels.tolist(),
                subwindows.tolist(),
            ):
                identity = (subject, str(session), int(label))
                previous = parent_assignments.setdefault(int(parent), identity)
                if previous != identity:
                    raise ValueError(
                        "source_lance_row_id crosses subject/logical split or label: "
                        f"{parent} appears in {previous} and {identity}."
                    )
                seen_subwindows = parent_subwindows.setdefault(int(parent), set())
                if int(subwindow) in seen_subwindows:
                    raise ValueError(
                        f"source_lance_row_id {parent} repeats subwindow {subwindow}."
                    )
                seen_subwindows.add(int(subwindow))

            subject_counts = {
                "total": n,
                "sessions": {session: int(np.sum(sessions == session)) for session in ("S1", "S2")},
                "classes": {
                    name: int(np.sum(labels == index))
                    for index, name in enumerate(AWAKENING_2S.class_names)
                },
            }
            counts_by_subject[str(subject)] = subject_counts
            total += n
            for session in counts_by_session:
                counts_by_session[session] += subject_counts["sessions"][session]
            for name in counts_by_class:
                counts_by_class[name] += subject_counts["classes"][name]

    if any(value <= 0 for value in counts_by_class.values()):
        raise ValueError(
            "The selected Awakening dataset must contain both classes globally; "
            f"counts={counts_by_class}."
        )
    for session in ("S1", "S2"):
        session_classes = {0: 0, 1: 0}
        for subject, path in subject_paths.items():
            del subject
            with h5py.File(path, "r") as handle:
                sessions = handle["session_ids"].asstr()[:]
                labels = handle["labels"][:]
                mask = sessions == session
                for label in session_classes:
                    session_classes[label] += int(np.sum(labels[mask] == label))
        if any(value <= 0 for value in session_classes.values()):
            raise ValueError(
                f"Logical split {session} must contain both classes globally; "
                f"counts={session_classes}."
            )

    return DatasetAudit(
        data_root=root,
        subject_paths=subject_paths,
        window_count=total,
        counts_by_session=counts_by_session,
        counts_by_class=counts_by_class,
        counts_by_subject=counts_by_subject,
        data_manifest_hashes=(_manifest_hashes(root) if require_manifests else {}),
    )


def build_cache_contract(
    *,
    audit: DatasetAudit,
    checkpoint_path: str | Path,
    target_subject: int,
    subjects: Sequence[int],
    window_order_seed: int = 42,
    feature_cache_dtype: str = "float16",
) -> dict[str, Any]:
    checkpoint = resolve_repo_path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"50M backbone checkpoint not found: {checkpoint}")
    config = awakening_model_config(checkpoint=checkpoint)
    normalized = tuple(sorted(set(int(value) for value in subjects)))
    if int(target_subject) not in normalized:
        raise ValueError("target_subject must be included in subjects.")
    return {
        "schema_version": 1,
        "dataset_name": AWAKENING_2S.name,
        "data_manifest_hashes": audit.data_manifest_hashes,
        "subjects": list(normalized),
        "target_subject": int(target_subject),
        "window_order_seed": int(window_order_seed),
        "feature_cache_dtype": str(feature_cache_dtype),
        "split": {
            "mode": "loso",
            "population_train": "S1",
            "population_validation": "S2",
            "target_final_test": "S2",
            "session_semantics": AWAKENING_2S.session_semantics,
            "group_key": "source_lance_row_id",
        },
        "window_seconds": 2.0,
        "source_sample_rate": 200.0,
        "target_sample_rate": 100.0,
        "channel_profile": channel_profile_contract(),
        "preprocessing": preprocessing_contract(config),
        "preprocessing_hash": stable_json_hash(preprocessing_contract(config)),
        "backbone_checkpoint": str(checkpoint),
        "backbone_checkpoint_sha256": sha256_file(checkpoint),
        "output_layer_idx": int(config.output_layer_idx),
        "aggregation": config.aggregation,
        "feature_dim": int(config.classifier_input_dim),
        "class_names": list(AWAKENING_2S.class_names),
        "label_mapping": {"0": "non_awakening", "1": "awakening"},
        "positive_class_index": AWAKENING_2S.positive_class_index,
    }


def cache_shard_relative_path(*, split_name: str, subject: int) -> Path:
    return Path(split_name) / f"subject_{subject:02d}.pt"


def _split_subject_sessions(
    *, subjects: Sequence[int], target_subject: int
) -> tuple[tuple[str, int, str], ...]:
    population = tuple(subject for subject in subjects if subject != target_subject)
    if not population:
        raise ValueError("LOSO caching requires at least one non-target subject.")
    rows = [
        *(('population_train', subject, 'S1') for subject in population),
        *(('population_validation', subject, 'S2') for subject in population),
        ('target_final_test', target_subject, 'S2'),
    ]
    return tuple(rows)


def generate_awakening_feature_cache(
    *,
    data_root: str | Path,
    cache_dir: str | Path,
    checkpoint_path: str | Path,
    subjects: Sequence[int],
    target_subject: int,
    device: str,
    feature_batch_size: int,
    cache_dtype_name: str,
    seed: int,
    overwrite: bool,
    max_windows_per_class_per_subject: int | None = None,
) -> Path:
    """Generate strict subject-sharded caches without training a head."""

    if feature_batch_size <= 0:
        raise ValueError("feature_batch_size must be positive.")
    normalized = tuple(sorted(set(int(value) for value in subjects)))
    audit = validate_awakening_dataset(data_root, subjects=normalized)
    contract = build_cache_contract(
        audit=audit,
        checkpoint_path=checkpoint_path,
        target_subject=target_subject,
        subjects=normalized,
        window_order_seed=seed,
        feature_cache_dtype=cache_dtype_name,
    )
    target = resolve_repo_path(cache_dir)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"Feature cache directory already exists: {target}. Use --overwrite explicitly."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    dtype = feature_cache_dtype_from_name(cache_dtype_name)
    bytes_per_value = torch.empty((), dtype=dtype).element_size()
    estimate = audit.window_count * int(contract["feature_dim"]) * bytes_per_value
    free = shutil.disk_usage(target.parent).free
    if free < int(estimate * 1.15):
        raise OSError(
            "Insufficient free space for Awakening feature cache: "
            f"estimated={estimate / 1024**3:.2f} GiB, free={free / 1024**3:.2f} GiB."
        )

    staging = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    backup: Path | None = None
    staging.mkdir(parents=False, exist_ok=False)
    try:
        config = awakening_model_config(checkpoint=checkpoint_path, device=device)
        backbone = Model50MBackbone(config=config, load_checkpoint=True, freeze=True)
        classifier = Model50MClassifier(config=config, backbone=backbone)
        classifier.eval()
        shards: list[dict[str, Any]] = []
        reference_metadata: HDF5Metadata | None = None
        split_rows = _split_subject_sessions(
            subjects=normalized, target_subject=int(target_subject)
        )
        for ordinal, (split_name, subject, session) in enumerate(split_rows):
            bundle, metadata, summary = build_subject_window_bundle(
                subject_id=subject,
                path=audit.subject_paths[subject],
                data_reader="eeg",
                session_name=session,
                reference_metadata=reference_metadata,
                window_seconds=2.0,
                stride_seconds=2.0,
                seed=seed + ordinal * 100,
                shuffle_trials_within_class=(split_name == "population_train"),
                max_windows_per_class=max_windows_per_class_per_subject,
                window_construction="direct_trial",
                direct_trial_anchor="start",
                require_all_classes=False,
            )
            if reference_metadata is None:
                reference_metadata = metadata
            features = extract_frozen_features(
                window_set=bundle.window_set,
                metadata=metadata,
                config=config,
                classifier=classifier,
                preprocess_batch_size=feature_batch_size,
                cache_dtype=dtype,
                split_name=f"{split_name}/subject_{subject:02d}",
                log_every=10,
            )
            relative = cache_shard_relative_path(
                split_name=split_name, subject=subject
            )
            with h5py.File(audit.subject_paths[subject], "r") as source_handle:
                source_subject_id = str(
                    source_handle.attrs.get("source_subject_id", subject)
                )
            save_population_feature_cache(
                dataset=features,
                bundle=bundle,
                path=staging / relative,
                split_name=split_name,
                class_names=AWAKENING_2S.class_names,
                subject_ids=[subject],
                data_reader="eeg",
                subject_identities={
                    str(subject): {
                        "canonical_subject_id": subject,
                        "source_subject_id": source_subject_id,
                    }
                },
                backbone_sha256=str(contract["backbone_checkpoint_sha256"]),
                preprocessing_hash=str(contract["preprocessing_hash"]),
                split_identity=contract["split"],
                cache_contract=contract,
                extra_metadata={
                    "canonical_subject_id": subject,
                    "logical_session": session,
                    "session_semantics": AWAKENING_2S.session_semantics,
                    "source_summary": summary,
                },
            )
            labels = features.tensors[1]
            shards.append(
                {
                    "path": str(relative),
                    "split": split_name,
                    "logical_session": session,
                    "subject": subject,
                    "samples": len(features),
                    "class_counts": {
                        name: int((labels == index).sum().item())
                        for index, name in enumerate(AWAKENING_2S.class_names)
                    },
                }
            )
            del features, bundle

        manifest = {
            "schema_version": 1,
            "created_unix_time": time.time(),
            "complete": max_windows_per_class_per_subject is None,
            "debug_max_windows_per_class_per_subject": max_windows_per_class_per_subject,
            "cache_contract": contract,
            "cache_contract_hash": feature_cache_contract_hash(contract),
            "shards": shards,
            "total_samples": sum(int(item["samples"]) for item in shards),
        }
        atomic_write_json(staging / CACHE_MANIFEST_NAME, manifest)
        if target.exists():
            backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
            os.replace(target, backup)
        os.replace(staging, target)
        if backup is not None:
            shutil.rmtree(backup)
        return target
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise


def read_cache_manifest(cache_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    root = resolve_repo_path(cache_dir)
    path = root / CACHE_MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Feature cache manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Feature cache manifest must be a mapping: {path}")
    return root, payload


def load_cache_split(
    *,
    cache_dir: str | Path,
    expected_contract: Mapping[str, Any],
    split_name: str,
) -> CacheBundle:
    root, manifest = read_cache_manifest(cache_dir)
    actual_contract = manifest.get("cache_contract")
    if not isinstance(actual_contract, Mapping):
        raise ValueError("Feature cache manifest has no cache_contract.")
    validate_feature_cache_contract(
        actual=actual_contract,
        expected=expected_contract,
        path=root / CACHE_MANIFEST_NAME,
    )
    if str(manifest.get("cache_contract_hash", "")) != feature_cache_contract_hash(actual_contract):
        raise ValueError("Feature cache manifest contract hash is corrupt.")
    if manifest.get("complete") is not True:
        raise ValueError(
            "Feature cache is marked incomplete/debug-only and cannot be used "
            "for training or formal evaluation."
        )
    rows = [row for row in manifest.get("shards", []) if row.get("split") == split_name]
    if not rows:
        raise ValueError(f"No cache shards found for split {split_name!r}.")
    all_subjects = {int(value) for value in expected_contract["subjects"]}
    target_subject = int(expected_contract["target_subject"])
    expected_subjects = (
        {target_subject}
        if split_name == "target_final_test"
        else all_subjects - {target_subject}
    )
    row_subjects = {int(row["subject"]) for row in rows}
    if row_subjects != expected_subjects:
        raise ValueError(
            f"Feature cache subjects for {split_name} are {sorted(row_subjects)}, "
            f"expected {sorted(expected_subjects)}."
        )
    expected_session = "S1" if split_name == "population_train" else "S2"
    datasets: list[TensorDataset] = []
    subjects: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for row in sorted(rows, key=lambda value: int(value["subject"])):
        loaded = load_population_feature_cache(
            root / str(row["path"]),
            expected_contract=expected_contract,
            expected_split=split_name,
            expected_subject=int(row["subject"]),
        )
        extra = loaded.metadata.get("extra_metadata")
        if not isinstance(extra, Mapping):
            raise ValueError(f"Cache shard {row['path']} has no extra_metadata.")
        if str(row.get("logical_session")) != expected_session or str(
            extra.get("logical_session")
        ) != expected_session:
            raise ValueError(
                f"Cache shard {row['path']} does not belong to logical "
                f"session {expected_session}."
            )
        if int(row.get("samples", -1)) != len(loaded.dataset):
            raise ValueError(f"Cache shard sample count mismatch: {row['path']}.")
        datasets.append(loaded.dataset)
        subjects.append(loaded.window_subject_ids)
        labels.append(loaded.dataset.tensors[1])
    return CacheBundle(
        manifest=manifest,
        contract=dict(actual_contract),
        dataset=ConcatDataset(datasets),
        subject_ids=torch.cat(subjects),
        labels=torch.cat(labels),
    )


def _class_weights(labels: torch.Tensor, *, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels.to(torch.int64), minlength=num_classes).float()
    if torch.any(counts <= 0):
        raise ValueError(f"Training split must contain every class; counts={counts.tolist()}.")
    weights = counts.sum() / (num_classes * counts)
    return weights / weights.mean()


def train_awakening_head_from_cache(
    *,
    data_root: str | Path,
    cache_dir: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    run_dir: str | Path,
    subjects: Sequence[int],
    target_subject: int,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    seed: int,
    cache_seed: int,
    feature_cache_dtype: str,
    class_weight: str,
    overwrite: bool,
) -> Path:
    """Train only the shared linear head from already frozen features."""

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    audit = validate_awakening_dataset(data_root, subjects=subjects)
    contract = build_cache_contract(
        audit=audit,
        checkpoint_path=checkpoint_path,
        target_subject=target_subject,
        subjects=subjects,
        window_order_seed=cache_seed,
        feature_cache_dtype=feature_cache_dtype,
    )
    train = load_cache_split(
        cache_dir=cache_dir, expected_contract=contract, split_name="population_train"
    )
    validation = load_cache_split(
        cache_dir=cache_dir,
        expected_contract=contract,
        split_name="population_validation",
    )
    if set(train.labels.tolist()) != {0, 1} or set(validation.labels.tolist()) != {0, 1}:
        raise ValueError("Population train and validation splits must each contain both classes.")

    output = resolve_repo_path(output_path)
    run = resolve_repo_path(run_dir)
    if (output.exists() or run.exists()) and not overwrite:
        raise FileExistsError("Training output exists; pass --overwrite explicitly to replace it.")
    run.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    config = awakening_model_config(checkpoint=checkpoint_path, device=device)
    backbone = Model50MBackbone(config=config, load_checkpoint=True, freeze=True)
    classifier = Model50MClassifier(config=config, backbone=backbone)
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise RuntimeError("Awakening linear-head training requires a frozen backbone.")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train.dataset,
        batch_size=min(batch_size, len(train.dataset)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(
        validation.dataset,
        batch_size=min(batch_size, len(validation.dataset)),
        shuffle=False,
        num_workers=0,
    )
    if class_weight == "balanced":
        loss_weights = _class_weights(train.labels, num_classes=2).to(classifier.device)
    elif class_weight == "none":
        loss_weights = None
    else:
        raise ValueError("class_weight must be 'balanced' or 'none'.")
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    head_parameters = list(classifier.head.parameters())
    optimizer = build_optimizer(
        classifier=classifier,
        backbone=backbone,
        head_parameters=head_parameters,
        trainable_backbone_parameters=[],
        lora_parameters=[],
        lora_parameter_ids=set(),
        head_lr=learning_rate,
        backbone_lr=1e-4,
        lora_lr=5e-4,
        weight_decay=weight_decay,
        partial_enabled=False,
        lora_enabled=False,
    )
    result = fit_with_early_stopping(
        classifier=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        num_classes=2,
        class_names=AWAKENING_2S.class_names,
        optimizer=optimizer,
        epochs=epochs,
        patience=patience,
        metric_for_best="val_bacc",
        live=False,
        partial_enabled=False,
        lora_enabled=False,
        extend_metrics=extend_metrics,
    )
    metadata = {
        "task_id": "awakening",
        "task": "awakening_classification",
        "dataset": "awakening",
        "class_names": list(AWAKENING_2S.class_names),
        "class_order": list(AWAKENING_2S.class_names),
        "label_mapping": {"0": "non_awakening", "1": "awakening"},
        "positive_class_index": 1,
        "feature_dim": 65536,
        "window_seconds": 2.0,
        "source_sample_rate": 200.0,
        "target_sample_rate": 100.0,
        "channel_profile": channel_profile_contract(),
        "split_semantics": AWAKENING_2S.session_semantics,
        "split_protocol": {
            "mode": "loso",
            "population_train": "non-target subjects S1",
            "population_validation": "non-target subjects S2",
            "target_final_test": "target subject S2",
        },
        "preprocessing": preprocessing_contract(config),
        "preprocessing_hash": contract["preprocessing_hash"],
        "backbone_checkpoint": contract["backbone_checkpoint"],
        "backbone_checkpoint_sha256": contract["backbone_checkpoint_sha256"],
        "feature_cache_contract_hash": feature_cache_contract_hash(contract),
        "seed": int(seed),
        "training_parameters": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "patience": int(patience),
            "class_weight": class_weight,
            "resolved_class_weights": None if loss_weights is None else loss_weights.cpu().tolist(),
            "backbone_frozen": True,
        },
    }
    saved = save_classifier_checkpoint(classifier, output, extra_metadata=metadata)
    with (run / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.epoch_rows[0]))
        writer.writeheader()
        writer.writerows(result.epoch_rows)
    atomic_write_json(
        run / "training_report.json",
        {
            **metadata,
            "checkpoint": str(saved),
            "best_epoch": result.best_epoch,
            "selected_validation": result.selected_val_metrics.to_dict(),
            "train_class_counts": torch.bincount(train.labels, minlength=2).tolist(),
            "validation_class_counts": torch.bincount(validation.labels, minlength=2).tolist(),
        },
    )
    return saved


def evaluate_awakening_checkpoint(
    *,
    data_root: str | Path,
    cache_dir: str | Path,
    backbone_checkpoint: str | Path,
    classifier_checkpoint: str | Path,
    output_path: str | Path,
    subjects: Sequence[int],
    target_subject: int,
    split_name: str,
    device: str,
    batch_size: int,
    cache_seed: int,
    feature_cache_dtype: str,
    overwrite: bool,
) -> dict[str, Any]:
    """Evaluate a frozen Awakening head on one persisted cache split."""

    audit = validate_awakening_dataset(data_root, subjects=subjects)
    contract = build_cache_contract(
        audit=audit,
        checkpoint_path=backbone_checkpoint,
        target_subject=target_subject,
        subjects=subjects,
        window_order_seed=cache_seed,
        feature_cache_dtype=feature_cache_dtype,
    )
    cached = load_cache_split(
        cache_dir=cache_dir,
        expected_contract=contract,
        split_name=split_name,
    )
    config = awakening_model_config(checkpoint=backbone_checkpoint, device=device)
    backbone = Model50MBackbone(config=config, load_checkpoint=True, freeze=True)
    classifier = Model50MClassifier(config=config, backbone=backbone)
    report = load_classifier_checkpoint(
        classifier, classifier_checkpoint, strict_metadata=True
    )
    metadata = report.metadata
    expected_metadata = {
        "task_id": "awakening",
        "class_order": list(AWAKENING_2S.class_names),
        "positive_class_index": 1,
        "feature_cache_contract_hash": feature_cache_contract_hash(contract),
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Awakening checkpoint metadata mismatch: {mismatches}.")
    metrics = evaluate_binary_feature_dataset(
        head=classifier.head,
        dataset=cached.dataset,
        subject_ids=cached.subject_ids,
        criterion=nn.CrossEntropyLoss(),
        device=classifier.device,
        class_names=AWAKENING_2S.class_names,
        positive_class_index=1,
        batch_size=batch_size,
    )
    payload = {
        "task_id": "awakening",
        "checkpoint": str(resolve_repo_path(classifier_checkpoint)),
        "backbone_checkpoint": str(resolve_repo_path(backbone_checkpoint)),
        "backbone_checkpoint_sha256": contract["backbone_checkpoint_sha256"],
        "data_split": split_name,
        "split_semantics": AWAKENING_2S.session_semantics,
        "class_order": list(AWAKENING_2S.class_names),
        "positive_class_index": 1,
        "channel_profile": channel_profile_contract(),
        "preprocessing": preprocessing_contract(config),
        "cache_contract_hash": feature_cache_contract_hash(contract),
        "metrics": metrics,
    }
    output = resolve_repo_path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Evaluation output exists: {output}")
    atomic_write_json(output, payload)
    return payload
