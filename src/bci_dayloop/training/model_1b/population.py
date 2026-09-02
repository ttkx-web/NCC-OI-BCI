"""First-version BNCI2014_001 population training for a frozen 1B backbone.

Only a final-layer, flatten, linear probe is trained here.  Split construction,
label validation, metrics, and source-trial leakage guards are intentionally
reused from the 50M population workflow so LOSO and within-subject semantics
remain identical.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bci_dayloop.models.model_1b import Model1BBackboneRunner, Model1BConfig
from bci_dayloop.models.model_1b.classifier import (
    Model1BFlattenLinearHead,
    classifier_input_dim,
    flatten_token_embeddings,
)
from bci_dayloop.runtime.types import RawEEGWindow
from bci_dayloop.training.model_50m.artifacts import (
    atomic_write_json,
    create_run_directory,
    current_git_commit,
    sha256_file,
)
from bci_dayloop.training.model_50m.data import (
    build_population_split,
    build_subject_identities,
    build_within_subject_split_metadata,
    build_within_subject_splits,
    build_within_subject_test_split,
    class_name_counts,
    normalize_subjects,
    resolve_class_names,
    resolve_subject_file,
    validate_labels,
    validate_no_source_leakage,
)
from bci_dayloop.training.model_50m.evaluation import extend_metrics
from bci_dayloop.training.model_50m.linear_head import (
    EpochMetrics,
    metric_is_better,
    resolve_repo_path,
    run_head_epoch,
    set_seed,
)
from bci_dayloop.training.model_50m.types import ExtendedMetrics, SplitBuildResult
from bci_dayloop.utils.paths import population_head_path, population_run_dir


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a frozen 1B flatten linear population head on BNCI2014_001."
    )
    parser.add_argument("--data-root", default="data/processed/bnci2014_001")
    parser.add_argument("--data-pattern", default="subject_{subject:02d}.h5")
    parser.add_argument("--data-reader", choices=("eeg",), default="eeg")
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--target-subject", type=int, default=1)
    parser.add_argument("--split-mode", choices=("loso", "within-subject"), default="loso")
    parser.add_argument("--train-session", nargs="+", default=["0train"])
    parser.add_argument("--validation-session", default="1test")
    parser.add_argument("--final-test-session", default="1test")
    parser.add_argument("--test-session", default=None)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--class-names", nargs="+", default=None)

    parser.add_argument(
        "--window-seconds", "--window-sec", dest="window_seconds", type=float,
        default=4.0, choices=(1.0, 2.0, 3.0, 4.0),
    )
    parser.add_argument("--window-stride-sec", type=float, default=None)
    parser.add_argument("--window-construction", choices=("direct_trial", "same_label_concat"), default="direct_trial")
    parser.add_argument("--direct-trial-anchor", choices=("start", "center", "end"), default="end")
    parser.add_argument("--max-windows-per-class-per-subject", type=int, default=None)
    parser.add_argument("--target-sample-rate", type=float, default=100.0)
    parser.add_argument("--patch-sec", type=float, default=1.0)
    parser.add_argument("--patch-stride-sec", type=float, default=1.0)
    parser.add_argument("--filter-low-hz", type=float, default=0.1)
    parser.add_argument("--filter-high-hz", type=float, default=75.0)
    parser.add_argument("--reference-mode", choices=("none", "average"), default="none")

    parser.add_argument(
        "--checkpoint", default="checkpoints/backbones/1b/pretrain_checkpoint_4.pt"
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="auto")
    parser.add_argument("--output", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--feature-log-every", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--head-batch-size", type=int, default=16)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--metric-for-best", choices=("val_bacc", "val_acc", "val_loss"), default="val_bacc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-seed", type=int, default=42)
    return parser


def _resolve_device(value: str) -> str:
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate_args(args: argparse.Namespace) -> None:
    if float(args.window_seconds) not in {1.0, 2.0, 3.0, 4.0}:
        raise ValueError("--window-seconds must be one of 1, 2, 3, or 4")
    if args.patch_sec != 1.0 or args.patch_stride_sec != 1.0:
        raise ValueError("1B population training requires 1-second non-overlapping patches")
    if args.target_sample_rate != 100.0:
        raise ValueError("1B population training requires target_sample_rate=100.0")
    if args.reference_mode != "none":
        raise ValueError("1B checkpoint contract requires --reference-mode none")
    if args.epochs <= 0 or args.head_batch_size <= 0 or args.feature_log_every <= 0:
        raise ValueError("epochs, head batch size, and feature log interval must be positive")
    if args.head_lr <= 0 or args.weight_decay < 0 or args.patience < 0:
        raise ValueError("invalid linear-head optimizer settings")
    if args.max_windows_per_class_per_subject is not None and args.max_windows_per_class_per_subject <= 0:
        raise ValueError("--max-windows-per-class-per-subject must be positive")
    if args.split_mode == "loso" and len(args.train_session) != 1:
        raise ValueError("LOSO accepts exactly one --train-session")
    if args.split_mode == "within-subject":
        if args.test_session is None:
            raise ValueError("--test-session is required for within-subject training")
        if not 0.0 < args.validation_ratio < 1.0:
            raise ValueError("--validation-ratio must be in (0, 1)")


def preprocessing_contract(config: Model1BConfig) -> dict[str, Any]:
    return {
        "input_type": "RawEEGWindow",
        "input_unit": "uV",
        "channel_mapping": "Model50MPreprocessor verified mapping",
        "standard_channels": list(config.standard_channels),
        "n_channels": config.n_channels,
        "filter_enabled": config.filter_enabled,
        "filter_low_hz": config.filter_low_hz,
        "filter_high_hz": config.filter_high_hz,
        "filter_order": config.filter_order,
        "reference_mode": config.reference_mode,
        "target_sample_rate": config.target_sample_rate,
        "window_seconds": config.window_seconds,
        "target_num_points": config.target_num_points,
        "patch_seconds": config.patch_seconds,
        "patch_stride_seconds": config.patch_stride_seconds,
        "patch_num_points": config.patch_num_points,
        "num_time_patches": config.num_time_patches,
        "num_tokens": config.num_tokens,
        "zscore_enabled": config.zscore_enabled,
        "zscore_eps": config.zscore_eps,
        "missing_channel_fill_value": config.missing_channel_fill_value,
        "window_tolerance_seconds": config.window_tolerance_seconds,
        "tokenization": "channel_major",
        "token_inputs_dtype": "torch.float32",
        "token_indices_dtype": "torch.int64",
    }


def _assert_frozen_backbone(runner: Model1BBackboneRunner) -> None:
    backbone = runner.backbone
    backbone.eval()
    parameters = list(backbone.parameters())
    if not parameters:
        raise RuntimeError("1B backbone has no parameters")
    if any(parameter.requires_grad for parameter in parameters):
        raise RuntimeError("1B backbone must be frozen before linear-head training")


def extract_frozen_features(
    *,
    runner: Model1BBackboneRunner,
    build: SplitBuildResult,
    split_name: str,
    log_every: int,
) -> TensorDataset:
    """Extract exactly the public 1B runner embeddings for each raw window."""
    _assert_frozen_backbone(runner)
    metadata = build.metadata
    features: list[torch.Tensor] = []
    windows = build.bundle.window_set.windows
    for index, window in enumerate(windows):
        prepared = runner.prepare(
            RawEEGWindow(
                data=np.asarray(window, dtype=np.float32),
                channel_names=list(metadata.channel_names),
                sample_rate=float(metadata.sample_rate),
                unit=str(metadata.unit),
                layout="CT",
                trial_id=str(index),
                window_id=f"{split_name}_{index}",
            )
        )
        embedding = runner.extract_embeddings(prepared)
        flattened = flatten_token_embeddings(
            embedding, prepared.token_valid_mask.to(embedding.device)
        )
        expected = classifier_input_dim(runner.config)
        if tuple(flattened.shape) != (1, expected):
            raise RuntimeError(
                f"{split_name}: expected flattened 1B feature [1,{expected}], "
                f"got {tuple(flattened.shape)}"
            )
        features.append(flattened.squeeze(0).cpu())
        if (index + 1) % log_every == 0 or index + 1 == len(windows):
            print(f"{split_name}: extracted {index + 1}/{len(windows)} frozen 1B features")
    labels = torch.from_numpy(build.bundle.window_set.labels.astype(np.int64, copy=True))
    return TensorDataset(torch.stack(features).to(torch.float32), labels)


def _fit_linear_head(
    *, head: Model1BFlattenLinearHead, train: TensorDataset, validation: TensorDataset,
    device: torch.device, num_classes: int, class_names: Sequence[str], args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], int, ExtendedMetrics, list[dict[str, Any]], float]:
    head = head.to(device)
    parameters = list(head.parameters())
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise RuntimeError("all linear-head parameters must require gradients")
    optimizer = torch.optim.AdamW(parameters, lr=args.head_lr, weight_decay=args.weight_decay)
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if optimizer_ids != {id(p) for p in parameters}:
        raise RuntimeError("optimizer must contain exactly the linear-head parameters")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=min(args.head_batch_size, len(train)), shuffle=True, generator=generator)
    validation_loader = DataLoader(validation, batch_size=min(args.head_batch_size, len(validation)), shuffle=False)
    criterion = nn.CrossEntropyLoss()
    best_value = float("inf") if args.metric_for_best == "val_loss" else -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_metrics: EpochMetrics | None = None
    without_improvement = 0
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = run_head_epoch(head=head, loader=train_loader, criterion=criterion, device=device, num_classes=num_classes, optimizer=optimizer)
        with torch.inference_mode():
            val_metrics = run_head_epoch(head=head, loader=validation_loader, criterion=criterion, device=device, num_classes=num_classes, optimizer=None)
        improved, value = metric_is_better(metric_name=args.metric_for_best, current=val_metrics, best_value=best_value)
        if improved:
            best_value, best_epoch, best_metrics, without_improvement = value, epoch, val_metrics, 0
            best_state = deepcopy({key: value.detach().cpu() for key, value in head.state_dict().items()})
        else:
            without_improvement += 1
        train_extended = extend_metrics(train_metrics, class_names=class_names)
        val_extended = extend_metrics(val_metrics, class_names=class_names)
        rows.append({
            "epoch": epoch, "train_loss": train_extended.loss, "train_acc": train_extended.accuracy,
            "train_bacc": train_extended.balanced_accuracy, "train_macro_f1": train_extended.macro_f1,
            "val_loss": val_extended.loss, "val_acc": val_extended.accuracy,
            "val_bacc": val_extended.balanced_accuracy, "val_macro_f1": val_extended.macro_f1,
            "is_best": improved, "epoch_seconds": time.perf_counter() - epoch_started,
        })
        if args.patience > 0 and without_improvement >= args.patience:
            break
    if best_state is None or best_metrics is None:
        raise RuntimeError("no validation-best 1B linear-head state was recorded")
    head.load_state_dict(best_state, strict=True)
    head.eval()
    return best_state, best_epoch, extend_metrics(best_metrics, class_names=class_names), rows, time.perf_counter() - started


def _evaluate_head(
    *, head: Model1BFlattenLinearHead, dataset: TensorDataset, device: torch.device,
    num_classes: int, class_names: Sequence[str], batch_size: int,
) -> ExtendedMetrics:
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)
    with torch.inference_mode():
        raw = run_head_epoch(head=head, loader=loader, criterion=nn.CrossEntropyLoss(), device=device, num_classes=num_classes, optimizer=None)
    return extend_metrics(raw, class_names=class_names)


def _prediction_records(head: Model1BFlattenLinearHead, dataset: TensorDataset, device: torch.device) -> list[dict[str, int]]:
    features, labels = dataset.tensors
    with torch.inference_mode():
        predicted = head(features.to(device=device, dtype=torch.float32)).argmax(dim=1).cpu()
    return [
        {"index": int(index), "label": int(label), "prediction": int(prediction)}
        for index, (label, prediction) in enumerate(zip(labels.tolist(), predicted.tolist(), strict=True))
    ]


def _head_payload(
    *, state: Mapping[str, torch.Tensor], config: Model1BConfig, class_names: Sequence[str],
    backbone_path: Path, backbone_sha256: str, split: Mapping[str, Any], args: argparse.Namespace,
    best_epoch: int, validation_metrics: ExtendedMetrics, final_test_metrics: ExtendedMetrics,
    training_seconds: float,
) -> dict[str, Any]:
    num_classes = len(class_names)
    return {
        "format_version": 1,
        "head_state_dict": {key: value.detach().cpu() for key, value in state.items()},
        "head_type": "linear",
        "aggregation": "flatten",
        "window_seconds": float(config.window_seconds),
        "num_time_patches": int(config.num_time_patches),
        "classifier_input_dim": classifier_input_dim(config),
        "num_classes": num_classes,
        "class_names": [str(name) for name in class_names],
        "label_mapping": {str(index): str(name) for index, name in enumerate(class_names)},
        "backbone_checkpoint_path": str(backbone_path),
        "backbone_checkpoint_sha256": backbone_sha256,
        "backbone_architecture": {
            "d_model": config.d_model, "n_heads": config.n_heads, "depth": config.depth,
            "mlp_ratio": config.mlp_ratio, "dropout": config.dropout,
            "output_layer_idx": config.output_layer_idx,
        },
        "preprocessing_contract": preprocessing_contract(config),
        "split_mode": str(args.split_mode),
        "subject_session_split": dict(split),
        "seed": int(args.seed),
        "training_hyperparameters": {
            "epochs": args.epochs, "head_batch_size": args.head_batch_size,
            "head_lr": args.head_lr, "weight_decay": args.weight_decay,
            "patience": args.patience, "metric_for_best": args.metric_for_best,
            "window_seed": args.window_seed,
        },
        "best_validation_epoch": int(best_epoch),
        "best_validation_metrics": validation_metrics.to_dict(),
        "final_test_metrics": final_test_metrics.to_dict(),
        "training_seconds": training_seconds,
        "contains_backbone_weights": False,
        "contains_optimizer_state": False,
        "contains_pretraining_head": False,
    }


def save_1b_head_checkpoint(path: Path | str, payload: Mapping[str, Any], *, overwrite: bool) -> Path:
    path = Path(path).expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"1B head checkpoint already exists: {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)
    return path


def validate_head_checkpoint_compatibility(
    payload: Mapping[str, Any], *, window_seconds: float | None = None,
    class_names: Sequence[str] | None = None, backbone_sha256: str | None = None,
) -> None:
    required = {
        "head_state_dict", "head_type", "aggregation", "window_seconds", "num_time_patches",
        "classifier_input_dim", "num_classes", "class_names", "label_mapping",
        "backbone_checkpoint_path", "backbone_checkpoint_sha256", "backbone_architecture",
        "preprocessing_contract", "split_mode", "subject_session_split", "seed",
        "training_hyperparameters", "best_validation_epoch", "best_validation_metrics", "final_test_metrics",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"1B head checkpoint missing metadata fields: {missing}")
    if payload["head_type"] != "linear" or payload["aggregation"] != "flatten":
        raise ValueError("1B first-version head must be linear with flatten aggregation")
    if window_seconds is not None and float(payload["window_seconds"]) != float(window_seconds):
        raise ValueError("1B head window_seconds does not match the requested contract")
    saved_names = [str(name) for name in payload["class_names"]]
    if class_names is not None and saved_names != [str(name) for name in class_names]:
        raise ValueError("1B head class_names order does not match the dataset contract")
    mapping = payload["label_mapping"]
    expected_mapping = {str(index): name for index, name in enumerate(saved_names)}
    if mapping != expected_mapping:
        raise ValueError("1B head label_mapping does not match class_names order")
    if backbone_sha256 is not None and payload["backbone_checkpoint_sha256"] != backbone_sha256:
        raise ValueError("1B head backbone checkpoint SHA-256 does not match")
    architecture = payload["backbone_architecture"]
    if {key: architecture.get(key) for key in ("d_model", "n_heads", "depth", "output_layer_idx")} != {
        "d_model": 2048, "n_heads": 16, "depth": 20, "output_layer_idx": 19,
    }:
        raise ValueError("1B head metadata does not match the formal 1B architecture")
    input_dim = int(payload["classifier_input_dim"])
    expected_dim = 64 * int(payload["num_time_patches"]) * 2048
    if input_dim != expected_dim:
        raise ValueError("1B head classifier_input_dim does not match token contract")
    state = payload["head_state_dict"]
    if set(state) != {"linear.weight", "linear.bias"}:
        raise ValueError("1B head checkpoint contains unexpected non-linear parameters")
    if tuple(state["linear.weight"].shape) != (int(payload["num_classes"]), input_dim):
        raise ValueError("1B head linear.weight shape does not match metadata")
    if tuple(state["linear.bias"].shape) != (int(payload["num_classes"]),):
        raise ValueError("1B head linear.bias shape does not match metadata")


def load_1b_head_checkpoint(
    path: Path | str, *, window_seconds: float | None = None,
    class_names: Sequence[str] | None = None, backbone_sha256: str | None = None,
    device: str | torch.device = "cpu",
) -> tuple[Model1BFlattenLinearHead, dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"1B head checkpoint was not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("1B head checkpoint must be a mapping")
    validate_head_checkpoint_compatibility(
        payload, window_seconds=window_seconds, class_names=class_names,
        backbone_sha256=backbone_sha256,
    )
    head = Model1BFlattenLinearHead(
        input_dim=int(payload["classifier_input_dim"]), num_classes=int(payload["num_classes"])
    ).to(device)
    head.load_state_dict(payload["head_state_dict"], strict=True)
    head.eval()
    return head, payload


def _default_paths(args: argparse.Namespace, timestamp: str) -> tuple[Path, Path]:
    dataset = "bnci2014_001" if args.split_mode == "loso" else "within_subject"
    output = (
        resolve_repo_path(args.output)
        if args.output is not None
        else population_head_path(
            stage="stage1_1b", dataset=dataset, subject_id=args.target_subject,
            window_seconds=args.window_seconds, aggregation="flatten",
        )
    )
    run_dir = (
        resolve_repo_path(args.run_dir)
        if args.run_dir is not None
        else population_run_dir(
            stage="stage1_1b", dataset=dataset, subject_id=args.target_subject,
            window_seconds=args.window_seconds, aggregation="flatten", run_id=timestamp,
        )
    )
    return output, run_dir


def _validate_existing_run_contract(
    run_dir: Path, *, window_seconds: float, backbone_sha256: str
) -> None:
    """Do not let --overwrite turn one run directory into a mixed contract."""
    metadata_path = run_dir / "head_metadata.json"
    if not metadata_path.is_file():
        return
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if float(payload.get("window_seconds", -1.0)) != float(window_seconds):
        raise ValueError("existing 1B run directory has a different window_seconds contract")
    if payload.get("backbone_checkpoint_sha256") != backbone_sha256:
        raise ValueError("existing 1B run directory has a different backbone checkpoint SHA-256")


def run_population_training(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    set_seed(args.seed)
    args.window_stride_sec = float(args.window_stride_sec or args.window_seconds)
    checkpoint_path = resolve_repo_path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"1B backbone checkpoint was not found: {checkpoint_path}")
    data_root = resolve_repo_path(args.data_root)
    target_subject = int(args.target_subject)
    subjects = normalize_subjects(args.subjects) if args.split_mode == "loso" else [target_subject]
    if args.split_mode == "loso" and target_subject not in subjects:
        raise ValueError("target subject must be included in --subjects for LOSO")
    population_subjects = [subject for subject in subjects if subject != target_subject]
    if args.split_mode == "loso" and not population_subjects:
        raise ValueError("LOSO requires at least one non-target population subject")
    if args.split_mode == "within-subject" and target_subject <= 0:
        raise ValueError("target subject must be positive")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path, run_dir = _default_paths(args, timestamp)
    create_run_directory(run_dir, overwrite=args.overwrite)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"1B head output already exists: {output_path}; use --overwrite")
    backbone_sha256 = sha256_file(checkpoint_path)
    _validate_existing_run_contract(
        run_dir, window_seconds=args.window_seconds, backbone_sha256=backbone_sha256
    )
    if output_path.is_file():
        existing = torch.load(output_path, map_location="cpu")
        if not isinstance(existing, dict):
            raise TypeError(f"existing 1B head output is not a checkpoint mapping: {output_path}")
        # --overwrite may repeat a run, but it must never mix another window
        # contract or backbone checkpoint into this head location.
        validate_head_checkpoint_compatibility(
            existing, window_seconds=args.window_seconds, backbone_sha256=backbone_sha256
        )

    subject_paths = {
        subject: resolve_subject_file(data_root=data_root, pattern=args.data_pattern, subject_id=subject)
        for subject in subjects
    }
    subject_identities = build_subject_identities(subject_paths=subject_paths, data_reader=args.data_reader)
    within_split = None
    within_trial_metadata = None
    if args.split_mode == "loso":
        train_build = build_population_split(
            subjects=population_subjects, data_root=data_root, data_pattern=args.data_pattern,
            data_reader=args.data_reader, session_name=args.train_session[0], window_seconds=args.window_seconds,
            stride_seconds=args.window_stride_sec, base_seed=args.window_seed + 1000,
            shuffle_trials_within_class=True, max_windows_per_class_per_subject=args.max_windows_per_class_per_subject,
            window_construction=args.window_construction, direct_trial_anchor=args.direct_trial_anchor,
        )
        val_build = build_population_split(
            subjects=population_subjects, data_root=data_root, data_pattern=args.data_pattern,
            data_reader=args.data_reader, session_name=args.validation_session, window_seconds=args.window_seconds,
            stride_seconds=args.window_stride_sec, base_seed=args.window_seed + 2000,
            shuffle_trials_within_class=False, max_windows_per_class_per_subject=args.max_windows_per_class_per_subject,
            reference_metadata=train_build.metadata, window_construction=args.window_construction,
            direct_trial_anchor=args.direct_trial_anchor,
        )
        metadata = train_build.metadata
        class_names = resolve_class_names(metadata=metadata, explicit_class_names=args.class_names)
        split_descriptor: dict[str, Any] = {
            "population_subjects": population_subjects, "excluded_target_subject": target_subject,
            "train_session": args.train_session[0], "validation_session": args.validation_session,
            "final_test_session": args.final_test_session,
        }
    else:
        train_build, val_build, metadata, class_names, within_split, within_trial_metadata = build_within_subject_splits(
            subject_id=target_subject, path=subject_paths[target_subject], data_reader=args.data_reader,
            train_sessions=args.train_session, test_session=args.test_session,
            validation_ratio=args.validation_ratio, seed=args.seed, window_seconds=args.window_seconds,
            stride_seconds=args.window_stride_sec, max_windows_per_class=args.max_windows_per_class_per_subject,
            window_construction=args.window_construction, direct_trial_anchor=args.direct_trial_anchor,
            explicit_class_names=args.class_names,
        )
        split_descriptor = build_within_subject_split_metadata(
            subject_id=target_subject, split=within_split, all_trial_metadata=within_trial_metadata,
            class_names=class_names, validation_ratio=args.validation_ratio, seed=args.seed,
        )
    num_classes = len(class_names)
    for name, build in (("train", train_build), ("validation", val_build)):
        validate_labels(build.bundle.window_set.labels, num_classes=num_classes, split_name=f"1B {name}")
    validate_no_source_leakage(train_build.bundle.window_set, val_build.bundle.window_set, left_name="1B train", right_name="1B validation")

    config = Model1BConfig(
        checkpoint_path=checkpoint_path, device=_resolve_device(args.device), window_seconds=args.window_seconds,
        target_sample_rate=args.target_sample_rate, patch_seconds=args.patch_sec,
        patch_stride_seconds=args.patch_stride_sec, filter_enabled=True,
        filter_low_hz=args.filter_low_hz, filter_high_hz=args.filter_high_hz,
        reference_mode=args.reference_mode, strict_window_duration=True,
    )
    load_started = time.perf_counter()
    runner = Model1BBackboneRunner(config)
    model_load_seconds = time.perf_counter() - load_started
    _assert_frozen_backbone(runner)
    head = Model1BFlattenLinearHead(input_dim=classifier_input_dim(config), num_classes=num_classes)
    backbone_parameter_count = sum(parameter.numel() for parameter in runner.backbone.parameters())
    head_parameter_count = sum(parameter.numel() for parameter in head.parameters())
    if any(parameter.requires_grad for parameter in runner.backbone.parameters()):
        raise RuntimeError("frozen 1B backbone unexpectedly has trainable parameters")
    print(f"1B backbone parameters: total={backbone_parameter_count}, trainable=0")
    print(f"1B linear head parameters: trainable={head_parameter_count}; input_dim={head.input_dim}")

    atomic_write_json(run_dir / "run_config.json", {
        "status": "started", "timestamp": timestamp, "git_commit": current_git_commit(),
        "arguments": vars(args), "backbone_checkpoint_path": str(checkpoint_path),
        "backbone_checkpoint_sha256": backbone_sha256, "output": str(output_path),
        "split_mode": args.split_mode, "subject_session_split": split_descriptor,
        "subject_identities": subject_identities,
    })
    train_features = extract_frozen_features(runner=runner, build=train_build, split_name="train", log_every=args.feature_log_every)
    val_features = extract_frozen_features(runner=runner, build=val_build, split_name="validation", log_every=args.feature_log_every)
    best_state, best_epoch, best_val, rows, training_seconds = _fit_linear_head(
        head=head, train=train_features, validation=val_features, device=runner.backbone.device_object,
        num_classes=num_classes, class_names=class_names, args=args,
    )

    if args.split_mode == "loso":
        target_build = build_population_split(
            subjects=[target_subject], data_root=data_root, data_pattern=args.data_pattern,
            data_reader=args.data_reader, session_name=args.final_test_session, window_seconds=args.window_seconds,
            stride_seconds=args.window_stride_sec, base_seed=args.window_seed + 3000,
            shuffle_trials_within_class=False, max_windows_per_class_per_subject=args.max_windows_per_class_per_subject,
            reference_metadata=metadata, window_construction=args.window_construction,
            direct_trial_anchor=args.direct_trial_anchor,
        )
        split_descriptor["final_test_session"] = args.final_test_session
    else:
        assert within_split is not None and within_trial_metadata is not None
        target_build = build_within_subject_test_split(
            subject_id=target_subject, path=subject_paths[target_subject], data_reader=args.data_reader,
            metadata=metadata, class_names=class_names, split=within_split,
            all_trial_metadata=within_trial_metadata, window_seconds=args.window_seconds,
            stride_seconds=args.window_stride_sec, max_windows_per_class=args.max_windows_per_class_per_subject,
            window_construction=args.window_construction, direct_trial_anchor=args.direct_trial_anchor,
        )
    validate_labels(target_build.bundle.window_set.labels, num_classes=num_classes, split_name="1B final test")
    validate_no_source_leakage(train_build.bundle.window_set, target_build.bundle.window_set, left_name="1B train", right_name="1B final test")
    validate_no_source_leakage(val_build.bundle.window_set, target_build.bundle.window_set, left_name="1B validation", right_name="1B final test")
    target_features = extract_frozen_features(runner=runner, build=target_build, split_name="final_test", log_every=args.feature_log_every)
    final_metrics = _evaluate_head(head=head, dataset=target_features, device=runner.backbone.device_object, num_classes=num_classes, class_names=class_names, batch_size=args.head_batch_size)
    payload = _head_payload(
        state=best_state, config=config, class_names=class_names, backbone_path=checkpoint_path,
        backbone_sha256=backbone_sha256, split=split_descriptor, args=args, best_epoch=best_epoch,
        validation_metrics=best_val, final_test_metrics=final_metrics, training_seconds=training_seconds,
    )
    checkpoint_output = save_1b_head_checkpoint(output_path, payload, overwrite=args.overwrite)
    atomic_write_json(run_dir / "head_metadata.json", {key: value for key, value in payload.items() if key != "head_state_dict"})
    with (run_dir / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    (run_dir / "final_test_predictions.json").write_text(
        json.dumps(_prediction_records(head, target_features, runner.backbone.device_object), indent=2) + "\n",
        encoding="utf-8",
    )
    atomic_write_json(run_dir / "summary.json", {
        "status": "completed", "head_checkpoint": str(checkpoint_output), "best_validation_epoch": best_epoch,
        "best_validation_metrics": best_val.to_dict(), "final_test_metrics": final_metrics.to_dict(),
        "backbone_parameter_count": backbone_parameter_count, "backbone_trainable_parameter_count": 0,
        "linear_head_trainable_parameter_count": head_parameter_count, "classifier_input_dim": head.input_dim,
        "model_load_seconds": model_load_seconds, "split_mode": args.split_mode,
        "train_window_counts": class_name_counts(train_build.bundle.window_set.labels, class_names),
        "validation_window_counts": class_name_counts(val_build.bundle.window_set.labels, class_names),
        "final_test_window_counts": class_name_counts(target_build.bundle.window_set.labels, class_names),
    })
    print(f"saved 1B linear head: {checkpoint_output}")
    print(
        "best validation:",
        f"epoch={best_epoch} acc={best_val.accuracy:.4f} "
        f"bacc={best_val.balanced_accuracy:.4f} macro_f1={best_val.macro_f1:.4f}",
    )
    print(
        "final test:",
        f"acc={final_metrics.accuracy:.4f} bacc={final_metrics.balanced_accuracy:.4f} "
        f"macro_f1={final_metrics.macro_f1:.4f}",
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    run_population_training(args)
    return 0
