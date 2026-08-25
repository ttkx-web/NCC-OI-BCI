"""Artifact primitives shared by the 50M population training stages."""

from __future__ import annotations

import hashlib
import json
import subprocess
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from bci_dayloop.training.model_50m.linear_head import json_default
from bci_dayloop.models.model_50m.classifier import (
    Model50MClassifier,
    save_classifier_checkpoint,
)
from bci_dayloop.models.model_50m.lora import (
    lora_state_dict,
    normalize_lora_target_modules,
)
from bci_dayloop.utils.config import project_root


@dataclass(frozen=True, slots=True)
class PreparedArtifactPaths:
    """The established run and classifier output locations."""

    run_dir: Path
    output_path: Path
    git_commit: str | None
    backbone_sha256: str


@dataclass(frozen=True, slots=True)
class TrainingArtifactResult:
    """Paths written for one completed population-training run."""

    output_dir: Path
    checkpoint_path: Path
    report_path: Path
    summary_path: Path
    epoch_metrics_path: Path


@dataclass(slots=True)
class TrainingArtifactInputs:
    """Completed-run values required by the stable Stage-1 artifact schema.

    This intentionally carries already-computed evaluation/split results.  The
    artifact layer serializes them but never performs model evaluation or split
    construction itself.
    """

    run_dir: Path
    output_path: Path
    classifier: Model50MClassifier
    args: Any
    config: Any
    backbone: Any
    checkpoint_path: Path
    git_commit: str | None
    backbone_sha256: str
    target_subject: int
    population_subjects: Sequence[int]
    subjects: Sequence[int]
    loso_train_session: str | None
    within_subject_train_sessions: Sequence[str]
    final_test_session: str
    embedding_layer: int
    subject_identities: Mapping[str, Any]
    all_subject_paths: Mapping[int, Path]
    metadata: Any
    class_names: Sequence[str]
    label_mapping: Mapping[Any, Any]
    num_classes: int
    within_subject_metadata: Mapping[str, Any] | None
    backbone_adaptation: str
    partial_finetuning_enabled: bool
    lora_enabled: bool
    trainable_backbone_parameters: Sequence[torch.nn.Parameter]
    trainable_block_indices: Sequence[int]
    lora_parameters: Sequence[torch.nn.Parameter]
    total_trainable_parameter_count: int
    head_parameters: Sequence[torch.nn.Parameter]
    selected_val_metrics: Any
    target_metrics: Any
    train_build: Any
    val_build: Any
    target_build: Any
    feature_cache_enabled: bool
    preprocessing_contract: Mapping[str, Any]
    preprocessing_hash: str
    model_load_seconds: float
    training_seconds: float
    best_epoch: int
    epoch_rows: Sequence[Mapping[str, Any]]


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write the JSON encoding used by existing training artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
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
            cwd=project_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def create_run_directory(run_dir: Path, *, overwrite: bool) -> None:
    """Create a run directory while preserving the no-overwrite guard."""
    run_dir.mkdir(parents=True, exist_ok=overwrite)


def prepare_training_artifact_paths(
    *,
    run_dir: Path,
    output_path: Path,
    overwrite: bool,
    backbone_checkpoint: Path,
) -> PreparedArtifactPaths:
    """Prepare the established output locations and reproducibility identity."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    create_run_directory(run_dir, overwrite=overwrite)
    return PreparedArtifactPaths(
        run_dir=run_dir,
        output_path=output_path,
        git_commit=current_git_commit(),
        backbone_sha256=sha256_file(backbone_checkpoint),
    )


def save_initial_run_config(
    *,
    paths: PreparedArtifactPaths,
    timestamp: str,
    target_subject: int,
    population_subjects: Sequence[int],
    split_mode: str,
    loso_train_session: str | None,
    within_subject_train_sessions: Sequence[str],
    validation_session: str,
    final_test_session: str,
    backbone_checkpoint: Path,
    validation_ratio: float,
    data_root: Path,
    data_pattern: str,
    data_reader: str,
    subject_identities: Mapping[str, Any],
    subject_paths: Mapping[int, Path],
    arguments: Mapping[str, Any],
) -> Path:
    """Write the unchanged ``started`` run-config artifact."""
    initial_run_config = {
        "status": "started",
        "timestamp": timestamp,
        "git_commit": paths.git_commit,
        "target_subject": target_subject,
        "population_subjects": list(population_subjects),
        "split_mode": split_mode,
        "sessions": (
            {
                "population_train": loso_train_session,
                "population_validation": validation_session,
                "final_target_test": final_test_session,
            }
            if split_mode == "loso"
            else {
                "within_subject_train_sessions": list(within_subject_train_sessions),
                "within_subject_test": final_test_session,
                "validation_ratio": validation_ratio,
            }
        ),
        "data_root": str(data_root),
        "data_pattern": data_pattern,
        "data_reader": data_reader,
        "subject_identities": subject_identities,
        "subject_paths": {
            str(subject): str(path) for subject, path in subject_paths.items()
        },
        "backbone_checkpoint": str(backbone_checkpoint),
        "backbone_sha256": paths.backbone_sha256,
        "output": str(paths.output_path),
        "arguments": dict(arguments),
    }
    path = paths.run_dir / "run_config.json"
    atomic_write_json(path, initial_run_config)
    return path


def _class_name_counts(labels: np.ndarray, class_names: Sequence[str]) -> dict[str, int]:
    return {
        str(class_name): int(np.sum(labels == index))
        for index, class_name in enumerate(class_names)
    }


def _concat_warning(window_construction: str) -> str | None:
    if window_construction == "direct_trial":
        return None
    return "Samples were constructed by concatenating same-label source trials."


def _write_epoch_metrics(path: Path, epoch_rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the legacy CSV exactly as the runner did."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(epoch_rows[0].keys()))
        writer.writeheader()
        writer.writerows(epoch_rows)


def save_training_artifacts(inputs: TrainingArtifactInputs) -> TrainingArtifactResult:
    """Assemble and persist the established Stage-1 checkpoint and reports."""
    args = inputs.args
    config = inputs.config
    lora_targets = (
        list(normalize_lora_target_modules(args.lora_target_modules))
        if inputs.lora_enabled
        else []
    )
    warning = _concat_warning(args.window_construction)
    trainable_backbone_parameter_count = sum(
        parameter.numel() for parameter in inputs.trainable_backbone_parameters
    )
    trainable_lora_parameter_count = sum(
        parameter.numel() for parameter in inputs.lora_parameters
    )
    trainable_head_parameter_count = sum(
        parameter.numel() for parameter in inputs.head_parameters
    )
    subject_data_paths = {
        str(subject): str(path) for subject, path in inputs.all_subject_paths.items()
    }
    selected_val = inputs.selected_val_metrics.to_dict()
    target_final = inputs.target_metrics.to_dict()
    saved_lora_state = (
        lora_state_dict(inputs.classifier.backbone.model)
        if inputs.lora_enabled
        else None
    )
    if inputs.lora_enabled and not saved_lora_state:
        raise RuntimeError(
            "Refusing to save a LoRA checkpoint without injected adapter state."
        )

    saved_path = save_classifier_checkpoint(
        classifier=inputs.classifier,
        checkpoint_path=inputs.output_path,
        extra_metadata={
            "task": (
                "BNCI2014_001_motor_imagery"
                if inputs.metadata.dataset_name == "bnci2014_001"
                and args.split_mode == "loso"
                else f"{inputs.metadata.dataset_name}_classification"
            ),
            "dataset": inputs.metadata.dataset_name,
            "data_reader": args.data_reader,
            "subject_identities": inputs.subject_identities,
            "mode": _experiment_name(
                split_mode=args.split_mode,
                lora_enabled=inputs.lora_enabled,
                partial_finetuning_enabled=inputs.partial_finetuning_enabled,
                head_type=config.head_type,
            ),
            "stage": "stage1",
            "split_mode": args.split_mode,
            "target_subject": inputs.target_subject,
            "excluded_subjects": ([inputs.target_subject] if args.split_mode == "loso" else []),
            "population_training_subjects": list(inputs.population_subjects),
            "population_validation_subjects": list(inputs.population_subjects),
            "population_train_session": inputs.loso_train_session,
            "population_validation_session": args.validation_session,
            "final_test_subject": inputs.target_subject,
            "final_test_session": inputs.final_test_session,
            "subject_data_paths": subject_data_paths,
            "class_names": list(inputs.class_names),
            "label_mapping": inputs.label_mapping,
            "num_classes": int(inputs.num_classes),
            "within_subject_split": inputs.within_subject_metadata,
            "backbone_sha256": inputs.backbone_sha256,
            "preprocessing_hash": inputs.preprocessing_hash,
            "backbone_adaptation": inputs.backbone_adaptation,
            "freeze_backbone": not inputs.partial_finetuning_enabled,
            "trainable_backbone_parameters": trainable_backbone_parameter_count,
            "embedding_layer_requested": str(args.embedding_layer),
            "embedding_layer_resolved": int(inputs.embedding_layer),
            "embedding_layer_internal_index": int(config.output_layer_idx),
            "unfreeze_last_n_blocks": int(args.unfreeze_last_n_blocks),
            "unfrozen_block_indices": [
                int(index)
                for index in (
                    inputs.trainable_block_indices
                    if inputs.partial_finetuning_enabled
                    else ()
                )
            ],
            "lora_last_n_blocks": int(args.lora_last_n_blocks) if inputs.lora_enabled else None,
            "lora_block_indices": [
                int(index)
                for index in (inputs.trainable_block_indices if inputs.lora_enabled else ())
            ],
            "lora_target_modules": lora_targets,
            "lora_rank": int(args.lora_rank) if inputs.lora_enabled else None,
            "lora_alpha": float(args.lora_alpha) if inputs.lora_enabled else None,
            "lora_dropout": float(args.lora_dropout) if inputs.lora_enabled else None,
            "head_type": config.head_type,
            "head_hidden_dim": int(config.head_hidden_dim),
            "head_dropout": float(config.head_dropout),
            "head_norm": config.head_norm,
            "head_trainable_parameters": sum(
                parameter.numel() for parameter in inputs.classifier.head.parameters()
            ),
            "optimizer": "AdamW",
            "head_lr": float(args.head_lr),
            "backbone_lr": float(args.backbone_lr),
            "lora_lr": float(args.lora_lr) if inputs.lora_enabled else None,
            "trainable_lora_parameters": trainable_lora_parameter_count,
            "trainable_total_parameters": inputs.total_trainable_parameter_count,
            "momentum": float(args.momentum),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "window_seed": int(args.window_seed),
            "metric_for_best": args.metric_for_best,
            "best_epoch": int(inputs.best_epoch),
            "best_validation": selected_val,
            "target_final_test": target_final,
            "window_construction": inputs.train_build.bundle.window_set.construction,
            "model_n_time_patches": int(config.model_n_time_patches),
            "warning": warning,
            "git_commit": inputs.git_commit,
        },
        backbone_state_dict=(
            inputs.classifier.backbone.model.state_dict()
            if inputs.partial_finetuning_enabled
            else None
        ),
        lora_state_dict=saved_lora_state,
    )

    metrics_csv = inputs.run_dir / "epoch_metrics.csv"
    _write_epoch_metrics(metrics_csv, inputs.epoch_rows)
    experiment_name = _experiment_name(
        split_mode=args.split_mode,
        lora_enabled=inputs.lora_enabled,
        partial_finetuning_enabled=inputs.partial_finetuning_enabled,
        head_type=config.head_type,
    )
    report = _build_report(
        inputs=inputs,
        experiment_name=experiment_name,
        warning=warning,
        metrics_csv=metrics_csv,
        saved_path=saved_path,
        subject_data_paths=subject_data_paths,
        selected_val=selected_val,
        target_final=target_final,
        lora_targets=lora_targets,
        trainable_backbone_parameter_count=trainable_backbone_parameter_count,
        trainable_lora_parameter_count=trainable_lora_parameter_count,
        trainable_head_parameter_count=trainable_head_parameter_count,
    )
    report_path = inputs.run_dir / "population_training_report.json"
    atomic_write_json(report_path, report)
    summary_path = inputs.run_dir / "summary.json"
    atomic_write_json(
        summary_path,
        _build_summary(
            inputs=inputs,
            saved_path=saved_path,
            report_path=report_path,
            selected_val=selected_val,
            target_final=target_final,
            lora_targets=lora_targets,
            trainable_backbone_parameter_count=trainable_backbone_parameter_count,
            trainable_lora_parameter_count=trainable_lora_parameter_count,
            trainable_head_parameter_count=trainable_head_parameter_count,
        ),
    )
    return TrainingArtifactResult(
        output_dir=inputs.run_dir,
        checkpoint_path=saved_path,
        report_path=report_path,
        summary_path=summary_path,
        epoch_metrics_path=metrics_csv,
    )


def _experiment_name(
    *, split_mode: str, lora_enabled: bool, partial_finetuning_enabled: bool, head_type: str
) -> str:
    prefix = "population_loso" if split_mode == "loso" else "within_subject"
    if lora_enabled:
        return f"{prefix}_lora"
    if partial_finetuning_enabled:
        return f"{prefix}_partial_finetune"
    return f"{prefix}_linear_probe" if head_type == "linear" else f"{prefix}_mlp_head"


def _build_report(
    *,
    inputs: TrainingArtifactInputs,
    experiment_name: str,
    warning: str | None,
    metrics_csv: Path,
    saved_path: Path,
    subject_data_paths: Mapping[str, str],
    selected_val: Mapping[str, Any],
    target_final: Mapping[str, Any],
    lora_targets: Sequence[str],
    trainable_backbone_parameter_count: int,
    trainable_lora_parameter_count: int,
    trainable_head_parameter_count: int,
) -> dict[str, Any]:
    """Build the legacy ``population_training_report.json`` payload."""
    args = inputs.args
    config = inputs.config
    return {
        "status": "completed",
        "stage": "stage1",
        "experiment": experiment_name,
        "split_mode": args.split_mode,
        "warning": warning,
        "files": {
            "backbone_checkpoint": str(inputs.checkpoint_path),
            "classifier_checkpoint": str(saved_path),
            "run_dir": str(inputs.run_dir),
            "epoch_metrics_csv": str(metrics_csv),
            "subject_data_paths": subject_data_paths,
        },
        "data_reader": args.data_reader,
        "subject_identities": inputs.subject_identities,
        "protocol": {
            "all_subjects": list(inputs.subjects),
            "target_subject": inputs.target_subject,
            "population_subjects": list(inputs.population_subjects),
            "population_train_session": inputs.loso_train_session,
            "population_validation_session": args.validation_session,
            "final_target_test_session": inputs.final_test_session,
            "target_subject_used_for_training": args.split_mode == "within-subject",
            "target_subject_used_for_validation": args.split_mode == "within-subject",
            "final_test_opened_after_model_selection": True,
            "within_subject": inputs.within_subject_metadata,
        },
        "dataset": {
            "name": inputs.metadata.dataset_name,
            "data_reader": args.data_reader,
            "sample_rate": inputs.metadata.sample_rate,
            "unit": inputs.metadata.unit,
            "channel_names": inputs.metadata.channel_names,
            "class_names": list(inputs.class_names),
            "label_mapping": inputs.label_mapping,
            "num_classes": inputs.num_classes,
        },
        "source_trials": {
            "population_train": inputs.train_build.source_trial_summary,
            "population_validation": inputs.val_build.source_trial_summary,
            "target_final_test": inputs.target_build.source_trial_summary,
        },
        "derived_windows": {
            "population_train_total": int(len(inputs.train_build.bundle.window_set.windows)),
            "population_train_per_class": _class_name_counts(
                inputs.train_build.bundle.window_set.labels, inputs.class_names
            ),
            "population_validation_total": int(len(inputs.val_build.bundle.window_set.windows)),
            "population_validation_per_class": _class_name_counts(
                inputs.val_build.bundle.window_set.labels, inputs.class_names
            ),
            "target_final_test_total": int(len(inputs.target_build.bundle.window_set.windows)),
            "target_final_test_per_class": _class_name_counts(
                inputs.target_build.bundle.window_set.labels, inputs.class_names
            ),
            "window_seconds": float(args.window_sec),
            "stride_seconds": float(args.window_stride_sec),
            "construction": inputs.train_build.bundle.window_set.construction,
        },
        "model": {
            "device": str(inputs.classifier.device),
            "load_seconds": inputs.model_load_seconds,
            "backbone_sha256": inputs.backbone_sha256,
            "trainable_backbone_parameters": inputs.backbone.trainable_parameters,
            "embedding_layer_requested": str(args.embedding_layer),
            "embedding_layer_resolved": int(inputs.embedding_layer),
            "embedding_layer_internal_index": int(config.output_layer_idx),
            "unfreeze_last_n_blocks": int(args.unfreeze_last_n_blocks),
            "unfrozen_blocks": [int(index + 1) for index in inputs.trainable_block_indices],
            "feature_cache_enabled": inputs.feature_cache_enabled,
            "backbone_adaptation": inputs.backbone_adaptation,
            "lora_last_n_blocks": int(args.lora_last_n_blocks) if inputs.lora_enabled else None,
            "lora_blocks": [
                int(index + 1)
                for index in (inputs.trainable_block_indices if inputs.lora_enabled else ())
            ],
            "lora_target_modules": list(lora_targets),
            "lora_rank": int(args.lora_rank) if inputs.lora_enabled else None,
            "lora_alpha": float(args.lora_alpha) if inputs.lora_enabled else None,
            "lora_dropout": float(args.lora_dropout) if inputs.lora_enabled else None,
            "trainable_lora_params": trainable_lora_parameter_count,
            "trainable_total_params": inputs.total_trainable_parameter_count,
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
            "preprocessing_contract": inputs.preprocessing_contract,
            "preprocessing_hash": inputs.preprocessing_hash,
        },
        "training": {
            "optimizer": "AdamW",
            "epochs_requested": args.epochs,
            "epochs_completed": len(inputs.epoch_rows),
            "training_seconds": inputs.training_seconds,
            "best_epoch": inputs.best_epoch,
            "metric_for_best": args.metric_for_best,
            "head_lr": args.head_lr,
            "backbone_lr": args.backbone_lr,
            "lora_lr": args.lora_lr if inputs.lora_enabled else None,
            "backbone_adaptation": inputs.backbone_adaptation,
            "lora_last_n_blocks": args.lora_last_n_blocks if inputs.lora_enabled else None,
            "lora_blocks": [
                int(index + 1)
                for index in (inputs.trainable_block_indices if inputs.lora_enabled else ())
            ],
            "lora_target_modules": list(lora_targets),
            "lora_rank": args.lora_rank if inputs.lora_enabled else None,
            "lora_alpha": args.lora_alpha if inputs.lora_enabled else None,
            "lora_dropout": args.lora_dropout if inputs.lora_enabled else None,
            "trainable_lora_params": trainable_lora_parameter_count,
            "trainable_total_params": inputs.total_trainable_parameter_count,
            "unfreeze_last_n_blocks": args.unfreeze_last_n_blocks,
            "embedding_layer": inputs.embedding_layer,
            "embedding_layer_internal_index": config.output_layer_idx,
            "unfrozen_blocks": [int(index + 1) for index in inputs.trainable_block_indices],
            "trainable_backbone_params": trainable_backbone_parameter_count,
            "trainable_head_params": trainable_head_parameter_count,
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
        "best_population_validation": dict(selected_val),
        "unseen_target_final_test": dict(target_final),
        "best_validation": dict(selected_val),
        "final_test": dict(target_final),
        "reproducibility": {
            "git_commit": inputs.git_commit,
            "backbone_sha256": inputs.backbone_sha256,
            "preprocessing_hash": inputs.preprocessing_hash,
            "source_trial_encoding": (
                "(subject_id << 32) | file_local_trial_id; Workload "
                "file_local_trial_id=(S<n> << 20) | trial_ordinal"
            ),
        },
    }


def _build_summary(
    *,
    inputs: TrainingArtifactInputs,
    saved_path: Path,
    report_path: Path,
    selected_val: Mapping[str, Any],
    target_final: Mapping[str, Any],
    lora_targets: Sequence[str],
    trainable_backbone_parameter_count: int,
    trainable_lora_parameter_count: int,
    trainable_head_parameter_count: int,
) -> dict[str, Any]:
    """Build the unchanged compact ``summary.json`` payload."""
    args = inputs.args
    config = inputs.config
    return {
        "status": "completed",
        "split_mode": args.split_mode,
        "data_reader": args.data_reader,
        "subject_identities": inputs.subject_identities,
        "target_subject": inputs.target_subject,
        "population_subjects": list(inputs.population_subjects),
        "within_subject": inputs.within_subject_metadata,
        "train_session": (
            inputs.loso_train_session
            if args.split_mode == "loso"
            else (
                inputs.within_subject_train_sessions[0]
                if len(inputs.within_subject_train_sessions) == 1
                else None
            )
        ),
        "train_sessions": (
            [inputs.loso_train_session]
            if args.split_mode == "loso"
            else list(inputs.within_subject_train_sessions)
        ),
        "test_session": inputs.final_test_session,
        "validation_ratio": float(args.validation_ratio) if args.split_mode == "within-subject" else None,
        "num_classes": inputs.num_classes,
        "class_names": list(inputs.class_names),
        "label_mapping": inputs.label_mapping,
        "best_epoch": inputs.best_epoch,
        "population_validation": dict(selected_val),
        "unseen_target_final_test": dict(target_final),
        "classifier_checkpoint": str(saved_path),
        "report": str(report_path),
        "head_type": config.head_type,
        "head_hidden_dim": config.head_hidden_dim,
        "head_dropout": config.head_dropout,
        "head_norm": config.head_norm,
        "head_lr": args.head_lr,
        "backbone_adaptation": inputs.backbone_adaptation,
        "backbone_lr": args.backbone_lr,
        "embedding_layer": inputs.embedding_layer,
        "embedding_layer_internal_index": config.output_layer_idx,
        "unfreeze_last_n_blocks": args.unfreeze_last_n_blocks,
        "unfrozen_blocks": [int(index + 1) for index in inputs.trainable_block_indices],
        "trainable_backbone_params": trainable_backbone_parameter_count,
        "trainable_head_params": trainable_head_parameter_count,
        "lora_last_n_blocks": args.lora_last_n_blocks if inputs.lora_enabled else None,
        "lora_blocks": [
            int(index + 1)
            for index in (inputs.trainable_block_indices if inputs.lora_enabled else ())
        ],
        "lora_target_modules": list(lora_targets),
        "lora_rank": args.lora_rank if inputs.lora_enabled else None,
        "lora_alpha": args.lora_alpha if inputs.lora_enabled else None,
        "lora_dropout": args.lora_dropout if inputs.lora_enabled else None,
        "lora_lr": args.lora_lr if inputs.lora_enabled else None,
        "trainable_lora_params": trainable_lora_parameter_count,
        "trainable_total_params": inputs.total_trainable_parameter_count,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
    }
