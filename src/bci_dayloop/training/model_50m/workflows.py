"""High-level Stage-1 50M training workflows.

This module sequences established data, feature, adaptation, engine,
evaluation and artifact components.  It intentionally does not reimplement
any of those component algorithms.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bci_dayloop.models.model_50m.backbone import Model50MBackbone
from bci_dayloop.models.model_50m.classifier import Model50MClassifier
from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.training.model_50m.adaptation import configure_adaptation, resolve_adaptation_plan, tokenize_windows_for_finetuning
from bci_dayloop.training.model_50m.artifacts import TrainingArtifactInputs, TrainingArtifactResult, prepare_training_artifact_paths, save_initial_run_config, save_training_artifacts, stable_json_hash
from bci_dayloop.training.model_50m.config import PopulationTrainingConfig
from bci_dayloop.training.model_50m.data import WithinSubjectTrialSplit, build_population_split, build_subject_identities, build_within_subject_split_metadata, build_within_subject_splits, build_within_subject_test_split, class_name_counts, normalize_subjects, resolve_class_names, resolve_subject_file, validate_labels, validate_no_source_leakage
from bci_dayloop.training.model_50m.engine import TrainingResult, build_optimizer, fit_with_early_stopping
from bci_dayloop.training.model_50m.evaluation import evaluate_heldout, extend_metrics
from bci_dayloop.training.model_50m.features import build_feature_cache_split_identity, extract_frozen_features, feature_cache_dtype_from_name, population_feature_cache_path, save_population_feature_cache
from bci_dayloop.training.model_50m.linear_head import resolve_repo_path, set_seed
from bci_dayloop.training.model_50m.types import ExtendedMetrics
from bci_dayloop.utils.paths import population_head_path, population_run_dir


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    """In-memory result of one completed Stage-1 workflow."""

    training_result: TrainingResult
    target_metrics: ExtendedMetrics
    artifacts: TrainingArtifactResult


def run_loso_workflow(config: PopulationTrainingConfig) -> WorkflowResult:
    """Run the established population LOSO protocol."""
    if config.split_mode != "loso":
        raise ValueError(f"LOSO workflow received split mode {config.split_mode!r}.")
    return _run_training_workflow(config)


def run_within_subject_workflow(config: PopulationTrainingConfig) -> WorkflowResult:
    """Run the established one-subject session-isolated protocol."""
    if config.split_mode != "within-subject":
        raise ValueError(
            "Within-subject workflow received split mode "
            f"{config.split_mode!r}."
        )
    return _run_training_workflow(config)


def _run_training_workflow(args: PopulationTrainingConfig) -> WorkflowResult:
    """Shared stages after the LOSO/within-subject split choice."""
    target_subject = int(args.target_subject)
    requested_train_sessions = [str(session) for session in args.train_session]
    if args.split_mode == "loso":
        if len(requested_train_sessions) != 1:
            raise ValueError(
                "--split-mode loso accepts exactly one --train-session. "
                f"Received {requested_train_sessions!r}."
            )
        loso_train_session = requested_train_sessions[0]
        within_subject_train_sessions: list[str] = []
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
        loso_train_session = None
        within_subject_train_sessions = requested_train_sessions
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
                "workload_pbci_hackathon"
                if args.data_reader == "workload"
                else "bnci2014_001"
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

    if args.run_dir is None:
        run_dir = population_run_dir(
            stage="stage1",
            dataset=(
                "workload_pbci_hackathon"
                if args.data_reader == "workload"
                else "bnci2014_001"
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

    artifact_paths = prepare_training_artifact_paths(
        run_dir=run_dir,
        output_path=output_path,
        overwrite=args.overwrite,
        backbone_checkpoint=checkpoint_path,
    )
    run_dir = artifact_paths.run_dir
    output_path = artifact_paths.output_path
    git_commit = artifact_paths.git_commit
    backbone_sha256 = artifact_paths.backbone_sha256

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
        print("population train session:", loso_train_session)
        print("population validation session:", args.validation_session)
        print("final unseen-subject test:", f"{target_tag}/{args.final_test_session}")
    else:
        print(
            "within-subject train source sessions:",
            ", ".join(within_subject_train_sessions),
        )
        print("within-subject final test session:", args.test_session)
        print("within-subject validation ratio:", args.validation_ratio)
        print("within-subject split seed:", args.seed)
    print("data root:", data_root)
    print("data pattern:", args.data_pattern)
    print("data reader:", args.data_reader)
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
    subject_identities = build_subject_identities(
        subject_paths=all_subject_paths,
        data_reader=args.data_reader,
    )

    save_initial_run_config(
        paths=artifact_paths,
        timestamp=timestamp,
        target_subject=target_subject,
        population_subjects=population_subjects,
        split_mode=args.split_mode,
        loso_train_session=loso_train_session,
        within_subject_train_sessions=within_subject_train_sessions,
        validation_session=args.validation_session,
        final_test_session=(
            args.final_test_session if args.split_mode == "loso" else args.test_session
        ),
        backbone_checkpoint=checkpoint_path,
        validation_ratio=args.validation_ratio,
        data_root=data_root,
        data_pattern=args.data_pattern,
        data_reader=args.data_reader,
        subject_identities=subject_identities,
        subject_paths=all_subject_paths,
        # PopulationTrainingConfig wraps the argparse Namespace.  Persist the
        # namespace fields themselves, matching the pre-refactor run config.
        arguments=vars(args.as_namespace()),
    )

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
            data_reader=args.data_reader,
            session_name=str(loso_train_session),
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
            data_reader=args.data_reader,
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
                data_reader=args.data_reader,
                train_sessions=within_subject_train_sessions,
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
        train_session_counts = {
            session: int(
                np.sum(
                    np.asarray(within_subject_all_trials["session_ids"]).astype(str)
                    == session
                )
            )
            for session in within_subject_split.train_sessions
        }
        print("Within-subject train source sessions:")
        for session, count in train_session_counts.items():
            print(f"  {session}: {count}")
        print(
            "Combined train-source trials:",
            sum(train_session_counts.values()),
            class_name_counts(
                within_subject_all_trials["labels"][
                    np.isin(
                        np.asarray(within_subject_all_trials["session_ids"]).astype(str),
                        within_subject_split.train_sessions,
                    )
                ],
                class_names,
            ),
        )
        print(
            "Train / validation source trials:",
            len(within_subject_split.train_indices),
            "/",
            len(within_subject_split.validation_indices),
        )
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

    adaptation_plan = resolve_adaptation_plan(
        config=config,
        requested_embedding_layer=str(args.embedding_layer),
        requested_adaptation=args.backbone_adaptation,
        unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
        lora_last_n_blocks=args.lora_last_n_blocks,
    )
    embedding_layer = adaptation_plan.embedding_layer
    # A manual CLI layer is user-facing 1-based; Model50MConfig and the
    # backbone remain 0-based internally.
    if int(config.output_layer_idx) != embedding_layer - 1:
        config = replace(
            config,
            output_layer_idx=embedding_layer - 1,
        )
    backbone_adaptation = adaptation_plan.mode
    partial_finetuning_enabled = adaptation_plan.partial_finetuning_enabled
    lora_enabled = adaptation_plan.lora_enabled
    requires_live_backbone_forward = adaptation_plan.requires_live_backbone_forward
    trainable_block_indices = adaptation_plan.trainable_block_indices
    feature_cache_enabled = adaptation_plan.feature_cache_enabled

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
    # Model50MClassifier initializes as a frozen probe. The formal adaptation
    # layer owns partial/LoRA scope selection and trainable-parameter checks.
    adaptation_setup = configure_adaptation(
        backbone=backbone,
        classifier=classifier,
        plan=adaptation_plan,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    lora_parameters = adaptation_setup.lora_parameters
    lora_parameter_ids = adaptation_setup.lora_parameter_ids
    trainable_backbone_parameters = adaptation_setup.trainable_backbone_parameters
    original_backbone_parameter_count = adaptation_setup.original_backbone_parameter_count
    trainable_lora_parameter_count = adaptation_setup.trainable_lora_parameter_count
    trainable_head_parameter_count = adaptation_setup.trainable_head_parameter_count
    trainable_original_backbone_parameter_count = (
        adaptation_setup.trainable_original_backbone_parameter_count
    )
    total_model_parameter_count = adaptation_setup.total_model_parameter_count
    total_trainable_parameter_count = adaptation_setup.total_trainable_parameter_count
    model_load_seconds = time.perf_counter() - load_start

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
    feature_cache_split_identity = build_feature_cache_split_identity(
        split_mode=args.split_mode,
        train_sessions=(
            [str(loso_train_session)]
            if args.split_mode == "loso"
            else within_subject_train_sessions
        ),
        test_session=(
            args.final_test_session if args.split_mode == "loso" else args.test_session
        ),
        validation_session=(
            args.validation_session if args.split_mode == "loso" else None
        ),
        validation_ratio=(
            None if args.split_mode == "loso" else args.validation_ratio
        ),
        split_seed=(None if args.split_mode == "loso" else args.seed),
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
            path=population_feature_cache_path(
                run_dir, split_name="population_train"
            ),
            split_name="population_train",
            class_names=class_names,
            subject_ids=train_validation_subjects,
            data_reader=args.data_reader,
            subject_identities=subject_identities,
            backbone_sha256=backbone_sha256,
            preprocessing_hash=preprocessing_hash,
            split_identity=feature_cache_split_identity,
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
            path=population_feature_cache_path(
                run_dir, split_name="population_validation"
            ),
            split_name="population_validation",
            class_names=class_names,
            subject_ids=train_validation_subjects,
            data_reader=args.data_reader,
            subject_identities=subject_identities,
            backbone_sha256=backbone_sha256,
            preprocessing_hash=preprocessing_hash,
            split_identity=feature_cache_split_identity,
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
    optimizer = build_optimizer(
        classifier=classifier, backbone=backbone, head_parameters=head_parameters,
        trainable_backbone_parameters=trainable_backbone_parameters,
        lora_parameters=lora_parameters, lora_parameter_ids=lora_parameter_ids,
        head_lr=args.head_lr, backbone_lr=args.backbone_lr, lora_lr=args.lora_lr,
        weight_decay=args.weight_decay, partial_enabled=partial_finetuning_enabled,
        lora_enabled=lora_enabled,
    )
    criterion = nn.CrossEntropyLoss()
    training_result = fit_with_early_stopping(
        classifier=classifier, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, num_classes=num_classes, class_names=class_names,
        optimizer=optimizer, epochs=args.epochs, patience=args.patience,
        metric_for_best=args.metric_for_best, live=requires_live_backbone_forward,
        partial_enabled=partial_finetuning_enabled, lora_enabled=lora_enabled,
        extend_metrics=extend_metrics,
    )
    epoch_rows = training_result.epoch_rows
    best_epoch = training_result.best_epoch
    selected_val_metrics = training_result.selected_val_metrics
    training_seconds = training_result.training_seconds

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
            data_reader=args.data_reader,
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
            data_reader=args.data_reader,
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
            path=population_feature_cache_path(
                run_dir, split_name="target_final_test"
            ),
            split_name="target_final_test",
            class_names=class_names,
            subject_ids=[target_subject],
            data_reader=args.data_reader,
            subject_identities=subject_identities,
            backbone_sha256=backbone_sha256,
            preprocessing_hash=preprocessing_hash,
            split_identity=feature_cache_split_identity,
        )

    target_metrics = evaluate_heldout(
        classifier=classifier,
        dataset=target_features,
        criterion=criterion,
        num_classes=num_classes,
        class_names=class_names,
        batch_size=args.head_batch_size,
        live=requires_live_backbone_forward,
    )

    if within_subject_split is not None:
        assert within_subject_all_trials is not None
        within_subject_metadata = build_within_subject_split_metadata(
            subject_id=target_subject,
            split=within_subject_split,
            all_trial_metadata=within_subject_all_trials,
            class_names=class_names,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
        )
    else:
        within_subject_metadata = None

    # ------------------------------------------------------------------
    # Save model and reports. Artifact assembly is owned by artifacts.py.
    # ------------------------------------------------------------------

    artifact_result = save_training_artifacts(
        TrainingArtifactInputs(
            run_dir=run_dir,
            output_path=output_path,
            classifier=classifier,
            args=args,
            config=config,
            backbone=backbone,
            checkpoint_path=checkpoint_path,
            git_commit=git_commit,
            backbone_sha256=backbone_sha256,
            target_subject=target_subject,
            population_subjects=population_subjects,
            subjects=subjects,
            loso_train_session=loso_train_session,
            within_subject_train_sessions=within_subject_train_sessions,
            final_test_session=final_test_session,
            embedding_layer=embedding_layer,
            subject_identities=subject_identities,
            all_subject_paths=all_subject_paths,
            metadata=metadata,
            class_names=class_names,
            label_mapping=label_mapping,
            num_classes=num_classes,
            within_subject_metadata=within_subject_metadata,
            backbone_adaptation=backbone_adaptation,
            partial_finetuning_enabled=partial_finetuning_enabled,
            lora_enabled=lora_enabled,
            trainable_backbone_parameters=trainable_backbone_parameters,
            trainable_block_indices=trainable_block_indices,
            lora_parameters=lora_parameters,
            total_trainable_parameter_count=total_trainable_parameter_count,
            head_parameters=head_parameters,
            selected_val_metrics=selected_val_metrics,
            target_metrics=target_metrics,
            train_build=train_build,
            val_build=val_build,
            target_build=target_build,
            feature_cache_enabled=feature_cache_enabled,
            preprocessing_contract=preprocessing_contract,
            preprocessing_hash=preprocessing_hash,
            model_load_seconds=model_load_seconds,
            training_seconds=training_seconds,
            best_epoch=best_epoch,
            epoch_rows=epoch_rows,
        )
    )
    saved_path = artifact_result.checkpoint_path
    report_path = artifact_result.report_path

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

    return WorkflowResult(
        training_result=training_result,
        target_metrics=target_metrics,
        artifacts=artifact_result,
    )
