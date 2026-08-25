"""Public entry point and compatibility facade for 50M Stage-1 training."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bci_dayloop.models.model_50m.backbone import Model50MBackbone
from bci_dayloop.models.model_50m.classifier import Model50MClassifier
from bci_dayloop.models.model_50m.config import STANDARD_64_CHANNELS, Model50MConfig
from bci_dayloop.training.model_50m.adaptation import (
    configure_adaptation,
    forward_live_logits,
    load_lora_state_dict,
    lora_state_dict,
    normalize_lora_target_modules,
    resolve_adaptation_plan,
    tokenize_windows_for_finetuning,
)
from bci_dayloop.training.model_50m.artifacts import (
    TrainingArtifactInputs,
    TrainingArtifactResult,
    atomic_write_json,
    create_run_directory,
    current_git_commit,
    prepare_training_artifact_paths,
    save_initial_run_config,
    save_training_artifacts,
    sha256_file,
    stable_json_hash,
)
from bci_dayloop.training.model_50m.config import (
    PopulationTrainingConfig,
    build_argument_parser,
    build_parser,
    parse_args,
    parse_training_config,
    training_config_from_namespace,
    validate_args,
)
from bci_dayloop.training.model_50m.data import (
    DataReaderName,
    WithinSubjectTrialSplit,
    build_population_split,
    build_subject_identities,
    build_subject_window_bundle,
    build_window_bundle_from_session_data,
    build_within_subject_split_metadata,
    build_within_subject_splits,
    build_within_subject_test_split,
    class_counts,
    class_name_counts,
    combine_window_bundles,
    concatenate_session_trial_data,
    encode_trial_ids,
    encoded_trial_id,
    normalize_subjects,
    resolve_class_names,
    resolve_subject_file,
    select_direct_trial_segment,
    select_trial_ids,
    select_trial_rows,
    source_id_set,
    validate_labels,
    validate_loaded_session,
    validate_metadata_compatibility,
    validate_no_source_leakage,
)
from bci_dayloop.training.model_50m.evaluation import evaluate_heldout, extend_metrics
from bci_dayloop.training.model_50m.features import (
    build_feature_cache_split_identity,
    extract_frozen_features,
    feature_cache_dtype_from_name,
    population_feature_cache_path,
    save_population_feature_cache,
)
from bci_dayloop.training.model_50m.linear_head import (
    EpochMetrics,
    WindowSet,
    confusion_to_metrics,
    metric_is_better,
    resolve_repo_path,
    run_head_epoch,
    set_seed,
)
from bci_dayloop.training.model_50m.types import ExtendedMetrics, SplitBuildResult, WindowBundle
from bci_dayloop.training.model_50m.workflows import (
    WorkflowResult,
    run_loso_workflow,
    run_within_subject_workflow,
)
from bci_dayloop.utils.config import project_root

ROOT = project_root()

# Imports above intentionally remain direct aliases: the historical population
# script was a de-facto helper module, and the personal trainer/tests still use
# these public names.  The only formal implementations live in their modules.


def run_training(config: PopulationTrainingConfig) -> WorkflowResult:
    """Dispatch a normalized training configuration to its split workflow."""
    if config.split_mode == "loso":
        return run_loso_workflow(config)
    if config.split_mode == "within-subject":
        return run_within_subject_workflow(config)
    raise ValueError(f"Unsupported split mode: {config.split_mode!r}.")


def run_population_training(
    args: PopulationTrainingConfig | object | None = None,
) -> WorkflowResult:
    """Compatibility entry point for the established population CLI/API."""
    if args is None:
        config = parse_training_config()
    elif isinstance(args, PopulationTrainingConfig):
        validate_args(args.as_namespace())
        config = args
    else:
        config = training_config_from_namespace(args)
    # This validation was formerly performed by runner before all workflow
    # side effects; it remains a model-contract check, not a parser branch.
    normalize_lora_target_modules(config.lora_target_modules)
    return run_training(config)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the population CLI while preserving direct-module execution."""
    run_population_training(parse_training_config(argv))
    return 0


__all__ = [name for name in globals() if not name.startswith("_")]


if __name__ == "__main__":
    main()
