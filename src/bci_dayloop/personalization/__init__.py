"""Stage-1 subject-adaptation utilities.

The initial package exposes trial-level splitting and frozen-feature classifier
training.  Model packaging and user-model registration will be added in
``package.py`` and ``registry.py`` without changing these public imports.
"""

from .split import (
    LeaveOneSubjectOutSplit,
    PersonalTrialSplit,
    build_loso_subject_split,
    build_personal_trial_split,
    encode_subject_trial_id,
    encode_subject_trial_ids,
    normalize_subjects,
    select_rows,
    trial_id_set,
    validate_disjoint_trial_ids,
    validate_nested_budgets,
    validate_three_way_trial_split,
)
from .trainer import (
    ClassifierMetrics,
    ClassifierTrainingConfig,
    ClassifierTrainingResult,
    EvaluationResult,
    build_optimizer,
    build_scheduler,
    clone_frozen_module,
    compare_classifier_heads,
    evaluate_classifier,
    reset_module_parameters,
    resolve_head_device,
    run_classifier_epoch,
    set_seed,
    train_classifier_head,
)

__all__ = [
    "ClassifierMetrics",
    "ClassifierTrainingConfig",
    "ClassifierTrainingResult",
    "EvaluationResult",
    "LeaveOneSubjectOutSplit",
    "PersonalTrialSplit",
    "build_loso_subject_split",
    "build_optimizer",
    "build_personal_trial_split",
    "build_scheduler",
    "clone_frozen_module",
    "compare_classifier_heads",
    "encode_subject_trial_id",
    "encode_subject_trial_ids",
    "evaluate_classifier",
    "normalize_subjects",
    "reset_module_parameters",
    "resolve_head_device",
    "run_classifier_epoch",
    "select_rows",
    "set_seed",
    "train_classifier_head",
    "trial_id_set",
    "validate_disjoint_trial_ids",
    "validate_nested_budgets",
    "validate_three_way_trial_split",
]
