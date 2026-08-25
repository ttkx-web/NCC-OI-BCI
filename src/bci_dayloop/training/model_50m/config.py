"""CLI contract and static configuration validation for 50M population training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence


@dataclass
class PopulationTrainingConfig:
    """Normalized argparse values; attribute access preserves legacy dest names."""
    namespace: argparse.Namespace

    def __getattr__(self, name: str):
        return getattr(self.namespace, name)

    def as_namespace(self) -> argparse.Namespace:
        return self.namespace


def build_parser() -> argparse.ArgumentParser:
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
        "--data-reader",
        choices=("eeg", "workload"),
        default="eeg",
        help=(
            "HDF5 reader format. Default 'eeg' preserves the existing "
            "flat EEGHDF5 behavior; choose 'workload' for grouped WorkloadHDF5."
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

    parser.add_argument(
        "--train-session",
        nargs="+",
        default=["0train"],
        help=(
            "One or more source sessions. LOSO accepts exactly one session; "
            "within-subject concatenates all requested train sessions before "
            "one global stratified validation split."
        ),
    )
    parser.add_argument("--validation-session", default="1test")
    parser.add_argument("--final-test-session", default="1test")
    parser.add_argument(
        "--test-session",
        default=None,
        help=(
            "Held-out final-test session for --split-mode within-subject. "
            "Must not appear in --train-session."
        ),
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help=(
            "Trial-level validation fraction of the combined --train-session "
            "source trials for "
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
        "--overwrite",
        action="store_true",
        help=(
            "Allow an existing --run-dir to be reused, overwriting files "
            "produced by this training run."
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


def build_argument_parser() -> argparse.ArgumentParser:
    """Compatibility alias for historical population trainer imports."""
    return build_parser()

def validate_args(args: argparse.Namespace) -> None:
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



def training_config_from_namespace(args: argparse.Namespace) -> PopulationTrainingConfig:
    validate_args(args)
    return PopulationTrainingConfig(args)

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)

def parse_training_config(argv: Sequence[str] | None = None) -> PopulationTrainingConfig:
    return training_config_from_namespace(parse_args(argv))
