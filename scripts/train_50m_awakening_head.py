"""Train the frozen 50M Awakening linear head from validated feature shards."""

from __future__ import annotations

import argparse

try:
    from _bootstrap import ROOT  # noqa: F401
except ModuleNotFoundError:
    from scripts._bootstrap import ROOT  # noqa: F401

from bci_dayloop.data.dataset_registry import AWAKENING_2S
from bci_dayloop.training.model_50m.awakening import (
    build_cache_contract,
    load_cache_split,
    train_awakening_head_from_cache,
    validate_awakening_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=AWAKENING_2S.default_data_root)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Default: data/features/awakening_2s/50m_frozen_targetNN.",
    )
    parser.add_argument(
        "--backbone-checkpoint",
        default="checkpoints/backbones/50m/model_deploy.pt",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Default uses the selected target subject under checkpoints/heads/stage1.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Default uses the selected target subject under runs/stage1.",
    )
    parser.add_argument(
        "--subjects", nargs="+", type=int, default=list(AWAKENING_2S.canonical_subjects)
    )
    parser.add_argument("--target-subject", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--head-batch-size", type=int, default=32)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-seed", type=int, default=42)
    parser.add_argument(
        "--feature-cache-dtype",
        choices=("float16", "float32", "bfloat16"),
        default="float16",
    )
    parser.add_argument(
        "--class-weight", choices=("balanced", "none"), default="balanced"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data/cache contracts and class coverage; do not train or write.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache_dir = args.cache_dir or (
        f"data/features/awakening_2s/50m_frozen_target{args.target_subject:02d}"
    )
    output = args.output or (
        "checkpoints/heads/stage1/awakening/"
        f"subject_{args.target_subject:02d}/population/2s_flatten/head.pt"
    )
    run_dir = args.run_dir or (
        "runs/stage1/awakening/"
        f"subject_{args.target_subject:02d}/population/2s_flatten"
    )
    if args.dry_run:
        audit = validate_awakening_dataset(args.data_root, subjects=args.subjects)
        contract = build_cache_contract(
            audit=audit,
            checkpoint_path=args.backbone_checkpoint,
            target_subject=args.target_subject,
            subjects=args.subjects,
            window_order_seed=args.cache_seed,
            feature_cache_dtype=args.feature_cache_dtype,
        )
        train = load_cache_split(
            cache_dir=cache_dir,
            expected_contract=contract,
            split_name="population_train",
        )
        validation = load_cache_split(
            cache_dir=cache_dir,
            expected_contract=contract,
            split_name="population_validation",
        )
        print(
            "Dry run OK: train/validation samples=",
            len(train.dataset),
            len(validation.dataset),
            "; no optimizer step or output write occurred.",
        )
        return 0
    saved = train_awakening_head_from_cache(
        data_root=args.data_root,
        cache_dir=cache_dir,
        checkpoint_path=args.backbone_checkpoint,
        output_path=output,
        run_dir=run_dir,
        subjects=args.subjects,
        target_subject=args.target_subject,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.head_batch_size,
        learning_rate=args.head_lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        cache_seed=args.cache_seed,
        feature_cache_dtype=args.feature_cache_dtype,
        class_weight=args.class_weight,
        overwrite=args.overwrite,
    )
    print(f"Awakening head written: {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
