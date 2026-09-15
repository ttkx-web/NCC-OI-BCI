"""Generate strict, subject-sharded frozen 50M features for Awakening."""

from __future__ import annotations

import argparse
import json

try:
    from _bootstrap import ROOT  # noqa: F401
except ModuleNotFoundError:
    from scripts._bootstrap import ROOT  # noqa: F401

from bci_dayloop.data.dataset_registry import AWAKENING_2S
from bci_dayloop.training.model_50m.awakening import (
    build_cache_contract,
    generate_awakening_feature_cache,
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
        "--subjects", nargs="+", type=int, default=list(AWAKENING_2S.canonical_subjects)
    )
    parser.add_argument("--target-subject", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="auto")
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument(
        "--feature-cache-dtype",
        choices=("float16", "float32", "bfloat16"),
        default="float16",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-windows-per-class-per-subject",
        type=int,
        default=None,
        help="Debug-only limit; omit for a complete cache.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache_dir = args.cache_dir or (
        f"data/features/awakening_2s/50m_frozen_target{args.target_subject:02d}"
    )
    if args.dry_run:
        audit = validate_awakening_dataset(args.data_root, subjects=args.subjects)
        contract = build_cache_contract(
            audit=audit,
            checkpoint_path=args.backbone_checkpoint,
            target_subject=args.target_subject,
            subjects=args.subjects,
            window_order_seed=args.seed,
            feature_cache_dtype=args.feature_cache_dtype,
        )
        print(json.dumps({"audit": audit.to_dict(), "cache_contract": contract}, indent=2))
        print("Dry run only: no feature cache was written.")
        return 0
    path = generate_awakening_feature_cache(
        data_root=args.data_root,
        cache_dir=cache_dir,
        checkpoint_path=args.backbone_checkpoint,
        subjects=args.subjects,
        target_subject=args.target_subject,
        device=args.device,
        feature_batch_size=args.feature_batch_size,
        cache_dtype_name=args.feature_cache_dtype,
        seed=args.seed,
        overwrite=args.overwrite,
        max_windows_per_class_per_subject=args.max_windows_per_class_per_subject,
    )
    print(f"Awakening feature cache written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
