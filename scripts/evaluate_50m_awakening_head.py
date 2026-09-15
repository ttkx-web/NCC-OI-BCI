"""Offline evaluation for a frozen 50M Awakening classifier checkpoint."""

from __future__ import annotations

import argparse
import json

try:
    from _bootstrap import ROOT  # noqa: F401
except ModuleNotFoundError:
    from scripts._bootstrap import ROOT  # noqa: F401

from bci_dayloop.data.dataset_registry import AWAKENING_2S
from bci_dayloop.training.model_50m.awakening import evaluate_awakening_checkpoint


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
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--subjects", nargs="+", type=int, default=list(AWAKENING_2S.canonical_subjects)
    )
    parser.add_argument("--target-subject", type=int, default=1)
    parser.add_argument(
        "--split",
        choices=("population_validation", "target_final_test"),
        default="target_final_test",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-seed", type=int, default=42)
    parser.add_argument(
        "--feature-cache-dtype",
        choices=("float16", "float32", "bfloat16"),
        default="float16",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache_dir = args.cache_dir or (
        f"data/features/awakening_2s/50m_frozen_target{args.target_subject:02d}"
    )
    result = evaluate_awakening_checkpoint(
        data_root=args.data_root,
        cache_dir=cache_dir,
        backbone_checkpoint=args.backbone_checkpoint,
        classifier_checkpoint=args.classifier_checkpoint,
        output_path=args.output,
        subjects=args.subjects,
        target_subject=args.target_subject,
        split_name=args.split,
        device=args.device,
        batch_size=args.batch_size,
        cache_seed=args.cache_seed,
        feature_cache_dtype=args.feature_cache_dtype,
        overwrite=args.overwrite,
    )
    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
