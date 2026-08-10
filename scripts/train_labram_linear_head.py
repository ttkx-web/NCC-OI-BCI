from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT  # noqa: F401

from bci_dayloop.training.labram_linear_head import (
    train_labram_linear_head,
)
from bci_dayloop.utils.config import (
    load_yaml,
    resolve_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the LaBraM encoder and train "
            "only the linear classification head."
        )
    )

    parser.add_argument(
        "--config",
        default=(
            "configs/stage0/"
            "day1_bnci_s01.yaml"
        ),
    )

    parser.add_argument(
        "--device",
        choices=(
            "cpu",
            "cuda",
            "mps",
        ),
        default=None,
        help=(
            "Optional device override. "
            "When omitted, use the YAML value."
        ),
    )

    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Smoke testing only.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = load_yaml(
        resolve_path(args.config)
    )

    if args.device is not None:
        config["model"]["device"] = (
            args.device
        )

    if args.random_init:
        config["model"]["random_init"] = True

    head_path, metrics = (
        train_labram_linear_head(config)
    )

    print()
    print("LaBraM linear-head training completed.")
    print("Head checkpoint:", head_path)
    print(
        json.dumps(
            metrics["final_test"],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()