from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT  # noqa: F401
from bci_dayloop.training.pipeline import train_linear_probe
from bci_dayloop.utils.config import load_yaml, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache LaBraM embeddings and train only nn.Linear")
    parser.add_argument("--config", default="configs/day1_bnci_s01.yaml")
    parser.add_argument("--random-init", action="store_true", help="Smoke testing only")
    args = parser.parse_args()
    config = load_yaml(resolve_path(args.config))
    if args.random_init:
        config["model"]["random_init"] = True
    package, metrics = train_linear_probe(config)
    print(f"Model package: {package}")
    print(json.dumps(metrics["test"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

