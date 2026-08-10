from __future__ import annotations

import argparse
import os
import subprocess
import sys

from _bootstrap import ROOT  # noqa: F401
from bci_dayloop.data.bnci import prepare_bnci2014_001_subject
from bci_dayloop.training.pipeline import train_linear_probe
from bci_dayloop.utils.config import load_yaml, resolve_path
from bci_dayloop.training.labram_linear_head import (
    train_labram_linear_head,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete BNCI S01 day-one pipeline")
    parser.add_argument("--config", default="configs/day1_bnci_s01.yaml")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--random-init", action="store_true", help="Smoke testing only; not a usable pretrained model")
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    config = load_yaml(config_path)
    if args.random_init:
        config["model"]["random_init"] = True
    data = config["data"]
    output = resolve_path(data["output_hdf5"])
    if args.force_prepare or not output.exists():
        print("[1/3] Preparing BNCI2014_001 Subject 1...")
        prepare_bnci2014_001_subject(
            int(data["subject"]),
            output,
            trial_tmin_sec=float(data.get("trial_tmin_sec", 2.0)),
            trial_tmax_sec=float(data.get("trial_tmax_sec", 6.0)),
        )
    else:
        print(f"[1/3] Reusing dataset: {output}")
    print("[2/3] Extracting/caching embeddings and training linear head...")
    head_path, metrics = (
        train_labram_linear_head(config)
    )

    print(
        f"Test accuracy: "
        f"{metrics['final_test']['accuracy']:.4f}"
    )

    print(
        "Linear head trained. "
        "Run export_labram_model_package.py "
        "to create the deployment package."
    )
    print(f"Test accuracy: {metrics['test']['accuracy']:.4f}")
    print("[3/3] Verifying package reload in a clean Python process...")
    verify_code = (
        "from bci_dayloop.models.factory import ModelFactory; "
        f"m=ModelFactory.load_package(r'{package}', device='cpu'); "
        "print(m.model_name, m.n_classes)"
    )
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run([sys.executable, "-c", verify_code], cwd=ROOT, env=child_env, check=True)
    print(f"Pipeline complete. Model package: {package}")


if __name__ == "__main__":
    main()
