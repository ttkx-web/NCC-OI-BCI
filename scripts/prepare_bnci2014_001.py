from __future__ import annotations

import argparse

from _bootstrap import ROOT  # noqa: F401
from bci_dayloop.data.bnci import prepare_bnci2014_001_subject
from bci_dayloop.utils.config import load_yaml, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BNCI2014_001 Subject 1 as standard HDF5")
    parser.add_argument("--config", default="configs/day1_bnci_s01.yaml")
    parser.add_argument("--subject", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_yaml(resolve_path(args.config))
    data = config["data"]
    output = resolve_path(args.output or data["output_hdf5"])
    path = prepare_bnci2014_001_subject(
        args.subject or int(data["subject"]),
        output,
        trial_tmin_sec=float(data.get("trial_tmin_sec", 2.0)),
        trial_tmax_sec=float(data.get("trial_tmax_sec", 6.0)),
    )
    print(f"Prepared dataset: {path}")


if __name__ == "__main__":
    main()

