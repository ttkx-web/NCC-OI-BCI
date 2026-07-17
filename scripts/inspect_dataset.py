from __future__ import annotations

import argparse
import json

import h5py
import numpy as np

from _bootstrap import ROOT  # noqa: F401
from bci_dayloop.data.hdf5_dataset import EEGHDF5
from bci_dayloop.utils.config import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect BCI DayLoop HDF5 metadata and splits")
    parser.add_argument("path", nargs="?", default="data/processed/bnci2014_001_s01.h5")
    args = parser.parse_args()
    path = resolve_path(args.path)
    dataset = EEGHDF5(path)
    metadata = dataset.metadata
    with h5py.File(path, "r") as handle:
        shape = list(handle["data"].shape)
    counts = {}
    for session in dataset.sessions():
        loaded = dataset.load(session)
        unique, values = np.unique(loaded["labels"], return_counts=True)
        counts[session] = {metadata.class_names[int(k)]: int(v) for k, v in zip(unique, values)}
    print(
        json.dumps(
            {"path": str(path), "shape": shape, "metadata": metadata.__dict__, "sessions": counts},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

