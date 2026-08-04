from __future__ import annotations

import argparse
import json
import time

import numpy as np

from _bootstrap import ROOT  # noqa: F401
from bci_dayloop.models.labram_linear import LaBraMLinearAdapter
from bci_dayloop.utils.config import resolve_path

DEFAULT_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one LaBraM Base forward-pass smoke test")
    parser.add_argument("--checkpoint", default="checkpoints/backbones/labram/labram_base.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    model = LaBraMLinearAdapter(
        DEFAULT_CHANNELS,
        checkpoint=resolve_path(args.checkpoint),
        device=args.device,
        amp=args.amp,
        random_init=args.random_init,
        n_patches=4,
        embedding_batch_size=1,
    )
    x = np.random.default_rng(42).standard_normal((1, len(DEFAULT_CHANNELS), 4, 200), dtype=np.float32)
    started = time.perf_counter()
    embeddings = model.extract_embeddings(x)
    elapsed = (time.perf_counter() - started) * 1000.0
    print(json.dumps({"shape": list(embeddings.shape), "device": str(model.device), "latency_ms": elapsed}, indent=2))


if __name__ == "__main__":
    main()

