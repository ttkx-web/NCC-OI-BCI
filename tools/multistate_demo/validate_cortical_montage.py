#!/usr/bin/env python3
"""Validate a named multi-state-demo cortical montage config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from bci_dayloop.demo.cortical_montage import load_cortical_montage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Registry name, for example bnci_22")
    args = parser.parse_args()
    montage = load_cortical_montage(args.config)
    counts = {"left": 0, "right": 0}
    for anchors in montage.channels.values():
        for anchor in anchors:
            counts[anchor.hemisphere] += 1
    print(f"valid: {montage.name}")
    print(f"device: {montage.device_name}")
    print(f"mapped channels: {len(montage.channels)}")
    print(f"anchors: left={counts['left']}, right={counts['right']}")


if __name__ == "__main__":
    main()
