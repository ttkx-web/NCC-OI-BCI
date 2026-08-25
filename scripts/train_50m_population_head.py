"""CLI compatibility entry point for Stage-1 50M population-head training."""
from _bootstrap import ROOT  # noqa: F401

# Compatibility re-exports: tests and downstream tooling historically import
# split/cache helpers from this script. Their implementation now lives in src.
from bci_dayloop.training.model_50m.runner import *  # noqa: F403
from bci_dayloop.training.model_50m.runner import main


if __name__ == "__main__":
    main()
