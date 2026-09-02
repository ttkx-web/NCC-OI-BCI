"""Train the first frozen-1B flatten linear population classification head."""
from _bootstrap import ROOT  # noqa: F401

# Keep 50M-compatible data/split utilities discoverable from the script.
from bci_dayloop.training.model_1b.population import *  # noqa: F403
from bci_dayloop.training.model_1b.population import main


if __name__ == "__main__":
    main()
