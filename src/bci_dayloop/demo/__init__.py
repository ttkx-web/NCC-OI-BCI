"""Self-contained EEG multi-state visualization demo.

This package deliberately does not depend on the production runtime, replay,
or model packages.  It is a small visualization layer with replaceable decoder
interfaces for product demonstrations.
"""

from bci_dayloop.demo.schemas import BrainStateResult
from bci_dayloop.demo.state_decoder import DemoStateDecoder

__all__ = ["BrainStateResult", "DemoStateDecoder"]
