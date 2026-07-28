"""50M EEG foundation model integration package."""

from .adapter import (
    AdapterTiming,
    Model50MAdapter,
    RawPredictionResult,
    build_model50m_adapter,
)
from .backbone import Model50MBackbone
from .classifier import Model50MClassifier
from .config import Model50MConfig
from .preprocessing import Model50MPreprocessor
from .tokenization import Model50MTokenizer

__all__ = [
    "AdapterTiming",
    "Model50MAdapter",
    "Model50MBackbone",
    "Model50MClassifier",
    "Model50MConfig",
    "Model50MPreprocessor",
    "Model50MTokenizer",
    "RawPredictionResult",
    "build_model50m_adapter",
]