"""50M EEG foundation model integration package."""

from importlib import import_module
from typing import Any

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
    "Model50MRuntime",
    "Model50MRuntimePrediction",
    "build_50m_runtime",
    "build_50m_runtime_from_metadata",
]


_EXPORTS: dict[str, tuple[str, str]] = {
    "AdapterTiming": (".adapter", "AdapterTiming"),
    "Model50MAdapter": (".adapter", "Model50MAdapter"),
    "RawPredictionResult": (".adapter", "RawPredictionResult"),
    "build_model50m_adapter": (".adapter", "build_model50m_adapter"),
    "Model50MBackbone": (".backbone", "Model50MBackbone"),
    "Model50MClassifier": (".classifier", "Model50MClassifier"),
    "Model50MConfig": (".config", "Model50MConfig"),
    "Model50MPreprocessor": (".preprocessing", "Model50MPreprocessor"),
    "Model50MTokenizer": (".tokenization", "Model50MTokenizer"),
    "Model50MRuntime": (".runtime", "Model50MRuntime"),
    "Model50MRuntimePrediction": (".runtime", "Model50MRuntimePrediction"),
    "build_50m_runtime": (".runtime", "build_50m_runtime"),
    "build_50m_runtime_from_metadata": (
        ".runtime",
        "build_50m_runtime_from_metadata",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
