"""1B EEG backbone embedding API; intentionally no classifier or package exporter."""

from .backbone import BackboneLoadReport, EEGBackbone1B, Model1BBackbone, load_backbone_checkpoint
from .config import Model1BConfig
from .classifier import Model1BFlattenLinearHead, classifier_input_dim, flatten_token_embeddings
from .preprocessing import Model1BInputTransform
from .runner import Model1BBackboneRunner, Model1BPreparedInput
from .tokenization import Model1BBatchedInput, Model1BTokenizedInput, Model1BTokenizer

__all__ = [
    "BackboneLoadReport",
    "EEGBackbone1B",
    "Model1BBackbone",
    "Model1BBackboneRunner",
    "Model1BConfig",
    "Model1BFlattenLinearHead",
    "Model1BInputTransform",
    "Model1BPreparedInput",
    "Model1BBatchedInput",
    "Model1BTokenizedInput",
    "Model1BTokenizer",
    "Model1BRuntime",
    "classifier_input_dim",
    "flatten_token_embeddings",
    "build_1b_runtime",
    "load_backbone_checkpoint",
]


def __getattr__(name: str):
    if name in {"Model1BRuntime", "build_1b_runtime"}:
        from .runtime import Model1BRuntime, build_1b_runtime
        return {"Model1BRuntime": Model1BRuntime, "build_1b_runtime": build_1b_runtime}[name]
    raise AttributeError(name)
