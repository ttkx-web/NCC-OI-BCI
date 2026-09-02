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
    "classifier_input_dim",
    "flatten_token_embeddings",
    "load_backbone_checkpoint",
]
