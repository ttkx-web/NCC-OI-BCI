"""Frozen-backbone downstream components for the 1B first-version probe.

These classes deliberately define only a flattening linear head.  They do not
provide runtime-package loading, probability serving, or a prediction API.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Model1BConfig


def classifier_input_dim(config: Model1BConfig) -> int:
    """Compute flatten-head width from the active 1B token contract."""
    return int(config.n_channels * config.num_time_patches * config.d_model)


class Model1BFlattenLinearHead(nn.Module):
    """A linear probe over masked final-token 1B encoder embeddings."""

    aggregation = "flatten"
    head_type = "linear"

    def __init__(self, *, input_dim: int, num_classes: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.linear = nn.Linear(self.input_dim, self.num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                "1B linear-head features must have shape "
                f"[B, {self.input_dim}], got {tuple(features.shape)}"
            )
        if features.dtype != torch.float32 or not torch.isfinite(features).all():
            raise ValueError("1B linear-head features must be finite torch.float32")
        return self.linear(features)


def flatten_token_embeddings(
    token_embeddings: torch.Tensor,
    token_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Zero invalid-channel tokens while retaining their stable flatten slots."""
    if token_embeddings.ndim != 3:
        raise ValueError("token_embeddings must have shape [B, S, D]")
    if token_valid_mask.shape != token_embeddings.shape[:2]:
        raise ValueError("token_valid_mask must have shape [B, S]")
    if not torch.isfinite(token_embeddings).all() or not torch.isfinite(token_valid_mask).all():
        raise ValueError("1B token embeddings and validity mask must be finite")
    return (
        token_embeddings
        * token_valid_mask.to(token_embeddings.device, token_embeddings.dtype).unsqueeze(-1)
    ).flatten(start_dim=1)
