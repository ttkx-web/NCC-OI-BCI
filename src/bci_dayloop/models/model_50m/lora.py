from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize

from .backbone import Model50MBackbone


VALID_LORA_TARGET_MODULES = ("q", "k", "v")


class FusedQKVLoRA(nn.Module):
    """Low-rank update for selected Q/K/V rows of fused MHA projection."""

    def __init__(
        self,
        *,
        embed_dim: int,
        rank: int,
        alpha: float,
        dropout: float,
        target_modules: Sequence[str],
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}.")
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")
        if alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {alpha}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {dropout}.")

        targets = tuple(dict.fromkeys(str(target) for target in target_modules))
        invalid_targets = [
            target for target in targets if target not in VALID_LORA_TARGET_MODULES
        ]
        if not targets or invalid_targets:
            raise ValueError(
                "LoRA fused-MHA targets must be one or more of "
                f"{VALID_LORA_TARGET_MODULES}, got {list(targets)}."
            )

        self.embed_dim = int(embed_dim)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.target_modules = targets
        self.lora_A = nn.Parameter(
            torch.empty(len(self.target_modules), self.rank, self.embed_dim)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(len(self.target_modules), self.embed_dim, self.rank)
        )
        self.dropout = nn.Dropout(float(dropout))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        expected_shape = (3 * self.embed_dim, self.embed_dim)
        if tuple(weight.shape) != expected_shape:
            raise ValueError(
                "Fused QKV projection has incompatible shape: "
                f"expected {expected_shape}, got {tuple(weight.shape)}."
            )

        # B starts at zero, making injection an exact identity initially.
        full_update = torch.zeros_like(weight)
        for offset, target in enumerate(self.target_modules):
            target_index = VALID_LORA_TARGET_MODULES.index(target)
            start = target_index * self.embed_dim
            end = start + self.embed_dim
            low_rank_update = self.dropout(
                self.lora_B[offset] @ self.lora_A[offset]
            )
            full_update[start:end] = low_rank_update
        return weight + full_update * self.scale


@dataclass(frozen=True, slots=True)
class LoRAInjectionReport:
    block_indices: tuple[int, ...]
    target_modules: tuple[str, ...]
    adapter_modules: tuple[FusedQKVLoRA, ...]

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for adapter in self.adapter_modules
            for parameter in adapter.parameters()
            if parameter.requires_grad
        )


def normalize_lora_target_modules(target_modules: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(target).lower() for target in target_modules))
    invalid = [
        target for target in normalized if target not in VALID_LORA_TARGET_MODULES
    ]
    if not normalized or invalid:
        raise ValueError(
            "--lora-target-modules must contain one or more of "
            f"{VALID_LORA_TARGET_MODULES}, got {list(normalized)}."
        )
    return normalized


def inject_lora_adapters(
    backbone: Model50MBackbone,
    *,
    block_indices: Sequence[int],
    target_modules: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> LoRAInjectionReport:
    """Inject LoRA into fused QKV attention projections of selected blocks."""
    normalized_blocks = tuple(sorted({int(index) for index in block_indices}))
    if not normalized_blocks:
        raise ValueError("LoRA requires at least one selected encoder block.")
    invalid_blocks = [
        index for index in normalized_blocks if not 0 <= index < backbone.config.depth
    ]
    if invalid_blocks:
        raise ValueError(
            "LoRA block indices must be in "
            f"[0, {backbone.config.depth - 1}], got {invalid_blocks}."
        )
    targets = normalize_lora_target_modules(target_modules)

    # All original 50M weights must stay frozen in the LoRA regime.
    backbone.set_trainable_encoder_blocks(())
    adapters: list[FusedQKVLoRA] = []
    for block_index in normalized_blocks:
        attention = backbone.model.encoder.encoder.layers[block_index].self_attn
        if not isinstance(attention, nn.MultiheadAttention):
            raise TypeError(
                "50M LoRA currently requires nn.MultiheadAttention, got "
                f"{type(attention)!r} in encoder block {block_index}."
            )
        if attention.in_proj_weight is None:
            raise ValueError(
                "50M attention uses separate Q/K/V projection weights; "
                "the fused-QKV LoRA injector cannot target this layout."
            )
        if parametrize.is_parametrized(attention, "in_proj_weight"):
            raise RuntimeError(
                f"LoRA is already injected into encoder block {block_index}."
            )

        attention.in_proj_weight.requires_grad = False
        adapter = FusedQKVLoRA(
            embed_dim=int(attention.embed_dim),
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=targets,
        )
        parametrize.register_parametrization(
            attention,
            "in_proj_weight",
            adapter,
        )
        adapters.append(adapter)

    backbone.set_trainable_adapters(adapters)
    return LoRAInjectionReport(
        block_indices=normalized_blocks,
        target_modules=targets,
        adapter_modules=tuple(adapters),
    )


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only trainable LoRA tensors, never a duplicate base backbone."""
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key.endswith(".lora_A") or key.endswith(".lora_B")
    }


def load_lora_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    expected_keys = set(lora_state_dict(model))
    received_keys = set(state_dict)
    if received_keys != expected_keys:
        missing = sorted(expected_keys - received_keys)
        unexpected = sorted(received_keys - expected_keys)
        raise ValueError(
            "LoRA checkpoint keys do not match the injected adapters: "
            f"missing={missing}, unexpected={unexpected}."
        )
    incompatible = model.load_state_dict(dict(state_dict), strict=False)
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if key.endswith(".lora_A") or key.endswith(".lora_B")
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected LoRA checkpoint keys: {unexpected}.")
