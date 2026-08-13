from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from bci_dayloop.runtime.types import (
    ModelTensor,
)


@dataclass(frozen=True, slots=True)
class OnlineFeatureSpec:
    """
    NeuroOnline 使用的统一 token 特征规格。

    token_count:
        每个窗口产生的 token 数量。

    embedding_dim:
        每个 token 的特征维度。
    """

    model_name: str
    token_count: int
    embedding_dim: int

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError(
                "model_name cannot be empty."
            )

        if self.token_count <= 0:
            raise ValueError(
                "token_count must be positive."
            )

        if self.embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive."
            )


@dataclass(frozen=True, slots=True)
class OnlineTokenContext:
    """
    一次在线 token 提取的无状态上下文。

    大多数 backend 只需要 ``tokens``。50M 的原始分类聚合还依赖
    每个样本的 token_valid_mask，因此将 mask 与本次提取结果显式绑定，
    避免通过 backend 的跨调用可变状态保存它。
    """

    tokens: torch.Tensor
    token_valid_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, torch.Tensor):
            raise TypeError(
                "tokens must be torch.Tensor, got "
                f"{type(self.tokens).__name__}."
            )

        if self.tokens.ndim != 3:
            raise ValueError(
                "tokens must have shape [B,N,D], got "
                f"{tuple(self.tokens.shape)}."
            )

        if self.token_valid_mask is None:
            return

        if not isinstance(self.token_valid_mask, torch.Tensor):
            raise TypeError(
                "token_valid_mask must be torch.Tensor, got "
                f"{type(self.token_valid_mask).__name__}."
            )

        expected_shape = self.tokens.shape[:2]

        if tuple(self.token_valid_mask.shape) != expected_shape:
            raise ValueError(
                "token_valid_mask must have shape "
                f"{expected_shape}, got "
                f"{tuple(self.token_valid_mask.shape)}."
            )


@runtime_checkable
class OnlineTokenContextFeatureBackend(Protocol):
    """
    可选扩展：分类时还需要 token 级附加上下文的 backend。

    该协议不替换 OnlineTrainableFeatureBackend。NeuroOnlineForward 仅在
    backend 实现本协议时使用它，故现有 LaBraM 和 CBraMod 不受影响。
    """

    def encode_online_token_context(
        self,
        model_input: ModelTensor,
        *,
        train_backbone: bool = False,
    ) -> OnlineTokenContext:
        ...

    def classify_online_tokens(
        self,
        tokens: torch.Tensor,
        *,
        token_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...


@runtime_checkable
class OnlineTrainableFeatureBackend(
    Protocol
):
    """
    支持 NeuroOnline token 适配的可选 Backend 接口。

    不要求所有 ModelBackend 都实现，当前先由：
    - LaBraMBackend
    - CBraModBackend

    实现。
    """

    @property
    def online_feature_spec(
        self,
    ) -> OnlineFeatureSpec:
        ...

    def encode_online_tokens(
        self,
        model_input: ModelTensor,
        *,
        train_backbone: bool = False,
    ) -> torch.Tensor:
        """
        提取统一形状的 token：

            [B, N, D]

        train_backbone=False：
            backbone 冻结，Generator 和分类头可训练。

        train_backbone=True：
            为未来 backbone 在线微调预留。
        """
        ...

    def classify_online_tokens(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        将 Generator 调整后的 [B,N,D] token
        转换成 [B,num_classes] logits。
        """
        ...

    def set_online_mode(
            self,
            *,
            training: bool,
            train_backbone: bool = False,
    ) -> None:
        """
        设置在线前向所需的模型状态。

        training=False:
            backbone.eval()
            classifier.eval()

        training=True, train_backbone=False:
            backbone.eval() 且冻结
            classifier.train()

        training=True, train_backbone=True:
            backbone.train()
            classifier.train()
        """
        ...
