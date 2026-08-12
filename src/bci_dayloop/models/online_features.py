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