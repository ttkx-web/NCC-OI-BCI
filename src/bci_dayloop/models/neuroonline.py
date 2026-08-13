from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bci_dayloop.models.online_features import (
    OnlineFeatureSpec,
)


class NeuroOnlineGenerator(nn.Module):
    """
    根据 backbone token 动态生成逐 token 的
    alpha 和 beta：

        adapted = tokens * alpha + beta

    输入：
        tokens: [B,N,D]

    输出：
        alpha:  [B,N,D]
        beta:   [B,N,D]

    其中：
        N = token_count
        D = embedding_dim
    """

    def __init__(
        self,
        feature_spec: OnlineFeatureSpec,
        *,
        num_subject_codes: int = 32,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.feature_spec = feature_spec
        self.token_count = int(
            feature_spec.token_count
        )
        self.embedding_dim = int(
            feature_spec.embedding_dim
        )
        self.num_subject_codes = int(
            num_subject_codes
        )
        self.num_attention_heads = int(
            num_attention_heads
        )

        if self.num_subject_codes <= 0:
            raise ValueError(
                "num_subject_codes must be positive."
            )

        if self.num_attention_heads <= 0:
            raise ValueError(
                "num_attention_heads must be positive."
            )

        if (
            self.embedding_dim
            % self.num_attention_heads
            != 0
        ):
            raise ValueError(
                "embedding_dim must be divisible "
                "by num_attention_heads: "
                f"embedding_dim={self.embedding_dim}, "
                f"num_attention_heads="
                f"{self.num_attention_heads}."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in [0,1)."
            )

        # K 个可学习隐变量原型。
        #
        # 它们不是 K 个真实被试 ID，而是模型自己学习的
        # K 种潜在适配模式。
        self.subject_codes = nn.Parameter(
            torch.randn(
                self.num_subject_codes,
                self.token_count,
                self.embedding_dim,
            )
            * 0.01
        )

        # 根据当前 EEG token 的全局平均特征，
        # 决定如何组合 subject codes。
        self.router = nn.Sequential(
            nn.Linear(
                self.embedding_dim,
                self.embedding_dim,
            ),
            nn.GELU(),
            nn.Linear(
                self.embedding_dim,
                self.num_subject_codes,
            ),
        )

        self.norm_q = nn.LayerNorm(
            self.embedding_dim
        )

        self.norm_kv = nn.LayerNorm(
            self.embedding_dim
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_mlp = nn.LayerNorm(
            self.embedding_dim
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                self.embedding_dim,
                2 * self.embedding_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                2 * self.embedding_dim,
                self.embedding_dim,
            ),
        )

        self.alpha_head = nn.Linear(
            self.embedding_dim,
            self.embedding_dim,
        )

        self.beta_head = nn.Linear(
            self.embedding_dim,
            self.embedding_dim,
        )

        # identity 初始化。
        #
        # 初始：
        #   alpha = 1
        #   beta = 0
        #
        # 因此 Generator 初始不会改变基线模型输出。
        self.gate_alpha = nn.Parameter(
            torch.tensor(0.0)
        )

        self.gate_beta = nn.Parameter(
            torch.tensor(0.0)
        )

    def _validate_tokens(
        self,
        tokens: torch.Tensor,
    ) -> None:
        if not isinstance(
            tokens,
            torch.Tensor,
        ):
            raise TypeError(
                "NeuroOnline tokens must be "
                "torch.Tensor, got "
                f"{type(tokens).__name__}."
            )

        if tokens.ndim != 3:
            raise ValueError(
                "NeuroOnline tokens must have "
                "shape [B,N,D], got "
                f"{tuple(tokens.shape)}."
            )

        expected_tail = (
            self.token_count,
            self.embedding_dim,
        )

        if tuple(tokens.shape[1:]) != (
            expected_tail
        ):
            raise ValueError(
                "NeuroOnline token shape mismatch: "
                f"expected=[B,{self.token_count},"
                f"{self.embedding_dim}], "
                f"actual={tuple(tokens.shape)}."
            )

        if tokens.shape[0] <= 0:
            raise ValueError(
                "NeuroOnline batch size "
                "must be positive."
            )

        if not tokens.is_floating_point():
            raise TypeError(
                "NeuroOnline tokens must use "
                "a floating-point dtype."
            )

        if not torch.isfinite(
            tokens
        ).all():
            raise ValueError(
                "NeuroOnline tokens contain "
                "NaN or Inf."
            )

    def route_subject_code(
        self,
        tokens: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        tokens:
            [B,N,D]

        router_probs:
            [B,K]

        dynamic_code:
            [B,N,D]
        """

        # [B,N,D] -> [B,D]
        pooled = tokens.mean(dim=1)

        router_logits = self.router(
            pooled
        )

        router_probs = F.softmax(
            router_logits,
            dim=-1,
        )

        # [B,K] × [K,N,D] -> [B,N,D]
        dynamic_code = torch.einsum(
            "bk,knd->bnd",
            router_probs,
            self.subject_codes,
        )

        return (
            dynamic_code,
            router_probs,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        return_aux: bool = False,
    ):
        self._validate_tokens(tokens)

        dynamic_code, router_probs = (
            self.route_subject_code(
                tokens
            )
        )

        query = self.norm_q(
            dynamic_code
        )

        key_value = self.norm_kv(
            tokens
        )

        attention_output, _ = (
            self.attention(
                query,
                key_value,
                key_value,
                need_weights=False,
            )
        )

        conditioned = (
            tokens
            + attention_output
        )

        conditioned = (
            conditioned
            + self.mlp(
                self.norm_mlp(
                    conditioned
                )
            )
        )

        alpha_raw = self.alpha_head(
            conditioned
        )

        beta_raw = self.beta_head(
            conditioned
        )

        alpha = (
            1.0
            + self.gate_alpha
            * alpha_raw
        )

        beta = (
            self.gate_beta
            * beta_raw
        )

        if return_aux:
            return (
                alpha,
                beta,
                router_probs,
            )

        return alpha, beta