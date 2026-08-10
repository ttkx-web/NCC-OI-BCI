from __future__ import annotations

import torch
import torch.nn as nn

from .config import CBraModConfig


class CBraModClassifier(nn.Module):
    """
    CBRaMod 下游分类头。

    输入：
        features: [B, 22, 4, 200]

    输出：
        logits: [B, num_classes]

    支持两种模式：

    - official_mlp
      CBRaMod 官方 quick-start / BCI-IV-2a 下游封装中的主头：
      Flatten -> Linear(17600, 800) -> ELU -> Dropout
              -> Linear(800, 200) -> ELU -> Dropout
              -> Linear(200, 4)

    - linear
      严格线性 probe：
      Flatten -> Linear(17600, 4)

    正式主基线应使用 official_mlp，并命名为
    cbramod-frozen-head；linear 仅作为独立补充实验。
    """

    def __init__(
        self,
        config: CBraModConfig,
    ) -> None:
        super().__init__()

        self.config = config
        self.input_dim = self.config.classifier_input_dim
        self.num_classes = self.config.num_classes
        self.head_type = self.config.head_type

        self.flatten = nn.Flatten(start_dim=1)

        if self.head_type == "official_mlp":
            self.head = nn.Sequential(
                nn.Linear(
                    self.input_dim,
                    self.config.head_hidden_dim_1,
                ),
                nn.ELU(),
                nn.Dropout(
                    p=self.config.head_dropout
                ),
                nn.Linear(
                    self.config.head_hidden_dim_1,
                    self.config.head_hidden_dim_2,
                ),
                nn.ELU(),
                nn.Dropout(
                    p=self.config.head_dropout
                ),
                nn.Linear(
                    self.config.head_hidden_dim_2,
                    self.num_classes,
                ),
            )

        elif self.head_type == "linear":
            self.head = nn.Linear(
                self.input_dim,
                self.num_classes,
            )

        else:
            raise ValueError(
                "Unsupported CBraMod head_type: "
                f"{self.head_type!r}."
            )

    @property
    def expected_feature_shape(
        self,
    ) -> tuple[int, int, int]:
        """不包含 batch 维度的 [C, S, D]。"""
        return (
            self.config.n_channels,
            self.config.time_segments,
            self.config.backbone_output_dim,
        )

    def _validate_features(
        self,
        features: torch.Tensor,
    ) -> None:
        if not isinstance(features, torch.Tensor):
            raise TypeError(
                "CBraMod classifier features must be "
                "torch.Tensor, got "
                f"{type(features).__name__}."
            )

        if features.ndim != 4:
            raise ValueError(
                "CBraMod classifier expects features with "
                "shape [B, C, S, D], got "
                f"{tuple(features.shape)}."
            )

        if features.shape[0] <= 0:
            raise ValueError(
                "CBraMod classifier batch size must be positive."
            )

        if tuple(features.shape[1:]) != (
            self.expected_feature_shape
        ):
            raise ValueError(
                "CBraMod classifier feature shape mismatch. "
                "Expected "
                f"[B, {self.expected_feature_shape[0]}, "
                f"{self.expected_feature_shape[1]}, "
                f"{self.expected_feature_shape[2]}], got "
                f"{tuple(features.shape)}."
            )

        if not features.is_floating_point():
            raise TypeError(
                "CBraMod classifier features must have a "
                "floating-point dtype, got "
                f"{features.dtype}."
            )

        if not torch.isfinite(features).all():
            raise ValueError(
                "CBraMod classifier features contain NaN or Inf."
            )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_features(features)

        flattened = self.flatten(features)

        if flattened.shape[1] != self.input_dim:
            raise RuntimeError(
                "CBraMod flattened feature dimension mismatch. "
                f"Expected {self.input_dim}, got "
                f"{flattened.shape[1]}."
            )

        logits = self.head(flattened)

        expected_shape = (
            features.shape[0],
            self.num_classes,
        )

        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                "CBraMod classifier produced an unexpected logits "
                f"shape. Expected {expected_shape}, got "
                f"{tuple(logits.shape)}."
            )

        if not torch.isfinite(logits).all():
            raise RuntimeError(
                "CBraMod classifier produced NaN or Inf logits."
            )

        return logits

    def extra_repr(self) -> str:
        return (
            f"head_type={self.head_type!r}, "
            f"input_dim={self.input_dim}, "
            f"num_classes={self.num_classes}"
        )


def build_cbramod_classifier(
    config: CBraModConfig,
) -> CBraModClassifier:
    """创建与 CBraModConfig 一致的分类头。"""
    return CBraModClassifier(config)