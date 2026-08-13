from __future__ import annotations

import torch
import torch.nn as nn

from bci_dayloop.models.base import (
    ModelBackend,
)
from bci_dayloop.models.online_features import (
    OnlineFeatureSpec,
)
from bci_dayloop.runtime.types import (
    ModelOutput,
    ModelTensor,
)

from .backbone import CBraModBackbone
from .config import CBraModConfig


class CBraModBackend(ModelBackend):
    """
    统一 Runtime 使用的 CBRaMod 计算后端。

    输入：
        - torch.Tensor: [B, 22, 4, 200]
        - 或 {"signal": torch.Tensor}

    分类头约定：
        classifier(features) -> logits
        features 的 shape 为 [B, 22, 4, 200]
        logits 的 shape 为 [B, num_classes]

    本类不负责原始 EEG 预处理；输入必须已由
    CBraModPipelinePreprocessor 转为 [B, 22, 4, 200]。
    """

    def __init__(
        self,
        backbone: CBraModBackbone,
        classifier: nn.Module,
        config: CBraModConfig,
    ) -> None:
        if not isinstance(backbone, CBraModBackbone):
            raise TypeError(
                "backbone must be CBraModBackbone, got "
                f"{type(backbone).__name__}."
            )

        if not isinstance(classifier, nn.Module):
            raise TypeError(
                "classifier must be torch.nn.Module, got "
                f"{type(classifier).__name__}."
            )

        self.backbone = backbone
        self.classifier = classifier.to(
            self.backbone.device,
        )
        self.config = config

        if self.backbone.config != self.config:
            raise ValueError(
                "CBraModBackend received a backbone and config "
                "that do not match."
            )

        if not any(
            True
            for _ in self.classifier.parameters()
        ):
            raise ValueError(
                "CBraMod classifier has no parameters."
            )

    @property
    def device(self) -> torch.device:
        return self.backbone.device

    @property
    def num_classes(self) -> int:
        return self.config.num_classes

    @property
    def online_feature_spec(
            self,
    ) -> OnlineFeatureSpec:
        """
        CBrAmod 的在线 token 特征规格。

        CBrAmod backbone 输出：
            [B,C,S,D]

        在线 Generator 使用：
            [B,C*S,D]
        """

        return OnlineFeatureSpec(
            model_name="cbramod",
            token_count=(
                self.config
                .num_feature_positions
            ),
            embedding_dim=(
                self.config
                .backbone_output_dim
            ),
        )

    def _unpack_signal(
        self,
        model_input: ModelTensor,
    ) -> torch.Tensor:
        if isinstance(model_input, torch.Tensor):
            signal = model_input

        elif isinstance(model_input, dict):
            if "signal" not in model_input:
                raise ValueError(
                    "CBraMod model_input is missing required key "
                    "'signal'."
                )

            signal = model_input["signal"]

        else:
            raise TypeError(
                "CBraMod model_input must be torch.Tensor or "
                "dict[str, torch.Tensor], got "
                f"{type(model_input).__name__}."
            )

        if not isinstance(signal, torch.Tensor):
            raise TypeError(
                "CBraMod signal must be torch.Tensor, got "
                f"{type(signal).__name__}."
            )

        return signal

    def _prepare_signal(
        self,
        model_input: ModelTensor,
    ) -> torch.Tensor:
        signal = self._unpack_signal(model_input)

        self.backbone._validate_input(signal)

        return signal.to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=True,
        )

    def _validate_logits(
        self,
        logits: torch.Tensor,
        *,
        expected_batch_size: int,
    ) -> None:
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                "CBraMod classifier output must be torch.Tensor, "
                f"got {type(logits).__name__}."
            )

        expected_shape = (
            expected_batch_size,
            self.num_classes,
        )

        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected CBraMod logits shape. Expected "
                f"{expected_shape}, got {tuple(logits.shape)}."
            )

        if not torch.isfinite(logits).all():
            raise RuntimeError(
                "CBraMod classifier produced NaN or Inf logits."
            )

    def encode_online_tokens(
            self,
            model_input: ModelTensor,
            *,
            train_backbone: bool = False,
    ) -> torch.Tensor:
        """
        提取供 NeuroOnline Generator 使用的 token。

        输入：
            signal: [B,C,S,P]

        CBrAmod backbone 输出：
            features: [B,C,S,D]

        本方法返回：
            tokens: [B,C*S,D]

        V1 使用：
            train_backbone=False
        """

        signal = self._prepare_signal(
            model_input
        )

        if train_backbone:
            # 为未来在线更新 backbone 预留。
            self.backbone.unfreeze()
        else:
            # NeuroOnline V1 冻结 backbone。
            self.backbone.freeze()

        # 不使用 torch.inference_mode()。
        #
        # train_backbone=False 时关闭 backbone 梯度；
        # 但输出仍是普通 Tensor，可以送入 Generator
        # 并计算 Generator/head 的梯度。
        with torch.set_grad_enabled(
                train_backbone
        ):
            features = self.backbone.encode(
                signal
            )

        expected_feature_shape = (
            signal.shape[0],
            self.config.n_channels,
            self.config.time_segments,
            self.config.backbone_output_dim,
        )

        if tuple(features.shape) != (
                expected_feature_shape
        ):
            raise RuntimeError(
                "Unexpected CBrAmod online feature "
                "shape: "
                f"expected={expected_feature_shape}, "
                f"actual={tuple(features.shape)}."
            )

        # [B,C,S,D] -> [B,C*S,D]
        tokens = features.flatten(
            start_dim=1,
            end_dim=2,
        )

        spec = self.online_feature_spec

        expected_token_shape = (
            signal.shape[0],
            spec.token_count,
            spec.embedding_dim,
        )

        if tuple(tokens.shape) != (
                expected_token_shape
        ):
            raise RuntimeError(
                "Unexpected CBrAmod online token "
                "shape: "
                f"expected={expected_token_shape}, "
                f"actual={tuple(tokens.shape)}."
            )

        if not torch.isfinite(
                tokens
        ).all():
            raise RuntimeError(
                "CBrAmod online tokens contain "
                "NaN or Inf."
            )

        return tokens

    def classify_online_tokens(
            self,
            tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        将 Generator 调整后的 [B,N,D] token
        重新转换成 CBrAmodClassifier 要求的
        [B,C,S,D]，然后计算 logits。

        输入：
            tokens: [B,C*S,D]

        输出：
            logits: [B,num_classes]
        """

        if not isinstance(
                tokens,
                torch.Tensor,
        ):
            raise TypeError(
                "CBrAmod online tokens must be "
                "torch.Tensor, got "
                f"{type(tokens).__name__}."
            )

        if tokens.ndim != 3:
            raise ValueError(
                "CBrAmod online tokens must have "
                "shape [B,N,D], got "
                f"{tuple(tokens.shape)}."
            )

        if tokens.shape[0] <= 0:
            raise ValueError(
                "CBrAmod online token batch size "
                "must be positive."
            )

        spec = self.online_feature_spec

        if tokens.shape[1] != spec.token_count:
            raise ValueError(
                "CBrAmod online token count "
                "does not match the Runtime "
                "Model Package: "
                f"expected={spec.token_count}, "
                f"actual={tokens.shape[1]}."
            )

        if tokens.shape[2] != spec.embedding_dim:
            raise ValueError(
                "CBrAmod online embedding "
                "dimension mismatch: "
                f"expected={spec.embedding_dim}, "
                f"actual={tokens.shape[2]}."
            )

        if not tokens.is_floating_point():
            raise TypeError(
                "CBrAmod online tokens must have "
                "a floating-point dtype, got "
                f"{tokens.dtype}."
            )

        if not torch.isfinite(
                tokens
        ).all():
            raise ValueError(
                "CBrAmod online tokens contain "
                "NaN or Inf."
            )

        tokens = tokens.to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=(
                    self.device.type
                    == "cuda"
            ),
        )

        # [B,C*S,D] -> [B,C,S,D]
        #
        # 使用 reshape 而不是 view，
        # 因为 Generator 输出不一定是 contiguous。
        features = tokens.reshape(
            tokens.shape[0],
            self.config.n_channels,
            self.config.time_segments,
            self.config.backbone_output_dim,
        )

        logits = self.classifier(
            features
        )

        self._validate_logits(
            logits,
            expected_batch_size=(
                tokens.shape[0]
            ),
        )

        return logits

    def predict_tensor(
        self,
        model_input: ModelTensor,
        return_features: bool = False,
    ) -> ModelOutput:
        """
        Runtime 单窗口预测。

        RuntimeModel 当前约定一次只预测一个窗口，因此 batch size 必须为 1。
        """

        signal = self._prepare_signal(model_input)

        if signal.shape[0] != 1:
            raise ValueError(
                "RuntimeModel currently expects one window per "
                "prediction, but got "
                f"batch_size={signal.shape[0]}."
            )

        self.backbone.freeze()
        self.classifier.eval()

        with torch.inference_mode():
            features = self.backbone.encode(signal)
            logits = self.classifier(features)
            self._validate_logits(
                logits,
                expected_batch_size=1,
            )

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=-1,
            )

        return ModelOutput(
            logits=logits,
            probabilities=probabilities,
            predicted_class=int(prediction[0].item()),
            confidence=float(confidence[0].item()),
            features=features if return_features else None,
            diagnostics={
                "model_name": "cbramod-frozen-head",
                "input_shape": list(signal.shape),
                "feature_shape": list(features.shape),
                "head_type": self.config.head_type,
            },
        )

    def encode_tensor(
        self,
        model_input: ModelTensor,
    ) -> torch.Tensor:
        """
        提供统一特征提取接口。

        与 predict_tensor 不同，此方法允许 batch size 大于 1，
        便于训练前缓存冻结骨干特征。
        """

        signal = self._prepare_signal(model_input)

        self.backbone.freeze()

        with torch.inference_mode():
            features = self.backbone.encode(signal)

        return features

    def set_online_mode(
            self,
            *,
            training: bool,
            train_backbone: bool = False,
    ) -> None:
        """
        设置 CBrAmod NeuroOnline 前向模式。
        """

        if training and train_backbone:
            self.backbone.unfreeze()
        else:
            self.backbone.freeze()

        self.classifier.train(
            mode=training
        )

    def get_trainable_parameters(
        self,
        scope: str,
    ) -> list[torch.nn.Parameter]:
        """
        返回指定训练范围内可优化的参数。

        当前正式基线只允许 scope='head'。
        其他两个范围仅为后续全量微调实验保留，不应写入
        cbramod-frozen-head 的训练脚本。
        """

        normalized_scope = scope.strip().lower()

        if normalized_scope == "head":
            self.backbone.freeze()

            for parameter in self.classifier.parameters():
                parameter.requires_grad = True

            self.classifier.train()

            return list(self.classifier.parameters())

        if normalized_scope == "backbone":
            self.backbone.unfreeze()

            for parameter in self.classifier.parameters():
                parameter.requires_grad = False

            return list(self.backbone.parameters())

        if normalized_scope == "full":
            self.backbone.unfreeze()

            for parameter in (
                    self.classifier.parameters()
            ):
                parameter.requires_grad = True

            self.classifier.train()

            return [
                *self.backbone.parameters(),
                *self.classifier.parameters(),
            ]

        raise ValueError(
            "Unsupported trainable scope "
            f"{scope!r}. Expected one of: "
            "'head', 'backbone', 'full'."
        )