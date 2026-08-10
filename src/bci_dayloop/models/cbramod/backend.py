from __future__ import annotations

import torch
import torch.nn as nn

from bci_dayloop.models.base import ModelBackend
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

            for parameter in self.classifier.parameters():
                parameter.requires_grad = True

            self.classifier.train()

            return [
                parameter
                for parameter in self.parameters()
                if parameter.requires_grad
            ]

        raise ValueError(
            "Unsupported trainable scope "
            f"{scope!r}. Expected one of: "
            "'head', 'backbone', 'full'."
        )