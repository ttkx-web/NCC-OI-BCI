from __future__ import annotations

import numpy as np
import torch

from bci_dayloop.models.base import ModelBackend
from bci_dayloop.models.model_50m.adapter import Model50MAdapter
from bci_dayloop.runtime.types import (
    ModelOutput,
    ModelTensor,
)


class Model50MBackend(ModelBackend):
    """
    统一 Runtime 使用的 50M 计算后端。

    当前复用 Model50MAdapter 内已经验证过的：
    - tokenizer；
    - backbone；
    - feature aggregator；
    - classification head。
    """

    def __init__(
        self,
        adapter: Model50MAdapter,
    ) -> None:
        self.adapter = adapter

    @property
    def device(self) -> torch.device:
        return self.adapter.device

    @property
    def num_classes(self) -> int:
        return self.adapter.num_classes

    @staticmethod
    def _to_numpy(
        tensor: torch.Tensor,
        *,
        name: str,
    ) -> np.ndarray:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{name} must be torch.Tensor, "
                f"got {type(tensor).__name__}."
            )

        if not torch.isfinite(tensor).all():
            raise ValueError(
                f"{name} contains NaN or Inf."
            )

        return (
            tensor.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

    def _unpack_input(
        self,
        model_input: ModelTensor,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        将统一输入拆成 50M 当前 Adapter 接受的 numpy 输入。
        """

        if isinstance(model_input, torch.Tensor):
            signal = self._to_numpy(
                model_input,
                name="signal",
            )
            return signal, None

        if not isinstance(model_input, dict):
            raise TypeError(
                "50M model_input must be torch.Tensor or "
                "dict[str, torch.Tensor]."
            )

        if "signal" not in model_input:
            raise ValueError(
                "50M model_input is missing required key "
                "'signal'."
            )

        if "channel_valid_mask" not in model_input:
            raise ValueError(
                "50M model_input is missing required key "
                "'channel_valid_mask'."
            )

        signal = self._to_numpy(
            model_input["signal"],
            name="signal",
        )

        channel_valid_mask = self._to_numpy(
            model_input["channel_valid_mask"],
            name="channel_valid_mask",
        )

        return signal, channel_valid_mask

    def _build_batch(
        self,
        model_input: ModelTensor,
    ):
        signal, channel_valid_mask = self._unpack_input(
            model_input
        )

        model_batch, _, _ = self.adapter._build_model_batch(
            X=signal,
            channel_valid_masks=channel_valid_mask,
        )

        return model_batch

    def predict_tensor(
        self,
        model_input: ModelTensor,
        return_features: bool = False,
    ) -> ModelOutput:
        model_batch = self._build_batch(model_input)

        self.adapter.classifier.eval()

        with torch.inference_mode():
            features = (
                self.adapter.classifier.extract_features(
                    model_batch
                )
            )

            logits = self.adapter.classifier.head(features)
            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            confidences, predictions = torch.max(
                probabilities,
                dim=-1,
            )

        if logits.ndim != 2:
            raise RuntimeError(
                "50M logits must have shape [B, classes], "
                f"got {tuple(logits.shape)}."
            )

        # 当前 RuntimeModel 一次处理一个 EEG 窗口。
        if logits.shape[0] != 1:
            raise ValueError(
                "RuntimeModel currently expects one window "
                "per prediction, but got "
                f"batch_size={logits.shape[0]}."
            )

        return ModelOutput(
            logits=logits,
            probabilities=probabilities,
            predicted_class=int(
                predictions[0].item()
            ),
            confidence=float(
                confidences[0].item()
            ),
            features=(
                features
                if return_features
                else None
            ),
        )

    def encode_tensor(
        self,
        model_input: ModelTensor,
    ) -> torch.Tensor:
        model_batch = self._build_batch(model_input)

        self.adapter.classifier.eval()

        with torch.inference_mode():
            features = (
                self.adapter.classifier.extract_features(
                    model_batch
                )
            )

        return features

    def get_trainable_parameters(
        self,
        scope: str,
    ) -> list[torch.nn.Parameter]:
        normalized_scope = scope.strip().lower()

        if normalized_scope == "head":
            self.adapter.backbone.freeze()

            for parameter in (
                self.adapter.classifier.head.parameters()
            ):
                parameter.requires_grad = True

            return list(
                self.adapter.classifier.head.parameters()
            )

        if normalized_scope == "backbone":
            self.adapter.backbone.unfreeze()

            for parameter in (
                self.adapter.classifier.head.parameters()
            ):
                parameter.requires_grad = False

            return list(
                self.adapter.backbone.parameters()
            )

        if normalized_scope == "full":
            self.adapter.backbone.unfreeze()

            for parameter in (
                self.adapter.classifier.head.parameters()
            ):
                parameter.requires_grad = True

            return [
                parameter
                for parameter
                in self.adapter.classifier.parameters()
                if parameter.requires_grad
            ]

        raise ValueError(
            "Unsupported trainable scope "
            f"{scope!r}. Expected one of: "
            "'head', 'backbone', 'full'."
        )