from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

from bci_dayloop.models.base import ModelBackend
from bci_dayloop.models.model_50m.tokenization import (
    Model50MBatchedInput,
)

if TYPE_CHECKING:
    from bci_dayloop.models.model_50m.adapter import Model50MAdapter
from bci_dayloop.models.online_features import (
    OnlineFeatureSpec,
    OnlineTokenContext,
)
from bci_dayloop.runtime.types import (
    ModelOutput,
    ModelTensor,
)


class Model50MBackend(ModelBackend):
    """
    统一 Runtime 使用的 50M 计算后端。

    原始 50M 路径为：

        signal + channel_valid_mask
        -> tokenizer -> Backbone [B, C*P, D]
        -> FeatureAggregator(token_valid_mask)
        -> LinearClassificationHead

    在线接口暴露的是真实、尚未聚合的 ``[B, C*P, D]`` token。token
    顺序与 Model50MTokenizer 一致：channel-major，且每个通道内 time-patch
    递增。由于原聚合器使用 token_valid_mask，本 backend 还实现可选的
    无状态 token-context 扩展；绝不缓存上一批 mask。
    """

    def __init__(
        self,
        adapter: "Model50MAdapter",
    ) -> None:
        self.adapter = adapter

    @property
    def device(self) -> torch.device:
        return self.adapter.device

    @property
    def num_classes(self) -> int:
        return self.adapter.num_classes

    @property
    def online_feature_spec(self) -> OnlineFeatureSpec:
        config = self.adapter.config
        return OnlineFeatureSpec(
            model_name="50m-linear",
            token_count=int(config.num_tokens),
            embedding_dim=int(config.d_model),
        )

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
        """将 Runtime 输入转换为 Adapter 的 numpy 批量输入。"""

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
    ) -> Model50MBatchedInput:
        signal, channel_valid_mask = self._unpack_input(
            model_input
        )

        model_batch, _, _ = self.adapter._build_model_batch(
            X=signal,
            channel_valid_masks=channel_valid_mask,
        )

        return model_batch

    def _validate_online_tokens(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(tokens, torch.Tensor):
            raise TypeError(
                "50M online tokens must be torch.Tensor, got "
                f"{type(tokens).__name__}."
            )

        if tokens.ndim != 3:
            raise ValueError(
                "50M online tokens must have shape [B,N,D], got "
                f"{tuple(tokens.shape)}."
            )

        if tokens.shape[0] <= 0:
            raise ValueError(
                "50M online token batch size must be positive."
            )

        spec = self.online_feature_spec
        expected_shape = (
            tokens.shape[0],
            spec.token_count,
            spec.embedding_dim,
        )

        if tuple(tokens.shape) != expected_shape:
            raise ValueError(
                "50M online token shape mismatch: expected "
                f"{expected_shape}, got {tuple(tokens.shape)}."
            )

        if not tokens.is_floating_point():
            raise TypeError(
                "50M online tokens must have a floating-point "
                f"dtype, got {tokens.dtype}."
            )

        if not torch.isfinite(tokens).all():
            raise ValueError(
                "50M online tokens contain NaN or Inf."
            )

        return tokens.to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=self.device.type == "cuda",
        )

    def _validate_token_valid_mask(
        self,
        token_valid_mask: torch.Tensor | None,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        if token_valid_mask is None:
            raise ValueError(
                "50M classify_online_tokens requires "
                "token_valid_mask to preserve the original "
                "mask-aware aggregation semantics. Use "
                "encode_online_token_context() with "
                "NeuroOnlineForward, or pass the mask explicitly."
            )

        if not isinstance(token_valid_mask, torch.Tensor):
            raise TypeError(
                "50M token_valid_mask must be torch.Tensor, got "
                f"{type(token_valid_mask).__name__}."
            )

        expected_shape = (
            batch_size,
            self.online_feature_spec.token_count,
        )

        if tuple(token_valid_mask.shape) != expected_shape:
            raise ValueError(
                "50M token_valid_mask shape mismatch: expected "
                f"{expected_shape}, got "
                f"{tuple(token_valid_mask.shape)}."
            )

        if not token_valid_mask.is_floating_point():
            raise TypeError(
                "50M token_valid_mask must have a floating-point "
                f"dtype, got {token_valid_mask.dtype}."
            )

        if not torch.isfinite(token_valid_mask).all():
            raise ValueError(
                "50M token_valid_mask contains NaN or Inf."
            )

        normalized_mask = token_valid_mask.to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=self.device.type == "cuda",
        )

        if torch.any(normalized_mask.sum(dim=1) <= 0):
            invalid_batch_indices = (
                (normalized_mask.sum(dim=1) <= 0)
                .nonzero(as_tuple=False)
                .flatten()
                .tolist()
            )
            raise ValueError(
                "At least one 50M sample contains no valid token. "
                f"Invalid batch indices: {invalid_batch_indices}."
            )

        return normalized_mask

    def encode_online_token_context(
        self,
        model_input: ModelTensor,
        *,
        train_backbone: bool = False,
    ) -> OnlineTokenContext:
        """
        提取 50M 的真实未池化 token 及其显式聚合 mask。

        ``train_backbone=False`` 是 NeuroOnline V1 的唯一支持范围。关闭
        backbone 计算图不会影响后续 Generator 和 head 对返回 token 的
        梯度路径；返回的是普通 no-grad tensor，而不是 inference tensor。
        """

        if train_backbone:
            raise NotImplementedError(
                "50M NeuroOnline V1 freezes the backbone; "
                "train_backbone=True is not supported."
            )

        model_batch = self._build_batch(model_input)
        # 提取 token 不改变 head 的 train/eval 状态。在线 update 已由
        # NeuroOnlineStrategy 在外层通过 set_online_mode() 设置 head.train().
        self.adapter.backbone.freeze()

        # Model50MBackbone.forward() uses torch.no_grad() while frozen.
        # Do not use inference_mode(): Generator/head training must still
        # accept these features as ordinary tensors.
        tokens = self.adapter.backbone.extract_embeddings(
            batch=model_batch,
            return_layer_idx=(
                self.adapter.config.output_layer_idx
            ),
        )
        tokens = self._validate_online_tokens(tokens)

        token_valid_mask = self._validate_token_valid_mask(
            model_batch.token_valid_mask,
            batch_size=tokens.shape[0],
        )

        return OnlineTokenContext(
            tokens=tokens,
            token_valid_mask=token_valid_mask,
        )

    def encode_online_tokens(
        self,
        model_input: ModelTensor,
        *,
        train_backbone: bool = False,
    ) -> torch.Tensor:
        """返回统一 ``[B,N,D]`` token，供直接检查或单独使用。"""

        return self.encode_online_token_context(
            model_input,
            train_backbone=train_backbone,
        ).tokens

    def classify_online_tokens(
        self,
        tokens: torch.Tensor,
        *,
        token_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        按原始 FeatureAggregator 和同一个线性 head 计算 logits。

        50M 的 flatten 与 mean 两种聚合都会显式使用 token_valid_mask，
        所以这里拒绝省略该参数，而不是静默把缺失通道当作有效 token。
        """

        tokens = self._validate_online_tokens(tokens)
        token_valid_mask = self._validate_token_valid_mask(
            token_valid_mask,
            batch_size=tokens.shape[0],
        )

        features = self.adapter.classifier.aggregator(
            token_embeddings=tokens,
            token_valid_mask=token_valid_mask,
        )
        logits = self.adapter.classifier.head(features)

        expected_shape = (tokens.shape[0], self.num_classes)
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected 50M online logits shape: expected "
                f"{expected_shape}, got {tuple(logits.shape)}."
            )

        if not torch.isfinite(logits).all():
            raise RuntimeError(
                "50M online classifier produced NaN or Inf logits."
            )

        return logits

    def set_online_mode(
        self,
        *,
        training: bool,
        train_backbone: bool = False,
    ) -> None:
        """设置 50M NeuroOnline V1 的冻结 backbone / 可训练 head 状态。"""

        if train_backbone:
            raise NotImplementedError(
                "50M NeuroOnline V1 freezes the backbone; "
                "train_backbone=True is not supported."
            )

        self.adapter.backbone.freeze()

        for parameter in self.adapter.classifier.head.parameters():
            parameter.requires_grad = True

        self.adapter.classifier.aggregator.eval()
        self.adapter.classifier.head.train(mode=training)

    def predict_tensor(
        self,
        model_input: ModelTensor,
        return_features: bool = False,
    ) -> ModelOutput:
        model_batch = self._build_batch(model_input)

        if model_batch.batch_size != 1:
            raise ValueError(
                "RuntimeModel currently expects one window per "
                "prediction, but got "
                f"batch_size={model_batch.batch_size}."
            )

        self.set_online_mode(
            training=False,
            train_backbone=False,
        )

        with torch.no_grad():
            tokens = self.adapter.backbone.extract_embeddings(
                batch=model_batch,
                return_layer_idx=(
                    self.adapter.config.output_layer_idx
                ),
            )
            tokens = self._validate_online_tokens(tokens)
            token_valid_mask = self._validate_token_valid_mask(
                model_batch.token_valid_mask,
                batch_size=tokens.shape[0],
            )
            features = self.adapter.classifier.aggregator(
                token_embeddings=tokens,
                token_valid_mask=token_valid_mask,
            )
            logits = self.classify_online_tokens(
                tokens,
                token_valid_mask=token_valid_mask,
            )
            probabilities = torch.softmax(logits, dim=-1)
            confidences, predictions = torch.max(
                probabilities,
                dim=-1,
            )

        return ModelOutput(
            logits=logits,
            probabilities=probabilities,
            predicted_class=int(predictions[0].item()),
            confidence=float(confidences[0].item()),
            features=features if return_features else None,
            diagnostics={
                "backend": "50m-linear",
                "token_shape": list(tokens.shape),
                "aggregation": self.adapter.config.aggregation,
            },
        )

    def encode_tensor(
        self,
        model_input: ModelTensor,
    ) -> torch.Tensor:
        """保持旧 Runtime 特征接口：返回聚合后的分类特征。"""

        model_batch = self._build_batch(model_input)
        self.set_online_mode(
            training=False,
            train_backbone=False,
        )

        with torch.no_grad():
            tokens = self.adapter.backbone.extract_embeddings(
                batch=model_batch,
                return_layer_idx=(
                    self.adapter.config.output_layer_idx
                ),
            )
            tokens = self._validate_online_tokens(tokens)
            token_valid_mask = self._validate_token_valid_mask(
                model_batch.token_valid_mask,
                batch_size=tokens.shape[0],
            )
            return self.adapter.classifier.aggregator(
                token_embeddings=tokens,
                token_valid_mask=token_valid_mask,
            )

    def get_trainable_parameters(
        self,
        scope: str,
    ) -> list[torch.nn.Parameter]:
        normalized_scope = scope.strip().lower()

        if normalized_scope != "head":
            raise ValueError(
                "50M NeuroOnline V1 only supports trainable "
                f"scope 'head', got {scope!r}."
            )

        self.set_online_mode(
            training=True,
            train_backbone=False,
        )

        parameters = list(
            self.adapter.classifier.head.parameters()
        )

        if not parameters:
            raise RuntimeError(
                "50M classification head has no parameters."
            )

        parameter_ids = [id(parameter) for parameter in parameters]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise RuntimeError(
                "50M classification head contains duplicate "
                "parameter references."
            )

        return parameters
