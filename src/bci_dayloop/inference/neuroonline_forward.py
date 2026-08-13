from __future__ import annotations

from dataclasses import dataclass

import torch

from bci_dayloop.models.neuroonline import (
    NeuroOnlineGenerator,
)
from bci_dayloop.models.online_features import (
    OnlineTokenContextFeatureBackend,
    OnlineTrainableFeatureBackend,
)
from bci_dayloop.runtime.model import (
    RuntimeModel,
)
from bci_dayloop.runtime.types import (
    ModelOutput,
    ModelTensor,
    PreparedModelInput,
)


@dataclass(slots=True)
class NeuroOnlineForwardResult:
    """
    一次 NeuroOnline 前向的中间结果。

    训练时需要 logits；
    调试和一致性 loss 可能需要 adapted_tokens、
    alpha、beta 和 router_probs。
    """

    logits: torch.Tensor

    original_tokens: torch.Tensor
    adapted_tokens: torch.Tensor

    alpha: torch.Tensor
    beta: torch.Tensor
    router_probs: torch.Tensor


class NeuroOnlineForward:
    """
    将模型专属 Backend 与统一 Generator 串联。

    Backend 负责：
        model_input -> tokens
        adapted_tokens -> logits

    Generator 负责：
        tokens -> alpha, beta
    """

    def __init__(
        self,
        *,
        runtime_model: RuntimeModel,
        generator: NeuroOnlineGenerator,
    ) -> None:
        self.runtime_model = runtime_model
        self.generator = generator

        backend = runtime_model.backend

        if not isinstance(
            backend,
            OnlineTrainableFeatureBackend,
        ):
            raise TypeError(
                "Runtime backend does not support "
                "online token adaptation: "
                f"{type(backend).__name__}."
            )

        self.backend = backend

        backend_spec = (
            self.backend
            .online_feature_spec
        )

        generator_spec = (
            self.generator
            .feature_spec
        )

        if backend_spec != generator_spec:
            raise ValueError(
                "Generator feature specification "
                "does not match Runtime backend: "
                f"backend={backend_spec}, "
                f"generator={generator_spec}."
            )

        self.generator.to(
            self.backend.device
        )

    def forward_batch(
        self,
        model_input: ModelTensor,
        *,
        train_backbone: bool = False,
    ) -> NeuroOnlineForwardResult:
        """
        通用 batch 前向。

        本方法不自动使用 no_grad，也不自动设置 train/eval。
        训练和预测由外层方法分别控制。
        """

        token_valid_mask: torch.Tensor | None = None

        # 50M 的原始聚合依赖每个样本的 token_valid_mask。该扩展把 mask
        # 与这次 token 提取结果显式传递，避免 backend 保存跨调用状态。
        if isinstance(
            self.backend,
            OnlineTokenContextFeatureBackend,
        ):
            token_context = (
                self.backend
                .encode_online_token_context(
                    model_input,
                    train_backbone=train_backbone,
                )
            )
            original_tokens = token_context.tokens
            token_valid_mask = (
                token_context.token_valid_mask
            )
        else:
            original_tokens = (
                self.backend
                .encode_online_tokens(
                    model_input,
                    train_backbone=(
                        train_backbone
                    ),
                )
            )

        (
            alpha,
            beta,
            router_probs,
        ) = self.generator(
            original_tokens,
            return_aux=True,
        )

        adapted_tokens = (
            original_tokens
            * alpha
            + beta
        )

        if token_valid_mask is None:
            logits = (
                self.backend
                .classify_online_tokens(
                    adapted_tokens
                )
            )
        else:
            logits = (
                self.backend
                .classify_online_tokens(
                    adapted_tokens,
                    token_valid_mask=(
                        token_valid_mask
                    ),
                )
            )

        expected_logits_shape = (
            original_tokens.shape[0],
            self.backend.num_classes,
        )

        if tuple(logits.shape) != (
            expected_logits_shape
        ):
            raise RuntimeError(
                "Unexpected NeuroOnline logits "
                "shape: "
                f"expected={expected_logits_shape}, "
                f"actual={tuple(logits.shape)}."
            )

        return NeuroOnlineForwardResult(
            logits=logits,
            original_tokens=(
                original_tokens
            ),
            adapted_tokens=(
                adapted_tokens
            ),
            alpha=alpha,
            beta=beta,
            router_probs=router_probs,
        )

    def predict_prepared(
        self,
        prepared: PreparedModelInput,
        *,
        return_features: bool = False,
    ) -> ModelOutput:
        """
        单窗口在线适配模型预测。

        此时 Generator 已参与前向，但不进行参数更新。
        """

        self.backend.set_online_mode(
            training=False,
            train_backbone=False,
        )

        self.generator.eval()

        with torch.no_grad():
            result = self.forward_batch(
                prepared.model_input,
                train_backbone=False,
            )

            probabilities = torch.softmax(
                result.logits,
                dim=-1,
            )

            confidence, prediction = (
                probabilities.max(dim=-1)
            )

        if result.logits.shape[0] != 1:
            raise ValueError(
                "NeuroOnline Runtime prediction "
                "expects one window, got "
                f"batch_size="
                f"{result.logits.shape[0]}."
            )

        return ModelOutput(
            logits=result.logits,
            probabilities=probabilities,
            predicted_class=int(
                prediction[0].item()
            ),
            confidence=float(
                confidence[0].item()
            ),
            features=(
                result.adapted_tokens
                if return_features
                else None
            ),
            diagnostics={
                "online_strategy": (
                    "neuroonline"
                ),
                "backend": (
                    self.backend
                    .online_feature_spec
                    .model_name
                ),
                "token_shape": list(
                    result
                    .original_tokens
                    .shape
                ),
                "adapted_token_shape": list(
                    result
                    .adapted_tokens
                    .shape
                ),
                "gate_alpha": float(
                    self.generator
                    .gate_alpha
                    .detach()
                    .cpu()
                    .item()
                ),
                "gate_beta": float(
                    self.generator
                    .gate_beta
                    .detach()
                    .cpu()
                    .item()
                ),
            },
        )

def build_neuroonline_forward(
    *,
    runtime_model: RuntimeModel,
    num_subject_codes: int = 32,
    num_attention_heads: int = 4,
    dropout: float = 0.1,
) -> NeuroOnlineForward:
    """
    根据 RuntimeModel 的 backend 特征规格，
    创建匹配的 NeuroOnlineGenerator 和前向桥接器。

    Generator 在这里创建一次，后续应持续复用。
    """

    backend = runtime_model.backend

    if not isinstance(
        backend,
        OnlineTrainableFeatureBackend,
    ):
        raise TypeError(
            "Runtime backend does not support "
            "NeuroOnline token adaptation: "
            f"{type(backend).__name__}."
        )

    feature_spec = (
        backend.online_feature_spec
    )

    generator = NeuroOnlineGenerator(
        feature_spec=feature_spec,
        num_subject_codes=(
            num_subject_codes
        ),
        num_attention_heads=(
            num_attention_heads
        ),
        dropout=dropout,
    ).to(
        backend.device
    )

    # 新创建后先进入推理模式。
    #
    # 真正在线更新时，再由训练逻辑切换成
    # generator.train()。
    generator.eval()

    backend.set_online_mode(
        training=False,
        train_backbone=False,
    )

    return NeuroOnlineForward(
        runtime_model=runtime_model,
        generator=generator,
    )
