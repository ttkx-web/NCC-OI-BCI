from __future__ import annotations

import torch

from bci_dayloop.models.base import (
    ModelBackend,
)
from bci_dayloop.models.labram_linear import (
    LaBraMLinearAdapter,
)
from bci_dayloop.models.online_features import (
    OnlineFeatureSpec,
)
from bci_dayloop.runtime.types import (
    ModelOutput,
    ModelTensor,
)


class LaBraMBackend(ModelBackend):
    """
    统一 Runtime 使用的 LaBraM 计算后端。

    输入必须已经经过 LaBraMInputTransform：
        signal: [1, C, patches, 200]
    """

    def __init__(
        self,
        adapter: LaBraMLinearAdapter,
    ) -> None:
        self.adapter = adapter

    @property
    def device(self) -> torch.device:
        return self.adapter.device

    @property
    def num_classes(self) -> int:
        return int(self.adapter.n_classes)

    @property
    def online_feature_spec(
            self,
    ) -> OnlineFeatureSpec:
        """
        返回 LaBraM 的在线 token 规格。

        LaBraM 每个通道、每个时间 patch 对应一个 token：

            token_count = channel_count * patch_count

        例如：
            22 channels * 4 patches = 88 tokens
        """

        return OnlineFeatureSpec(
            model_name="labram",
            token_count=(
                    len(self.adapter.channel_names)
                    * self.adapter.n_patches
            ),
            embedding_dim=(
                self.adapter.embedding_dim
            ),
        )

    @staticmethod
    def _unpack_signal(
        model_input: ModelTensor,
    ) -> torch.Tensor:
        if isinstance(model_input, torch.Tensor):
            signal = model_input
        elif isinstance(model_input, dict):
            if "signal" not in model_input:
                raise ValueError(
                    "LaBraM model_input is missing "
                    "required key 'signal'."
                )

            signal = model_input["signal"]
        else:
            raise TypeError(
                "LaBraM model_input must be a Tensor "
                "or dict[str, Tensor]."
            )

        if not isinstance(signal, torch.Tensor):
            raise TypeError(
                "LaBraM signal must be torch.Tensor."
            )
        if signal.shape[0] <= 0:
            raise ValueError(
                "LaBraM batch size must be positive."
            )

        if signal.ndim != 4:
            raise ValueError(
                "LaBraM signal must have shape "
                "[B,C,patches,200], got "
                f"{tuple(signal.shape)}."
            )

        if signal.shape[-1] != 200:
            raise ValueError(
                "LaBraM patch length must be 200, got "
                f"{signal.shape[-1]}."
            )

        if not torch.isfinite(signal).all():
            raise ValueError(
                "LaBraM signal contains NaN or Inf."
            )

        return signal

    def _extract_features(
            self,
            signal: torch.Tensor,
    ) -> torch.Tensor:
        signal = signal.to(
            self.adapter.device,
            non_blocking=(
                    self.adapter.device.type
                    == "cuda"
            ),
        )

        self.adapter.encoder.eval()

        with self.adapter._autocast():
            if hasattr(
                    self.adapter.encoder,
                    "forward_features",
            ):
                features = (
                    self.adapter.encoder
                    .forward_features(
                        signal,
                        input_chans=(
                            self.adapter
                            .input_chans
                        ),
                    )
                )
            else:
                features = (
                    self.adapter.encoder(
                        signal
                    )
                )

        return features.float()

    def _extract_patch_tokens(
            self,
            signal: torch.Tensor,
            *,
            train_backbone: bool,
    ) -> torch.Tensor:
        """
        提取未池化的 LaBraM patch tokens。

        输入：
            signal: [B,C,S,200]

        输出：
            tokens: [B,C*S,D]

        V1 中 train_backbone=False：
            backbone 不建立梯度图，但返回的普通 Tensor
            仍然可以作为 Generator 的输入参与反向传播。
        """

        signal = signal.to(
            self.adapter.device,
            dtype=torch.float32,
            non_blocking=(
                    self.adapter.device.type
                    == "cuda"
            ),
        )

        encoder = self.adapter.encoder

        if not hasattr(
                encoder,
                "forward_features",
        ):
            raise RuntimeError(
                "LaBraM encoder does not provide "
                "forward_features(), so patch tokens "
                "cannot be extracted."
            )

        if train_backbone:
            encoder.train()
        else:
            encoder.eval()

        # train_backbone=False 时等价于 no_grad，
        # 但不会生成 inference tensor。
        #
        # 不能使用 torch.inference_mode()：
        # inference tensor 后续送入 Generator 时，
        # 可能无法参与 Generator 参数的反向传播。
        with torch.set_grad_enabled(
                train_backbone
        ):
            with self.adapter._autocast():
                tokens = (
                    encoder.forward_features(
                        signal,
                        input_chans=(
                            self.adapter.input_chans
                        ),
                        return_patch_tokens=True,
                    )
                )

        tokens = tokens.float()

        if tokens.ndim != 3:
            raise RuntimeError(
                "LaBraM patch tokens must have "
                "shape [B,N,D], got "
                f"{tuple(tokens.shape)}."
            )

        if not torch.isfinite(
                tokens
        ).all():
            raise RuntimeError(
                "LaBraM patch tokens contain "
                "NaN or Inf."
            )

        return tokens

    def encode_online_tokens(
            self,
            model_input: ModelTensor,
            *,
            train_backbone: bool = False,
    ) -> torch.Tensor:
        """
        为 NeuroOnline 提取统一的 [B,N,D] token。

        当前 V1 应始终使用：
            train_backbone=False
        """

        signal = self._unpack_signal(
            model_input
        )

        tokens = self._extract_patch_tokens(
            signal,
            train_backbone=train_backbone,
        )

        spec = self.online_feature_spec

        expected_shape = (
            signal.shape[0],
            spec.token_count,
            spec.embedding_dim,
        )

        if tuple(tokens.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected LaBraM online token "
                "shape: "
                f"expected={expected_shape}, "
                f"actual={tuple(tokens.shape)}."
            )

        return tokens

    def classify_online_tokens(
            self,
            tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        将 Generator 调整后的 patch tokens
        送回现有 LaBraM 线性分类头。

        输入：
            [B,N,D]

        处理：
            mean pooling -> [B,D]
            linear head  -> [B,num_classes]
        """

        if not isinstance(
                tokens,
                torch.Tensor,
        ):
            raise TypeError(
                "LaBraM online tokens must be "
                "torch.Tensor, got "
                f"{type(tokens).__name__}."
            )

        if tokens.ndim != 3:
            raise ValueError(
                "LaBraM online tokens must have "
                "shape [B,N,D], got "
                f"{tuple(tokens.shape)}."
            )

        if tokens.shape[0] <= 0:
            raise ValueError(
                "LaBraM online token batch size "
                "must be positive."
            )

        spec = self.online_feature_spec

        if tokens.shape[1] != spec.token_count:
            raise ValueError(
                "LaBraM online token count "
                "does not match the Runtime "
                "Model Package: "
                f"expected={spec.token_count}, "
                f"actual={tokens.shape[1]}."
            )

        if tokens.shape[2] != spec.embedding_dim:
            raise ValueError(
                "LaBraM online embedding "
                "dimension mismatch: "
                f"expected={spec.embedding_dim}, "
                f"actual={tokens.shape[2]}."
            )

        if not tokens.is_floating_point():
            raise TypeError(
                "LaBraM online tokens must have "
                "a floating-point dtype, got "
                f"{tokens.dtype}."
            )

        if not torch.isfinite(
                tokens
        ).all():
            raise ValueError(
                "LaBraM online tokens contain "
                "NaN or Inf."
            )

        tokens = tokens.to(
            self.device,
            non_blocking=(
                    self.device.type == "cuda"
            ),
        )

        # 恢复 LaBraM 当前静态前向中的 mean pooling。
        pooled_features = tokens.mean(
            dim=1
        )

        logits = self.adapter.head(
            pooled_features
        )

        expected_logits_shape = (
            tokens.shape[0],
            self.num_classes,
        )

        if tuple(logits.shape) != (
                expected_logits_shape
        ):
            raise RuntimeError(
                "Unexpected LaBraM online logits "
                "shape: "
                f"expected={expected_logits_shape}, "
                f"actual={tuple(logits.shape)}."
            )

        if not torch.isfinite(
                logits
        ).all():
            raise RuntimeError(
                "LaBraM online classifier "
                "produced NaN or Inf logits."
            )

        return logits

    def set_online_mode(
            self,
            *,
            training: bool,
            train_backbone: bool = False,
    ) -> None:
        """
        设置 LaBraM NeuroOnline 前向模式。
        """

        if training and train_backbone:
            for parameter in (
                    self.adapter.encoder.parameters()
            ):
                parameter.requires_grad = True

            self.adapter.encoder.train()

        else:
            for parameter in (
                    self.adapter.encoder.parameters()
            ):
                parameter.requires_grad = False

            self.adapter.encoder.eval()

        self.adapter.head.train(
            mode=training
        )

    def predict_tensor(
        self,
        model_input: ModelTensor,
        return_features: bool = False,
    ) -> ModelOutput:
        signal = self._unpack_signal(
            model_input
        )

        if signal.shape[0] != 1:
            raise ValueError(
                "RuntimeModel currently expects one "
                "LaBraM window per prediction, got "
                f"batch_size={signal.shape[0]}."
            )

        self.adapter.encoder.eval()
        self.adapter.head.eval()

        with torch.inference_mode():
            features = self._extract_features(
                signal
            )

            logits = self.adapter.head(
                features
            )

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            confidence, prediction = (
                probabilities.max(dim=-1)
            )

        if logits.shape != (
            1,
            self.num_classes,
        ):
            raise RuntimeError(
                "Unexpected LaBraM logits shape: "
                f"{tuple(logits.shape)}."
            )

        return ModelOutput(
            logits=logits,
            probabilities=probabilities,
            predicted_class=int(
                prediction[0].item()
            ),
            confidence=float(
                confidence[0].item()
            ),
            features=(
                features
                if return_features
                else None
            ),
            diagnostics={
                "backend": "labram",
                "embedding_dim": int(
                    features.shape[-1]
                ),
            },
        )

    def encode_tensor(
            self,
            model_input: ModelTensor,
    ) -> torch.Tensor:
        signal = self._unpack_signal(
            model_input
        )

        self.adapter.encoder.eval()

        with torch.inference_mode():
            return self._extract_features(
                signal
            )

    def get_trainable_parameters(
        self,
        scope: str,
    ) -> list[torch.nn.Parameter]:
        normalized = scope.strip().lower()

        if normalized == "head":
            for parameter in (
                self.adapter.encoder.parameters()
            ):
                parameter.requires_grad = False

            for parameter in (
                self.adapter.head.parameters()
            ):
                parameter.requires_grad = True

            self.adapter.encoder.eval()
            self.adapter.head.train()

            return list(
                self.adapter.head.parameters()
            )

        if normalized in {
            "backbone",
            "encoder",
        }:
            for parameter in (
                self.adapter.encoder.parameters()
            ):
                parameter.requires_grad = True

            for parameter in (
                self.adapter.head.parameters()
            ):
                parameter.requires_grad = False

            self.adapter.encoder.train()
            self.adapter.head.eval()

            return list(
                self.adapter.encoder.parameters()
            )

        if normalized == "full":
            for parameter in (
                self.adapter.encoder.parameters()
            ):
                parameter.requires_grad = True

            for parameter in (
                self.adapter.head.parameters()
            ):
                parameter.requires_grad = True

            self.adapter.encoder.train()
            self.adapter.head.train()

            return [
                *self.adapter.encoder.parameters(),
                *self.adapter.head.parameters(),
            ]

        raise ValueError(
            f"Unsupported LaBraM trainable scope "
            f"{scope!r}. Expected one of: "
            "'head', 'backbone', 'encoder', 'full'."
        )