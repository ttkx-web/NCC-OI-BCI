from __future__ import annotations

import torch

from bci_dayloop.models.base import (
    ModelBackend,
)
from bci_dayloop.models.labram_linear import (
    LaBraMLinearAdapter,
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

        if signal.ndim != 4:
            raise ValueError(
                "LaBraM signal must have shape "
                "[B,C,patches,200], got "
                f"{tuple(signal.shape)}."
            )

        if signal.shape[0] != 1:
            raise ValueError(
                "RuntimeModel currently expects one "
                "LaBraM window per prediction, got "
                f"batch_size={signal.shape[0]}."
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

    def predict_tensor(
        self,
        model_input: ModelTensor,
        return_features: bool = False,
    ) -> ModelOutput:
        signal = self._unpack_signal(
            model_input
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