from __future__ import annotations

import numpy as np
import torch

from bci_dayloop.models.base import ModelBackend
from bci_dayloop.preprocessing.base import (
    ModelInputTransform,
)
from bci_dayloop.preprocessing.canonical import (
    SignalCanonicalizer,
)
from bci_dayloop.runtime.model import RuntimeModel
from bci_dayloop.runtime.types import (
    CanonicalEEGWindow,
    InputContract,
    ModelOutput,
    ModelTensor,
    PreparedModelInput,
)


class IdentityInputTransform(ModelInputTransform):
    """
    测试专用 Transform。

    默认只输出 signal。
    include_scale=True 时额外输出 scale，
    用于测试 Runtime 是否支持 dict[str, Tensor] 输入。
    """

    def __init__(
        self,
        *,
        channel_names: tuple[str, ...],
        sample_rate: float,
        window_sec: float,
        include_scale: bool = False,
    ) -> None:
        self.include_scale = bool(include_scale)

        model_input_keys = (
            ("signal", "scale")
            if self.include_scale
            else ("signal",)
        )

        self._input_contract = InputContract(
            channel_names=channel_names,
            sample_rate=float(sample_rate),
            window_sec=float(window_sec),
            num_samples=int(
                round(sample_rate * window_sec)
            ),
            input_unit="uV",
            tensor_layout="BCT",
            strict_window_duration=True,
            model_input_keys=model_input_keys,
        )

    @property
    def input_contract(self) -> InputContract:
        return self._input_contract

    def transform(
        self,
        window: CanonicalEEGWindow,
    ) -> PreparedModelInput:
        signal = torch.from_numpy(
            np.ascontiguousarray(
                window.data,
                dtype=np.float32,
            )
        ).unsqueeze(0)

        model_input: dict[str, torch.Tensor] = {
            "signal": signal,
        }

        if self.include_scale:
            model_input["scale"] = torch.ones(
                (1, 1),
                dtype=torch.float32,
            )

        return PreparedModelInput(
            model_input=model_input,
            canonical_window=window,
            preprocessing_trace=[
                *window.processing_history,
                "identity_input_transform",
            ],
        )


class FixedBackend(ModelBackend):
    """
    测试专用 Backend。

    probabilities:
        固定返回的类别概率。

    expect_scale:
        是否要求 model_input 中存在 scale。

    error_message:
        非 None 时，predict_tensor 主动抛出 ValueError，
        用于测试 PipelineController 的失败传播。
    """

    def __init__(
        self,
        probabilities: tuple[float, ...] = (
            0.05,
            0.05,
            0.85,
            0.05,
        ),
        *,
        expect_scale: bool = False,
        error_message: str | None = None,
    ) -> None:
        self._probabilities = torch.tensor(
            [probabilities],
            dtype=torch.float32,
        )

        self.expect_scale = bool(expect_scale)
        self.error_message = error_message

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def num_classes(self) -> int:
        return int(
            self._probabilities.shape[-1]
        )

    def predict_tensor(
        self,
        model_input: ModelTensor,
        return_features: bool = False,
    ) -> ModelOutput:
        if self.error_message is not None:
            raise ValueError(
                self.error_message
            )

        if not isinstance(model_input, dict):
            raise TypeError(
                "Expected dict[str, torch.Tensor] "
                "model input."
            )

        if "signal" not in model_input:
            raise KeyError(
                "Model input is missing 'signal'."
            )

        signal = model_input["signal"]

        if not isinstance(signal, torch.Tensor):
            raise TypeError(
                "model_input['signal'] must be "
                "a torch.Tensor."
            )

        if signal.ndim != 3:
            raise ValueError(
                "Expected signal shape [B, C, T], "
                f"got {tuple(signal.shape)}."
            )

        if self.expect_scale:
            if "scale" not in model_input:
                raise KeyError(
                    "Model input is missing 'scale'."
                )

            scale = model_input["scale"]

            if not isinstance(
                scale,
                torch.Tensor,
            ):
                raise TypeError(
                    "model_input['scale'] must be "
                    "a torch.Tensor."
                )

            if scale.shape != (1, 1):
                raise ValueError(
                    "Expected scale shape [1, 1], "
                    f"got {tuple(scale.shape)}."
                )

        probabilities = (
            self._probabilities.clone()
        )

        logits = torch.log(
            probabilities.clamp_min(1e-8)
        )

        confidence, prediction = (
            probabilities.max(dim=-1)
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
                signal
                if return_features
                else None
            ),
        )

    def encode_tensor(
        self,
        model_input: ModelTensor,
    ) -> torch.Tensor:
        if not isinstance(model_input, dict):
            raise TypeError(
                "Expected dictionary model input."
            )

        if "signal" not in model_input:
            raise KeyError(
                "Model input is missing 'signal'."
            )

        return model_input["signal"]

    def get_trainable_parameters(
        self,
        scope: str,
    ) -> list[torch.nn.Parameter]:
        del scope
        return []


def build_fixed_runtime(
    *,
    channel_names: tuple[str, ...],
    sample_rate: float,
    window_sec: float,
    probabilities: tuple[float, ...] = (
        0.05,
        0.05,
        0.85,
        0.05,
    ),
    include_scale: bool = False,
    expect_scale: bool = False,
    error_message: str | None = None,
) -> RuntimeModel:
    """
    构建测试用 RuntimeModel。

    参数会分别传递给 Transform 和 Backend：

    include_scale
        控制 Transform 是否输出 scale。

    expect_scale
        控制 Backend 是否检查 scale。

    error_message
        控制 Backend 是否主动抛出异常。
    """

    return RuntimeModel(
        canonicalizer=SignalCanonicalizer(
            target_unit="uV",
        ),
        input_transform=IdentityInputTransform(
            channel_names=channel_names,
            sample_rate=sample_rate,
            window_sec=window_sec,
            include_scale=include_scale,
        ),
        backend=FixedBackend(
            probabilities=probabilities,
            expect_scale=expect_scale,
            error_message=error_message,
        ),
    )