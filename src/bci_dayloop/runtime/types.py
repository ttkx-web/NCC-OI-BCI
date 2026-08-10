from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import numpy as np
import torch


ArrayLayout = Literal["CT", "TC"]
TensorLayout = Literal["BCT", "BTC", "BCTP"]


ModelTensor: TypeAlias = (
    torch.Tensor
    | dict[str, torch.Tensor]
)


@dataclass(slots=True)
class RawEEGWindow:
    """数据源提供的一个原始 EEG 窗口。"""

    data: np.ndarray
    channel_names: list[str]
    sample_rate: float
    unit: str
    layout: ArrayLayout = "CT"

    start_time_sec: float | None = None
    trial_id: str | None = None
    window_id: str | None = None
    label: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CanonicalEEGWindow:
    """
    通用标准化后的 EEG 窗口。

    data 固定为 [channel, time]。
    """

    data: np.ndarray
    channel_names: list[str]
    sample_rate: float
    unit: str

    start_time_sec: float | None = None
    trial_id: str | None = None
    window_id: str | None = None
    label: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    processing_history: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class InputContract:
    """
    模型专属输入要求。

    sample_rate、window_sec 和 num_samples 描述模型处理完成后的目标输入，
    不代表数据源必须天然满足这些值。
    """

    channel_names: tuple[str, ...]
    sample_rate: float
    window_sec: float
    num_samples: int
    input_unit: str
    tensor_layout: TensorLayout
    strict_window_duration: bool = True

    # 50M 为 ("signal", "channel_valid_mask")；
    # 普通模型可以只使用 ("signal",)。
    model_input_keys: tuple[str, ...] = ("signal",)

    def __post_init__(self) -> None:
        if not self.channel_names:
            raise ValueError(
                "InputContract.channel_names cannot be empty."
            )

        normalized_names = [
            name.strip().upper()
            for name in self.channel_names
        ]

        if any(not name for name in normalized_names):
            raise ValueError(
                "InputContract contains an empty channel name."
            )

        if len(normalized_names) != len(set(normalized_names)):
            raise ValueError(
                "InputContract channel_names contains duplicates."
            )

        if (
            not math.isfinite(self.sample_rate)
            or self.sample_rate <= 0
        ):
            raise ValueError(
                f"sample_rate must be finite and positive, "
                f"got {self.sample_rate}."
            )

        if (
            not math.isfinite(self.window_sec)
            or self.window_sec <= 0
        ):
            raise ValueError(
                f"window_sec must be finite and positive, "
                f"got {self.window_sec}."
            )

        if self.num_samples <= 0:
            raise ValueError(
                f"num_samples must be positive, "
                f"got {self.num_samples}."
            )

        expected_num_samples = (
            self.sample_rate * self.window_sec
        )

        if abs(self.num_samples - expected_num_samples) > 1.0:
            raise ValueError(
                "InputContract is internally inconsistent: "
                f"sample_rate × window_sec = "
                f"{expected_num_samples}, "
                f"but num_samples = {self.num_samples}."
            )

        if not self.input_unit.strip():
            raise ValueError(
                "InputContract.input_unit cannot be empty."
            )

        if not self.model_input_keys:
            raise ValueError(
                "InputContract.model_input_keys cannot be empty."
            )


@dataclass(slots=True)
class PreparedModelInput:
    """
    模型专属预处理的最终结果。

    model_input 可以是一个 Tensor，也可以是包含多个 Tensor 的字典。
    """

    model_input: ModelTensor
    canonical_window: CanonicalEEGWindow
    preprocessing_trace: list[str]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def get_tensor(
        self,
        name: str = "signal",
    ) -> torch.Tensor:
        if isinstance(self.model_input, torch.Tensor):
            if name != "signal":
                raise KeyError(
                    "Prepared input is a single Tensor; "
                    f"requested key={name!r}."
                )
            return self.model_input

        if name not in self.model_input:
            raise KeyError(
                f"Prepared model input does not contain {name!r}. "
                f"Available keys: "
                f"{sorted(self.model_input.keys())}."
            )

        return self.model_input[name]


@dataclass(slots=True)
class ModelOutput:
    """单个 Runtime 窗口的统一预测输出。"""

    logits: torch.Tensor
    probabilities: torch.Tensor
    predicted_class: int
    confidence: float
    features: torch.Tensor | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)