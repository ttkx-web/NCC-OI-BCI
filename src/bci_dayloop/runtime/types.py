from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch


ArrayLayout = Literal["CT", "TC"]
TensorLayout = Literal["BCT", "BTC"]


@dataclass
class RawEEGWindow:
    """数据源提供的原始 EEG 窗口。"""

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


@dataclass
class CanonicalEEGWindow:
    """完成通用标准化后的 EEG，固定为 channel × time。"""

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


@dataclass(frozen=True)
class InputContract:
    channel_names: tuple[str, ...]
    sample_rate: float
    window_sec: float
    num_samples: int
    input_unit: str
    tensor_layout: TensorLayout
    strict_window_duration: bool = True


@dataclass
class PreparedModelInput:
    tensor: torch.Tensor
    canonical_window: CanonicalEEGWindow
    preprocessing_trace: list[str]


@dataclass
class ModelOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    predicted_class: int
    confidence: float
    features: torch.Tensor | None = None