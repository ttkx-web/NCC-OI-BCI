from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import Model50MConfig
from .preprocessing import Model50MPreprocessor


@dataclass(frozen=True, slots=True)
class Model50MPreprocessingDiagnostics:
    mapped_channel_count: int
    missing_channel_count: int
    unknown_channel_names: tuple[str, ...]
    notes: tuple[str, ...]


class Model50MPipelinePreprocessor:
    """
    兼容当前 SlidingWindowDecoder 的 50M 预处理包装器。

    输入：
        原始 EEG [C, T]

    输出：
        50M 标准化输入 [64, 1000]
    """

    def __init__(
        self,
        config: Model50MConfig,
        *,
        channel_names: Sequence[str],
        sample_rate: float,
        input_unit: str,
    ) -> None:
        self.config = config
        self.channel_names = tuple(channel_names)
        self.sample_rate = float(sample_rate)
        self.input_unit = str(input_unit)

        self.preprocessor = Model50MPreprocessor(config)

        self.last_channel_valid_mask: np.ndarray | None = None
        self.last_result = None
        self.last_diagnostics: Model50MPreprocessingDiagnostics | None = None

    def transform(
        self,
        samples: np.ndarray,
        sample_rate: float,
        input_unit: str,
        *,
        reshape: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Args:
            raw_window:
                原始 EEG，[C, T]。
                阶段 0.5 必须是真实 10 秒数据，不能用 4 秒补零。

        Returns:
            [64, 1000]，np.float32。
        """
        if not reshape:
            raise ValueError("Model50MPipelinePreprocessor only supports reshape=True")
        raw_window = np.asarray(samples)
        if raw_window.ndim != 2:
            raise ValueError(f"50M samples must have shape [C,T], got {raw_window.shape}")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be greater than zero, got {sample_rate}")

        result = self.preprocessor(
            signal=raw_window,
            channel_names=self.channel_names,
            original_sample_rate=float(sample_rate),
            input_unit=str(input_unit),
        )

        self.last_result = result
        self.last_channel_valid_mask = result.channel_valid_mask

        self.last_diagnostics = Model50MPreprocessingDiagnostics(
            mapped_channel_count=result.mapped_channel_count,
            missing_channel_count=result.missing_channel_count,
            unknown_channel_names=result.unknown_channel_names,
            notes=result.notes,
        )
        return {
            "signal": result.signal,
            "channel_valid_mask": result.channel_valid_mask,
        }
