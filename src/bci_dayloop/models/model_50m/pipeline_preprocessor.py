from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import Model50MConfig
from .preprocessing import Model50MPreprocessor


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

    def transform(self, raw_window: np.ndarray) -> np.ndarray:
        """
        Args:
            raw_window:
                原始 EEG，[C, T]。
                阶段 0.5 必须是真实 10 秒数据，不能用 4 秒补零。

        Returns:
            [64, 1000]，np.float32。
        """
        result = self.preprocessor(
            signal=raw_window,
            channel_names=self.channel_names,
            original_sample_rate=self.sample_rate,
            input_unit=self.input_unit,
        )

        self.last_result = result
        self.last_channel_valid_mask = result.channel_valid_mask

        return result.signal