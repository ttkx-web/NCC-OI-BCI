from __future__ import annotations

import numpy as np
import torch

from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.models.model_50m.preprocessing import (
    Model50MPreprocessor,
)
from bci_dayloop.preprocessing.base import ModelInputTransform
from bci_dayloop.runtime.types import (
    CanonicalEEGWindow,
    InputContract,
    PreparedModelInput,
)


class Model50MInputTransform(ModelInputTransform):
    """
    将 CanonicalEEGWindow 转换为 50M 模型输入。

    复用现有 Model50MPreprocessor，避免新旧 Runtime、
    训练脚本和 Replay 使用不同的预处理实现。
    """

    def __init__(self, config: Model50MConfig) -> None:
        self.config = config
        self.preprocessor = Model50MPreprocessor(config)

        self._contract = InputContract(
            channel_names=tuple(config.standard_channels),
            sample_rate=float(config.target_sample_rate),
            window_sec=float(config.window_seconds),
            num_samples=int(config.target_num_points),
            input_unit="uV",
            tensor_layout="BCT",
            strict_window_duration=bool(
                config.strict_window_duration
            ),
            model_input_keys=(
                "signal",
                "channel_valid_mask",
            ),
        )

    @property
    def input_contract(self) -> InputContract:
        return self._contract

    def transform(
        self,
        window: CanonicalEEGWindow,
    ) -> PreparedModelInput:
        """
        输入：
            window.data: [source_channels, source_time]

        输出：
            signal: [1, 64, target_time]
            channel_valid_mask: [1, 64]
        """

        result = self.preprocessor(
            signal=window.data,
            channel_names=window.channel_names,
            original_sample_rate=window.sample_rate,
            input_unit=window.unit,
        )

        signal = torch.from_numpy(
            np.ascontiguousarray(
                result.signal,
                dtype=np.float32,
            )
        ).unsqueeze(0)

        channel_valid_mask = torch.from_numpy(
            np.ascontiguousarray(
                result.channel_valid_mask,
                dtype=np.float32,
            )
        ).unsqueeze(0)

        trace = list(window.processing_history)
        trace.append("50m:convert_to_microvolts")
        trace.append("50m:align_to_standard_channels")

        if self.config.reference_mode == "average":
            trace.append("50m:average_reference")

        if self.config.filter_enabled:
            trace.append(
                "50m:bandpass_filter:"
                f"{self.config.filter_low_hz}-"
                f"{self.config.filter_high_hz}Hz"
            )

        trace.append(
            "50m:resample:"
            f"{window.sample_rate}->"
            f"{self.config.target_sample_rate}Hz"
        )

        trace.append(
            "50m:fix_window_length:"
            f"{self.config.target_num_points}"
        )

        if self.config.zscore_enabled:
            trace.append("50m:channelwise_zscore")

        trace.append("50m:add_batch_dimension")

        return PreparedModelInput(
            model_input={
                "signal": signal,
                "channel_valid_mask": channel_valid_mask,
            },
            canonical_window=window,
            preprocessing_trace=trace,
            diagnostics={
                "original_channel_names": (
                    result.original_channel_names
                ),
                "canonical_channel_names": (
                    result.canonical_channel_names
                ),
                "unknown_channel_names": (
                    result.unknown_channel_names
                ),
                "original_sample_rate": (
                    result.original_sample_rate
                ),
                "target_sample_rate": (
                    result.target_sample_rate
                ),
                "mapped_channel_count": (
                    result.mapped_channel_count
                ),
                "missing_channel_count": (
                    result.missing_channel_count
                ),
                "duplicate_channel_count": (
                    result.duplicate_channel_count
                ),
                "padded_points": result.padded_points,
                "cropped_points": result.cropped_points,
                "notes": result.notes,
            },
        )