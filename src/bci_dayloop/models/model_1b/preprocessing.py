"""1B latency-only input transform reusing the verified 50M EEG preprocessor."""

from __future__ import annotations

import numpy as np
import torch

from bci_dayloop.models.model_50m.preprocessing import Model50MPreprocessor
from bci_dayloop.preprocessing.base import ModelInputTransform
from bci_dayloop.runtime.types import CanonicalEEGWindow, InputContract, PreparedModelInput

from .config import Model1BConfig


class Model1BInputTransform(ModelInputTransform):
    """Prepare only; this class has no classification/prediction operation."""

    def __init__(self, config: Model1BConfig) -> None:
        self.config = config
        self.preprocessor = Model50MPreprocessor(config)
        self._contract = InputContract(
            channel_names=tuple(config.standard_channels),
            sample_rate=float(config.target_sample_rate),
            window_sec=float(config.window_seconds),
            num_samples=int(config.target_num_points),
            input_unit="uV",
            tensor_layout="BCT",
            strict_window_duration=bool(config.strict_window_duration),
            model_input_keys=("signal", "channel_valid_mask"),
        )

    @property
    def input_contract(self) -> InputContract:
        return self._contract

    def transform(self, window: CanonicalEEGWindow) -> PreparedModelInput:
        result = self.preprocessor(
            signal=window.data,
            channel_names=window.channel_names,
            original_sample_rate=window.sample_rate,
            input_unit=window.unit,
        )
        signal = torch.from_numpy(np.ascontiguousarray(result.signal, dtype=np.float32)).unsqueeze(0)
        mask = torch.from_numpy(np.ascontiguousarray(result.channel_valid_mask, dtype=np.float32)).unsqueeze(0)
        trace = list(window.processing_history) + [
            "1b:reuse_50m_preprocessing",
            "1b:add_batch_dimension",
        ]
        return PreparedModelInput(
            model_input={"signal": signal, "channel_valid_mask": mask},
            canonical_window=window,
            preprocessing_trace=trace,
            diagnostics={
                "original_channel_names": result.original_channel_names,
                "canonical_channel_names": result.canonical_channel_names,
                "unknown_channel_names": result.unknown_channel_names,
                "original_sample_rate": result.original_sample_rate,
                "target_sample_rate": result.target_sample_rate,
                "mapped_channel_count": result.mapped_channel_count,
                "missing_channel_count": result.missing_channel_count,
                "duplicate_channel_count": result.duplicate_channel_count,
                "padded_points": result.padded_points,
                "cropped_points": result.cropped_points,
                "notes": result.notes,
            },
        )
