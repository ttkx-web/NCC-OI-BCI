from __future__ import annotations

import numpy as np
import torch

from bci_dayloop.data.preprocessing import (
    EEGPreprocessor,
    PreprocessingConfig,
)
from bci_dayloop.preprocessing.base import (
    ModelInputTransform,
)
from bci_dayloop.runtime.types import (
    CanonicalEEGWindow,
    InputContract,
    PreparedModelInput,
)


class LaBraMInputTransform(ModelInputTransform):
    """
    将 CanonicalEEGWindow 转换为 LaBraM 输入。

    最终 Tensor：
        [B, C, n_patches, patch_samples]
    """

    def __init__(
        self,
        *,
        channel_names: tuple[str, ...],
        preprocessing_config: PreprocessingConfig,
        n_patches: int,
        strict_window_duration: bool = True,
    ) -> None:
        if not channel_names:
            raise ValueError(
                "LaBraM channel_names cannot be empty."
            )

        if n_patches <= 0:
            raise ValueError(
                "LaBraM n_patches must be positive."
            )

        self.channel_names = tuple(
            str(name) for name in channel_names
        )
        self.config = preprocessing_config
        self.n_patches = int(n_patches)
        self.strict_window_duration = bool(
            strict_window_duration
        )

        self.preprocessor = EEGPreprocessor(
            preprocessing_config
        )

        target_num_samples = (
            self.n_patches
            * self.config.patch_samples
        )

        window_sec = (
            target_num_samples
            / self.config.target_sample_rate
        )

        self._contract = InputContract(
            channel_names=self.channel_names,
            sample_rate=float(
                self.config.target_sample_rate
            ),
            window_sec=float(window_sec),
            num_samples=int(target_num_samples),
            input_unit="uV",
            tensor_layout="BCTP",
            strict_window_duration=(
                self.strict_window_duration
            ),
            model_input_keys=("signal",),
        )

    @property
    def input_contract(self) -> InputContract:
        return self._contract

    def _align_channels(
        self,
        window: CanonicalEEGWindow,
    ) -> tuple[np.ndarray, list[str]]:
        """
        根据 Package 中的通道顺序选择并重排输入。

        LaBraM 不建议像 50M 那样自动补零：
        缺少模型训练通道时直接报错更安全。
        """

        source_index: dict[str, int] = {}

        for index, name in enumerate(
            window.channel_names
        ):
            key = str(name).strip().upper()

            if key in source_index:
                raise ValueError(
                    "Duplicate source EEG channel after "
                    f"normalization: {name!r}."
                )

            source_index[key] = index

        missing = [
            name
            for name in self.channel_names
            if name.strip().upper()
            not in source_index
        ]

        if missing:
            raise ValueError(
                "LaBraM input is missing required channels: "
                f"{missing}."
            )

        indexes = [
            source_index[name.strip().upper()]
            for name in self.channel_names
        ]

        aligned = np.asarray(
            window.data[indexes, :],
            dtype=np.float32,
        )

        return aligned, missing

    def transform(
        self,
        window: CanonicalEEGWindow,
    ) -> PreparedModelInput:
        aligned, missing = self._align_channels(
            window
        )

        # 复用原有训练和 Replay 已验证的 EEGPreprocessor。
        processed = self.preprocessor.transform(
            aligned,
            sample_rate=float(
                window.sample_rate
            ),
            input_unit=str(window.unit),
            reshape=True,
        )

        # 单窗口预期：[C, patches, 200]
        if processed.ndim != 3:
            raise RuntimeError(
                "LaBraM preprocessing must produce "
                "[C, patches, patch_samples], got "
                f"{processed.shape}."
            )

        expected_shape = (
            len(self.channel_names),
            self.n_patches,
            self.config.patch_samples,
        )

        if processed.shape != expected_shape:
            raise ValueError(
                "LaBraM preprocessed shape does not "
                "match the model contract: "
                f"expected={expected_shape}, "
                f"actual={processed.shape}."
            )

        signal = torch.from_numpy(
            np.ascontiguousarray(
                processed,
                dtype=np.float32,
            )
        ).unsqueeze(0)

        trace = list(window.processing_history)

        trace.extend(
            [
                "labram:align_channels",
                (
                    "labram:bandpass:"
                    f"{self.config.bandpass_hz[0]}-"
                    f"{self.config.bandpass_hz[1]}Hz"
                ),
                (
                    "labram:notch:"
                    f"{self.config.notch_hz}Hz"
                ),
                (
                    "labram:resample:"
                    f"{window.sample_rate}->"
                    f"{self.config.target_sample_rate}Hz"
                ),
                "labram:ensure_microvolts",
                "labram:channelwise_zscore",
                (
                    "labram:reshape_patches:"
                    f"{self.n_patches}x"
                    f"{self.config.patch_samples}"
                ),
                "labram:add_batch_dimension",
            ]
        )

        return PreparedModelInput(
            model_input={
                "signal": signal,
            },
            canonical_window=window,
            preprocessing_trace=trace,
            diagnostics={
                "source_channel_count": len(
                    window.channel_names
                ),
                "target_channel_count": len(
                    self.channel_names
                ),
                "missing_channel_names": missing,
                "source_sample_rate": float(
                    window.sample_rate
                ),
                "target_sample_rate": float(
                    self.config.target_sample_rate
                ),
                "n_patches": self.n_patches,
                "patch_samples": int(
                    self.config.patch_samples
                ),
                "output_shape": list(
                    signal.shape
                ),
            },
        )