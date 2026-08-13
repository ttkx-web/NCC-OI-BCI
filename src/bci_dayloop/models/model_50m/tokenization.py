from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np
import torch

try:
    # 作为 package 导入时
    from .config import Model50MConfig
except ImportError:
    # 直接运行当前文件时
    from config import Model50MConfig

if TYPE_CHECKING:
    from .preprocessing import PreprocessResult


@dataclass(frozen=True, slots=True)
class Model50MTokenizedInput:
    """
    单个 EEG 样本 Token 化后的结果，不包含 batch 维。

    默认 50M 原始配置下：

        token_inputs:
            [640, 100], torch.float32

        token_channel_indices:
            [640], torch.int64

        token_time_indices:
            [640], torch.int64

        token_valid_mask:
            [640], torch.float32

        channel_valid_mask:
            [64], torch.float32
    """

    token_inputs: torch.Tensor
    token_channel_indices: torch.Tensor
    token_time_indices: torch.Tensor
    token_valid_mask: torch.Tensor
    channel_valid_mask: torch.Tensor

    num_channels: int
    num_time_patches: int
    patch_len: int
    patch_stride: int

    @property
    def num_tokens(self) -> int:
        return int(self.token_inputs.shape[0])

    @property
    def valid_token_count(self) -> int:
        return int((self.token_valid_mask > 0.5).sum().item())

    def validate(self) -> None:
        """检查单样本 Token 张量的形状和 dtype。"""
        expected_num_tokens = self.num_channels * self.num_time_patches

        if self.token_inputs.ndim != 2:
            raise ValueError(
                "token_inputs must have shape [S, L], "
                f"got {tuple(self.token_inputs.shape)}."
            )

        if tuple(self.token_inputs.shape) != (
            expected_num_tokens,
            self.patch_len,
        ):
            raise ValueError(
                "Unexpected token_inputs shape: "
                f"expected {(expected_num_tokens, self.patch_len)}, "
                f"got {tuple(self.token_inputs.shape)}."
            )

        expected_vector_shape = (expected_num_tokens,)

        for name, tensor in (
            ("token_channel_indices", self.token_channel_indices),
            ("token_time_indices", self.token_time_indices),
            ("token_valid_mask", self.token_valid_mask),
        ):
            if tuple(tensor.shape) != expected_vector_shape:
                raise ValueError(
                    f"{name} should have shape {expected_vector_shape}, "
                    f"got {tuple(tensor.shape)}."
                )

        if tuple(self.channel_valid_mask.shape) != (self.num_channels,):
            raise ValueError(
                "channel_valid_mask should have shape "
                f"{(self.num_channels,)}, "
                f"got {tuple(self.channel_valid_mask.shape)}."
            )

        if self.token_inputs.dtype != torch.float32:
            raise TypeError(
                "token_inputs must be torch.float32, "
                f"got {self.token_inputs.dtype}."
            )

        if self.token_channel_indices.dtype != torch.int64:
            raise TypeError(
                "token_channel_indices must be torch.int64, "
                f"got {self.token_channel_indices.dtype}."
            )

        if self.token_time_indices.dtype != torch.int64:
            raise TypeError(
                "token_time_indices must be torch.int64, "
                f"got {self.token_time_indices.dtype}."
            )

        if self.token_valid_mask.dtype != torch.float32:
            raise TypeError(
                "token_valid_mask must be torch.float32, "
                f"got {self.token_valid_mask.dtype}."
            )

        if self.channel_valid_mask.dtype != torch.float32:
            raise TypeError(
                "channel_valid_mask must be torch.float32, "
                f"got {self.channel_valid_mask.dtype}."
            )

        if not torch.isfinite(self.token_inputs).all():
            raise ValueError("token_inputs contains NaN or Inf.")

        if not torch.isfinite(self.token_valid_mask).all():
            raise ValueError("token_valid_mask contains NaN or Inf.")

        if self.token_channel_indices.numel() > 0:
            min_channel_index = int(
                self.token_channel_indices.min().item()
            )
            max_channel_index = int(
                self.token_channel_indices.max().item()
            )

            if min_channel_index < 0:
                raise ValueError(
                    "token_channel_indices contains a negative value."
                )

            if max_channel_index >= self.num_channels:
                raise ValueError(
                    "token_channel_indices exceeds channel range: "
                    f"max={max_channel_index}, "
                    f"num_channels={self.num_channels}."
                )

        if self.token_time_indices.numel() > 0:
            min_time_index = int(self.token_time_indices.min().item())
            max_time_index = int(self.token_time_indices.max().item())

            if min_time_index < 0:
                raise ValueError(
                    "token_time_indices contains a negative value."
                )

            if max_time_index >= self.num_time_patches:
                raise ValueError(
                    "token_time_indices exceeds time-patch range: "
                    f"max={max_time_index}, "
                    f"num_time_patches={self.num_time_patches}."
                )

    def as_batch(
        self,
        device: torch.device | str | None = None,
        *,
        non_blocking: bool = False,
    ) -> "Model50MBatchedInput":
        """
        添加 batch 维，转换成 50M 模型接口需要的 [B, ...] 形状。

        返回的 batch size 为 1。
        """
        self.validate()

        batched = Model50MBatchedInput(
            token_inputs=self.token_inputs.unsqueeze(0),
            token_channel_indices=(
                self.token_channel_indices.unsqueeze(0)
            ),
            token_time_indices=self.token_time_indices.unsqueeze(0),
            token_valid_mask=self.token_valid_mask.unsqueeze(0),
            channel_valid_mask=self.channel_valid_mask.unsqueeze(0),
        )

        if device is not None:
            batched = batched.to(
                device=device,
                non_blocking=non_blocking,
            )

        batched.validate()
        return batched


@dataclass(frozen=True, slots=True)
class Model50MBatchedInput:
    """
    带 batch 维的 50M 输入。

    默认单样本形状：

        token_inputs:           [1, 640, 100]
        token_channel_indices:  [1, 640]
        token_time_indices:     [1, 640]
        token_valid_mask:       [1, 640]
        channel_valid_mask:     [1, 64]
    """

    token_inputs: torch.Tensor
    token_channel_indices: torch.Tensor
    token_time_indices: torch.Tensor
    token_valid_mask: torch.Tensor
    channel_valid_mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.token_inputs.shape[0])

    @property
    def num_tokens(self) -> int:
        return int(self.token_inputs.shape[1])

    @property
    def patch_len(self) -> int:
        return int(self.token_inputs.shape[2])

    def validate(self) -> None:
        if self.token_inputs.ndim != 3:
            raise ValueError(
                "Batched token_inputs must have shape [B, S, L], "
                f"got {tuple(self.token_inputs.shape)}."
            )

        batch_size, num_tokens, _ = self.token_inputs.shape

        expected_token_shape = (batch_size, num_tokens)

        for name, tensor in (
            ("token_channel_indices", self.token_channel_indices),
            ("token_time_indices", self.token_time_indices),
            ("token_valid_mask", self.token_valid_mask),
        ):
            if tuple(tensor.shape) != expected_token_shape:
                raise ValueError(
                    f"{name} should have shape "
                    f"{expected_token_shape}, "
                    f"got {tuple(tensor.shape)}."
                )

        if self.channel_valid_mask.ndim != 2:
            raise ValueError(
                "channel_valid_mask must have shape [B, C], "
                f"got {tuple(self.channel_valid_mask.shape)}."
            )

        if self.channel_valid_mask.shape[0] != batch_size:
            raise ValueError(
                "channel_valid_mask batch size does not match "
                "token_inputs."
            )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "Model50MBatchedInput":
        """将全部输入移动到 CPU、CUDA 或 MPS。"""
        return Model50MBatchedInput(
            token_inputs=self.token_inputs.to(
                device=device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            ),
            token_channel_indices=self.token_channel_indices.to(
                device=device,
                dtype=torch.long,
                non_blocking=non_blocking,
            ),
            token_time_indices=self.token_time_indices.to(
                device=device,
                dtype=torch.long,
                non_blocking=non_blocking,
            ),
            token_valid_mask=self.token_valid_mask.to(
                device=device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            ),
            channel_valid_mask=self.channel_valid_mask.to(
                device=device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            ),
        )

    def model_kwargs(self) -> dict[str, torch.Tensor]:
        """
        返回 extract_token_embeddings() 可以直接接收的参数。

        token_valid_mask 不传给 Backbone，而是在下游 Flatten 或
        Mean Pooling 时使用。
        """
        self.validate()

        return {
            "token_inputs": self.token_inputs,
            "token_channel_indices": self.token_channel_indices,
            "token_time_indices": self.token_time_indices,
        }


def _validate_channel_valid_mask(
    channel_valid_mask: np.ndarray,
    num_channels: int,
) -> np.ndarray:
    mask = np.asarray(channel_valid_mask)

    if mask.shape != (num_channels,):
        raise ValueError(
            "channel_valid_mask should have shape "
            f"{(num_channels,)}, got {mask.shape}."
        )

    if not np.isfinite(mask).all():
        raise ValueError(
            "channel_valid_mask contains NaN or Inf."
        )

    mask = mask.astype(np.float32, copy=False)

    # 允许输入 bool、0/1 整数或 0.0/1.0 浮点数。
    is_binary = np.logical_or(
        np.isclose(mask, 0.0),
        np.isclose(mask, 1.0),
    )

    if not is_binary.all():
        invalid_values = np.unique(mask[~is_binary])
        raise ValueError(
            "channel_valid_mask must contain only 0 or 1. "
            f"Invalid values: {invalid_values.tolist()}."
        )

    return (mask > 0.5).astype(np.float32)


def make_channel_time_patches(
    signal: np.ndarray,
    *,
    patch_len: int,
    patch_stride: int,
) -> np.ndarray:
    """
    将 EEG 按通道和时间切成 Patch。

    Args:
        signal:
            [C, T]。

        patch_len:
            单个 Patch 的采样点数。
            默认 100 Hz × 1 秒 = 100 点。

        patch_stride:
            Patch 步长的采样点数。
            默认 100 Hz × 1 秒 = 100 点。

    Returns:
        patches:
            [C, N, L]，np.float32。

    不进行补零或裁剪。输入长度不足时直接报错。
    """
    signal = np.asarray(signal)

    if signal.ndim != 2:
        raise ValueError(
            f"signal must have shape [C, T], got {signal.shape}."
        )

    if not np.issubdtype(signal.dtype, np.number):
        raise TypeError(
            f"signal must be numeric, got dtype={signal.dtype}."
        )

    if not np.isfinite(signal).all():
        raise ValueError("signal contains NaN or Inf.")

    if patch_len <= 0:
        raise ValueError(
            f"patch_len must be positive, got {patch_len}."
        )

    if patch_stride <= 0:
        raise ValueError(
            "patch_stride must be positive, "
            f"got {patch_stride}."
        )

    num_channels, total_points = signal.shape

    if total_points < patch_len:
        raise ValueError(
            "EEG window is shorter than one Patch: "
            f"total_points={total_points}, patch_len={patch_len}."
        )

    num_patches = (
        (total_points - patch_len) // patch_stride
    ) + 1

    if num_patches <= 0:
        raise ValueError(
            "No time Patch can be generated from the input."
        )

    patches = np.empty(
        (num_channels, num_patches, patch_len),
        dtype=np.float32,
    )

    for patch_index in range(num_patches):
        start = patch_index * patch_stride
        end = start + patch_len

        patches[:, patch_index, :] = signal[:, start:end]

    return np.ascontiguousarray(
        patches,
        dtype=np.float32,
    )


class Model50MTokenizer:
    """
    将 Model50MPreprocessor 输出转换为 50M Token。

    默认原始配置：

        输入 signal: [64, 1000]
        patches:      [64, 10, 100]
        tokens:       [640, 100]
    """

    def __init__(self, config: Model50MConfig):
        self.config = config

    def __call__(
        self,
        preprocessed: PreprocessResult,
    ) -> Model50MTokenizedInput:
        return self.tokenize(
            signal=preprocessed.signal,
            channel_valid_mask=(
                preprocessed.channel_valid_mask
            ),
        )

    def tokenize(
        self,
        signal: np.ndarray,
        channel_valid_mask: np.ndarray,
    ) -> Model50MTokenizedInput:
        config = self.config

        signal = np.asarray(signal)

        expected_signal_shape = (
            config.n_channels,
            config.target_num_points,
        )

        if signal.shape != expected_signal_shape:
            actual_duration = (
                signal.shape[-1] / config.target_sample_rate
                if signal.ndim == 2
                else None
            )

            duration_text = (
                f"{actual_duration:.3f}s"
                if actual_duration is not None
                else "unknown"
            )

            raise ValueError(
                "The preprocessed EEG shape does not match the "
                "50M original configuration. "
                f"Expected {expected_signal_shape} "
                f"({config.window_seconds:.1f}s at "
                f"{config.target_sample_rate:.1f} Hz), "
                f"got {signal.shape} ({duration_text}). "
                "Do not pad a 4-second Pipeline window to 10 seconds. "
                "Set replay.window_sec=10.0 and ensure that the "
                "Model50MPreprocessor receives a real 10-second window."
            )

        if signal.dtype != np.float32:
            signal = signal.astype(np.float32)

        if not np.isfinite(signal).all():
            raise ValueError(
                "Preprocessed EEG contains NaN or Inf."
            )

        channel_valid_mask = _validate_channel_valid_mask(
            channel_valid_mask=channel_valid_mask,
            num_channels=config.n_channels,
        )

        invalid_channels = channel_valid_mask < 0.5

        # 缺失通道应当由预处理器保持为全 0。
        if invalid_channels.any():
            invalid_signal = signal[invalid_channels]

            if not np.allclose(
                invalid_signal,
                0.0,
                atol=1e-7,
                rtol=0.0,
            ):
                raise ValueError(
                    "Missing channels contain non-zero values. "
                    "Model50MPreprocessor should keep missing "
                    "channels at zero."
                )

        patches = make_channel_time_patches(
            signal=signal,
            patch_len=config.patch_num_points,
            patch_stride=config.patch_stride_points,
        )

        num_channels, num_time_patches, patch_len = (
            patches.shape
        )

        if num_channels != config.n_channels:
            raise RuntimeError(
                f"Expected {config.n_channels} channels, "
                f"got {num_channels}."
            )

        if num_time_patches != config.num_time_patches:
            raise RuntimeError(
                "Unexpected number of time patches: "
                f"expected {config.num_time_patches}, "
                f"got {num_time_patches}."
            )

        if patch_len != config.patch_num_points:
            raise RuntimeError(
                "Unexpected Patch length: "
                f"expected {config.patch_num_points}, "
                f"got {patch_len}."
            )

        # Token 顺序：
        # channel 0: time 0, 1, ..., N-1
        # channel 1: time 0, 1, ..., N-1
        # ...
        token_inputs_np = patches.reshape(
            num_channels * num_time_patches,
            patch_len,
        ).astype(np.float32, copy=False)

        token_channel_indices_np = np.repeat(
            np.arange(num_channels, dtype=np.int64),
            num_time_patches,
        )

        token_time_indices_np = np.tile(
            np.arange(num_time_patches, dtype=np.int64),
            num_channels,
        )

        token_valid_mask_np = channel_valid_mask[
            token_channel_indices_np
        ].astype(np.float32, copy=False)

        result = Model50MTokenizedInput(
            token_inputs=torch.from_numpy(
                np.ascontiguousarray(token_inputs_np)
            ),
            token_channel_indices=torch.from_numpy(
                np.ascontiguousarray(
                    token_channel_indices_np
                )
            ),
            token_time_indices=torch.from_numpy(
                np.ascontiguousarray(
                    token_time_indices_np
                )
            ),
            token_valid_mask=torch.from_numpy(
                np.ascontiguousarray(token_valid_mask_np)
            ),
            channel_valid_mask=torch.from_numpy(
                np.ascontiguousarray(channel_valid_mask)
            ),
            num_channels=num_channels,
            num_time_patches=num_time_patches,
            patch_len=patch_len,
            patch_stride=config.patch_stride_points,
        )

        result.validate()
        return result


def stack_model50m_tokens(
    samples: Iterable[Model50MTokenizedInput],
    *,
    device: torch.device | str | None = None,
    non_blocking: bool = False,
) -> Model50MBatchedInput:
    """
    将多个已 Token 化的样本堆叠成 batch。

    所有样本必须具有相同的 Token 数量和 Patch 长度。
    """
    sample_list = list(samples)

    if not sample_list:
        raise ValueError(
            "At least one tokenized sample is required."
        )

    for sample in sample_list:
        sample.validate()

    reference_shape = sample_list[0].token_inputs.shape
    reference_channel_mask_shape = (
        sample_list[0].channel_valid_mask.shape
    )

    for index, sample in enumerate(
        sample_list[1:],
        start=1,
    ):
        if sample.token_inputs.shape != reference_shape:
            raise ValueError(
                "All tokenized samples must have the same "
                "token_inputs shape. "
                f"Sample 0: {tuple(reference_shape)}, "
                f"sample {index}: "
                f"{tuple(sample.token_inputs.shape)}."
            )

        if (
            sample.channel_valid_mask.shape
            != reference_channel_mask_shape
        ):
            raise ValueError(
                "All channel_valid_mask tensors must have "
                "the same shape."
            )

    batch = Model50MBatchedInput(
        token_inputs=torch.stack(
            [sample.token_inputs for sample in sample_list],
            dim=0,
        ),
        token_channel_indices=torch.stack(
            [
                sample.token_channel_indices
                for sample in sample_list
            ],
            dim=0,
        ),
        token_time_indices=torch.stack(
            [
                sample.token_time_indices
                for sample in sample_list
            ],
            dim=0,
        ),
        token_valid_mask=torch.stack(
            [
                sample.token_valid_mask
                for sample in sample_list
            ],
            dim=0,
        ),
        channel_valid_mask=torch.stack(
            [
                sample.channel_valid_mask
                for sample in sample_list
            ],
            dim=0,
        ),
    )

    if device is not None:
        batch = batch.to(
            device=device,
            non_blocking=non_blocking,
        )

    batch.validate()
    return batch


def tokenize_preprocessed_eeg(
    preprocessed: PreprocessResult,
    config: Model50MConfig,
) -> Model50MTokenizedInput:
    """函数式 Token 化入口。"""
    tokenizer = Model50MTokenizer(config)
    return tokenizer(preprocessed)
