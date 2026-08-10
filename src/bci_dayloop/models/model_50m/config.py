from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# 与 50M 原仓库 channel_config.py 保持相同顺序。
STANDARD_64_CHANNELS: tuple[str, ...] = (
    "Fp1", "Fpz", "Fp2",
    "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
    "Iz",
    "F9", "F10",
)


# 50M 原仓库明确包含的通道别名。
CHANNEL_ALIASES: dict[str, str] = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
    "FP1": "Fp1",
    "FPZ": "Fpz",
    "FP2": "Fp2",
    "FZ": "Fz",
    "CZ": "Cz",
    "PZ": "Pz",
    "OZ": "Oz",
    "AFZ": "AFz",
    "FCZ": "FCz",
    "CPZ": "CPz",
    "POZ": "POz",
}


ReferenceMode = Literal["none", "average"]
AggregationMode = Literal["flatten", "mean"]


@dataclass(frozen=True, slots=True)
class Model50MConfig:
    """
    50M 模型部署侧配置。

    第一版保持原始配置：
    - 64 通道
    - 100 Hz
    - 10 秒窗口
    - 1 秒 Patch
    - 1 秒 Patch stride

    注意：
    checkpoint 的真实模型配置仍应以权重随附信息为准。
    """

    # ------------------------------------------------------------------
    # 模型文件
    # ------------------------------------------------------------------
    checkpoint_path: Path | str
    classifier_path: Path | str | None = None
    device: str = "cpu"

    # ------------------------------------------------------------------
    # EEG 输入配置
    # ------------------------------------------------------------------
    target_sample_rate: float = 100.0
    window_seconds: float = 10.0

    patch_seconds: float = 1.0
    patch_stride_seconds: float = 1.0

    n_channels: int = 64
    standard_channels: tuple[str, ...] = STANDARD_64_CHANNELS

    # 要求输入窗口就是约 10 秒，避免把 4 秒数据静默补成 10 秒。
    strict_window_duration: bool = True
    window_tolerance_seconds: float = 0.02

    # ------------------------------------------------------------------
    # 物理信号预处理
    # ------------------------------------------------------------------
    filter_enabled: bool = True
    filter_low_hz: float = 0.1
    filter_high_hz: float = 75.0
    filter_order: int = 4

    # 大规模数据构建脚本中未执行平均参考，因此第一版默认 none。
    # 确认实际 checkpoint 使用平均参考后，可改为 "average"。
    reference_mode: ReferenceMode = "none"

    # 有效通道沿时间维 Z-score。
    zscore_enabled: bool = True
    zscore_eps: float = 1e-8

    # 缺失通道补零。
    missing_channel_fill_value: float = 0.0

    # ------------------------------------------------------------------
    # Backbone 结构
    # ------------------------------------------------------------------
    d_model: int = 512
    n_heads: int = 8
    depth: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    # 预训练 checkpoint 使用 10 个时间位置。
    # 4 秒下游输入只使用位置 0、1、2、3，但模型结构仍保留 10 个。
    model_n_time_patches: int = 10

    # 当前 finetune 仓库的默认 probe_layer_ratio=0.75，
    # depth=12 时对应 0-based index 8。
    output_layer_idx: int = 8

    # 第一版与当前 finetune 代码保持一致，使用 flatten。
    aggregation: AggregationMode = "flatten"
    num_classes: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))

        if self.model_n_time_patches <= 0:
            raise ValueError(
                "model_n_time_patches must be positive."
            )

        if self.model_n_time_patches < self.num_time_patches:
            raise ValueError(
                "model_n_time_patches cannot be smaller than the number "
                "of input time patches: "
                f"{self.model_n_time_patches} < {self.num_time_patches}."
            )

        if self.classifier_path is not None:
            object.__setattr__(
                self,
                "classifier_path",
                Path(self.classifier_path),
            )

        if self.n_channels != len(self.standard_channels):
            raise ValueError(
                f"n_channels={self.n_channels}, but "
                f"len(standard_channels)={len(self.standard_channels)}."
            )

        if self.target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive.")

        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")

        if self.patch_seconds <= 0:
            raise ValueError("patch_seconds must be positive.")

        if self.patch_stride_seconds <= 0:
            raise ValueError("patch_stride_seconds must be positive.")

        if self.patch_seconds > self.window_seconds:
            raise ValueError(
                "patch_seconds cannot be greater than window_seconds."
            )

        if self.filter_enabled:
            if self.filter_low_hz <= 0:
                raise ValueError("filter_low_hz must be greater than 0.")

            if self.filter_high_hz <= self.filter_low_hz:
                raise ValueError(
                    "filter_high_hz must be greater than filter_low_hz."
                )

            if self.filter_order <= 0:
                raise ValueError("filter_order must be positive.")

        if self.reference_mode not in {"none", "average"}:
            raise ValueError(
                f"Unsupported reference_mode: {self.reference_mode}"
            )

        if self.aggregation not in {"flatten", "mean"}:
            raise ValueError(
                f"Unsupported aggregation mode: {self.aggregation}"
            )

        if not 0 <= self.output_layer_idx < self.depth:
            raise ValueError(
                f"output_layer_idx={self.output_layer_idx} must be in "
                f"[0, {self.depth - 1}]."
            )

    @property
    def target_num_points(self) -> int:
        """10 秒 × 100 Hz = 1000 点。"""
        return int(round(self.window_seconds * self.target_sample_rate))

    @property
    def patch_num_points(self) -> int:
        """1 秒 × 100 Hz = 100 点。"""
        return int(round(self.patch_seconds * self.target_sample_rate))

    @property
    def patch_stride_points(self) -> int:
        return int(
            round(self.patch_stride_seconds * self.target_sample_rate)
        )

    @property
    def num_time_patches(self) -> int:
        """
        默认：
        (1000 - 100) / 100 + 1 = 10 个时间 Patch。
        """
        return (
            self.target_num_points - self.patch_num_points
        ) // self.patch_stride_points + 1

    @property
    def num_tokens(self) -> int:
        """默认 64 × 10 = 640 Token。"""
        return self.n_channels * self.num_time_patches

    @property
    def classifier_input_dim(self) -> int:
        if self.aggregation == "mean":
            return self.d_model

        return self.num_tokens * self.d_model

    @property
    def target_nyquist_hz(self) -> float:
        return self.target_sample_rate / 2.0


def default_50m_config(
    checkpoint_path: Path | str,
    classifier_path: Path | str | None = None,
    device: str = "cpu",
) -> Model50MConfig:
    """创建第一版 50M Adapter 默认配置。"""
    return Model50MConfig(
        checkpoint_path=checkpoint_path,
        classifier_path=classifier_path,
        device=device,
    )