from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# BCI Competition IV-2a 的 22 个 EEG 通道。
#
# 注意：
# 这是 CBRaMod 接入时的候选标准顺序。正式训练前必须通过
# CBRaMod 官方数据集实现、checkpoint 前向测试和预处理记录核验。
BCICIV2A_22_CHANNELS: tuple[str, ...] = (
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
)


# 常见别名只用于输入数据通道名匹配，最终喂给模型时仍使用
# BCICIV2A_22_CHANNELS 中定义的标准名称和顺序。
CHANNEL_ALIASES: dict[str, str] = {
    "FZ": "Fz",
    "FCZ": "FCz",
    "CZ": "Cz",
    "CPZ": "CPz",
    "PZ": "Pz",
    "POZ": "POz",
}


ReferenceMode = Literal["none", "average"]
NormalizationMode = Literal["none", "per_window_zscore"]
MissingChannelPolicy = Literal["error", "spherical_spline"]
HeadType = Literal["official_mlp", "linear"]


CBRAMOD_STRICT22_PROFILE = "strict22"
CBRAMOD_NEURACLE_LIVE19_SPLINE22_PROFILE = (
    "neuracle_live19_spline22"
)
CBRAMOD_NEURACLE_SIMULATED_MISSING_CHANNELS: tuple[str, ...] = (
    "CPz",
    "P1",
    "P2",
)


@dataclass(frozen=True, slots=True)
class CBraModDeploymentProfile:
    name: str
    training_channel_source_count: int
    observed_channel_names: tuple[str, ...]
    simulated_missing_channels: tuple[str, ...]
    missing_channel_policy: MissingChannelPolicy
    min_observed_channels: int
    spline_alpha: float
    channel_completion_source: str


def resolve_cbramod_deployment_profile(
    name: str,
    *,
    target_channel_names: tuple[str, ...] = BCICIV2A_22_CHANNELS,
) -> CBraModDeploymentProfile:
    profile_name = str(name).strip()
    target_channels = tuple(str(value) for value in target_channel_names)

    if len(target_channels) != len(set(target_channels)):
        raise ValueError("CBRaMod target channels contain duplicates.")

    if profile_name == CBRAMOD_STRICT22_PROFILE:
        return CBraModDeploymentProfile(
            name=profile_name,
            training_channel_source_count=len(target_channels),
            observed_channel_names=target_channels,
            simulated_missing_channels=(),
            missing_channel_policy="error",
            min_observed_channels=len(target_channels),
            spline_alpha=1e-5,
            channel_completion_source="shared_runtime_preprocessor",
        )

    if profile_name == CBRAMOD_NEURACLE_LIVE19_SPLINE22_PROFILE:
        missing = CBRAMOD_NEURACLE_SIMULATED_MISSING_CHANNELS
        absent_from_target = tuple(
            channel for channel in missing if channel not in target_channels
        )
        if absent_from_target:
            raise ValueError(
                "Neuracle Live simulated missing channels are absent from "
                f"the CBRaMod target montage: {absent_from_target}."
            )
        observed = tuple(
            channel for channel in target_channels if channel not in missing
        )
        if len(target_channels) != 22 or len(observed) != 19:
            raise ValueError(
                "Neuracle Live CBRaMod profile requires 22 target channels "
                "and exactly 19 observed channels."
            )
        return CBraModDeploymentProfile(
            name=profile_name,
            training_channel_source_count=22,
            observed_channel_names=observed,
            simulated_missing_channels=missing,
            missing_channel_policy="spherical_spline",
            min_observed_channels=19,
            spline_alpha=1e-5,
            channel_completion_source="shared_runtime_preprocessor",
        )

    raise ValueError(f"Unsupported CBRaMod deployment profile: {profile_name!r}.")

@dataclass(frozen=True, slots=True)
class CBraModConfig:
    """
    CBRaMod 在 NCC-OI-BCI Runtime 中的配置。

    主基线定义：
    - 输入：4 秒、22 通道、200 Hz；
    - 模型输入 shape：[B, 22, 4, 200]；
    - CBRaMod backbone 完全冻结；
    - 默认训练官方 quick-start 对应的 MLP 分类头；
    - 不应将 official_mlp 结果称为 linear probe。

    说明：
    `target_sample_rate`、`standard_channels`、滤波和归一化必须在
    第一次正式训练前，依据实际使用的官方 checkpoint 再次核验。
    """

    # ------------------------------------------------------------------
    # 模型文件
    # ------------------------------------------------------------------
    checkpoint_path: Path | str
    classifier_path: Path | str | None = None
    device: str = "cpu"

    # 可选：导出和训练时记录实际使用权重的 SHA-256。
    checkpoint_sha256: str | None = None

    # ------------------------------------------------------------------
    # EEG 输入契约
    # ------------------------------------------------------------------
    target_sample_rate: float = 200.0
    window_seconds: float = 4.0

    n_channels: int = 22
    standard_channels: tuple[str, ...] = BCICIV2A_22_CHANNELS

    # CBRaMod 输入为 [B, C, S, P]：
    # S = time_segments，P = points_per_patch。
    time_segments: int = 4
    points_per_patch: int = 200

    # Runtime 输入窗口必须是真实完整 4 秒，不能补零或静默截断。
    strict_window_duration: bool = True
    window_tolerance_seconds: float = 0.02

    # Pipeline 将原始单位转换到该单位后，再执行模型专属预处理。
    input_unit: str = "uV"

    # ------------------------------------------------------------------
    # 物理信号预处理
    # ------------------------------------------------------------------
    # 默认不额外滤波：在 CBRaMod 官方预处理规范核验前，避免复制
    # 50M 的滤波设置或引入未经记录的处理。
    filter_enabled: bool = False
    filter_low_hz: float = 0.1
    filter_high_hz: float = 75.0
    filter_order: int = 4

    reference_mode: ReferenceMode = "none"

    # 正式实验前根据官方 checkpoint 的预处理规范显式设定。
    # 可选值：
    # - none: 不做数值归一化；
    # - per_window_zscore: 每条 trial、每个通道沿时间维标准化。
    normalization: NormalizationMode = "none"
    zscore_eps: float = 1e-8

    # 通道匹配规则：
    # - 输入中不属于 standard_channels 的通道会直接丢弃；
    # - 映射到同一目标通道的多个输入通道会取平均；
    # - 缺失目标通道的处理由 missing_channel_policy 决定。
    #
    # error:
    #   保持原始严格 CBraMod 基线：只要缺少任一目标通道即报错。
    #
    # spherical_spline:
    #   使用已观测到的目标通道，通过通用 standard_1005 电极坐标进行
    #   球面样条插值，生成缺失目标通道；不会把全 0 通道送入 backbone。
    missing_channel_policy: MissingChannelPolicy = "error"

    # 仅在 missing_channel_policy="spherical_spline" 时生效。
    # 这是最低质量门槛，不是设备通道数；必须由训练/部署协议明确记录。
    min_observed_channels: int | None = None

    # 插值矩阵的 Tikhonov 正则项，固定记录到模型包，保证可复现。
    spline_alpha: float = 1e-5

    # ------------------------------------------------------------------
    # 分类头
    # ------------------------------------------------------------------
    num_classes: int = 4

    # CBRaMod 官方 quick-start 在 model.proj_out = nn.Identity() 后，
    # 使用每个位置 200 维输出构建分类头。
    backbone_output_dim: int = 200

    # 主协议固定为 official_mlp。
    #
    # official_mlp:
    #   Flatten(22 * 4 * 200)
    #   -> Linear(..., 800) -> ELU -> Dropout
    #   -> Linear(800, 200) -> ELU -> Dropout
    #   -> Linear(200, 4)
    #
    # linear:
    #   Flatten(22 * 4 * 200) -> Linear(..., 4)
    #
    # linear 仅可作为独立补充实验 cbramod-frozen-linear。
    head_type: HeadType = "official_mlp"
    head_hidden_dim_1: int = 800
    head_hidden_dim_2: int = 200
    head_dropout: float = 0.1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "checkpoint_path",
            Path(self.checkpoint_path),
        )

        required_observed = (
            self.n_channels
            if self.min_observed_channels is None
            else int(self.min_observed_channels)
        )
        object.__setattr__(
            self,
            "min_observed_channels",
            required_observed,
        )

        if not 1 <= required_observed <= self.n_channels:
            raise ValueError(
                "min_observed_channels must be in "
                f"[1, {self.n_channels}], got "
                f"{required_observed}."
            )

        if (
                self.missing_channel_policy == "error"
                and required_observed != self.n_channels
        ):
            raise ValueError(
                "missing_channel_policy='error' requires "
                "min_observed_channels == n_channels."
            )

        if self.spline_alpha <= 0:
            raise ValueError("spline_alpha must be positive.")

        if self.classifier_path is not None:
            object.__setattr__(
                self,
                "classifier_path",
                Path(self.classifier_path),
            )

        if self.target_sample_rate <= 0:
            raise ValueError(
                "target_sample_rate must be positive."
            )

        if self.window_seconds <= 0:
            raise ValueError(
                "window_seconds must be positive."
            )

        if self.n_channels <= 0:
            raise ValueError(
                "n_channels must be positive."
            )

        if self.n_channels != len(self.standard_channels):
            raise ValueError(
                f"n_channels={self.n_channels}, but "
                f"len(standard_channels)="
                f"{len(self.standard_channels)}."
            )
        if self.missing_channel_policy not in {
            "error",
            "spherical_spline",
        }:
            raise ValueError(
                "Unsupported missing_channel_policy: "
                f"{self.missing_channel_policy!r}."
            )

        required_observed_channels = (
            self.n_channels
            if self.min_observed_channels is None
            else int(self.min_observed_channels)
        )

        if not 1 <= required_observed_channels <= self.n_channels:
            raise ValueError(
                "min_observed_channels must be in "
                f"[1, {self.n_channels}], got "
                f"{required_observed_channels}."
            )

        if (
                self.missing_channel_policy == "error"
                and required_observed_channels != self.n_channels
        ):
            raise ValueError(
                "missing_channel_policy='error' requires "
                "min_observed_channels to be None or n_channels."
            )

        if self.spline_alpha <= 0:
            raise ValueError("spline_alpha must be positive.")

        normalized_names = tuple(
            channel.strip().upper()
            for channel in self.standard_channels
        )

        if any(not channel for channel in normalized_names):
            raise ValueError(
                "standard_channels contains an empty name."
            )

        if len(normalized_names) != len(set(normalized_names)):
            raise ValueError(
                "standard_channels contains duplicate names."
            )

        if self.time_segments <= 0:
            raise ValueError(
                "time_segments must be positive."
            )

        if self.points_per_patch <= 0:
            raise ValueError(
                "points_per_patch must be positive."
            )

        if self.num_samples != (
            self.time_segments * self.points_per_patch
        ):
            raise ValueError(
                "CBraMod input shape is inconsistent: "
                f"num_samples={self.num_samples}, but "
                f"time_segments × points_per_patch="
                f"{self.time_segments * self.points_per_patch}."
            )

        if self.filter_enabled:
            if self.filter_low_hz <= 0:
                raise ValueError(
                    "filter_low_hz must be greater than 0."
                )

            if self.filter_high_hz <= self.filter_low_hz:
                raise ValueError(
                    "filter_high_hz must be greater than "
                    "filter_low_hz."
                )

            if self.filter_high_hz >= self.target_nyquist_hz:
                raise ValueError(
                    "filter_high_hz must be lower than the "
                    "target Nyquist frequency."
                )

            if self.filter_order <= 0:
                raise ValueError(
                    "filter_order must be positive."
                )

        if self.reference_mode not in {"none", "average"}:
            raise ValueError(
                f"Unsupported reference_mode: "
                f"{self.reference_mode!r}."
            )

        if self.normalization not in {
            "none",
            "per_window_zscore",
        }:
            raise ValueError(
                f"Unsupported normalization: "
                f"{self.normalization!r}."
            )

        if self.zscore_eps <= 0:
            raise ValueError(
                "zscore_eps must be positive."
            )

        if self.missing_channel_policy not in {
            "error",
            "spherical_spline",
        }:
            raise ValueError(
                "Unsupported missing_channel_policy: "
                f"{self.missing_channel_policy!r}."
            )

        if not (
            2 <= self.min_observed_channels
            <= self.n_channels
        ):
            raise ValueError(
                "min_observed_channels must be in "
                f"[2, {self.n_channels}], got "
                f"{self.min_observed_channels}."
            )

        if self.spline_alpha <= 0:
            raise ValueError(
                "spline_alpha must be positive."
            )

        if self.head_type not in {"official_mlp", "linear"}:
            raise ValueError(
                f"Unsupported head_type: {self.head_type!r}."
            )

        if self.backbone_output_dim <= 0:
            raise ValueError(
                "backbone_output_dim must be positive."
            )

        if self.num_classes <= 1:
            raise ValueError(
                "num_classes must be greater than 1."
            )

        if self.head_type == "official_mlp":
            if self.head_hidden_dim_1 <= 0:
                raise ValueError(
                    "head_hidden_dim_1 must be positive."
                )

            if self.head_hidden_dim_2 <= 0:
                raise ValueError(
                    "head_hidden_dim_2 must be positive."
                )

            if not 0.0 <= self.head_dropout < 1.0:
                raise ValueError(
                    "head_dropout must be in [0, 1)."
                )

        if not self.input_unit.strip():
            raise ValueError(
                "input_unit cannot be empty."
            )

    @property
    def required_observed_channels(self) -> int:
        return (
            self.n_channels
            if self.min_observed_channels is None
            else int(self.min_observed_channels)
        )

    @property
    def num_samples(self) -> int:
        """4 秒 × 200 Hz = 800 点。"""
        return int(
            round(
                self.window_seconds
                * self.target_sample_rate
            )
        )

    @property
    def patch_seconds(self) -> float:
        """默认每个 time segment 为 1 秒。"""
        return self.window_seconds / self.time_segments

    @property
    def expected_unbatched_shape(self) -> tuple[int, int, int]:
        """模型预处理后、加入 batch 维前的 [C, S, P]。"""
        return (
            self.n_channels,
            self.time_segments,
            self.points_per_patch,
        )

    @property
    def expected_batched_shape(self) -> tuple[int | None, int, int, int]:
        """模型前向输入的 [B, C, S, P]。"""
        return (
            None,
            self.n_channels,
            self.time_segments,
            self.points_per_patch,
        )

    @property
    def num_feature_positions(self) -> int:
        """22 通道 × 4 个时间 segment。"""
        return self.n_channels * self.time_segments

    @property
    def classifier_input_dim(self) -> int:
        """官方分类头 Flatten 后的维度：22 × 4 × 200。"""
        return (
            self.num_feature_positions
            * self.backbone_output_dim
        )

    @property
    def target_nyquist_hz(self) -> float:
        return self.target_sample_rate / 2.0


def default_cbramod_config(
    checkpoint_path: Path | str,
    classifier_path: Path | str | None = None,
    device: str = "cpu",
) -> CBraModConfig:
    """创建 CBRaMod 冻结骨干主基线的默认配置。"""
    return CBraModConfig(
        checkpoint_path=checkpoint_path,
        classifier_path=classifier_path,
        device=device,
    )
