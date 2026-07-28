from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

try:
    # 作为 package 使用
    from .config import (
        CHANNEL_ALIASES,
        STANDARD_64_CHANNELS,
        Model50MConfig,
    )
except ImportError:
    # 直接运行当前文件时使用
    from config import (
        CHANNEL_ALIASES,
        STANDARD_64_CHANNELS,
        Model50MConfig,
    )


_STANDARD_CHANNEL_BY_UPPER = {
    name.upper(): name for name in STANDARD_64_CHANNELS
}


@dataclass(frozen=True, slots=True)
class PreprocessResult:
    """
    50M 预处理结果。

    signal:
        [64, 1000]，float32。
        已完成通道映射、滤波、重采样和 Z-score。

    channel_valid_mask:
        [64]，float32。
        真实存在通道为 1，缺失通道为 0。
    """

    signal: np.ndarray
    channel_valid_mask: np.ndarray

    original_channel_names: tuple[str, ...]
    canonical_channel_names: tuple[str, ...]
    unknown_channel_names: tuple[str, ...]

    original_sample_rate: float
    target_sample_rate: float

    mapped_channel_count: int
    missing_channel_count: int
    duplicate_channel_count: int

    padded_points: int
    cropped_points: int

    notes: tuple[str, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return self.signal.shape  # type: ignore[return-value]


def canonicalize_channel_name(name: str | None) -> str:
    """
    将常见 EEG 通道名称规范化。

    示例：
        EEG Fp1-Ref -> Fp1
        FP1         -> Fp1
        Cz.         -> Cz
        T3          -> T7
    """
    if name is None:
        return ""

    value = str(name).strip()

    # 去除 EEG 前缀，兼容 "EEG Fp1"、"EEG-Fp1"。
    value = re.sub(
        r"^EEG[\s_\-]*",
        "",
        value,
        flags=re.IGNORECASE,
    )

    # 去除常见参考后缀。
    value = re.sub(
        r"[-_ ]?(REF|REFERENCE|AVG|A1|A2|M1|M2)$",
        "",
        value,
        flags=re.IGNORECASE,
    )

    # 只保留字母和数字。
    value = re.sub(r"[^A-Za-z0-9]", "", value)

    if not value:
        return ""

    upper_name = value.upper()

    # 先处理 T3/T4/T5/T6 及 z 通道别名。
    if upper_name in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[upper_name]

    # 将标准通道名称恢复到标准大小写。
    if upper_name in _STANDARD_CHANNEL_BY_UPPER:
        return _STANDARD_CHANNEL_BY_UPPER[upper_name]

    # 未识别通道保留清洗后的名称，后续会被标记为 unknown。
    return value


def _validate_input(
    signal: np.ndarray,
    channel_names: Sequence[str],
    original_sample_rate: float,
    input_unit: str,
) -> np.ndarray:
    signal = np.asarray(signal)

    if signal.ndim != 2:
        raise ValueError(
            f"EEG signal must have shape [C, T], got {signal.shape}."
        )

    if signal.shape[0] != len(channel_names):
        raise ValueError(
            "Channel count mismatch: "
            f"signal has {signal.shape[0]} channels, "
            f"but channel_names has {len(channel_names)} entries."
        )

    if signal.shape[1] <= 1:
        raise ValueError(
            f"EEG signal contains too few time points: {signal.shape[1]}."
        )

    if original_sample_rate <= 0:
        raise ValueError(
            f"Invalid original_sample_rate: {original_sample_rate}."
        )

    if not np.issubdtype(signal.dtype, np.number):
        raise TypeError(
            f"EEG signal must be numeric, got dtype={signal.dtype}."
        )

    if not np.isfinite(signal).all():
        raise ValueError("EEG signal contains NaN or Inf.")

    normalized_unit = (
        str(input_unit)
        .strip()
        .replace("μ", "u")
        .replace("µ", "u")
        .lower()
    )

    if normalized_unit not in {"v", "mv", "uv"}:
        raise ValueError(
            f"Unsupported EEG unit: {input_unit!r}. "
            "Supported values are V, mV, uV or µV."
        )

    return signal.astype(np.float32, copy=False)


def _convert_to_microvolts(
    signal: np.ndarray,
    input_unit: str,
) -> np.ndarray:
    """
    将物理信号统一转换为 µV。

    转换后还会进行 Z-score，因此模型最终接收的是无量纲值。
    """
    unit = (
        str(input_unit)
        .strip()
        .replace("μ", "u")
        .replace("µ", "u")
        .lower()
    )

    if unit == "v":
        scale = 1e6
    elif unit == "mv":
        scale = 1e3
    elif unit == "uv":
        scale = 1.0
    else:
        raise ValueError(f"Unsupported EEG unit: {input_unit!r}.")

    return (signal.astype(np.float32) * scale).astype(np.float32)


def _align_to_standard_channels(
    signal: np.ndarray,
    channel_names: Sequence[str],
    fill_value: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
    int,
]:
    """
    将任意 EEG 通道映射到固定的 STANDARD_64_CHANNELS。

    如果多个输入通道规范化后映射到同一标准通道，则取平均值。

    Returns:
        aligned_signal:
            [64, T]
        channel_valid_mask:
            [64]
        canonical_names:
            规范化后的输入通道名称
        unknown_names:
            无法映射的原始输入通道名称
        duplicate_channel_count:
            被合并的重复通道数量
    """
    canonical_names = tuple(
        canonicalize_channel_name(name)
        for name in channel_names
    )

    standard_index = {
        name: index
        for index, name in enumerate(STANDARD_64_CHANNELS)
    }

    n_standard_channels = len(STANDARD_64_CHANNELS)
    n_times = signal.shape[1]

    # float64 累加，最后再转换成 float32。
    channel_sum = np.zeros(
        (n_standard_channels, n_times),
        dtype=np.float64,
    )
    channel_count = np.zeros(
        n_standard_channels,
        dtype=np.int64,
    )

    unknown_names: list[str] = []

    for source_index, canonical_name in enumerate(canonical_names):
        if not canonical_name:
            unknown_names.append(str(channel_names[source_index]))
            continue

        destination_index = standard_index.get(canonical_name)

        if destination_index is None:
            unknown_names.append(str(channel_names[source_index]))
            continue

        channel_sum[destination_index] += signal[source_index]
        channel_count[destination_index] += 1

    valid = channel_count > 0

    if not valid.any():
        raise ValueError(
            "None of the input EEG channels could be mapped to "
            "STANDARD_64_CHANNELS."
        )

    aligned_signal = np.full(
        (n_standard_channels, n_times),
        fill_value=fill_value,
        dtype=np.float32,
    )

    aligned_signal[valid] = (
        channel_sum[valid]
        / channel_count[valid, None]
    ).astype(np.float32)

    channel_valid_mask = valid.astype(np.float32)

    duplicate_channel_count = int(
        np.maximum(channel_count - 1, 0).sum()
    )

    return (
        aligned_signal,
        channel_valid_mask,
        canonical_names,
        tuple(unknown_names),
        duplicate_channel_count,
    )


def _apply_average_reference(
    signal: np.ndarray,
    channel_valid_mask: np.ndarray,
) -> np.ndarray:
    """仅使用真实存在的通道计算平均参考。"""
    valid = channel_valid_mask.astype(bool)

    if not valid.any():
        return signal.astype(np.float32)

    output = signal.copy().astype(np.float32)

    average_reference = output[valid].mean(
        axis=0,
        keepdims=True,
    )

    output[valid] -= average_reference

    # 缺失通道继续保持 0。
    output[~valid] = 0.0

    return output.astype(np.float32)


def _bandpass_filter(
    signal: np.ndarray,
    channel_valid_mask: np.ndarray,
    sample_rate: float,
    low_hz: float,
    high_hz: float,
    order: int,
) -> np.ndarray:
    """
    对真实存在的通道执行零相位 Butterworth 带通滤波。

    注意：
    当前按照要求在重采样前执行 0.1–75 Hz。
    """
    nyquist = sample_rate / 2.0

    if high_hz >= nyquist:
        raise ValueError(
            f"filter_high_hz={high_hz} Hz must be below the "
            f"original Nyquist frequency {nyquist:.3f} Hz. "
            f"Current original sample rate is {sample_rate} Hz."
        )

    if low_hz <= 0 or low_hz >= high_hz:
        raise ValueError(
            f"Invalid bandpass range: {low_hz}–{high_hz} Hz."
        )

    valid = channel_valid_mask.astype(bool)

    output = np.zeros_like(signal, dtype=np.float32)

    if not valid.any():
        return output

    sos = butter(
        N=order,
        Wn=(low_hz / nyquist, high_hz / nyquist),
        btype="bandpass",
        output="sos",
    )

    try:
        filtered = sosfiltfilt(
            sos,
            signal[valid].astype(np.float64),
            axis=-1,
        )
    except ValueError as error:
        raise ValueError(
            "Bandpass filtering failed. The EEG window may be too short "
            "for the configured filter. For production streaming, filter "
            "continuous EEG before window slicing."
        ) from error

    output[valid] = filtered.astype(np.float32)
    return output


def _resample_signal(
    signal: np.ndarray,
    original_sample_rate: float,
    target_sample_rate: float,
) -> np.ndarray:
    """使用 polyphase 方法沿时间维重采样。"""
    if abs(original_sample_rate - target_sample_rate) < 1e-6:
        return signal.astype(np.float32)

    ratio = Fraction(
        target_sample_rate / original_sample_rate
    ).limit_denominator(10_000)

    output = resample_poly(
        signal,
        up=ratio.numerator,
        down=ratio.denominator,
        axis=1,
    )

    return output.astype(np.float32)


def _crop_or_pad(
    signal: np.ndarray,
    target_num_points: int,
    fill_value: float,
) -> tuple[np.ndarray, int, int]:
    """
    调整到固定长度。

    Returns:
        output
        padded_points
        cropped_points
    """
    n_channels, n_times = signal.shape

    if n_times == target_num_points:
        return signal.astype(np.float32), 0, 0

    if n_times > target_num_points:
        cropped_points = n_times - target_num_points
        return (
            signal[:, :target_num_points].astype(np.float32),
            0,
            cropped_points,
        )

    padded_points = target_num_points - n_times

    output = np.full(
        (n_channels, target_num_points),
        fill_value=fill_value,
        dtype=np.float32,
    )

    output[:, :n_times] = signal

    return output, padded_points, 0


def _standardize_valid_channels(
    signal: np.ndarray,
    channel_valid_mask: np.ndarray,
    eps: float,
) -> np.ndarray:
    """
    对每个有效通道沿时间维执行 Z-score。

    缺失通道保持全 0。
    """
    valid = channel_valid_mask.astype(bool)

    output = np.zeros_like(signal, dtype=np.float32)

    if not valid.any():
        return output

    valid_signal = signal[valid].astype(np.float64)

    channel_mean = valid_signal.mean(
        axis=1,
        keepdims=True,
    )

    channel_std = valid_signal.std(
        axis=1,
        keepdims=True,
    )

    output[valid] = (
        (valid_signal - channel_mean)
        / (channel_std + eps)
    ).astype(np.float32)

    return output


class Model50MPreprocessor:
    """将 Pipeline 原始 EEG 窗口转换为 50M 标准整段输入。"""

    def __init__(self, config: Model50MConfig):
        self.config = config

    def __call__(
        self,
        signal: np.ndarray,
        channel_names: Sequence[str],
        original_sample_rate: float,
        input_unit: str,
    ) -> PreprocessResult:
        config = self.config
        notes: list[str] = []

        signal = _validate_input(
            signal=signal,
            channel_names=channel_names,
            original_sample_rate=original_sample_rate,
            input_unit=input_unit,
        )

        actual_duration_seconds = (
            signal.shape[1] / float(original_sample_rate)
        )

        if config.strict_window_duration:
            duration_error = abs(
                actual_duration_seconds - config.window_seconds
            )

            if duration_error > config.window_tolerance_seconds:
                raise ValueError(
                    "Input EEG window duration does not match the "
                    "50M original configuration: "
                    f"expected {config.window_seconds:.3f}s, "
                    f"got {actual_duration_seconds:.3f}s. "
                    "For stage 0.5, change the Pipeline window to 10 seconds "
                    "instead of padding a 4-second window."
                )

        # 1. 物理单位统一为 µV。
        signal_uv = _convert_to_microvolts(
            signal=signal,
            input_unit=input_unit,
        )

        # 2. 映射到固定 64 通道。
        (
            signal_64,
            channel_valid_mask,
            canonical_names,
            unknown_names,
            duplicate_channel_count,
        ) = _align_to_standard_channels(
            signal=signal_uv,
            channel_names=channel_names,
            fill_value=config.missing_channel_fill_value,
        )

        if unknown_names:
            notes.append(
                f"{len(unknown_names)} input channel(s) were ignored "
                "because they are not in STANDARD_64_CHANNELS."
            )

        if duplicate_channel_count > 0:
            notes.append(
                f"{duplicate_channel_count} duplicate mapped channel(s) "
                "were averaged."
            )

        # 3. 可选平均参考。
        if config.reference_mode == "average":
            signal_64 = _apply_average_reference(
                signal=signal_64,
                channel_valid_mask=channel_valid_mask,
            )

        # 4. 在原始采样率下执行带通滤波。
        if config.filter_enabled:
            signal_64 = _bandpass_filter(
                signal=signal_64,
                channel_valid_mask=channel_valid_mask,
                sample_rate=original_sample_rate,
                low_hz=config.filter_low_hz,
                high_hz=config.filter_high_hz,
                order=config.filter_order,
            )

            if config.filter_high_hz >= config.target_nyquist_hz:
                notes.append(
                    f"The configured high cutoff is "
                    f"{config.filter_high_hz} Hz, but the target sample "
                    f"rate is {config.target_sample_rate} Hz. Frequencies "
                    f"above {config.target_nyquist_hz} Hz cannot remain "
                    "after resampling."
                )

        # 5. 直接从原始采样率重采样至 100 Hz。
        signal_64 = _resample_signal(
            signal=signal_64,
            original_sample_rate=original_sample_rate,
            target_sample_rate=config.target_sample_rate,
        )

        # 6. 固定为 1000 点。
        signal_64, padded_points, cropped_points = _crop_or_pad(
            signal=signal_64,
            target_num_points=config.target_num_points,
            fill_value=config.missing_channel_fill_value,
        )

        if padded_points > 0:
            notes.append(
                f"The resampled signal was padded by "
                f"{padded_points} point(s)."
            )

        if cropped_points > 0:
            notes.append(
                f"The resampled signal was cropped by "
                f"{cropped_points} point(s)."
            )

        # 7. 每个有效通道沿时间维做 Z-score。
        if config.zscore_enabled:
            signal_64 = _standardize_valid_channels(
                signal=signal_64,
                channel_valid_mask=channel_valid_mask,
                eps=config.zscore_eps,
            )
        else:
            # 即使关闭标准化，也必须保证缺失通道仍为 0。
            invalid = ~channel_valid_mask.astype(bool)
            signal_64[invalid] = config.missing_channel_fill_value

        signal_64 = signal_64.astype(np.float32, copy=False)
        channel_valid_mask = channel_valid_mask.astype(
            np.float32,
            copy=False,
        )

        expected_shape = (
            config.n_channels,
            config.target_num_points,
        )

        if signal_64.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected preprocessed EEG shape: "
                f"expected {expected_shape}, got {signal_64.shape}."
            )

        if channel_valid_mask.shape != (config.n_channels,):
            raise RuntimeError(
                "Unexpected channel_valid_mask shape: "
                f"{channel_valid_mask.shape}."
            )

        if not np.isfinite(signal_64).all():
            raise RuntimeError(
                "Preprocessed EEG contains NaN or Inf."
            )

        mapped_channel_count = int(channel_valid_mask.sum())

        return PreprocessResult(
            signal=signal_64,
            channel_valid_mask=channel_valid_mask,
            original_channel_names=tuple(str(x) for x in channel_names),
            canonical_channel_names=canonical_names,
            unknown_channel_names=unknown_names,
            original_sample_rate=float(original_sample_rate),
            target_sample_rate=float(config.target_sample_rate),
            mapped_channel_count=mapped_channel_count,
            missing_channel_count=(
                config.n_channels - mapped_channel_count
            ),
            duplicate_channel_count=duplicate_channel_count,
            padded_points=padded_points,
            cropped_points=cropped_points,
            notes=tuple(notes),
        )


def preprocess_eeg_window(
    signal: np.ndarray,
    channel_names: Sequence[str],
    original_sample_rate: float,
    input_unit: str,
    config: Model50MConfig,
) -> PreprocessResult:
    """函数式调用入口。"""
    preprocessor = Model50MPreprocessor(config)

    return preprocessor(
        signal=signal,
        channel_names=channel_names,
        original_sample_rate=original_sample_rate,
        input_unit=input_unit,
    )