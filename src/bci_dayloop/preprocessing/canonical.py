from __future__ import annotations

import math
import re

import numpy as np

from bci_dayloop.runtime.types import (
    CanonicalEEGWindow,
    RawEEGWindow,
)


_CHANNEL_PREFIX_CASE = {
    "FP": "Fp",
    "AF": "AF",
    "F": "F",
    "FT": "FT",
    "FC": "FC",
    "T": "T",
    "C": "C",
    "TP": "TP",
    "CP": "CP",
    "P": "P",
    "PO": "PO",
    "O": "O",
}

_CHANNEL_ALIASES = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}


def normalize_eeg_unit(unit: str) -> str:
    """将 EEG 单位转换为内部统一写法。"""

    value = (
        str(unit)
        .strip()
        .replace("μ", "u")
        .replace("µ", "u")
        .lower()
    )

    aliases = {
        "v": "V",
        "volt": "V",
        "volts": "V",
        "mv": "mV",
        "millivolt": "mV",
        "millivolts": "mV",
        "uv": "uV",
        "microvolt": "uV",
        "microvolts": "uV",
    }

    normalized = aliases.get(value)

    if normalized is None:
        raise ValueError(
            f"Unsupported EEG unit: {unit!r}. "
            "Supported units are V, mV and uV/µV."
        )

    return normalized


def normalize_channel_name(name: str) -> str:
    """
    清理常见 EEG 通道名称。

    示例：
        EEG Fp1-Ref -> Fp1
        FPZ         -> Fpz
        CZ.         -> Cz
        T3          -> T7
    """

    value = str(name).strip()

    if not value:
        raise ValueError("EEG channel name cannot be empty.")

    value = re.sub(
        r"^EEG[\s_\-]*",
        "",
        value,
        flags=re.IGNORECASE,
    )

    value = re.sub(
        r"[-_ ]?(REF|REFERENCE|AVG)$",
        "",
        value,
        flags=re.IGNORECASE,
    )

    value = re.sub(
        r"[^A-Za-z0-9]",
        "",
        value,
    )

    if not value:
        raise ValueError(
            f"Invalid EEG channel name: {name!r}."
        )

    upper = value.upper()

    if upper in _CHANNEL_ALIASES:
        return _CHANNEL_ALIASES[upper]

    match = re.fullmatch(
        r"([A-Z]+)(Z|\d+)",
        upper,
    )

    if match is None:
        # 无法安全标准化时保留清理后的名字，
        # 由模型专属 Transform 判断能否映射。
        return value

    prefix, suffix = match.groups()
    canonical_prefix = _CHANNEL_PREFIX_CASE.get(prefix)

    if canonical_prefix is None:
        return value

    canonical_suffix = (
        "z"
        if suffix == "Z"
        else suffix
    )

    return f"{canonical_prefix}{canonical_suffix}"


class SignalCanonicalizer:
    """
    将来自 HDF5、Replay 或实时设备的窗口转换为统一 CT 格式。

    这里只做通用、模型无关处理：
    - 数据维度检查
    - TC -> CT
    - 通道名称基础规范化
    - 物理单位统一
    - dtype 和有限值检查

    不负责：
    - 滤波
    - 重采样
    - 模型通道补齐
    - 模型通道重排
    - Z-score
    - Patch/token 构造
    """

    def __init__(
        self,
        target_unit: str = "uV",
    ) -> None:
        self.target_unit = normalize_eeg_unit(
            target_unit
        )

    def transform(
        self,
        window: RawEEGWindow,
    ) -> CanonicalEEGWindow:
        data = np.asarray(window.data)

        if data.ndim != 2:
            raise ValueError(
                "EEG data must be two-dimensional, "
                f"got shape={data.shape}."
            )

        if not np.issubdtype(data.dtype, np.number):
            raise TypeError(
                "EEG data must be numeric, "
                f"got dtype={data.dtype}."
            )

        if (
            not math.isfinite(float(window.sample_rate))
            or float(window.sample_rate) <= 0
        ):
            raise ValueError(
                "sample_rate must be finite and positive, "
                f"got {window.sample_rate}."
            )

        history: list[str] = []

        if window.layout == "TC":
            data = data.T
            history.append("transpose:TC->CT")
        elif window.layout == "CT":
            history.append("layout:CT")
        else:
            raise ValueError(
                f"Unsupported EEG layout: "
                f"{window.layout!r}."
            )

        if data.shape[0] == 0:
            raise ValueError(
                "EEG window contains no channels."
            )

        if data.shape[1] <= 1:
            raise ValueError(
                "EEG window contains too few time points: "
                f"{data.shape[1]}."
            )

        if data.shape[0] != len(window.channel_names):
            raise ValueError(
                "Channel dimension does not match "
                "channel_names: "
                f"data_channels={data.shape[0]}, "
                f"channel_names={len(window.channel_names)}."
            )

        channel_names = [
            normalize_channel_name(name)
            for name in window.channel_names
        ]
        history.append("normalize_channel_names")

        if not np.isfinite(data).all():
            raise ValueError(
                "EEG window contains NaN or Inf values."
            )

        source_unit = normalize_eeg_unit(window.unit)

        data = self._convert_unit(
            data=data,
            source_unit=source_unit,
            target_unit=self.target_unit,
        )

        history.append(
            f"convert_unit:{source_unit}"
            f"->{self.target_unit}"
        )

        data = np.asarray(
            data,
            dtype=np.float32,
        )

        history.append("cast:float32")

        return CanonicalEEGWindow(
            data=data,
            channel_names=channel_names,
            sample_rate=float(window.sample_rate),
            unit=self.target_unit,
            start_time_sec=window.start_time_sec,
            trial_id=window.trial_id,
            window_id=window.window_id,
            label=window.label,
            metadata=dict(window.metadata or {}),
            processing_history=history,
        )

    @staticmethod
    def _convert_unit(
        data: np.ndarray,
        source_unit: str,
        target_unit: str,
    ) -> np.ndarray:
        scale_to_volts = {
            "V": 1.0,
            "mV": 1e-3,
            "uV": 1e-6,
        }

        source_scale = scale_to_volts[source_unit]
        target_scale = scale_to_volts[target_unit]

        scale = source_scale / target_scale

        return (
            data.astype(np.float64, copy=False)
            * scale
        )