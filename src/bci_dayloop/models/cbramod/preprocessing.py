from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt

from bci_dayloop.preprocessing.base import ModelInputTransform
from bci_dayloop.preprocessing.canonical import (
    normalize_channel_name,
    normalize_eeg_unit,
)
from bci_dayloop.runtime.types import (
    CanonicalEEGWindow,
    InputContract,
    PreparedModelInput,
)

from .config import (
    CHANNEL_ALIASES,
    CBraModConfig,
)


@dataclass(frozen=True, slots=True)
class CBraModChannelAlignment:
    """按 CBraMod 目标通道顺序对齐后的中间结果。"""

    signal: np.ndarray
    channel_valid_mask: np.ndarray
    canonical_channel_names: tuple[str, ...]
    unknown_channel_names: tuple[str, ...]
    duplicate_channel_count: int


@dataclass(frozen=True, slots=True)
class CBraModPreprocessingDiagnostics:
    source_channel_names: tuple[str, ...]
    target_channel_names: tuple[str, ...]
    unknown_channel_names: tuple[str, ...]
    observed_channel_names: tuple[str, ...]
    missing_channel_names: tuple[str, ...]
    channel_valid_mask: tuple[float, ...]
    duplicate_channel_count: int
    completion_policy: str
    completion_matrix_sha256: str | None

    source_sample_rate_hz: float
    target_sample_rate_hz: float
    source_num_samples: int
    target_num_samples: int
    filter_applied: bool
    reference_mode: str
    normalization: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source_channel_names": list(
                self.source_channel_names
            ),
            "target_channel_names": list(
                self.target_channel_names
            ),
            "unknown_channel_names": list(
                self.unknown_channel_names
            ),
            "observed_channel_names": list(
                self.observed_channel_names
            ),
            "missing_channel_names": list(
                self.missing_channel_names
            ),
            "channel_valid_mask": list(
                self.channel_valid_mask
            ),
            "observed_channel_count": len(
                self.observed_channel_names
            ),
            "missing_channel_count": len(
                self.missing_channel_names
            ),
            "duplicate_channel_count": (
                self.duplicate_channel_count
            ),
            "completion_policy": self.completion_policy,
            "completion_matrix_sha256": (
                self.completion_matrix_sha256
            ),
            "source_sample_rate_hz": (
                self.source_sample_rate_hz
            ),
            "target_sample_rate_hz": (
                self.target_sample_rate_hz
            ),
            "source_num_samples": self.source_num_samples,
            "target_num_samples": self.target_num_samples,
            "filter_applied": self.filter_applied,
            "reference_mode": self.reference_mode,
            "normalization": self.normalization,
        }


class CBraModPipelinePreprocessor(ModelInputTransform):
    """
    将统一 CanonicalEEGWindow 转为 CBRaMod 输入。

    输入：
        CanonicalEEGWindow.data，shape [C, T]。

    输出：
        PreparedModelInput.model_input["signal"]，
        shape [1, 22, 4, 200]，dtype torch.float32。

    执行顺序：
        1. 严格验证 4 秒窗口；
        2. 通道名规范化、检查、重排为 CBRaMod 的 22 通道顺序；
        3. 可选带通滤波；
        4. 可选平均参考；
        5. 重采样至 200 Hz；
        6. 可选逐 trial、逐通道 Z-score；
        7. [22, 800] -> [22, 4, 200]；
        8. 增加 batch 维度。

    不允许：
        - 缺失通道补零；
        - 重复通道取平均；
        - 跨 trial / session 拼接；
        - 对短窗口补零；
        - 对长窗口静默截断。
    """

    def __init__(
        self,
        config: CBraModConfig,
    ) -> None:
        self.config = config

        self._target_channels = tuple(
            self._canonical_channel_name(name)
            for name in self.config.standard_channels
        )

        if len(self._target_channels) != self.config.n_channels:
            raise RuntimeError(
                "CBraMod target channel count is inconsistent "
                "with config.n_channels."
            )

        if len(set(self._target_channels)) != len(
            self._target_channels
        ):
            raise ValueError(
                "CBraMod target channel names contain duplicates."
            )

        self._target_channel_indices = {
            channel_name: index
            for index, channel_name in enumerate(
                self._target_channels
            )
        }

        # key 是“按目标通道顺序排列的已观测通道名”。
        # 同一种设备布局只计算一次插值矩阵，后续窗口直接复用。
        self._completion_matrix_cache: dict[
            tuple[str, ...],
            tuple[np.ndarray, str],
        ] = {}

        self._target_position_cache: (
            dict[str, np.ndarray] | None
        ) = None

        self.last_diagnostics: (
            CBraModPreprocessingDiagnostics | None
        ) = None

    @property
    def input_contract(self) -> InputContract:
        return InputContract(
            channel_names=self.config.standard_channels,
            sample_rate=self.config.target_sample_rate,
            window_sec=self.config.window_seconds,
            num_samples=self.config.num_samples,
            input_unit=self.config.input_unit,
            tensor_layout="BCTP",
            strict_window_duration=(
                self.config.strict_window_duration
            ),
            model_input_keys=("signal",),
        )

    @staticmethod
    def _canonical_channel_name(
        name: str,
    ) -> str:
        normalized = normalize_channel_name(name)

        alias = CHANNEL_ALIASES.get(
            normalized.upper()
        )

        if alias is not None:
            normalized = alias

        return normalized.upper()

    def _validate_window(
        self,
        window: CanonicalEEGWindow,
    ) -> np.ndarray:
        if not isinstance(window, CanonicalEEGWindow):
            raise TypeError(
                "CBraModPipelinePreprocessor expects "
                "CanonicalEEGWindow, got "
                f"{type(window).__name__}."
            )

        signal = np.asarray(window.data)

        if signal.ndim != 2:
            raise ValueError(
                "Canonical EEG data must have shape [C, T], got "
                f"{signal.shape}."
            )

        if signal.shape[0] != len(window.channel_names):
            raise ValueError(
                "Channel count does not match channel_names: "
                f"data_channels={signal.shape[0]}, "
                f"channel_names={len(window.channel_names)}."
            )

        if signal.shape[1] <= 1:
            raise ValueError(
                "CBraMod input contains too few time points: "
                f"{signal.shape[1]}."
            )

        if not np.issubdtype(signal.dtype, np.number):
            raise TypeError(
                "CBraMod input must be numeric, got "
                f"dtype={signal.dtype}."
            )

        if not np.isfinite(signal).all():
            raise ValueError(
                "CBraMod input contains NaN or Inf."
            )

        if (
            not math.isfinite(float(window.sample_rate))
            or float(window.sample_rate) <= 0
        ):
            raise ValueError(
                "window.sample_rate must be finite and positive, "
                f"got {window.sample_rate}."
            )

        canonical_unit = normalize_eeg_unit(window.unit)
        expected_unit = normalize_eeg_unit(
            self.config.input_unit
        )

        if canonical_unit != expected_unit:
            raise ValueError(
                "CBraMod expects canonical EEG unit "
                f"{expected_unit!r}, but received "
                f"{canonical_unit!r}. Construct RuntimeModel "
                "with SignalCanonicalizer("
                f"target_unit={expected_unit!r})."
            )

        duration_seconds = (
            signal.shape[1] / float(window.sample_rate)
        )

        duration_error = abs(
            duration_seconds - self.config.window_seconds
        )

        if (
            self.config.strict_window_duration
            and duration_error
            > self.config.window_tolerance_seconds
        ):
            raise ValueError(
                "CBraMod requires one real "
                f"{self.config.window_seconds:.3f}-second window, "
                f"but got {duration_seconds:.6f} seconds "
                f"({signal.shape[1]} samples at "
                f"{window.sample_rate} Hz)."
            )

        return signal.astype(np.float32, copy=False)

    def _align_channels(
        self,
        signal: np.ndarray,
        source_channel_names: Sequence[str],
    ) -> CBraModChannelAlignment:
        """
        将任意设备通道映射到 config.standard_channels。

        - unknown 输入通道：直接丢弃；
        - duplicate 输入通道：按样本平均；
        - missing 目标通道：暂时保留为 0，并通过 valid_mask 标记；
          后续由 _complete_missing_channels 决定报错或空间插值。
        """
        canonical_names = tuple(
            self._canonical_channel_name(str(name))
            for name in source_channel_names
        )

        n_target_channels = self.config.n_channels
        n_times = signal.shape[1]

        channel_sum = np.zeros(
            (n_target_channels, n_times),
            dtype=np.float64,
        )
        channel_count = np.zeros(
            n_target_channels,
            dtype=np.int64,
        )

        unknown_channel_names: list[str] = []

        for source_index, canonical_name in enumerate(
            canonical_names
        ):
            target_index = self._target_channel_indices.get(
                canonical_name
            )

            if target_index is None:
                unknown_channel_names.append(
                    str(source_channel_names[source_index])
                )
                continue

            channel_sum[target_index] += signal[source_index]
            channel_count[target_index] += 1

        valid = channel_count > 0

        if not valid.any():
            raise ValueError(
                "None of the input channels matched any CBraMod "
                "target channel. Check device channel names and "
                "CHANNEL_ALIASES."
            )

        aligned_signal = np.zeros(
            (n_target_channels, n_times),
            dtype=np.float32,
        )

        aligned_signal[valid] = (
            channel_sum[valid]
            / channel_count[valid, None]
        ).astype(np.float32)

        return CBraModChannelAlignment(
            signal=aligned_signal,
            channel_valid_mask=valid.astype(np.float32),
            canonical_channel_names=canonical_names,
            unknown_channel_names=tuple(
                unknown_channel_names
            ),
            duplicate_channel_count=int(
                np.maximum(channel_count - 1, 0).sum()
            ),
        )

    def _filter(
        self,
        signal: np.ndarray,
        sample_rate: float,
        channel_valid_mask: np.ndarray,
    ) -> np.ndarray:
        valid = channel_valid_mask.astype(bool)

        output = np.zeros_like(signal, dtype=np.float32)

        if not valid.any():
            raise RuntimeError(
                "No valid CBraMod channels are available for "
                "filtering."
            )

        if not self.config.filter_enabled:
            output[valid] = signal[valid]
            return output

        nyquist_hz = sample_rate / 2.0

        if self.config.filter_high_hz >= nyquist_hz:
            raise ValueError(
                "CBraMod filter_high_hz must be below input "
                "Nyquist frequency. Got "
                f"{self.config.filter_high_hz} Hz with "
                f"input sample_rate={sample_rate} Hz."
            )

        sos = butter(
            self.config.filter_order,
            [
                self.config.filter_low_hz / nyquist_hz,
                self.config.filter_high_hz / nyquist_hz,
            ],
            btype="bandpass",
            output="sos",
        )

        try:
            filtered = sosfiltfilt(
                sos,
                signal[valid].astype(np.float64, copy=False),
                axis=-1,
            )
        except ValueError as error:
            raise ValueError(
                "CBraMod band-pass filtering failed. The input "
                "window may be too short for the configured "
                "filter order."
            ) from error

        output[valid] = filtered.astype(np.float32)
        return output

    def _apply_reference(
        self,
        signal: np.ndarray,
        channel_valid_mask: np.ndarray,
    ) -> np.ndarray:
        valid = channel_valid_mask.astype(bool)

        output = np.zeros_like(signal, dtype=np.float32)

        if not valid.any():
            raise RuntimeError(
                "No valid CBraMod channels are available for "
                "referencing."
            )

        output[valid] = signal[valid]

        if self.config.reference_mode == "none":
            return output

        if self.config.reference_mode == "average":
            average_reference = output[valid].mean(
                axis=0,
                keepdims=True,
            )
            output[valid] -= average_reference
            return output.astype(np.float32)

        raise RuntimeError(
            "Unsupported CBraMod reference mode: "
            f"{self.config.reference_mode!r}."
        )

    @staticmethod
    def _spherical_spline_weights(
        source_positions: np.ndarray,
        destination_positions: np.ndarray,
        *,
        alpha: float,
        stiffness: int = 4,
        n_legendre_terms: int = 50,
    ) -> np.ndarray:
        """
        返回 [n_destination, n_source] 的球面样条插值矩阵。

        实现放在本文件，避免依赖 MNE 私有 API。
        """
        source = np.asarray(
            source_positions,
            dtype=np.float64,
        )
        destination = np.asarray(
            destination_positions,
            dtype=np.float64,
        )

        if source.ndim != 2 or source.shape[1] != 3:
            raise ValueError(
                "source_positions must have shape [N, 3], got "
                f"{source.shape}."
            )

        if (
            destination.ndim != 2
            or destination.shape[1] != 3
        ):
            raise ValueError(
                "destination_positions must have shape [M, 3], "
                f"got {destination.shape}."
            )

        if source.shape[0] < 2:
            raise ValueError(
                "Spherical-spline completion requires at least "
                "two observed channels."
            )

        source_norm = np.linalg.norm(
            source,
            axis=1,
            keepdims=True,
        )
        destination_norm = np.linalg.norm(
            destination,
            axis=1,
            keepdims=True,
        )

        if (
            np.any(source_norm == 0.0)
            or np.any(destination_norm == 0.0)
        ):
            raise ValueError(
                "Electrode coordinates must be non-zero."
            )

        source = source / source_norm
        destination = destination / destination_norm

        terms = np.arange(
            1,
            n_legendre_terms + 1,
            dtype=np.float64,
        )

        coefficients = np.zeros(
            n_legendre_terms + 1,
            dtype=np.float64,
        )
        coefficients[1:] = (
            (2.0 * terms + 1.0)
            / (
                4.0
                * np.pi
                * np.power(terms, stiffness)
                * np.power(terms + 1.0, stiffness)
            )
        )

        def spline_kernel(cosine: np.ndarray) -> np.ndarray:
            clipped = np.clip(cosine, -1.0, 1.0)
            return np.polynomial.legendre.legval(
                clipped,
                coefficients,
            )

        source_kernel = spline_kernel(source @ source.T)
        destination_kernel = spline_kernel(
            destination @ source.T
        )

        n_source = source.shape[0]

        system = np.empty(
            (n_source + 1, n_source + 1),
            dtype=np.float64,
        )
        system[:n_source, :n_source] = source_kernel
        system[
            :n_source,
            :n_source,
        ] = system[:n_source, :n_source] + (
            np.eye(n_source, dtype=np.float64) * alpha
        )
        system[:n_source, n_source] = 1.0
        system[n_source, :n_source] = 1.0
        system[n_source, n_source] = 0.0

        destination_system = np.concatenate(
            [
                destination_kernel,
                np.ones(
                    (destination.shape[0], 1),
                    dtype=np.float64,
                ),
            ],
            axis=1,
        )

        weights = destination_system @ np.linalg.pinv(
            system
        )[:, :n_source]

        if not np.isfinite(weights).all():
            raise RuntimeError(
                "Spherical-spline completion produced NaN or Inf "
                "weights."
            )

        return weights.astype(np.float32)

    def _target_channel_positions(
        self,
    ) -> dict[str, np.ndarray]:
        """
        从通用 standard_1005 坐标表按名称取得 CBraMod target 坐标。

        没有设备专属 montage；设备只需提供真实通道名。
        """
        if self._target_position_cache is not None:
            return self._target_position_cache

        try:
            from mne.channels import make_standard_montage
        except ImportError as error:
            raise RuntimeError(
                "Missing-channel completion requires MNE. "
                "Install the project dependencies, including "
                "mne>=1.6."
            ) from error

        montage = make_standard_montage("standard_1005")
        channel_positions = montage.get_positions()["ch_pos"]

        positions: dict[str, np.ndarray] = {}

        for channel_name, position in channel_positions.items():
            canonical_name = self._canonical_channel_name(
                channel_name
            )

            if canonical_name in self._target_channel_indices:
                positions[canonical_name] = np.asarray(
                    position,
                    dtype=np.float64,
                )

        unavailable = [
            channel_name
            for channel_name in self._target_channels
            if channel_name not in positions
        ]

        if unavailable:
            raise RuntimeError(
                "standard_1005 does not provide coordinates for "
                "CBraMod target channels: "
                f"{unavailable}."
            )

        self._target_position_cache = positions
        return positions

    def _completion_matrix(
        self,
        observed_channel_names: tuple[str, ...],
        missing_channel_names: tuple[str, ...],
    ) -> tuple[np.ndarray, str]:
        """
        返回当前“观测通道布局”专属的插值矩阵。

        同一设备/同一通道布局只计算一次，后续滑窗直接复用。
        """
        cached = self._completion_matrix_cache.get(
            observed_channel_names
        )

        if cached is not None:
            return cached

        positions = self._target_channel_positions()

        source_positions = np.stack(
            [
                positions[channel_name]
                for channel_name in observed_channel_names
            ],
            axis=0,
        )

        destination_positions = np.stack(
            [
                positions[channel_name]
                for channel_name in missing_channel_names
            ],
            axis=0,
        )

        matrix = self._spherical_spline_weights(
            source_positions,
            destination_positions,
            alpha=self.config.spline_alpha,
        )

        expected_shape = (
            len(missing_channel_names),
            len(observed_channel_names),
        )

        if tuple(matrix.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected CBraMod completion matrix shape. "
                f"Expected {expected_shape}, got "
                f"{tuple(matrix.shape)}."
            )

        matrix_sha256 = hashlib.sha256(
            np.ascontiguousarray(matrix).tobytes()
        ).hexdigest()

        cached = (matrix, matrix_sha256)
        self._completion_matrix_cache[
            observed_channel_names
        ] = cached

        return cached

    def _complete_missing_channels(
        self,
        signal: np.ndarray,
        channel_valid_mask: np.ndarray,
    ) -> tuple[
        np.ndarray,
        tuple[str, ...],
        tuple[str, ...],
        str | None,
    ]:
        """
        在重采样后补全缺失目标通道。

        返回：
            completed_signal,
            observed_channel_names,
            missing_channel_names,
            completion_matrix_sha256
        """
        valid = channel_valid_mask.astype(bool)

        if valid.shape != (self.config.n_channels,):
            raise ValueError(
                "channel_valid_mask has unexpected shape: "
                f"{valid.shape}."
            )

        observed_channel_names = tuple(
            channel_name
            for index, channel_name in enumerate(
                self._target_channels
            )
            if valid[index]
        )

        missing_channel_names = tuple(
            channel_name
            for index, channel_name in enumerate(
                self._target_channels
            )
            if not valid[index]
        )

        if not missing_channel_names:
            return (
                signal.astype(np.float32, copy=False),
                observed_channel_names,
                missing_channel_names,
                None,
            )

        if self.config.missing_channel_policy == "error":
            raise ValueError(
                "CBraMod requires all target channels in strict "
                "mode. Missing channels: "
                f"{list(missing_channel_names)}. To run the "
                "device-adapted variant, set "
                "missing_channel_policy='spherical_spline' and "
                "use a head trained/evaluated with the same "
                "completion protocol."
            )

        if (
            len(observed_channel_names)
            < self.config.min_observed_channels
        ):
            raise ValueError(
                "Too few observed channels for CBraMod spherical "
                "spline completion: "
                f"observed={len(observed_channel_names)}, "
                "required_at_least="
                f"{self.config.min_observed_channels}."
            )

        if (
            self.config.missing_channel_policy
            != "spherical_spline"
        ):
            raise RuntimeError(
                "Unsupported missing_channel_policy: "
                f"{self.config.missing_channel_policy!r}."
            )

        matrix, matrix_sha256 = self._completion_matrix(
            observed_channel_names,
            missing_channel_names,
        )

        completed = signal.astype(
            np.float32,
            copy=True,
        )

        completed[~valid] = matrix @ completed[valid]

        if not np.isfinite(completed).all():
            raise RuntimeError(
                "CBraMod channel completion produced NaN or Inf."
            )

        return (
            completed,
            observed_channel_names,
            missing_channel_names,
            matrix_sha256,
        )

    def _resample(
        self,
        signal: np.ndarray,
        source_sample_rate: float,
    ) -> np.ndarray:
        target_sample_rate = self.config.target_sample_rate

        if math.isclose(
            source_sample_rate,
            target_sample_rate,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            resampled = signal

        else:
            ratio = Fraction(
                target_sample_rate / source_sample_rate
            ).limit_denominator(10_000)

            resampled = resample_poly(
                signal.astype(np.float64, copy=False),
                up=ratio.numerator,
                down=ratio.denominator,
                axis=-1,
            )

        resampled = np.asarray(
            resampled,
            dtype=np.float32,
        )

        expected_shape = (
            self.config.n_channels,
            self.config.num_samples,
        )

        if tuple(resampled.shape) != expected_shape:
            raise ValueError(
                "CBraMod resampling produced an unexpected shape. "
                f"Expected {expected_shape}, got "
                f"{tuple(resampled.shape)}. This usually means "
                "the raw window was not exactly 4 seconds."
            )

        if not np.isfinite(resampled).all():
            raise RuntimeError(
                "CBraMod resampling produced NaN or Inf."
            )

        return resampled

    def _normalize(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        if self.config.normalization == "none":
            return signal.astype(np.float32, copy=False)

        if self.config.normalization == "fixed_100uv":
            # The canonical window has already been converted to uV.
            return (signal / 100.0).astype(np.float32, copy=False)

        if self.config.normalization == "per_window_zscore":
            mean = signal.mean(
                axis=-1,
                keepdims=True,
                dtype=np.float64,
            )

            std = signal.std(
                axis=-1,
                keepdims=True,
                dtype=np.float64,
            )

            normalized = (
                signal.astype(np.float64, copy=False) - mean
            ) / np.maximum(
                std,
                self.config.zscore_eps,
            )

            return normalized.astype(np.float32)

        raise RuntimeError(
            "Unsupported CBraMod normalization mode: "
            f"{self.config.normalization!r}."
        )

    def _patchify(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        expected_signal_shape = (
            self.config.n_channels,
            self.config.num_samples,
        )

        if tuple(signal.shape) != expected_signal_shape:
            raise ValueError(
                "CBraMod patchify expects signal shape "
                f"{expected_signal_shape}, got "
                f"{tuple(signal.shape)}."
            )

        patched = np.ascontiguousarray(signal).reshape(
            self.config.n_channels,
            self.config.time_segments,
            self.config.points_per_patch,
        )

        expected_patched_shape = (
            self.config.n_channels,
            self.config.time_segments,
            self.config.points_per_patch,
        )

        if tuple(patched.shape) != expected_patched_shape:
            raise RuntimeError(
                "Unexpected CBraMod patched signal shape. "
                f"Expected {expected_patched_shape}, got "
                f"{tuple(patched.shape)}."
            )

        return patched.astype(np.float32, copy=False)

    def transform(
        self,
        window: CanonicalEEGWindow,
    ) -> PreparedModelInput:
        signal = self._validate_window(window)

        alignment = self._align_channels(
            signal=signal,
            source_channel_names=window.channel_names,
        )

        processed = self._filter(
            alignment.signal,
            sample_rate=float(window.sample_rate),
            channel_valid_mask=alignment.channel_valid_mask,
        )

        processed = self._apply_reference(
            processed,
            channel_valid_mask=alignment.channel_valid_mask,
        )

        processed = self._resample(
            processed,
            source_sample_rate=float(window.sample_rate),
        )

        (
            processed,
            observed_channel_names,
            missing_channel_names,
            completion_matrix_sha256,
        ) = self._complete_missing_channels(
            processed,
            channel_valid_mask=alignment.channel_valid_mask,
        )

        processed = self._normalize(processed)

        patched = self._patchify(processed)

        model_input = torch.from_numpy(
            patched
        ).unsqueeze(0)

        expected_batched_shape = (
            1,
            self.config.n_channels,
            self.config.time_segments,
            self.config.points_per_patch,
        )

        if tuple(model_input.shape) != expected_batched_shape:
            raise RuntimeError(
                "Unexpected CBraMod model input shape. "
                f"Expected {expected_batched_shape}, got "
                f"{tuple(model_input.shape)}."
            )

        if model_input.dtype != torch.float32:
            raise RuntimeError(
                "CBraMod model input must be float32, got "
                f"{model_input.dtype}."
            )

        diagnostics = CBraModPreprocessingDiagnostics(
            source_channel_names=(
                alignment.canonical_channel_names
            ),
            target_channel_names=self._target_channels,
            unknown_channel_names=(
                alignment.unknown_channel_names
            ),
            observed_channel_names=observed_channel_names,
            missing_channel_names=missing_channel_names,
            channel_valid_mask=tuple(
                float(value)
                for value in alignment.channel_valid_mask
            ),
            duplicate_channel_count=(
                alignment.duplicate_channel_count
            ),
            completion_policy=(
                "none"
                if not missing_channel_names
                else self.config.missing_channel_policy
            ),
            completion_matrix_sha256=(
                completion_matrix_sha256
            ),
            source_sample_rate_hz=float(
                window.sample_rate
            ),
            target_sample_rate_hz=float(
                self.config.target_sample_rate
            ),
            source_num_samples=int(signal.shape[1]),
            target_num_samples=int(processed.shape[1]),
            filter_applied=bool(
                self.config.filter_enabled
            ),
            reference_mode=self.config.reference_mode,
            normalization=self.config.normalization,
        )

        self.last_diagnostics = diagnostics

        trace = list(window.processing_history)
        trace.extend(
            [
                "cbramod:align_channels_by_name",
                (
                    "cbramod:unknown_channels_dropped="
                    f"{len(alignment.unknown_channel_names)}"
                ),
                (
                    "cbramod:duplicate_channels_averaged="
                    f"{alignment.duplicate_channel_count}"
                ),
                (
                    "cbramod:observed_channels="
                    f"{len(observed_channel_names)}"
                ),
                (
                    "cbramod:missing_channels="
                    f"{len(missing_channel_names)}"
                ),
                (
                    "cbramod:completion="
                    f"{diagnostics.completion_policy}"
                ),
                (
                    "cbramod:filter="
                    f"{self.config.filter_enabled}"
                ),
                (
                    "cbramod:reference="
                    f"{self.config.reference_mode}"
                ),
                (
                    "cbramod:resample="
                    f"{window.sample_rate}"
                    f"->{self.config.target_sample_rate}"
                ),
                (
                    "cbramod:normalization="
                    f"{self.config.normalization}"
                ),
                "cbramod:patchify=[C,T]->[C,S,P]",
                "cbramod:add_batch_dimension",
            ]
        )

        return PreparedModelInput(
            model_input={"signal": model_input},
            canonical_window=window,
            preprocessing_trace=trace,
            diagnostics=diagnostics.to_dict(),
        )
