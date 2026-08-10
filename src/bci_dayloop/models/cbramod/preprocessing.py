from __future__ import annotations

import math
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
class CBraModPreprocessingDiagnostics:
    source_channel_names: tuple[str, ...]
    target_channel_names: tuple[str, ...]
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
        source_channel_names: list[str],
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """
        严格重排。

        任何 unknown、duplicate 或 missing channel 都报错；
        CBRaMod 不采用 50M 的缺失通道填零策略。
        """

        source_channels = tuple(
            self._canonical_channel_name(name)
            for name in source_channel_names
        )

        source_indices: dict[str, int] = {}
        duplicate_channels: list[str] = []
        unknown_channels: list[str] = []

        for source_index, channel_name in enumerate(
            source_channels
        ):
            if channel_name not in self._target_channel_indices:
                unknown_channels.append(
                    str(source_channel_names[source_index])
                )
                continue

            if channel_name in source_indices:
                duplicate_channels.append(channel_name)
                continue

            source_indices[channel_name] = source_index

        missing_channels = [
            target_channel
            for target_channel in self._target_channels
            if target_channel not in source_indices
        ]

        if unknown_channels:
            raise ValueError(
                "CBraMod received channels outside its required "
                "22-channel montage: "
                f"{unknown_channels}."
            )

        if duplicate_channels:
            raise ValueError(
                "CBraMod received duplicate channels after "
                "canonicalization: "
                f"{duplicate_channels}."
            )

        if missing_channels:
            raise ValueError(
                "CBraMod requires all 22 channels; missing "
                f"{missing_channels}."
            )

        reorder_indices = [
            source_indices[target_channel]
            for target_channel in self._target_channels
        ]

        aligned = signal[reorder_indices]

        expected_shape = (
            self.config.n_channels,
            signal.shape[1],
        )

        if tuple(aligned.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected aligned CBRaMod signal shape. "
                f"Expected {expected_shape}, got "
                f"{tuple(aligned.shape)}."
            )

        return (
            aligned.astype(np.float32, copy=False),
            source_channels,
        )

    def _filter(
        self,
        signal: np.ndarray,
        sample_rate: float,
    ) -> np.ndarray:
        if not self.config.filter_enabled:
            return signal.astype(np.float32, copy=False)

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
                signal.astype(np.float64, copy=False),
                axis=-1,
            )
        except ValueError as error:
            raise ValueError(
                "CBraMod band-pass filtering failed. The input "
                "window may be too short for the configured "
                "filter order."
            ) from error

        return filtered.astype(np.float32)

    def _apply_reference(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        if self.config.reference_mode == "none":
            return signal.astype(np.float32, copy=False)

        if self.config.reference_mode == "average":
            reference = signal.mean(
                axis=0,
                keepdims=True,
            )

            return (
                signal - reference
            ).astype(np.float32)

        raise RuntimeError(
            "Unsupported CBraMod reference mode: "
            f"{self.config.reference_mode!r}."
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

        aligned_signal, source_channels = self._align_channels(
            signal=signal,
            source_channel_names=window.channel_names,
        )

        processed = self._filter(
            aligned_signal,
            sample_rate=float(window.sample_rate),
        )

        processed = self._apply_reference(processed)

        processed = self._resample(
            processed,
            source_sample_rate=float(window.sample_rate),
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
            source_channel_names=source_channels,
            target_channel_names=self._target_channels,
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
                "cbramod:strict_channel_reorder",
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