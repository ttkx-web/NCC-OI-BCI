from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any

import numpy as np
from scipy.signal import butter, iirnotch, resample_poly, sosfiltfilt, tf2sos


@dataclass(frozen=True)
class PreprocessingConfig:
    bandpass_hz: tuple[float, float] = (0.1, 75.0)
    notch_hz: float = 50.0
    target_sample_rate: float = 200.0
    output_unit: str = "uV"
    zscore_epsilon: float = 1.0e-6
    patch_samples: int = 200

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PreprocessingConfig":
        values = dict(payload)
        values["bandpass_hz"] = tuple(values.get("bandpass_hz", (0.1, 75.0)))
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["bandpass_hz"] = list(self.bandpass_hz)
        return result


class EEGPreprocessor:
    """Shared deterministic preprocessing for training and replay."""

    def __init__(self, config: PreprocessingConfig | dict[str, Any]) -> None:
        self.config = config if isinstance(config, PreprocessingConfig) else PreprocessingConfig.from_dict(config)

    @staticmethod
    def select_eeg_channels(
        data: np.ndarray, channel_names: list[str], channel_types: list[str] | None = None
    ) -> tuple[np.ndarray, list[str]]:
        values = np.asarray(data)
        if channel_types is None:
            keep = [i for i, name in enumerate(channel_names) if not name.upper().startswith(("EOG", "ECG", "EMG", "STI"))]
        else:
            keep = [i for i, kind in enumerate(channel_types) if kind.lower() == "eeg"]
        if not keep:
            raise ValueError("No EEG channels remain after channel selection")
        return values[..., keep, :] if values.ndim == 3 else values[keep], [channel_names[i] for i in keep]

    def transform(
        self,
        data: np.ndarray,
        sample_rate: float,
        input_unit: str = "V",
        *,
        reshape: bool = True,
    ) -> np.ndarray:
        x = np.asarray(data, dtype=np.float64)
        single = x.ndim == 2
        if single:
            x = x[None, ...]
        if x.ndim != 3:
            raise ValueError(f"Expected [N,C,T] or [C,T], got {x.shape}")
        low, high = self.config.bandpass_hz
        nyquist = float(sample_rate) / 2.0
        if not 0 < low < high < nyquist:
            raise ValueError(f"Invalid bandpass {low}-{high} Hz for {sample_rate} Hz data")
        band_sos = butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
        x = sosfiltfilt(band_sos, x, axis=-1)
        if 0 < self.config.notch_hz < nyquist:
            b, a = iirnotch(self.config.notch_hz / nyquist, Q=30.0)
            x = sosfiltfilt(tf2sos(b, a), x, axis=-1)

        ratio = Fraction(self.config.target_sample_rate / float(sample_rate)).limit_denominator(1000)
        if ratio.numerator != ratio.denominator:
            x = resample_poly(x, ratio.numerator, ratio.denominator, axis=-1)
        unit = input_unit.strip().lower().replace("μ", "u").replace("µ", "u")
        scale_to_uv = {"v": 1e6, "mv": 1e3, "uv": 1.0}.get(unit)
        if scale_to_uv is None:
            raise ValueError(f"Unsupported EEG input unit: {input_unit}")
        x *= scale_to_uv
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        x = (x - mean) / np.maximum(std, self.config.zscore_epsilon)
        x = x.astype(np.float32, copy=False)

        if reshape:
            patch = self.config.patch_samples
            remainder = x.shape[-1] % patch
            if remainder:
                x = x[..., : x.shape[-1] - remainder]
            if x.shape[-1] < patch:
                raise ValueError(f"Window has only {x.shape[-1]} samples; need at least {patch}")
            x = x.reshape(x.shape[0], x.shape[1], x.shape[-1] // patch, patch)
        return x[0] if single else x

