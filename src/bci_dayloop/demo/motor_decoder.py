"""Replaceable motor-intent decoders for the multi-state demo."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np

from bci_dayloop.demo.schemas import MOTOR_LABELS_CN
from bci_dayloop.runtime.types import RawEEGWindow


class MotorIntentDecoder(Protocol):
    """The small Motor Intent interface consumed by :class:`DemoStateDecoder`."""

    display_name: str

    def predict(
        self,
        *,
        band_power: dict[str, float],
        rms_uv: float,
        samples: np.ndarray | None = None,
        sample_rate: float | None = None,
        channel_names: list[str] | None = None,
        unit: str = "V",
    ) -> dict[str, object]:
        ...


def _canonical_motor_label(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "left_hand": "left_hand",
        "lefthand": "left_hand",
        "right_hand": "right_hand",
        "righthand": "right_hand",
        "feet": "feet",
        "foot": "feet",
        "both_feet": "feet",
        "tongue": "tongue",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported Motor Intent class name: {value!r}")
    return aliases[normalized]


class ModelPackageMotorIntentDecoder:
    """Adapter from an existing Runtime Model Package to demo motor intent.

    ``RuntimeModel.predict`` owns the package's canonicalisation, channel
    mapping, resampling and model-input transform.  This adapter deliberately
    does not reproduce any model-specific preprocessing.
    """

    def __init__(self, package_path: str | Path, *, device: str = "cpu") -> None:
        # Keep the lightweight Demo Decoder importable when optional heavy model
        # dependencies are absent; real-package loading still uses the official loader.
        from bci_dayloop.packages.loader import load_runtime_package

        self.package = load_runtime_package(package_path, device=device, verify_hashes=True)
        self.package_path = self.package.package_path
        self.device = device
        self.class_names = tuple(_canonical_motor_label(name) for name in self.package.class_names)
        if len(self.class_names) != 4 or set(self.class_names) != set(MOTOR_LABELS_CN):
            raise ValueError(
                "Model Package is not a compatible four-class motor-imagery package: "
                f"class_names={self.package.class_names}."
            )
        family = {"model_50m": "50M", "labram": "LaBraM", "cbramod": "CBraMod"}.get(
            self.package.model_type,
            self.package.model_name,
        )
        self.display_name = str(family)
        self.window_sec = self.package.window_sec
        self.target_sample_rate = self.package.target_sample_rate

    def predict(
        self,
        *,
        band_power: dict[str, float],
        rms_uv: float,
        samples: np.ndarray | None = None,
        sample_rate: float | None = None,
        channel_names: list[str] | None = None,
        unit: str = "V",
    ) -> dict[str, object]:
        del band_power, rms_uv
        if samples is None or sample_rate is None or channel_names is None:
            raise ValueError("Model Package motor decoder requires samples, sample_rate and channel_names")
        raw_window = RawEEGWindow(
            data=np.asarray(samples, dtype=np.float32),
            channel_names=list(channel_names),
            sample_rate=float(sample_rate),
            unit=str(unit),
            layout="CT",
            metadata={"source": "multistate_demo"},
        )
        started = perf_counter()
        output = self.package.runtime_model.predict(raw_window)
        inference_ms = (perf_counter() - started) * 1000.0
        probabilities_array = output.probabilities.detach().cpu().numpy().reshape(-1)
        if probabilities_array.shape[0] != len(self.class_names):
            raise RuntimeError("Model Package returned a probability count inconsistent with its class_names")
        probabilities = {
            label: float(value)
            for label, value in zip(self.class_names, probabilities_array, strict=True)
        }
        label = self.class_names[int(output.predicted_class)]
        return {
            "label": label,
            "label_cn": MOTOR_LABELS_CN[label],
            "confidence": float(output.confidence),
            "probabilities": probabilities,
            "decoder_type": "runtime_model_package",
            "decoder_display_name": self.display_name,
            "motor_inference_ms": inference_ms,
        }


class DemoMotorIntentDecoder:
    """Generate stable four-class motor intent probabilities from EEG features."""

    class_names = ("left_hand", "right_hand", "feet", "tongue")
    display_name = "Demo"

    def __init__(self, *, hold_windows: int = 12, smoothing: float = 0.24) -> None:
        self.hold_windows = max(2, hold_windows)
        self.smoothing = float(np.clip(smoothing, 0.01, 1.0))
        self._window_index = 0
        self._probabilities: np.ndarray | None = None

    def reset(self) -> None:
        self._window_index = 0
        self._probabilities = None

    def predict(
        self,
        *,
        band_power: dict[str, float],
        rms_uv: float,
        samples: np.ndarray | None = None,
        sample_rate: float | None = None,
        channel_names: list[str] | None = None,
        unit: str = "V",
    ) -> dict[str, object]:
        """Predict a smooth demo probability distribution.

        A future adapter may preserve this method and return real logits here.
        """
        del samples, sample_rate, channel_names, unit
        phase = self._window_index // self.hold_windows
        dominant = phase % len(self.class_names)
        feature_bias = np.array(
            [band_power.get("beta", 0.0), band_power.get("alpha", 0.0), band_power.get("theta", 0.0), np.log1p(max(0.0, rms_uv)) / 10.0],
            dtype=np.float64,
        )
        feature_bias = feature_bias - feature_bias.mean()
        target = np.full(4, 0.055, dtype=np.float64)
        target[dominant] = 0.76
        target += 0.05 * np.tanh(feature_bias)
        target = np.clip(target, 0.02, None)
        target /= target.sum()
        if self._probabilities is None:
            self._probabilities = target
        else:
            self._probabilities = (1.0 - self.smoothing) * self._probabilities + self.smoothing * target
        self._probabilities /= self._probabilities.sum()
        self._window_index += 1
        index = int(np.argmax(self._probabilities))
        label = self.class_names[index]
        probabilities = {name: float(value) for name, value in zip(self.class_names, self._probabilities, strict=True)}
        return {
            "label": label,
            "label_cn": MOTOR_LABELS_CN[label],
            "confidence": probabilities[label],
            "probabilities": probabilities,
            "decoder_type": "demo_smooth_visualization",
            "decoder_display_name": self.display_name,
        }
