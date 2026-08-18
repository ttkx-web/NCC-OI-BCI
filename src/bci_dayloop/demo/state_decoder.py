"""Independent traditional-feature multi-state decoder used by the demo."""

from __future__ import annotations

from collections import deque
from time import perf_counter, time

import numpy as np

from bci_dayloop.demo.interpretation import make_interpretation
from bci_dayloop.demo.motor_decoder import DemoMotorIntentDecoder, MotorIntentDecoder
from bci_dayloop.demo.schemas import BrainStateResult, STATE_LABELS_CN
from bci_dayloop.demo.signal_features import SignalFeatures, extract_signal_features
from bci_dayloop.demo.utils import RollingLatency, bounded_score, standard_1020_positions


class DemoStateDecoder:
    """Produce a single :class:`BrainStateResult` from each EEG window.

    The class is deliberately stateful only for display smoothing, attention
    stability, motor transition and recent latency metrics.  It can later be
    replaced by a learned decoder without changing the Streamlit page contract.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        smoothing: float = 0.34,
        history_size: int = 24,
        motor_decoder: MotorIntentDecoder | None = None,
    ) -> None:
        self.device = device.upper()
        self.smoothing = float(np.clip(smoothing, 0.01, 1.0))
        self.motor_decoder: MotorIntentDecoder = motor_decoder or DemoMotorIntentDecoder()
        self._smoothed_states: dict[str, float] = {}
        self._engagement_history: deque[float] = deque(maxlen=max(4, history_size))
        self._latency = RollingLatency()

    def reset(self) -> None:
        reset_motor_decoder = getattr(self.motor_decoder, "reset", None)
        if callable(reset_motor_decoder):
            reset_motor_decoder()
        self._smoothed_states.clear()
        self._engagement_history.clear()
        self._latency = RollingLatency()

    def set_motor_decoder(self, motor_decoder: MotorIntentDecoder) -> None:
        """Swap only the motor head while retaining neural-state history."""
        self.motor_decoder = motor_decoder

    @staticmethod
    def _raw_states(features: SignalFeatures) -> dict[str, float]:
        bands = features.relative_band_power
        alpha = bands["alpha"]
        beta = bands["beta"]
        theta = bands["theta"]
        activation = bounded_score(np.log1p(features.rms_uv), center=2.5, scale=1.5)
        arousal = bounded_score(np.log((beta + 1e-5) / (alpha + 1e-5)), center=-0.2, scale=0.85)
        relaxation = bounded_score(alpha, center=0.20, scale=0.12)
        engagement = bounded_score(np.log((beta + 1e-5) / (alpha + theta + 1e-5)), center=-1.25, scale=0.65)
        load = bounded_score(np.log((theta + 1e-5) / (alpha + 1e-5)), center=-0.45, scale=0.85)
        complexity = bounded_score(features.spectral_entropy, center=0.72, scale=0.20)
        synchrony = bounded_score(features.mean_abs_correlation, center=0.50, scale=0.45)
        return {
            "neural_activation": activation,
            "cortical_arousal": arousal,
            "neural_relaxation": relaxation,
            "cognitive_engagement": engagement,
            "cognitive_load": load,
            "neural_complexity": complexity,
            "neural_synchrony": synchrony,
        }

    def _smooth_states(self, raw: dict[str, float]) -> dict[str, float]:
        result: dict[str, float] = {}
        for name, value in raw.items():
            old = self._smoothed_states.get(name, value)
            result[name] = float(np.clip((1.0 - self.smoothing) * old + self.smoothing * value, 0.0, 100.0))
        self._smoothed_states.update(result)
        self._engagement_history.append(result["cognitive_engagement"])
        history = np.asarray(self._engagement_history, dtype=np.float64)
        if history.size < 4:
            stability = 65.0
        else:
            coefficient_of_variation = float(history.std() / (history.mean() + 1e-6))
            stability = float(np.clip(100.0 * (1.0 - coefficient_of_variation / 0.35), 0.0, 100.0))
        result["attention_stability"] = stability
        self._smoothed_states["attention_stability"] = stability
        return result

    def decode(
        self,
        samples: np.ndarray,
        *,
        sample_rate: float,
        channel_names: list[str],
        unit: str = "V",
        timestamp: float | None = None,
    ) -> BrainStateResult:
        """Decode one [channel, sample] window into the stable presentation API."""
        started = perf_counter()
        values = np.asarray(samples, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"samples must be [channels, samples], got {values.shape}")
        if values.shape[0] != len(channel_names):
            raise ValueError("channel_names length does not match samples")
        features = extract_signal_features(values, sample_rate, unit=unit)
        states = self._smooth_states(self._raw_states(features))
        motor = self.motor_decoder.predict(
            band_power=features.relative_band_power,
            rms_uv=features.rms_uv,
            samples=values,
            sample_rate=sample_rate,
            channel_names=channel_names,
            unit=unit,
        )
        positions, positioned_names = standard_1020_positions(channel_names)
        if positioned_names is None:
            topomap_values = None
        else:
            by_name = {name: value for name, value in zip(channel_names, features.channel_alpha_power, strict=True)}
            topomap_values = np.asarray([by_name[name] for name in positioned_names], dtype=np.float64)
            topomap_values = np.log10(topomap_values + 1e-12)
        latency_ms = (perf_counter() - started) * 1000.0
        average_latency_ms, p95_latency_ms = self._latency.add(latency_ms)
        interpretation = make_interpretation(states, motor, features.relative_band_power)
        return BrainStateResult(
            timestamp=time() if timestamp is None else float(timestamp),
            states=states,
            motor_intent=motor,
            band_power=features.relative_band_power,
            signal_quality=features.signal_quality,
            latency_ms=latency_ms,
            device=self.device,
            interpretation=interpretation,
            topomap_values=topomap_values,
            topomap_positions=positions,
            topomap_channel_names=positioned_names,
            waveform=values.copy(),
            channel_names=list(channel_names),
            sample_rate=float(sample_rate),
            waveform_unit=unit,
            psd_frequencies=features.frequencies,
            psd_values=np.mean(features.psd, axis=0),
            average_latency_ms=average_latency_ms,
            p95_latency_ms=p95_latency_ms,
        )


__all__ = ["DemoStateDecoder", "STATE_LABELS_CN"]
