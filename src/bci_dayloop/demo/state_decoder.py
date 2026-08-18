"""Independent traditional-feature multi-state decoder used by the demo."""

from __future__ import annotations

from collections import deque
from time import perf_counter, time

import numpy as np

from bci_dayloop.demo.interpretation import make_interpretation
from bci_dayloop.demo.cortical_activity import CorticalActivityMapper
from bci_dayloop.demo.motor_decoder import DemoMotorIntentDecoder, MotorIntentDecoder
from bci_dayloop.demo.schemas import BrainStateResult, DemoEEGWindow, EmotionState, STATE_LABELS_CN
from bci_dayloop.demo.signal_features import SignalFeatures, extract_signal_features
from bci_dayloop.demo.utils import RollingLatency, bounded_score


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
        compute_motor_intent: bool = True,
    ) -> None:
        self.device = device.upper()
        self.smoothing = float(np.clip(smoothing, 0.01, 1.0))
        self.motor_decoder: MotorIntentDecoder = motor_decoder or DemoMotorIntentDecoder()
        self.compute_motor_intent = compute_motor_intent
        self._smoothed_states: dict[str, float] = {}
        self._engagement_history: deque[float] = deque(maxlen=max(4, history_size))
        self._state_vector_history: deque[np.ndarray] = deque(maxlen=max(4, history_size))
        self._previous_mean_psd: np.ndarray | None = None
        self._emotion_label: str | None = None
        self._emotion_candidate: str | None = None
        self._emotion_candidate_count = 0
        # The mapper loads static RGBA assets and precomputes masks once.  It is
        # deliberately independent of MNE/Nilearn and retained across trials.
        self._cortical_mapper = CorticalActivityMapper()
        self._latency = RollingLatency()

    def reset(self) -> None:
        reset_motor_decoder = getattr(self.motor_decoder, "reset", None)
        if callable(reset_motor_decoder):
            reset_motor_decoder()
        self._smoothed_states.clear()
        self._engagement_history.clear()
        self._state_vector_history.clear()
        self._previous_mean_psd = None
        self._emotion_label = None
        self._emotion_candidate = None
        self._emotion_candidate_count = 0
        self._cortical_mapper.reset()
        self._latency = RollingLatency()

    def set_motor_decoder(self, motor_decoder: MotorIntentDecoder) -> None:
        """Swap only the motor head while retaining neural-state history."""
        self.motor_decoder = motor_decoder

    def set_compute_motor_intent(self, enabled: bool) -> None:
        """Toggle motor inference without removing the injected decoder."""
        self.compute_motor_intent = enabled

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

    def _rhythm_stability(self, psd: np.ndarray) -> float:
        mean_psd = np.mean(psd, axis=0)
        if self._previous_mean_psd is None or self._previous_mean_psd.shape != mean_psd.shape:
            similarity = 0.72
        else:
            similarity = float(
                np.dot(mean_psd, self._previous_mean_psd)
                / (np.linalg.norm(mean_psd) * np.linalg.norm(self._previous_mean_psd) + 1e-12)
            )
        self._previous_mean_psd = mean_psd.copy()
        return float(np.clip(similarity * 100.0, 0.0, 100.0))

    @staticmethod
    def _additional_raw_states(features: SignalFeatures, rhythm_stability: float) -> dict[str, float]:
        return {
            "rhythm_stability": rhythm_stability,
            "spatial_balance": float(np.clip(features.spatial_balance * 100.0, 0.0, 100.0)),
            "neural_mobility": bounded_score(features.hjorth_mobility, center=0.55, scale=0.38),
            "dynamic_complexity": bounded_score(features.hjorth_complexity, center=1.45, scale=0.72),
            "signal_activity": bounded_score(np.log1p(features.temporal_activity), center=1.35, scale=0.9),
            "regional_consistency": float(np.clip(features.regional_consistency * 100.0, 0.0, 100.0)),
        }

    def _state_stability(self, states: dict[str, float]) -> float:
        names = [name for name in STATE_LABELS_CN if name not in {"emotion_state", "state_stability", "attention_stability"}]
        vector = np.asarray([states[name] for name in names], dtype=np.float64)
        if len(self._state_vector_history) < 3:
            raw_stability = 70.0
        else:
            reference = np.mean(np.asarray(self._state_vector_history), axis=0)
            distance = float(np.linalg.norm(vector - reference) / np.sqrt(vector.size))
            raw_stability = float(np.clip(100.0 * np.exp(-distance / 22.0), 0.0, 100.0))
        self._state_vector_history.append(vector)
        old = self._smoothed_states.get("state_stability", raw_stability)
        score = float(np.clip((1.0 - self.smoothing) * old + self.smoothing * raw_stability, 0.0, 100.0))
        self._smoothed_states["state_stability"] = score
        return score

    def _emotion(self, states: dict[str, float]) -> EmotionState:
        relaxation = states["neural_relaxation"]
        arousal = states["cortical_arousal"]
        engagement = states["cognitive_engagement"]
        load = states["cognitive_load"]
        if relaxation >= 65.0 and arousal < 45.0:
            candidate, score = "relaxed", 82.0
        elif relaxation >= 55.0 and 40.0 <= arousal < 65.0:
            candidate, score = "positive", 72.0
        elif engagement >= 65.0 and load < 68.0:
            candidate, score = "focused", 74.0
        elif arousal >= 65.0 and load >= 60.0:
            candidate, score = "tense", 28.0
        else:
            candidate, score = "neutral", 55.0
        if candidate == self._emotion_candidate:
            self._emotion_candidate_count += 1
        else:
            self._emotion_candidate, self._emotion_candidate_count = candidate, 1
        if self._emotion_label is None or candidate == self._emotion_label or self._emotion_candidate_count >= 3:
            self._emotion_label = candidate
        metadata = {
            "relaxed": ("放松", "😌"),
            "positive": ("愉悦", "🙂"),
            "neutral": ("平稳", "😐"),
            "focused": ("专注", "🤔"),
            "tense": ("紧张", "😣"),
        }
        label = self._emotion_label or "neutral"
        label_cn, emoji = metadata[label]
        display_score = score if label == candidate else {"relaxed": 82.0, "positive": 72.0, "neutral": 55.0, "focused": 74.0, "tense": 28.0}[label]
        return EmotionState(label=label, label_cn=label_cn, emoji=emoji, score=display_score)

    def decode_window(self, window: DemoEEGWindow) -> BrainStateResult:
        """Decode a source-neutral :class:`DemoEEGWindow` into presentation data."""
        started = perf_counter()
        values = np.asarray(window.samples, dtype=np.float32)
        channel_names = list(window.channel_names)
        sample_rate = float(window.sample_rate)
        features = extract_signal_features(
            values,
            sample_rate,
            unit=window.unit,
            channel_names=channel_names,
            channel_valid_mask=window.valid_mask,
        )
        raw_states = self._raw_states(features)
        raw_states.update(self._additional_raw_states(features, self._rhythm_stability(features.psd)))
        states = self._smooth_states(raw_states)
        states["state_stability"] = self._state_stability(states)
        emotion = self._emotion(states)
        emotion_old = self._smoothed_states.get("emotion_state", emotion.score)
        states["emotion_state"] = float(
            np.clip((1.0 - self.smoothing) * emotion_old + self.smoothing * emotion.score, 0.0, 100.0)
        )
        self._smoothed_states["emotion_state"] = states["emotion_state"]
        states = {name: states[name] for name in STATE_LABELS_CN}
        if self.compute_motor_intent:
            motor = self.motor_decoder.predict(
                band_power=features.relative_band_power,
                rms_uv=features.rms_uv,
                samples=values,
                sample_rate=sample_rate,
                channel_names=channel_names,
                unit=window.unit,
            )
        else:
            motor = {
                "label": "",
                "label_cn": "",
                "confidence": 0.0,
                "probabilities": {},
                "decoder_type": "disabled",
                "decoder_display_name": "disabled",
            }
        cortical_activity = self._cortical_mapper.update(
            channel_names,
            features.channel_power_1_30,
            channel_valid_mask=features.channel_valid_mask,
            montage_name=window.montage_name,
        )
        latency_ms = (perf_counter() - started) * 1000.0
        average_latency_ms, p95_latency_ms = self._latency.add(latency_ms)
        interpretation = make_interpretation(states, motor, features.relative_band_power)
        return BrainStateResult(
            timestamp=float(window.timestamp),
            states=states,
            motor_intent=motor,
            band_power=features.relative_band_power,
            signal_quality=features.signal_quality,
            latency_ms=latency_ms,
            device=self.device,
            interpretation=interpretation,
            waveform=values.copy(),
            channel_names=list(channel_names),
            sample_rate=float(sample_rate),
            waveform_unit=window.unit,
            psd_frequencies=features.frequencies,
            psd_values=np.mean(features.psd, axis=0),
            average_latency_ms=average_latency_ms,
            p95_latency_ms=p95_latency_ms,
            emotion=emotion,
            cortical_activity=cortical_activity,
        )

    def decode(
        self,
        samples: np.ndarray,
        *,
        sample_rate: float,
        channel_names: list[str],
        unit: str = "V",
        timestamp: float | None = None,
        channel_valid_mask: np.ndarray | None = None,
        device_name: str | None = None,
        montage_name: str | None = None,
    ) -> BrainStateResult:
        """Backward-compatible raw-array adapter for existing HDF5 callers."""
        return self.decode_window(
            DemoEEGWindow(
                samples=samples,
                sample_rate=sample_rate,
                channel_names=channel_names,
                unit=unit,
                timestamp=time() if timestamp is None else float(timestamp),
                channel_valid_mask=channel_valid_mask,
                device_name=device_name,
                montage_name=montage_name,
            )
        )


__all__ = ["DemoStateDecoder", "STATE_LABELS_CN"]
