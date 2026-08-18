"""Public data contracts for the multi-state demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bci_dayloop.demo.cortical_activity import CorticalActivityResult


STATE_LABELS_CN: dict[str, str] = {
    "neural_activation": "神经激活度",
    "cortical_arousal": "皮层唤醒度",
    "neural_relaxation": "神经放松度",
    "cognitive_engagement": "认知参与度",
    "cognitive_load": "认知负荷",
    "attention_stability": "注意稳定度",
    "neural_complexity": "神经复杂度",
    "neural_synchrony": "神经同步度",
    "emotion_state": "情绪状态",
    "rhythm_stability": "节律稳定度",
    "spatial_balance": "空间平衡度",
    "neural_mobility": "神经动态性",
    "dynamic_complexity": "动态复杂度",
    "signal_activity": "信号活跃度",
    "regional_consistency": "脑区一致性",
    "state_stability": "状态稳定度",
}

MOTOR_LABELS_CN: dict[str, str] = {
    "left_hand": "左手",
    "right_hand": "右手",
    "feet": "双脚",
    "tongue": "舌部",
}


@dataclass(frozen=True, slots=True)
class EmotionState:
    """Display metadata for the demo-only emotion-state estimator."""

    label: str
    label_cn: str
    emoji: str
    score: float


@dataclass(slots=True)
class BrainStateResult:
    """All presentation data for one EEG window.

    Values in ``states`` and ``signal_quality`` are bounded to 0--100.  The
    raw waveform is included so UI code never needs to run feature extraction.
    ``cortical_activity`` is a sensor-derived visualization over static
    templates; it is not an EEG source-localization result.
    """

    timestamp: float
    states: dict[str, float]
    motor_intent: dict[str, object]
    band_power: dict[str, float]
    signal_quality: float
    latency_ms: float
    device: str
    interpretation: str
    topomap_values: np.ndarray | None = None
    topomap_positions: np.ndarray | None = None
    topomap_channel_names: list[str] | None = None
    waveform: np.ndarray | None = None
    channel_names: list[str] | None = None
    sample_rate: float | None = None
    waveform_unit: str | None = None
    psd_frequencies: np.ndarray | None = None
    psd_values: np.ndarray | None = None
    average_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    emotion: EmotionState | None = None
    cortical_activity: CorticalActivityResult | None = None

    @property
    def brain_state_score(self) -> float:
        """Visualization-only aggregate; it is not a clinical measure."""
        return float(np.mean(list(self.states.values()))) if self.states else 0.0
