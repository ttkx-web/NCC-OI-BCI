"""Stable contract for the workload, attention, emotion application."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

TASKS = ("workload", "attention", "emotion")
TASK_OUTPUT_DIMS: dict[str, int] = {"workload": 2, "attention": 3, "emotion": 3}
SHARED_FEATURE_CONTRACT: dict[str, Any] = {
    "feature_dim": 65_536, "window_seconds": 2.0, "target_sample_rate": 100.0,
    "embedding_layer_resolved": 9, "embedding_layer_internal_index": 8,
    "output_layer_idx": 8, "aggregation": "flatten", "d_model": 512,
    "num_tokens": 128, "n_channels": 64, "model_n_time_patches": 10,
    "patch_seconds": 1.0, "patch_stride_seconds": 1.0,
    "backbone_adaptation": "frozen", "freeze_backbone": True,
    "missing_channel_strategy": "zero_with_valid_mask",
}

DEFAULT_PATHS = {
    "input_h5": "data/processed/yaxin/smr_control_yaxin_0819_combined.h5",
    "model_package": "model_packages/50m_three_mental_states",
    "session": "S6",
}

@dataclass(frozen=True, slots=True)
class HeadPrediction:
    label_id: int
    label: str
    confidence: float
    probabilities: tuple[float, ...]

@dataclass(frozen=True, slots=True)
class ThreeMentalStatePrediction:
    workload: HeadPrediction
    attention: HeadPrediction
    emotion: HeadPrediction

@dataclass(frozen=True, slots=True)
class ThreeMentalStateDiagnostics:
    preprocessing_calls: int
    backbone_forwards: int
    head_forwards: dict[str, int]
    preprocessed_shape: tuple[int, ...]
    selected_embedding_shape: tuple[int, ...]
    shared_feature_shape: tuple[int, ...]
    logit_shapes: dict[str, tuple[int, ...]]
    mapped_channel_count: int
    missing_channel_count: int
    missing_standard_channel_names: tuple[str, ...]
    unknown_channel_names: tuple[str, ...]
    duplicate_channel_count: int
    preprocessing_latency_ms: float
    backbone_latency_ms: float
    heads_latency_ms: float

@dataclass(frozen=True, slots=True)
class HeadCheckpointInfo:
    task: str
    checkpoint_path: Path
    class_names: tuple[str, ...]
    input_dim: int
    output_dim: int
    metadata: Mapping[str, Any]

# Compatibility names remain type-identical, not wrappers.
MultiHeadPrediction = ThreeMentalStatePrediction
MultiHeadInferenceDiagnostics = ThreeMentalStateDiagnostics
