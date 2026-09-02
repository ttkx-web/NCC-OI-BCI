"""Configuration for the latency-only 1B EEG backbone.

This module intentionally does not define a classifier, labels, aggregation,
or a Runtime Model Package contract.  It describes only the checkpoint-backed
preprocessing/tokenization and encoder path used for latency measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bci_dayloop.models.model_50m.config import STANDARD_64_CHANNELS


@dataclass(frozen=True, slots=True)
class Model1BConfig:
    checkpoint_path: Path | str
    device: str = "cpu"

    target_sample_rate: float = 100.0
    window_seconds: float = 4.0
    patch_seconds: float = 1.0
    patch_stride_seconds: float = 1.0
    n_channels: int = 64
    standard_channels: tuple[str, ...] = STANDARD_64_CHANNELS
    strict_window_duration: bool = True
    window_tolerance_seconds: float = 0.02

    # These are deliberately independent fields, even though their defaults
    # match the 50M preprocessing contract.
    filter_enabled: bool = True
    filter_low_hz: float = 0.1
    filter_high_hz: float = 75.0
    filter_order: int = 4
    reference_mode: str = "none"
    zscore_enabled: bool = True
    zscore_eps: float = 1e-8
    missing_channel_fill_value: float = 0.0

    # Actual formal 1B checkpoint architecture.
    d_model: int = 2048
    n_heads: int = 16
    depth: int = 20
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    # Checkpoint has ten learned temporal positions.  A latency run may use
    # only positions 0..N-1 for a 1/2/3/4 second window.
    model_n_time_patches: int = 10
    output_layer_idx: int = 19

    # Explicit safety marker used by the latency-only runtime/policy.  It is
    # not an inference-classification mode.
    latency_only: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        if self.n_channels != 64 or len(self.standard_channels) != 64:
            raise ValueError("1B backbone requires exactly the confirmed 64 channels")
        if self.target_sample_rate != 100.0:
            raise ValueError("1B backbone requires target_sample_rate=100.0")
        if self.patch_seconds != 1.0 or self.patch_stride_seconds != 1.0:
            raise ValueError("1B backbone requires non-overlapping 1.0 second patches")
        if not 1.0 <= self.window_seconds <= 10.0:
            raise ValueError("1B window_seconds must be in [1.0, 10.0]")
        if self.target_num_points % self.patch_num_points != 0:
            raise ValueError("1B window must split exactly into 1-second patches")
        if self.model_n_time_patches != 10:
            raise ValueError("formal 1B checkpoint requires model_n_time_patches=10")
        if self.num_time_patches > self.model_n_time_patches:
            raise ValueError("1B input has more patches than checkpoint time embeddings")
        if (self.d_model, self.n_heads, self.depth, self.mlp_ratio, self.dropout) != (2048, 16, 20, 4.0, 0.1):
            raise ValueError("1B architecture must match the formal checkpoint")
        if self.output_layer_idx != 19:
            raise ValueError("latency-only 1B extraction must use final layer index 19")
        if self.reference_mode != "none":
            raise ValueError("confirmed 1B preprocessing requires reference_mode='none'")
        if not self.latency_only:
            raise ValueError("Model1BConfig is latency-only; classification is not supported")

    @property
    def target_num_points(self) -> int:
        return int(round(self.window_seconds * self.target_sample_rate))

    @property
    def patch_num_points(self) -> int:
        return int(round(self.patch_seconds * self.target_sample_rate))

    @property
    def patch_stride_points(self) -> int:
        return int(round(self.patch_stride_seconds * self.target_sample_rate))

    @property
    def num_time_patches(self) -> int:
        return ((self.target_num_points - self.patch_num_points) // self.patch_stride_points) + 1

    @property
    def num_tokens(self) -> int:
        return self.n_channels * self.num_time_patches

    @property
    def target_nyquist_hz(self) -> float:
        return self.target_sample_rate / 2.0
