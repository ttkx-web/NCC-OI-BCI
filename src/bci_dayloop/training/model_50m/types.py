"""Lightweight, cross-stage types for 50M population training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bci_dayloop.data.hdf5_dataset import HDF5Metadata
from bci_dayloop.training.model_50m.linear_head import WindowSet


@dataclass(frozen=True, slots=True)
class WindowBundle:
    """A window set and the source subject identity of each derived window."""

    window_set: WindowSet
    window_subject_ids: np.ndarray

    def __post_init__(self) -> None:
        expected = (len(self.window_set.windows),)
        if self.window_subject_ids.shape != expected:
            raise ValueError(
                "window_subject_ids shape mismatch: "
                f"expected {expected}, got {self.window_subject_ids.shape}."
            )


@dataclass(frozen=True, slots=True)
class SplitBuildResult:
    """Prepared windows together with their reproducibility provenance."""

    bundle: WindowBundle
    source_trial_summary: dict[str, Any]
    subject_paths: dict[int, Path]
    metadata: HDF5Metadata


@dataclass(frozen=True, slots=True)
class ExtendedMetrics:
    """Classification metrics persisted in the established artifact schema."""

    loss: float
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    confusion_matrix: list[list[int]]
    per_class: list[dict[str, float | int | None]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "confusion_matrix": self.confusion_matrix,
            "per_class": self.per_class,
        }
