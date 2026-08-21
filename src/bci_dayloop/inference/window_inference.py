"""The shared one-window inference entry point used by direct and HTTP paths."""

from __future__ import annotations

from dataclasses import fields
from typing import Sequence

import numpy as np

from bci_dayloop.inference.inference_schema import Prediction
from bci_dayloop.inference.multi_head import HeadPrediction, MultiHeadPrediction
from bci_dayloop.inference.predictor import RawWindowPredictor
from bci_dayloop.runtime.types import RawEEGWindow


def infer_eeg_window(
    predictor: RawWindowPredictor,
    *,
    eeg: np.ndarray,
    sample_rate_hz: float,
    channel_names: Sequence[str],
) -> MultiHeadPrediction:
    """Run the formal raw-window predictor once without changing its semantics."""
    data = np.asarray(eeg, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"EEG must have [C, T] layout, got {data.shape}.")
    if data.shape[0] != len(channel_names):
        raise ValueError(
            "EEG channel count does not match channel_names: "
            f"{data.shape[0]} != {len(channel_names)}."
        )
    if data.shape[1] <= 0:
        raise ValueError("EEG must contain at least one sample.")
    if not np.isfinite(data).all():
        raise ValueError("EEG contains NaN or Inf.")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be finite and greater than zero.")

    return predictor.predict(
        RawEEGWindow(
            data=np.ascontiguousarray(data),
            channel_names=[str(name) for name in channel_names],
            sample_rate=float(sample_rate_hz),
            unit="uV",
            layout="CT",
            metadata={"source": "one_window_inference"},
        )
    )


def named_predictions(prediction: MultiHeadPrediction) -> tuple[Prediction, ...]:
    """Serialize predictor-defined task names and probabilities without hardcoding them."""
    entries: list[Prediction] = []
    for field in fields(prediction):
        task_id = field.name
        head = getattr(prediction, task_id)
        if not isinstance(head, HeadPrediction):
            raise TypeError(f"{task_id}: expected HeadPrediction, got {type(head).__name__}.")
        entries.append(
            Prediction(
                task_id=task_id,
                class_id=int(head.label_id),
                label=str(head.label),
                confidence=float(head.confidence),
                probabilities=tuple(float(value) for value in head.probabilities),
            )
        )
    return tuple(entries)
