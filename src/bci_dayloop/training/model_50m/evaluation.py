"""Held-out evaluation for the 50M Stage-1 classification workflows."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from bci_dayloop.models.model_50m.classifier import Model50MClassifier
from bci_dayloop.training.model_50m.engine import run_finetune_epoch
from bci_dayloop.training.model_50m.linear_head import EpochMetrics, run_head_epoch
from bci_dayloop.training.model_50m.types import ExtendedMetrics


def extend_metrics(
    metrics: EpochMetrics,
    *,
    class_names: Sequence[str],
) -> ExtendedMetrics:
    """Add the established per-class precision/recall/F1 report fields."""
    confusion = np.asarray(metrics.confusion_matrix, dtype=np.int64)
    if confusion.shape != (len(class_names), len(class_names)):
        raise ValueError(
            "Confusion matrix shape does not match class names: "
            f"{confusion.shape} vs {len(class_names)}."
        )

    true_support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    per_class: list[dict[str, float | int | None]] = []
    f1_values: list[float] = []
    for index, class_name in enumerate(class_names):
        tp = int(true_positive[index])
        support = int(true_support[index])
        predicted = int(predicted_support[index])
        precision = (tp / predicted) if predicted > 0 else 0.0
        recall = (tp / support) if support > 0 else 0.0
        denominator = precision + recall
        f1 = 2.0 * precision * recall / denominator if denominator > 0 else 0.0
        f1_values.append(float(f1))
        per_class.append(
            {
                "class_index": index,
                "class_name": str(class_name),
                "support": support,
                "predicted": predicted,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    return ExtendedMetrics(
        loss=float(metrics.loss),
        accuracy=float(metrics.accuracy),
        balanced_accuracy=float(metrics.balanced_accuracy),
        macro_f1=float(np.mean(f1_values)),
        confusion_matrix=metrics.confusion_matrix,
        per_class=per_class,
    )


def evaluate_heldout(
    *,
    classifier: Model50MClassifier,
    dataset: Dataset,
    criterion: nn.Module,
    num_classes: int,
    class_names: Sequence[str],
    batch_size: int,
    live: bool,
) -> ExtendedMetrics:
    """Evaluate exactly one held-out dataset without entering training mode."""
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        num_workers=0,
        pin_memory=classifier.device.type == "cuda",
        drop_last=False,
    )
    with torch.no_grad():
        if live:
            raw = run_finetune_epoch(
                classifier=classifier,
                loader=loader,
                criterion=criterion,
                num_classes=num_classes,
                optimizer=None,
            )
        else:
            raw = run_head_epoch(
                head=classifier.head,
                loader=loader,
                criterion=criterion,
                device=classifier.device,
                num_classes=num_classes,
                optimizer=None,
            )
    return extend_metrics(raw, class_names=class_names)
