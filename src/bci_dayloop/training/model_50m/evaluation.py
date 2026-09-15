"""Held-out evaluation for the 50M Stage-1 classification workflows."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from bci_dayloop.models.model_50m.classifier import Model50MClassifier
from bci_dayloop.training.model_50m.engine import run_finetune_epoch
from bci_dayloop.training.model_50m.linear_head import EpochMetrics, run_head_epoch
from bci_dayloop.training.model_50m.types import ExtendedMetrics


def binary_auroc(
    labels: np.ndarray | torch.Tensor,
    positive_scores: np.ndarray | torch.Tensor,
) -> float:
    """Compute tie-aware binary AUROC from rank statistics.

    This is the Mann-Whitney formulation used by standard metric libraries,
    kept local so offline head evaluation does not acquire a pandas dependency.
    """

    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    if y.shape != scores.shape or y.size == 0:
        raise ValueError("AUROC labels and scores must be non-empty equal vectors.")
    if not np.isfinite(scores).all():
        raise ValueError("AUROC scores contain NaN or Inf.")
    if not set(y.tolist()).issubset({0, 1}):
        raise ValueError("Binary AUROC labels must be 0 or 1.")
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("Binary AUROC requires both labels 0 and 1.")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[y == 1].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


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


def evaluate_binary_feature_dataset(
    *,
    head: nn.Module,
    dataset: Dataset,
    subject_ids: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    class_names: Sequence[str],
    positive_class_index: int,
    batch_size: int,
) -> dict[str, Any]:
    """Evaluate cached binary features, including explicit positive-class AUROC."""

    if tuple(class_names) != ("non_awakening", "awakening"):
        raise ValueError(
            "Awakening binary evaluation requires class order "
            "['non_awakening', 'awakening']."
        )
    if positive_class_index != 1:
        raise ValueError("Awakening AUROC positive_class_index must be 1.")
    if len(subject_ids) != len(dataset):
        raise ValueError("subject_ids length must match the evaluation dataset.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    classifier_view = SimpleNamespace(head=head, device=device)
    established = evaluate_heldout(
        classifier=classifier_view,
        dataset=dataset,
        criterion=criterion,
        num_classes=2,
        class_names=class_names,
        batch_size=batch_size,
        live=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    logits_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    head.eval()
    with torch.no_grad():
        for features, labels in loader:
            logits_rows.append(
                head(features.to(device=device, dtype=torch.float32)).cpu()
            )
            label_rows.append(labels.to(torch.int64).cpu())
    logits = torch.cat(logits_rows)
    labels = torch.cat(label_rows)
    probabilities = torch.softmax(logits, dim=-1)
    predictions = logits.argmax(dim=-1)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Overall binary AUROC requires both labels 0 and 1.")
    auroc = binary_auroc(
        labels.numpy(), probabilities[:, positive_class_index].numpy()
    )

    per_subject: dict[str, dict[str, Any]] = {}
    subject_ids_cpu = subject_ids.to(torch.int64).cpu()
    for subject in sorted(set(subject_ids_cpu.tolist())):
        mask = subject_ids_cpu == subject
        subject_labels = labels[mask]
        subject_logits = logits[mask]
        subject_predictions = predictions[mask]
        confusion = torch.zeros((2, 2), dtype=torch.long)
        flat = subject_labels * 2 + subject_predictions
        confusion += torch.bincount(flat, minlength=4).reshape(2, 2)
        support = confusion.sum(dim=1)
        recalls = confusion.diag().float() / support.clamp_min(1)
        valid = support > 0
        raw = EpochMetrics(
            loss=float(criterion(subject_logits, subject_labels).item()),
            accuracy=float((subject_predictions == subject_labels).float().mean().item()),
            balanced_accuracy=float(recalls[valid].mean().item()),
            confusion_matrix=confusion.tolist(),
            per_class_recall=[
                float(recalls[index].item()) if support[index] > 0 else None
                for index in range(2)
            ],
        )
        values = extend_metrics(raw, class_names=class_names).to_dict()
        values["class_counts"] = {
            str(class_names[index]): int((subject_labels == index).sum().item())
            for index in range(2)
        }
        values["auroc"] = (
            binary_auroc(
                subject_labels.numpy(),
                torch.softmax(subject_logits, dim=-1)[:, positive_class_index].numpy(),
            )
            if set(subject_labels.tolist()) == {0, 1}
            else None
        )
        per_subject[str(subject)] = values

    return {
        **established.to_dict(),
        "auroc": auroc,
        "positive_class_index": positive_class_index,
        "class_counts": {
            str(class_names[index]): int((labels == index).sum().item())
            for index in range(2)
        },
        "per_subject": per_subject,
    }
