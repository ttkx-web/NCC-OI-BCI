from __future__ import annotations

"""Reusable trainer for Stage-1 head-only subject adaptation.

This module intentionally trains a classifier head on frozen features.  It
contains no 50M preprocessing or checkpoint-packaging logic, so the same
trainer can be reused by population-head and personal-head scripts.

When Backbone parameters become trainable in a later stage, do not use cached
features; introduce a separate end-to-end trainer or extend this module with a
full-model training path.
"""

import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


MetricName = Literal["val_bacc", "val_acc", "val_loss"]
OptimizerName = Literal["sgd", "adamw"]
SchedulerName = Literal["none", "plateau"]


@dataclass(frozen=True, slots=True)
class ClassifierTrainingConfig:
    num_classes: int
    epochs: int = 100
    learning_rate: float = 1e-3
    optimizer: OptimizerName = "sgd"
    momentum: float = 0.0
    weight_decay: float = 1e-3
    patience: int = 15
    metric_for_best: MetricName = "val_bacc"
    scheduler: SchedulerName = "none"
    scheduler_factor: float = 0.3
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-5
    gradient_clip_norm: float | None = None
    device: str = "cpu"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if self.momentum < 0:
            raise ValueError("momentum must be non-negative.")
        if self.patience < 0:
            raise ValueError("patience must be non-negative.")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("scheduler_factor must be in (0, 1).")
        if self.scheduler_patience < 0:
            raise ValueError("scheduler_patience must be non-negative.")
        if self.scheduler_min_lr < 0:
            raise ValueError("scheduler_min_lr must be non-negative.")
        if self.gradient_clip_norm is not None:
            if self.gradient_clip_norm <= 0:
                raise ValueError("gradient_clip_norm must be positive.")


@dataclass(frozen=True, slots=True)
class ClassifierMetrics:
    loss: float
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    confusion_matrix: list[list[int]]
    per_class: list[dict[str, float | int | str | None]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "confusion_matrix": self.confusion_matrix,
            "per_class": self.per_class,
        }


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    metrics: ClassifierMetrics
    labels: list[int]
    predictions: list[int]
    confidences: list[float]
    probabilities: list[list[float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.metrics.to_dict(),
            "labels": self.labels,
            "predictions": self.predictions,
            "confidences": self.confidences,
            "probabilities": self.probabilities,
        }


@dataclass(slots=True)
class ClassifierTrainingResult:
    best_epoch: int
    best_metric_value: float
    best_state_dict: dict[str, torch.Tensor]
    selected_validation: EvaluationResult
    history: list[dict[str, Any]] = field(default_factory=list)
    training_seconds: float = 0.0
    stopped_early: bool = False

    def to_dict(self, *, include_state_dict: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "best_epoch": self.best_epoch,
            "best_metric_value": self.best_metric_value,
            "selected_validation": self.selected_validation.to_dict(),
            "history": self.history,
            "training_seconds": self.training_seconds,
            "stopped_early": self.stopped_early,
        }
        if include_state_dict:
            payload["best_state_dict"] = self.best_state_dict
        return payload


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_head_device(
    requested: str,
    *,
    feature_device: torch.device | str | None = None,
    classifier_input_dim: int | None = None,
) -> torch.device:
    """Resolve the device used for linear-head optimization.

    ``auto`` keeps the head on CUDA when features are extracted on CUDA;
    otherwise it uses CPU.  This avoids the native MPS/ANE crash observed for
    very wide Flatten heads such as Linear(327680, 4).
    """

    requested = str(requested).lower()
    feature_device = (
        torch.device(feature_device)
        if feature_device is not None
        else torch.device("cpu")
    )

    if requested == "auto":
        device = (
            feature_device
            if feature_device.type == "cuda"
            else torch.device("cpu")
        )
    else:
        device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if device.type == "mps":
        if not hasattr(torch.backends, "mps"):
            raise RuntimeError("This PyTorch build has no MPS backend.")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        if classifier_input_dim is not None and classifier_input_dim > 16_384:
            raise ValueError(
                "Refusing to train a very wide classifier on MPS. "
                f"classifier_input_dim={classifier_input_dim}. Use CPU for "
                "the head while keeping MPS for frozen feature extraction."
            )
    return device


def clone_frozen_module(module: nn.Module, *, device: torch.device) -> nn.Module:
    clone = deepcopy(module).to(device)
    clone.eval()
    for parameter in clone.parameters():
        parameter.requires_grad = False
    return clone


def reset_module_parameters(module: nn.Module) -> None:
    """Reset a head recursively for the random-initialization control."""

    reset_count = 0
    for child in module.modules():
        if child is module:
            continue
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()
            reset_count += 1
    if reset_count == 0:
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()
            reset_count += 1
    if reset_count == 0:
        raise RuntimeError("No reset_parameters() method was found in module.")


def build_optimizer(
    *,
    head: nn.Module,
    config: ClassifierTrainingConfig,
) -> torch.optim.Optimizer:
    parameters = [
        parameter for parameter in head.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("The classifier head has no trainable parameters.")

    if config.optimizer == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer!r}.")


def build_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    config: ClassifierTrainingConfig,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if config.scheduler == "none":
        return None
    if config.scheduler != "plateau":
        raise ValueError(f"Unsupported scheduler: {config.scheduler!r}.")
    mode = "min" if config.metric_for_best == "val_loss" else "max"
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
    )


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, Mapping):
        feature_key = next(
            (key for key in ("features", "feature", "x") if key in batch),
            None,
        )
        label_key = next(
            (key for key in ("labels", "label", "y") if key in batch),
            None,
        )
        if feature_key is None or label_key is None:
            raise KeyError(
                "A mapping batch must contain features/labels (or x/y). "
                f"Available keys: {list(batch.keys())}."
            )
        return batch[feature_key], batch[label_key]
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError(
        "Unsupported batch type. Expected mapping or (features, labels)."
    )


def _metric_value(metrics: ClassifierMetrics, metric_name: MetricName) -> float:
    if metric_name == "val_loss":
        return metrics.loss
    if metric_name == "val_acc":
        return metrics.accuracy
    if metric_name == "val_bacc":
        return metrics.balanced_accuracy
    raise ValueError(f"Unsupported metric_for_best: {metric_name!r}.")


def _is_better(
    *,
    current: float,
    best: float,
    metric_name: MetricName,
) -> bool:
    if metric_name == "val_loss":
        return current < best
    return current > best


def _metrics_from_confusion(
    *,
    total_loss: float,
    total_count: int,
    confusion: torch.Tensor,
    class_names: Sequence[str],
) -> ClassifierMetrics:
    if total_count <= 0:
        raise RuntimeError("Cannot compute metrics for an empty loader.")

    confusion = confusion.to(dtype=torch.long, device="cpu")
    num_classes = len(class_names)
    support = confusion.sum(dim=1)
    predicted_support = confusion.sum(dim=0)
    true_positive = confusion.diag()
    accuracy = float(true_positive.sum().item() / total_count)

    recall = true_positive.float() / support.clamp_min(1).float()
    valid_classes = support > 0
    balanced_accuracy = (
        float(recall[valid_classes].mean().item())
        if bool(valid_classes.any())
        else float("nan")
    )

    per_class: list[dict[str, float | int | str | None]] = []
    f1_values: list[float] = []
    for class_index, class_name in enumerate(class_names):
        tp = int(true_positive[class_index].item())
        true_count = int(support[class_index].item())
        predicted_count = int(predicted_support[class_index].item())
        precision = tp / predicted_count if predicted_count > 0 else 0.0
        class_recall = tp / true_count if true_count > 0 else 0.0
        denominator = precision + class_recall
        f1 = (
            2.0 * precision * class_recall / denominator
            if denominator > 0
            else 0.0
        )
        f1_values.append(float(f1))
        per_class.append(
            {
                "class_index": class_index,
                "class_name": str(class_name),
                "support": true_count,
                "predicted": predicted_count,
                "precision": float(precision),
                "recall": float(class_recall),
                "f1": float(f1),
            }
        )

    return ClassifierMetrics(
        loss=float(total_loss / total_count),
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        macro_f1=float(np.mean(f1_values)),
        confusion_matrix=confusion.tolist(),
        per_class=per_class,
    )


def run_classifier_epoch(
    *,
    head: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Sequence[str],
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float | None = None,
) -> ClassifierMetrics:
    is_train = optimizer is not None
    head.train(is_train)
    num_classes = len(class_names)
    total_loss = 0.0
    total_count = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for batch in loader:
        features, labels = _unpack_batch(batch)
        features = features.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        labels = labels.to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        ).view(-1)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = head(features)
        if logits.ndim != 2 or logits.shape[-1] != num_classes:
            raise ValueError(
                f"Expected logits [B,{num_classes}], got {tuple(logits.shape)}."
            )
        if labels.numel() != logits.shape[0]:
            raise ValueError(
                "labels batch size does not match logits batch size."
            )
        if labels.numel() == 0:
            continue
        if int(labels.min()) < 0 or int(labels.max()) >= num_classes:
            raise ValueError(
                f"Labels must be in [0,{num_classes - 1}]."
            )

        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss detected: {loss.item()}.")

        if is_train:
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(),
                    max_norm=gradient_clip_norm,
                )
            optimizer.step()

        predictions = logits.argmax(dim=-1)
        batch_size = int(labels.numel())
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

        labels_cpu = labels.detach().cpu()
        predictions_cpu = predictions.detach().cpu()
        flat = labels_cpu * num_classes + predictions_cpu
        confusion += torch.bincount(
            flat,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)

    return _metrics_from_confusion(
        total_loss=total_loss,
        total_count=total_count,
        confusion=confusion,
        class_names=class_names,
    )


@torch.no_grad()
def evaluate_classifier(
    *,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device | str,
    class_names: Sequence[str],
    criterion: nn.Module | None = None,
) -> EvaluationResult:
    device = torch.device(device)
    criterion = criterion or nn.CrossEntropyLoss()
    head.eval()
    num_classes = len(class_names)

    total_loss = 0.0
    total_count = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    labels_all: list[int] = []
    predictions_all: list[int] = []
    confidences_all: list[float] = []
    probabilities_all: list[list[float]] = []

    for batch in loader:
        features, labels = _unpack_batch(batch)
        features = features.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long).view(-1)
        logits = head(features)
        loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=-1)
        confidences, predictions = probabilities.max(dim=-1)

        batch_size = int(labels.numel())
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

        labels_cpu = labels.detach().cpu()
        predictions_cpu = predictions.detach().cpu()
        probabilities_cpu = probabilities.detach().cpu()
        confidences_cpu = confidences.detach().cpu()

        labels_all.extend(int(value) for value in labels_cpu.tolist())
        predictions_all.extend(int(value) for value in predictions_cpu.tolist())
        confidences_all.extend(float(value) for value in confidences_cpu.tolist())
        probabilities_all.extend(
            [[float(value) for value in row] for row in probabilities_cpu.tolist()]
        )

        flat = labels_cpu * num_classes + predictions_cpu
        confusion += torch.bincount(
            flat,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)

    metrics = _metrics_from_confusion(
        total_loss=total_loss,
        total_count=total_count,
        confusion=confusion,
        class_names=class_names,
    )
    return EvaluationResult(
        metrics=metrics,
        labels=labels_all,
        predictions=predictions_all,
        confidences=confidences_all,
        probabilities=probabilities_all,
    )


def train_classifier_head(
    *,
    head: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    class_names: Sequence[str],
    config: ClassifierTrainingConfig,
    criterion: nn.Module | None = None,
    verbose: bool = True,
) -> ClassifierTrainingResult:
    """Train a classifier head and restore the best validation checkpoint."""

    if len(class_names) != config.num_classes:
        raise ValueError(
            f"class_names has {len(class_names)} entries but "
            f"num_classes={config.num_classes}."
        )

    set_seed(config.seed)
    device = torch.device(config.device)
    head.to(device)
    criterion = criterion or nn.CrossEntropyLoss()
    optimizer = build_optimizer(head=head, config=config)
    scheduler = build_scheduler(optimizer=optimizer, config=config)

    best_value = (
        float("inf")
        if config.metric_for_best == "val_loss"
        else -float("inf")
    )
    best_epoch = -1
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    stopped_early = False
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = run_classifier_epoch(
            head=head,
            loader=train_loader,
            criterion=criterion,
            device=device,
            class_names=class_names,
            optimizer=optimizer,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        with torch.no_grad():
            validation_metrics = run_classifier_epoch(
                head=head,
                loader=validation_loader,
                criterion=criterion,
                device=device,
                class_names=class_names,
                optimizer=None,
            )

        current_value = _metric_value(
            validation_metrics,
            config.metric_for_best,
        )
        improved = _is_better(
            current=current_value,
            best=best_value,
            metric_name=config.metric_for_best,
        )
        if improved:
            best_value = current_value
            best_epoch = epoch
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step(current_value)

        learning_rate = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_metrics.loss,
            "train_acc": train_metrics.accuracy,
            "train_bacc": train_metrics.balanced_accuracy,
            "train_macro_f1": train_metrics.macro_f1,
            "val_loss": validation_metrics.loss,
            "val_acc": validation_metrics.accuracy,
            "val_bacc": validation_metrics.balanced_accuracy,
            "val_macro_f1": validation_metrics.macro_f1,
            "is_best": improved,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)

        if verbose:
            marker = " *" if improved else ""
            print(
                f"epoch={epoch:03d} lr={learning_rate:.6g} "
                f"train_loss={train_metrics.loss:.4f} "
                f"train_bacc={train_metrics.balanced_accuracy:.4f} "
                f"train_f1={train_metrics.macro_f1:.4f} "
                f"val_loss={validation_metrics.loss:.4f} "
                f"val_bacc={validation_metrics.balanced_accuracy:.4f} "
                f"val_f1={validation_metrics.macro_f1:.4f}{marker}",
                flush=True,
            )

        if (
            config.patience > 0
            and epochs_without_improvement >= config.patience
        ):
            stopped_early = True
            break

    if best_state_dict is None or best_epoch < 1:
        raise RuntimeError("No best classifier state was selected.")

    head.load_state_dict(best_state_dict, strict=True)
    head.to(device)
    head.eval()
    selected_validation = evaluate_classifier(
        head=head,
        loader=validation_loader,
        device=device,
        class_names=class_names,
        criterion=criterion,
    )

    return ClassifierTrainingResult(
        best_epoch=best_epoch,
        best_metric_value=float(best_value),
        best_state_dict=best_state_dict,
        selected_validation=selected_validation,
        history=history,
        training_seconds=time.perf_counter() - started,
        stopped_early=stopped_early,
    )


def compare_classifier_heads(
    *,
    population_head: nn.Module,
    personal_head: nn.Module,
    loader: DataLoader,
    device: torch.device | str,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Evaluate population and personal heads on the exact same features."""

    population = evaluate_classifier(
        head=population_head,
        loader=loader,
        device=device,
        class_names=class_names,
    )
    personal = evaluate_classifier(
        head=personal_head,
        loader=loader,
        device=device,
        class_names=class_names,
    )
    if population.labels != personal.labels:
        raise RuntimeError(
            "Population and personal evaluations used different labels."
        )
    return {
        "population": population.to_dict(),
        "personal": personal.to_dict(),
        "gain": {
            "accuracy": (
                personal.metrics.accuracy - population.metrics.accuracy
            ),
            "balanced_accuracy": (
                personal.metrics.balanced_accuracy
                - population.metrics.balanced_accuracy
            ),
            "macro_f1": (
                personal.metrics.macro_f1 - population.metrics.macro_f1
            ),
        },
    }
