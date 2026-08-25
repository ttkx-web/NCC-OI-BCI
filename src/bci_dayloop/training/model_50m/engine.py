"""Optimizer, epoch execution, and early stopping for 50M population training."""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bci_dayloop.training.model_50m.adaptation import load_lora_state_dict, lora_state_dict
from bci_dayloop.training.model_50m.linear_head import EpochMetrics, metric_is_better, run_head_epoch


def run_finetune_epoch(
    *,
    classifier: Model50MClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    num_classes: int,
    optimizer: torch.optim.Optimizer | None,
) -> EpochMetrics:
    """Run an epoch that recomputes backbone features from token inputs."""
    is_train = optimizer is not None
    classifier.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for (
        token_inputs,
        token_channel_indices,
        token_time_indices,
        token_valid_mask,
        channel_valid_mask,
        labels,
    ) in loader:
        labels = labels.to(
            device=classifier.device,
            dtype=torch.long,
            non_blocking=True,
        )
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = forward_live_logits(
            classifier=classifier,
            token_inputs=token_inputs,
            token_channel_indices=token_channel_indices,
            token_time_indices=token_time_indices,
            token_valid_mask=token_valid_mask,
            channel_valid_mask=channel_valid_mask,
        )
        loss = criterion(logits, labels)
        if is_train:
            loss.backward()
            optimizer.step()

        predictions = logits.argmax(dim=-1)
        batch_size = int(labels.numel())
        total_loss += float(loss.item()) * batch_size
        total_correct += int((predictions == labels).sum().item())
        total_count += batch_size
        flat_indices = labels.detach().cpu() * num_classes + predictions.detach().cpu()
        confusion += torch.bincount(
            flat_indices,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)

    if total_count <= 0:
        raise RuntimeError("Empty token-input loader.")

    balanced_accuracy, per_class_recall = confusion_to_metrics(confusion)
    return EpochMetrics(
        loss=total_loss / total_count,
        accuracy=total_correct / total_count,
        balanced_accuracy=balanced_accuracy,
        confusion_matrix=confusion.tolist(),
        per_class_recall=per_class_recall,
    )


@dataclass(slots=True)
class TrainingResult:
    optimizer: torch.optim.Optimizer
    epoch_rows: list[dict[str, Any]]
    best_epoch: int
    best_value: float
    best_val_metrics: EpochMetrics
    selected_val_metrics: Any
    training_seconds: float
    best_head_state: dict[str, torch.Tensor]
    best_backbone_state: dict[str, torch.Tensor] | None
    best_lora_state: dict[str, torch.Tensor] | None


def build_optimizer(
    *, classifier, backbone, head_parameters, trainable_backbone_parameters,
    lora_parameters, lora_parameter_ids, head_lr: float, backbone_lr: float,
    lora_lr: float, weight_decay: float, partial_enabled: bool, lora_enabled: bool,
) -> torch.optim.Optimizer:
    groups = [{"params": head_parameters, "lr": head_lr}]
    if trainable_backbone_parameters:
        groups.append({"params": trainable_backbone_parameters, "lr": backbone_lr})
    if lora_parameters:
        groups.append({"params": lora_parameters, "lr": lora_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    backbone_ids = {id(p) for p in backbone.parameters()}
    head_ids = {id(p) for p in head_parameters}
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    expected = head_ids | {id(p) for p in trainable_backbone_parameters} | lora_parameter_ids
    if not head_ids or optimizer_ids != expected:
        raise RuntimeError("Optimizer parameters must be exactly the classification head and selected trainable backbone blocks.")
    if not partial_enabled and not lora_enabled and backbone_ids & optimizer_ids:
        raise RuntimeError("Frozen backbone parameters were added to the optimizer.")
    if partial_enabled and backbone_ids & optimizer_ids != {id(p) for p in trainable_backbone_parameters}:
        raise RuntimeError("Optimizer backbone scope differs from selected blocks.")
    if lora_enabled:
        missing = lora_parameter_ids - optimizer_ids
        unexpected = (backbone_ids - lora_parameter_ids) & optimizer_ids
        if missing or unexpected:
            raise RuntimeError(f"Optimizer LoRA scope differs from injected adapters: missing_lora={len(missing)}, unexpected_original_backbone={len(unexpected)}.")
    if any(not p.requires_grad for p in head_parameters):
        raise RuntimeError("Classification head parameters must require gradients.")
    return optimizer


def fit_with_early_stopping(
    *, classifier, train_loader: DataLoader, val_loader: DataLoader,
    criterion: nn.Module, num_classes: int, class_names: Sequence[str],
    optimizer: torch.optim.Optimizer, epochs: int, patience: int,
    metric_for_best: str, live: bool, partial_enabled: bool, lora_enabled: bool,
    extend_metrics: Callable[..., Any],
) -> TrainingResult:
    best_value = float("inf") if metric_for_best == "val_loss" else -float("inf")
    best_epoch = -1; best_head_state = None; best_backbone_state = None; best_lora_state = None
    best_val_metrics = None; without_improvement = 0; rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        if live:
            train_metrics = run_finetune_epoch(classifier=classifier, loader=train_loader, criterion=criterion, num_classes=num_classes, optimizer=optimizer)
        else:
            train_metrics = run_head_epoch(head=classifier.head, loader=train_loader, criterion=criterion, device=classifier.device, num_classes=num_classes, optimizer=optimizer)
        with torch.no_grad():
            if live:
                val_metrics = run_finetune_epoch(classifier=classifier, loader=val_loader, criterion=criterion, num_classes=num_classes, optimizer=None)
            else:
                val_metrics = run_head_epoch(head=classifier.head, loader=val_loader, criterion=criterion, device=classifier.device, num_classes=num_classes, optimizer=None)
        train_extended = extend_metrics(train_metrics, class_names=class_names)
        val_extended = extend_metrics(val_metrics, class_names=class_names)
        improved, current_value = metric_is_better(metric_name=metric_for_best, current=val_metrics, best_value=best_value)
        if improved:
            best_value = current_value; best_epoch = epoch
            best_head_state = deepcopy({k: v.detach().cpu() for k, v in classifier.head.state_dict().items()})
            if partial_enabled:
                best_backbone_state = deepcopy({k: v.detach().cpu() for k, v in classifier.backbone.model.state_dict().items()})
            if lora_enabled:
                best_lora_state = deepcopy(lora_state_dict(classifier.backbone.model))
            best_val_metrics = val_metrics; without_improvement = 0
        else:
            without_improvement += 1
        rows.append({"epoch": epoch, "train_loss": train_extended.loss, "train_acc": train_extended.accuracy, "train_bacc": train_extended.balanced_accuracy, "train_macro_f1": train_extended.macro_f1, "val_loss": val_extended.loss, "val_acc": val_extended.accuracy, "val_bacc": val_extended.balanced_accuracy, "val_macro_f1": val_extended.macro_f1, "is_best": improved, "epoch_seconds": time.perf_counter() - epoch_start})
        if patience > 0 and without_improvement >= patience:
            break
    if best_head_state is None or best_val_metrics is None:
        raise RuntimeError("No best population-head state was recorded.")
    classifier.head.load_state_dict(best_head_state, strict=True)
    if best_backbone_state is not None:
        classifier.backbone.model.load_state_dict(best_backbone_state, strict=True)
    if best_lora_state is not None:
        load_lora_state_dict(classifier.backbone.model, best_lora_state)
    classifier.eval()
    with torch.no_grad():
        if live:
            raw = run_finetune_epoch(classifier=classifier, loader=val_loader, criterion=criterion, num_classes=num_classes, optimizer=None)
        else:
            raw = run_head_epoch(head=classifier.head, loader=val_loader, criterion=criterion, device=classifier.device, num_classes=num_classes, optimizer=None)
    return TrainingResult(optimizer=optimizer, epoch_rows=rows, best_epoch=best_epoch, best_value=best_value, best_val_metrics=best_val_metrics, selected_val_metrics=extend_metrics(raw, class_names=class_names), training_seconds=time.perf_counter()-started, best_head_state=best_head_state, best_backbone_state=best_backbone_state, best_lora_state=best_lora_state)
