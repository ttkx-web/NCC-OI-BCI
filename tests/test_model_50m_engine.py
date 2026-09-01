from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from bci_dayloop.training.model_50m import engine
from bci_dayloop.training.model_50m.engine import build_optimizer


def test_optimizer_groups_preserve_head_backbone_and_lora_order() -> None:
    class Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            self.partial = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    class Classifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = torch.nn.Linear(1, 1)

    backbone = Backbone()
    classifier = Classifier()
    adapter = torch.nn.Parameter(torch.ones(1), requires_grad=True)
    optimizer = build_optimizer(
        classifier=classifier, backbone=backbone,
        head_parameters=list(classifier.head.parameters()),
        trainable_backbone_parameters=[backbone.partial],
        lora_parameters=[adapter], lora_parameter_ids={id(adapter)},
        head_lr=1e-3, backbone_lr=1e-4, lora_lr=5e-4, weight_decay=1e-3,
        partial_enabled=True, lora_enabled=False,
    )
    assert isinstance(optimizer, torch.optim.AdamW)
    assert [group["lr"] for group in optimizer.param_groups] == [1e-3, 1e-4, 5e-4]
    assert all(group["weight_decay"] == 1e-3 for group in optimizer.param_groups)
    assert id(backbone.base) not in {id(p) for g in optimizer.param_groups for p in g["params"]}


def test_live_epoch_resolves_adaptation_forward(monkeypatch) -> None:
    """Keep partial/LoRA live token evaluation wired to adaptation.py."""
    class Classifier(torch.nn.Module):
        device = torch.device("cpu")

        def __init__(self) -> None:
            super().__init__()
            self.head = torch.nn.Linear(1, 2)

    classifier = Classifier()
    monkeypatch.setattr(
        engine,
        "forward_live_logits",
        lambda *, classifier, token_inputs, **_: classifier.head(token_inputs.float()),
    )
    dataset = TensorDataset(
        torch.tensor([[1.0], [-1.0]]),
        torch.zeros((2, 1), dtype=torch.long),
        torch.zeros((2, 1), dtype=torch.long),
        torch.ones((2, 1), dtype=torch.bool),
        torch.ones((2, 1), dtype=torch.bool),
        torch.tensor([0, 1]),
    )
    result = engine.run_finetune_epoch(
        classifier=classifier,
        loader=DataLoader(dataset, batch_size=2),
        criterion=torch.nn.CrossEntropyLoss(),
        num_classes=2,
        optimizer=None,
    )
    assert classifier.training is False
    assert len(result.confusion_matrix) == 2
