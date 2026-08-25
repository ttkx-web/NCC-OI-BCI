from __future__ import annotations

import torch

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
