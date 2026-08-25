from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import TensorDataset

from bci_dayloop.training.model_50m import evaluation
from bci_dayloop.training.model_50m.linear_head import EpochMetrics


def test_frozen_heldout_evaluation_preserves_order_metrics_and_eval_mode() -> None:
    head = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    classifier = SimpleNamespace(head=head, device=torch.device("cpu"))
    dataset = TensorDataset(
        torch.tensor([[3.0, 0.0], [0.0, 4.0], [2.0, 0.0]]),
        torch.tensor([0, 1, 1]),
    )
    result = evaluation.evaluate_heldout(
        classifier=classifier,
        dataset=dataset,
        criterion=torch.nn.CrossEntropyLoss(),
        num_classes=2,
        class_names=("left", "right"),
        batch_size=2,
        live=False,
    )
    assert head.training is False
    assert result.accuracy == pytest.approx(2 / 3)
    assert result.balanced_accuracy == pytest.approx(0.75)
    assert result.confusion_matrix == [[1, 0], [1, 1]]
    assert [item["class_name"] for item in result.per_class] == ["left", "right"]
    assert [item["support"] for item in result.per_class] == [1, 2]


@pytest.mark.parametrize("mode", ["partial", "lora"])
def test_live_heldout_evaluation_uses_no_grad_and_live_epoch(monkeypatch, mode) -> None:
    calls: list[bool] = []

    def fake_live_epoch(**kwargs):
        calls.append(torch.is_grad_enabled())
        assert kwargs["optimizer"] is None
        return EpochMetrics(
            loss=0.25,
            accuracy=1.0,
            balanced_accuracy=1.0,
            confusion_matrix=[[1, 0], [0, 1]],
            per_class_recall=[1.0, 1.0],
        )

    monkeypatch.setattr(evaluation, "run_finetune_epoch", fake_live_epoch)
    classifier = SimpleNamespace(head=torch.nn.Linear(1, 2), device=torch.device("cpu"))
    dataset = TensorDataset(torch.zeros((2, 1)), torch.tensor([0, 1]))
    result = evaluation.evaluate_heldout(
        classifier=classifier,
        dataset=dataset,
        criterion=torch.nn.CrossEntropyLoss(),
        num_classes=2,
        class_names=("left", "right"),
        batch_size=2,
        live=True,
    )
    assert calls == [False]
    assert result.macro_f1 == pytest.approx(1.0)
