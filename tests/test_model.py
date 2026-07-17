from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from bci_dayloop.models.factory import ModelFactory
from bci_dayloop.models.labram_linear import LaBraMLinearAdapter


class TinyEncoder(nn.Module):
    embed_dim = 8

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, self.embed_dim)

    def forward_features(self, x, input_chans=None):
        del input_chans
        return self.projection(x.mean(dim=(-1, -2)))


def test_labram_linear_adapter_fit_predict_update_and_package(tmp_path):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(16, 2, 1, 200)).astype(np.float32)
    y = np.array([0, 1] * 8, dtype=np.int64)
    adapter = LaBraMLinearAdapter(
        ["C3", "C4"],
        n_classes=2,
        device="cpu",
        random_init=True,
        n_patches=1,
        encoder=TinyEncoder(),
        embedding_batch_size=4,
    )
    metrics = adapter.fit(
        X[:12],
        y[:12],
        validation_data=(X[12:], y[12:]),
        epochs=3,
        batch_size=4,
        patience=2,
        cache_dir=tmp_path / "cache",
    )
    probabilities = adapter.predict_proba(X[:2])
    assert probabilities.shape == (2, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert metrics["epochs_ran"] >= 1
    assert adapter.update(X[:2], y[:2])["updated"] == 2.0

    package = adapter.save(
        tmp_path / "package",
        preprocessing={"target_sample_rate": 200},
        label_map={0: "left_hand", 1: "right_hand"},
        command_map={"left_hand": "LEFT", "right_hand": "RIGHT"},
        metrics={"accuracy": 0.5},
    )
    required = {
        "head.pt", "model.yaml", "preprocessing.yaml", "label_map.json",
        "command_map.json", "metrics.json", "base_model.json",
    }
    assert required == {path.name for path in package.iterdir()}
    assert json.loads((package / "base_model.json").read_text(encoding="utf-8"))["random_init"] is True


def test_model_factory_is_dynamic():
    assert "labram-linear" in ModelFactory.list_models()


def test_missing_checkpoint_has_actionable_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="LaBraM Base checkpoint is missing"):
        LaBraMLinearAdapter(
            ["C3", "C4"],
            checkpoint=tmp_path / "missing-labram-base.pth",
            device="cpu",
            random_init=False,
            n_patches=1,
        )
