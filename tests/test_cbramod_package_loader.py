from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from bci_dayloop.models.cbramod.config import (
    BCICIV2A_22_CHANNELS,
)
from bci_dayloop.packages.loader import load_runtime_package
from bci_dayloop.runtime.types import ModelOutput, RawEEGWindow


FIXTURE_PACKAGE = (
    Path(__file__).parent
    / "fixtures"
    / "cbramod_package_minimal"
)


class _LightweightBackbone:
    def __init__(self, config: object) -> None:
        self.config = config
        self.device = torch.device("cpu")


class _LightweightBackend:
    def __init__(
        self,
        *,
        backbone: _LightweightBackbone,
        classifier: nn.Module,
        config: object,
    ) -> None:
        self.backbone = backbone
        self.classifier = classifier
        self.config = config

    def predict_tensor(
        self,
        model_input: object,
        return_features: bool = False,
    ) -> ModelOutput:
        assert isinstance(model_input, dict)
        signal = model_input["signal"]
        assert isinstance(signal, torch.Tensor)

        logits = torch.tensor(
            [[0.1, 0.2, 0.7]],
            dtype=torch.float32,
        ).repeat(signal.shape[0], 1)
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = int(probabilities.argmax(dim=-1)[0])

        return ModelOutput(
            logits=logits,
            probabilities=probabilities,
            predicted_class=predicted_class,
            confidence=float(probabilities[0, predicted_class]),
            features=None,
            diagnostics={"fixture_backend": True},
        )


def test_load_minimal_cbramod_package_and_predict(
    monkeypatch,
) -> None:
    import bci_dayloop.models.cbramod.backend as backend_module
    import bci_dayloop.models.cbramod.backbone as backbone_module
    import bci_dayloop.models.cbramod.classifier as classifier_module
    import bci_dayloop.models.cbramod.runtime as runtime_module

    monkeypatch.setattr(
        backbone_module,
        "CBraModBackbone",
        _LightweightBackbone,
    )
    monkeypatch.setattr(
        backend_module,
        "CBraModBackend",
        _LightweightBackend,
    )
    monkeypatch.setattr(
        classifier_module,
        "build_cbramod_classifier",
        lambda config: nn.Linear(1, config.num_classes),
    )
    monkeypatch.setattr(
        runtime_module,
        "load_cbramod_classifier_checkpoint",
        lambda *args, **kwargs: None,
    )

    loaded = load_runtime_package(
        FIXTURE_PACKAGE,
        device="cpu",
        verify_hashes=True,
    )

    assert loaded.model_type == "cbramod"
    assert loaded.class_names == (
        "negative",
        "neutral",
        "positive",
    )

    output = loaded.runtime_model.predict(
        RawEEGWindow(
            data=np.zeros((22, 200), dtype=np.float32),
            channel_names=list(BCICIV2A_22_CHANNELS),
            sample_rate=200.0,
            unit="uV",
            window_id="minimal-fixture-window",
        )
    )
    result = {
        "prediction": output.predicted_class,
        "probabilities": output.probabilities[0].tolist(),
    }

    assert result["prediction"] == 2
    assert len(result["probabilities"]) == len(loaded.class_names)
    assert np.isfinite(result["probabilities"]).all()
    assert sum(result["probabilities"]) == pytest.approx(1.0)
