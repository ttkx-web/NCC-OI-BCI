from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bci_dayloop.demo.model_package_discovery import discover_motor_intent_packages
from bci_dayloop.demo.motor_decoder import ModelPackageMotorIntentDecoder


def test_model_package_adapter_uses_runtime_model_and_preserves_label_mapping(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeRuntimeModel:
        def predict(self, raw_window):
            captured["raw_window"] = raw_window
            return SimpleNamespace(
                probabilities=torch.tensor([[0.10, 0.20, 0.60, 0.10]]),
                predicted_class=2,
                confidence=0.60,
            )

    fake_loaded_package = SimpleNamespace(
        package_path=Path("/tmp/fake-mi-package"),
        runtime_model=FakeRuntimeModel(),
        class_names=("left_hand", "right_hand", "feet", "tongue"),
        model_type="model_50m",
        model_name="50m-linear",
        window_sec=2.0,
        target_sample_rate=100.0,
    )
    fake_loader = types.ModuleType("bci_dayloop.packages.loader")
    fake_loader.load_runtime_package = lambda *args, **kwargs: fake_loaded_package
    fake_packages = types.ModuleType("bci_dayloop.packages")
    monkeypatch.setitem(sys.modules, "bci_dayloop.packages", fake_packages)
    monkeypatch.setitem(sys.modules, "bci_dayloop.packages.loader", fake_loader)

    decoder = ModelPackageMotorIntentDecoder("/tmp/fake-mi-package", device="cpu")
    result = decoder.predict(
        band_power={},
        rms_uv=0.0,
        samples=np.ones((3, 20), dtype=np.float32),
        sample_rate=200.0,
        channel_names=["F3", "C3", "C4"],
        unit="V",
    )

    raw_window = captured["raw_window"]
    assert raw_window.sample_rate == 200.0
    assert raw_window.channel_names == ["F3", "C3", "C4"]
    assert raw_window.unit == "V"
    assert result["label"] == "feet"
    assert result["label_cn"] == "双脚"
    assert result["probabilities"] == pytest.approx(
        {
            "left_hand": 0.10,
            "right_hand": 0.20,
            "feet": 0.60,
            "tongue": 0.10,
        }
    )
    assert result["decoder_display_name"] == "50M"


def test_discovery_filters_to_four_class_motor_imagery_packages(tmp_path: Path) -> None:
    package_root = tmp_path / "model_packages" / "mi" / "v1"
    package_root.mkdir(parents=True)
    (package_root / "package.yaml").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "model": {
                    "type": "labram",
                    "name": "labram-linear",
                    "task": "motor_imagery",
                    "dataset": "BNCI2014_001",
                    "num_classes": 4,
                    "class_names": ["left_hand", "right_hand", "feet", "tongue"],
                },
                "input_contract": {"window_sec": 2.0},
            }
        ),
        encoding="utf-8",
    )
    invalid_root = tmp_path / "runs" / "workload" / "v1"
    invalid_root.mkdir(parents=True)
    (invalid_root / "package.yaml").write_text(
        json.dumps(
            {"schema_version": 2, "model": {"task": "workload", "num_classes": 2}},
        ),
        encoding="utf-8",
    )

    options = discover_motor_intent_packages(tmp_path)
    assert len(options) == 1
    assert options[0].model_type == "labram"
    assert "LaBraM · BNCI · 2s" in options[0].label
