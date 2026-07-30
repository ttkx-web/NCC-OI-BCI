from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bci_dayloop.models.base import BaseModelAdapter, add_batch_dimension
from bci_dayloop.models.model_50m.classifier import save_classifier_checkpoint
from bci_dayloop.models.factory import ModelFactory
from bci_dayloop.models.model_50m import adapter as adapter_module
from bci_dayloop.models.model_50m.adapter import Model50MAdapter
from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.models.model_50m.pipeline_preprocessor import Model50MPipelinePreprocessor
from bci_dayloop.models.model_50m.runtime import Model50MRuntime


class FakeBackbone:
    def __init__(self, config, **kwargs):
        self.config = config


class FakeClassifier:
    def __init__(self, config, backbone):
        self.config = config
        self.backbone = backbone
        self.device = torch.device("cpu")
        self.head = torch.nn.Linear(config.classifier_input_dim, config.num_classes)

    def eval(self):
        return self

    def predict_batch(self, batch):
        features = torch.zeros((batch.batch_size, self.config.classifier_input_dim), dtype=torch.float32)
        logits = self.head(features)
        probabilities = torch.softmax(logits, dim=-1)
        return SimpleNamespace(
            probabilities=probabilities,
            backbone_ms=1.0,
            aggregation_ms=2.0,
            classifier_ms=3.0,
        )


@pytest.fixture
def fake_adapter_dependencies(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter_module, "Model50MBackbone", FakeBackbone)
    monkeypatch.setattr(adapter_module, "Model50MClassifier", FakeClassifier)
    monkeypatch.setattr(
        Model50MAdapter,
        "_build_model_batch",
        lambda self, X, channel_valid_masks=None: (
            SimpleNamespace(batch_size=np.asarray(X).shape[0]),
            0.1,
            0.2,
        ),
    )
    checkpoint = tmp_path / "backbone.pt"
    checkpoint.write_bytes(b"small test backbone")
    return checkpoint


def make_adapter(checkpoint: Path) -> Model50MAdapter:
    config = Model50MConfig(
        checkpoint_path=checkpoint,
        classifier_path=checkpoint.with_name("initial_classifier.pt"),
        device="cpu",
        aggregation="mean",
        filter_enabled=False,
    )
    seed_classifier = FakeClassifier(config, FakeBackbone(config))
    save_classifier_checkpoint(seed_classifier, config.classifier_path)
    return Model50MAdapter(
        config,
        class_names=("left_hand", "right_hand", "feet", "tongue"),
        strict_head_metadata=False,
    )


def test_adapter_is_a_complete_base_adapter_and_training_is_explicitly_unavailable(fake_adapter_dependencies):
    adapter = make_adapter(fake_adapter_dependencies)

    assert issubclass(Model50MAdapter, BaseModelAdapter)
    assert not inspect.isabstract(Model50MAdapter)
    assert adapter.model_name == "50m-linear"
    with pytest.raises(NotImplementedError, match="loading an existing classifier head and inference only"):
        adapter.fit(np.empty((1, 64, 1000), dtype=np.float32), np.array([0]))
    with pytest.raises(NotImplementedError, match="formal 50M classifier training script"):
        adapter.update(np.empty((1, 64, 1000), dtype=np.float32), np.array([0]))


def test_pipeline_preprocessor_returns_explicit_dict_and_validates_arguments():
    config = Model50MConfig(checkpoint_path="unused.pt", filter_enabled=False)
    preprocessor = Model50MPipelinePreprocessor(
        config,
        channel_names=("C3", "C4"),
        sample_rate=200.0,
        input_unit="uV",
    )
    preprocessor.preprocessor = lambda **kwargs: SimpleNamespace(
        signal=np.ones((64, 1000), dtype=np.float32),
        channel_valid_mask=np.ones(64, dtype=np.float32),
        mapped_channel_count=2,
        missing_channel_count=62,
        unknown_channel_names=(),
        notes=("test",),
    )

    value = preprocessor.transform(np.ones((2, 20), dtype=np.float32), 200.0, "uV")

    assert set(value) == {"signal", "channel_valid_mask"}
    assert add_batch_dimension(value)["signal"].shape == (1, 64, 1000)
    with pytest.raises(ValueError, match="reshape=True"):
        preprocessor.transform(np.ones((2, 20)), 200.0, "uV", reshape=False)
    with pytest.raises(ValueError, match=r"\[C,T\]"):
        preprocessor.transform(np.ones(20), 200.0, "uV")
    with pytest.raises(ValueError, match="sample_rate"):
        preprocessor.transform(np.ones((2, 20)), 0.0, "uV")


def test_adapter_accepts_batched_dict_and_rejects_missing_required_fields(fake_adapter_dependencies):
    adapter = make_adapter(fake_adapter_dependencies)
    model_input = {
        "signal": np.ones((1, 64, 1000), dtype=np.float32),
        "channel_valid_mask": np.ones((1, 64), dtype=np.float32),
    }

    probabilities = adapter.predict_proba(model_input)

    assert probabilities.shape == (1, 4)
    with pytest.raises(ValueError, match="'signal'"):
        adapter.predict_proba({"channel_valid_mask": model_input["channel_valid_mask"]})
    with pytest.raises(ValueError, match="'channel_valid_mask'"):
        adapter.predict_proba({"signal": model_input["signal"]})


def test_save_load_package_round_trip_and_metadata_rejection(fake_adapter_dependencies, tmp_path):
    adapter = make_adapter(fake_adapter_dependencies)
    package = adapter.save(tmp_path / "package", command_map={"feet": "FORWARD"})
    required = {"model.yaml", "preprocessing.yaml", "label_map.json", "command_map.json", "base_model.json", "classifier.pt"}
    assert required <= {item.name for item in package.iterdir()}

    loaded = Model50MAdapter.from_package(package, device="cpu")
    model_input = {
        "signal": np.ones((1, 64, 1000), dtype=np.float32),
        "channel_valid_mask": np.ones((1, 64), dtype=np.float32),
    }
    np.testing.assert_allclose(adapter.predict_proba(model_input), loaded.predict_proba(model_input))

    import yaml

    model_path = package / "model.yaml"
    payload = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    payload["aggregation"] = "flatten"
    model_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata mismatch"):
        adapter.load(package)


def test_factory_registers_both_models_and_runtime_uses_explicit_dict(monkeypatch):
    assert {"labram-linear", "50m-linear"} <= set(ModelFactory.list_models())
    assert {"labram-linear", "50m-linear"} <= set(ModelFactory.list_package_loaders())
    with pytest.raises(ValueError, match="50m-linear"):
        ModelFactory.create("unknown")

    captured = {}

    class Preprocessor:
        sample_rate = 200.0
        input_unit = "uV"
        last_diagnostics = SimpleNamespace(mapped_channel_count=64, missing_channel_count=0, unknown_channel_names=(), notes=())

        def transform(self, samples, sample_rate, input_unit, *, reshape=True):
            return {"signal": np.ones((64, 1000), dtype=np.float32), "channel_valid_mask": np.ones(64, dtype=np.float32)}

    class Adapter:
        last_timing = None

        def predict_proba(self, value):
            captured["value"] = value
            return np.array([[0.1, 0.2, 0.6, 0.1]], dtype=np.float32)

    runtime = Model50MRuntime(
        config=SimpleNamespace(num_classes=4),
        adapter=Adapter(),
        preprocessor=Preprocessor(),
        class_names=("left_hand", "right_hand", "feet", "tongue"),
    )
    result = runtime.predict_raw_window(np.ones((2, 20), dtype=np.float32))
    assert captured["value"]["signal"].shape == (1, 64, 1000)
    assert result.prediction == 2
    assert "last_channel_valid_mask" not in inspect.getsource(Model50MRuntime.predict_raw_window)
