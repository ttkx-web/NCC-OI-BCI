from __future__ import annotations

import json

import pytest

from bci_dayloop.data.hdf5_dataset import HDF5Metadata
from bci_dayloop.models.factory import ModelFactory
from bci_dayloop.models.runtime_package import validate_runtime_request
from bci_dayloop.utils.config import dump_yaml


def metadata():
    return HDF5Metadata(200.0, ["C3", "C4"], ["left_hand", "right_hand", "feet", "tongue"], "uV", "test")


def write_package(path, name: str):
    path.mkdir()
    payload = {"name": name, "class_names": metadata().class_names}
    if name == "50m-linear":
        payload.update({"window_seconds": 10.0, "step_sec": 0.5, "target_sample_rate": 100.0})
    dump_yaml(payload, path / "model.yaml")
    dump_yaml({}, path / "preprocessing.yaml")
    (path / "label_map.json").write_text(json.dumps({str(i): value for i, value in enumerate(metadata().class_names)}), encoding="utf-8")
    (path / "command_map.json").write_text("{}", encoding="utf-8")
    (path / "base_model.json").write_text(json.dumps({"is_test_head": True}), encoding="utf-8")


def test_runtime_loaders_are_registered_and_unknown_error_lists_them(tmp_path):
    assert {"labram-linear", "50m-linear"} <= set(ModelFactory.list_runtime_loaders())
    write_package(tmp_path / "unknown", "unknown")
    with pytest.raises(ValueError, match="50m-linear"):
        ModelFactory.load_runtime_package(tmp_path / "unknown", metadata())


def test_50m_runtime_loader_uses_metadata_and_validates_contract(monkeypatch, tmp_path):
    import bci_dayloop.models.runtime_package as runtime_module

    package = tmp_path / "50m"
    write_package(package, "50m-linear")
    captured = {}

    class Adapter:
        config = object()

    class Preprocessor:
        def __init__(self, config, *, channel_names, sample_rate, input_unit):
            captured.update(channel_names=channel_names, sample_rate=sample_rate, input_unit=input_unit)

    monkeypatch.setattr(runtime_module.Model50MAdapter, "from_package", lambda path, device: Adapter())
    monkeypatch.setattr(runtime_module, "Model50MPipelinePreprocessor", Preprocessor)
    loaded = ModelFactory.load_runtime_package(package, metadata())
    assert loaded.model_name == "50m-linear"
    assert loaded.is_test_head and "链路验证" in loaded.warning_message
    assert captured == {"channel_names": ["C3", "C4"], "sample_rate": 200.0, "input_unit": "uV"}
    validate_runtime_request(loaded, window_sec=10.0, step_sec=0.5)
    with pytest.raises(ValueError, match="window_sec"):
        validate_runtime_request(loaded, window_sec=4.0, step_sec=0.5)
    with pytest.raises(ValueError, match="step_sec"):
        validate_runtime_request(loaded, window_sec=10.0, step_sec=1.0)


def test_labram_runtime_loader_returns_shared_eeg_preprocessor(monkeypatch, tmp_path):
    import bci_dayloop.models.runtime_package as runtime_module
    from bci_dayloop.data.preprocessing import EEGPreprocessor

    package = tmp_path / "labram"
    write_package(package, "labram-linear")
    dump_yaml({"target_sample_rate": 200.0, "patch_samples": 200}, package / "preprocessing.yaml")
    fake_model = object()
    monkeypatch.setattr(runtime_module.LaBraMLinearAdapter, "from_package", lambda path, device: fake_model)
    loaded = ModelFactory.load_runtime_package(package, metadata())
    assert loaded.model is fake_model
    assert isinstance(loaded.preprocessor, EEGPreprocessor)


def test_50m_loader_rejects_class_order_mismatch(monkeypatch, tmp_path):
    import bci_dayloop.models.runtime_package as runtime_module

    package = tmp_path / "bad"
    write_package(package, "50m-linear")
    (package / "label_map.json").write_text(json.dumps({"0": "right_hand", "1": "left_hand", "2": "feet", "3": "tongue"}), encoding="utf-8")
    monkeypatch.setattr(runtime_module.Model50MAdapter, "from_package", lambda path, device: object())
    with pytest.raises(ValueError, match="class order"):
        ModelFactory.load_runtime_package(package, metadata())
