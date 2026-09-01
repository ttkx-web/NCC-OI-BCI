from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from bci_dayloop.packages import load_inference_package
from bci_dayloop.packages import inference
from bci_dayloop.runtime.types import InputContract
from bci_dayloop.utils.config import dump_yaml


def _contract() -> InputContract:
    return InputContract(("C3",), 100.0, 2.0, 200, "uV", "BCT")


def _single_manifest(path: Path, model_type: str) -> None:
    path.mkdir()
    dump_yaml({"schema_version": 2, "model": {"type": model_type}}, path / "package.yaml")


@pytest.mark.parametrize("model_type", ["model_50m", "labram", "cbramod"])
def test_single_head_types_dispatch_to_existing_loader(monkeypatch, tmp_path: Path, model_type: str) -> None:
    package = tmp_path / model_type; _single_manifest(package, model_type)
    calls: list[Path] = []
    fake = SimpleNamespace(
        runtime_model=SimpleNamespace(input_contract=_contract()),
        window_sec=2.0, step_sec=0.5, class_names=("a", "b"),
    )
    monkeypatch.setattr(inference, "load_runtime_package", lambda path, **kwargs: calls.append(Path(path)) or fake)
    loaded = load_inference_package(package)
    assert calls == [package]
    assert loaded.model_type == model_type and loaded.prediction_mode == "classification"
    assert loaded.tasks[0].task_id == "classification" and loaded.tasks[0].class_names == ("a", "b")


def test_three_state_package_dispatches_and_keeps_stable_task_order(monkeypatch, tmp_path: Path) -> None:
    package = tmp_path / "three"; package.mkdir()
    dump_yaml({
        "schema_version": 2, "package": {"prediction_mode": "multi_head"},
        "model": {"type": "model_50m_multi_head"},
        "input_contract": {"channel_names": ["C3"], "sample_rate": 100.0, "window_sec": 2.0, "num_samples": 200, "input_unit": "uV"},
        "runtime": {"step_sec": 2.0},
        "heads": {"emotion": {"class_names": ["n", "z", "p"]}, "workload": {"class_names": ["l", "h"]}, "attention": {"class_names": ["r", "n", "c"]}},
    }, package / "package.yaml")
    predictor = object()
    monkeypatch.setattr(inference, "load_multi_head_runtime_package", lambda path, **kwargs: predictor)
    loaded = load_inference_package(package)
    assert loaded.predictor is predictor
    assert loaded.window_sec == loaded.input_contract.window_sec == 2.0 and loaded.step_sec == 2.0
    assert [task.task_id for task in loaded.tasks] == ["workload", "attention", "emotion"]
    assert loaded.tasks[0].class_names == ("l", "h")


def test_unknown_or_missing_manifest_fails_before_any_loader(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="package.yaml"):
        load_inference_package(tmp_path)
    package = tmp_path / "unknown"; _single_manifest(package, "other")
    with pytest.raises(ValueError, match="Supported types"):
        load_inference_package(package)
