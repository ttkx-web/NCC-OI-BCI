from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bci_dayloop.data.hdf5_dataset import HDF5Metadata
from bci_dayloop.data.preprocessing import EEGPreprocessor
from bci_dayloop.models.base import BaseModelAdapter, ModelPreprocessor
from bci_dayloop.models.labram_linear import LaBraMLinearAdapter
from bci_dayloop.models.model_50m.adapter import Model50MAdapter
from bci_dayloop.models.model_50m.pipeline_preprocessor import Model50MPipelinePreprocessor
from bci_dayloop.utils.config import load_yaml


@dataclass(frozen=True, slots=True)
class ModelRuntimePackage:
    model: BaseModelAdapter
    preprocessor: ModelPreprocessor
    model_name: str
    class_names: tuple[str, ...]
    command_map: dict[str, str]
    label_map: dict[str, str]
    window_sec: float
    step_sec: float
    target_sample_rate: float
    is_test_head: bool
    warning_message: str | None
    package_path: Path


def _json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as handle:
        return dict(json.load(handle))


def _labels(package: Path, metadata: HDF5Metadata) -> tuple[dict[str, str], tuple[str, ...]]:
    label_map = _json(package / "label_map.json")
    if label_map:
        try:
            names = tuple(str(label_map[str(index)]) for index in range(len(label_map)))
        except KeyError as error:
            raise ValueError(f"Model package label_map has no stable numeric class order: {package / 'label_map.json'}") from error
    else:
        names = tuple(metadata.class_names)
        label_map = {str(index): name for index, name in enumerate(names)}
    if tuple(metadata.class_names) != names:
        raise ValueError(f"HDF5 class order {metadata.class_names} does not match model package class order {list(names)}")
    return label_map, names


def load_labram_runtime_package(package_path: str | Path, metadata: HDF5Metadata, *, device: str = "cpu") -> ModelRuntimePackage:
    package = Path(package_path).resolve()
    model = LaBraMLinearAdapter.from_package(package, device=device)
    preprocessing = load_yaml(package / "preprocessing.yaml")
    labels, names = _labels(package, metadata)
    model_yaml = load_yaml(package / "model.yaml")
    return ModelRuntimePackage(model, EEGPreprocessor(preprocessing), "labram-linear", names, _json(package / "command_map.json"), labels,
                               float(model_yaml.get("window_sec", 4.0)), float(model_yaml.get("step_sec", 0.5)),
                               float(preprocessing.get("target_sample_rate", metadata.sample_rate)), False, None, package)


def load_50m_runtime_package(package_path: str | Path, metadata: HDF5Metadata, *, device: str = "cpu") -> ModelRuntimePackage:
    package = Path(package_path).resolve()
    if not getattr(metadata, "channel_names", None):
        raise ValueError("50M runtime package requires metadata.channel_names")
    model_yaml = load_yaml(package / "model.yaml")
    if model_yaml.get("name") != "50m-linear":
        raise ValueError(f"Expected 50m-linear package, got {model_yaml.get('name')!r}")
    labels, names = _labels(package, metadata)
    package_names = tuple(str(item) for item in model_yaml.get("class_names", names))
    if package_names != names:
        raise ValueError(f"50M model.yaml class_names {list(package_names)} do not match label_map order {list(names)}")
    model = Model50MAdapter.from_package(package, device=device)
    preprocessor = Model50MPipelinePreprocessor(model.config, channel_names=metadata.channel_names, sample_rate=metadata.sample_rate, input_unit=metadata.unit)
    base = _json(package / "base_model.json")
    is_test_head = bool(base.get("is_test_head", "test" in str(base.get("classifier_path", "")).lower()))
    warning = base.get("warning_message")
    if is_test_head and not warning:
        warning = "仅用于链路验证，预测和置信度无准确率意义"
    return ModelRuntimePackage(model, preprocessor, "50m-linear", names, _json(package / "command_map.json"), labels,
                               float(model_yaml["window_seconds"]), float(model_yaml.get("step_sec", 0.5)),
                               float(model_yaml["target_sample_rate"]), is_test_head, warning, package)


def validate_runtime_request(runtime: ModelRuntimePackage, *, window_sec: float, step_sec: float) -> None:
    if runtime.model_name == "50m-linear":
        if window_sec != runtime.window_sec:
            raise ValueError(f"50M window_sec must be {runtime.window_sec}, got {window_sec}")
        if step_sec != runtime.step_sec:
            raise ValueError(f"50M step_sec must be {runtime.step_sec}, got {step_sec}")
