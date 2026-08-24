"""Validated export-only configuration for the three-state Package."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from bci_dayloop.applications.three_mental_states.contract import TASKS
from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.utils.config import load_yaml, project_root, resolve_path

EXPORT_CONFIG_SCHEMA_VERSION = 1
EXPORT_CONFIG_APPLICATION = "three_mental_states"
DEFAULT_EXPORT_CONFIG_PATH = "configs/three_mental_states/export.yaml"


@dataclass(frozen=True, slots=True)
class ThreeMentalStateSourceConfig:
    backbone_checkpoint: Path
    workload_head: Path
    attention_head: Path
    emotion_head: Path

    @property
    def heads(self) -> dict[str, Path]:
        return {"workload": self.workload_head, "attention": self.attention_head, "emotion": self.emotion_head}


@dataclass(frozen=True, slots=True)
class ThreeMentalStateRuntimeConfig:
    target_sample_rate_hz: float
    window_sec: float
    step_sec: float
    patch_sec: float
    patch_stride_sec: float
    model_n_time_patches: int
    output_layer_idx: int
    aggregation: str

    def model_config(self, *, checkpoint_path: Path) -> Model50MConfig:
        return Model50MConfig(
            checkpoint_path=checkpoint_path,
            device="cpu",
            target_sample_rate=self.target_sample_rate_hz,
            window_seconds=self.window_sec,
            patch_seconds=self.patch_sec,
            patch_stride_seconds=self.patch_stride_sec,
            model_n_time_patches=self.model_n_time_patches,
            output_layer_idx=self.output_layer_idx,
            aggregation=self.aggregation,
            num_classes=3,
            head_type="linear",
        )


@dataclass(frozen=True, slots=True)
class ThreeMentalStatePackageConfig:
    output_dir: Path
    package_id: str
    package_version: str


@dataclass(frozen=True, slots=True)
class ThreeMentalStateExportConfig:
    sources: ThreeMentalStateSourceConfig
    runtime: ThreeMentalStateRuntimeConfig
    package: ThreeMentalStatePackageConfig
    config_path: Path


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"export config field {field!r} must be a mapping.")
    return value


def _required(mapping: Mapping[str, Any], field: str) -> Any:
    if field not in mapping:
        raise ValueError(f"export config is missing required field {field!r}.")
    return mapping[field]


def _string(mapping: Mapping[str, Any], field: str) -> str:
    value = _required(mapping, field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"export config field {field!r} must be a non-empty string.")
    return value.strip()


def _positive(mapping: Mapping[str, Any], field: str) -> float:
    value = _required(mapping, field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise ValueError(f"export config field {field!r} must be greater than zero.")
    return float(value)


def _positive_int(mapping: Mapping[str, Any], field: str) -> int:
    value = _required(mapping, field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"export config field {field!r} must be a positive integer.")
    return int(value)


def load_three_mental_state_export_config(path: str | Path = DEFAULT_EXPORT_CONFIG_PATH) -> ThreeMentalStateExportConfig:
    """Load export settings; all relative source paths resolve from repository root."""
    config_path = resolve_path(path, project_root())
    payload = load_yaml(config_path)
    if payload.get("schema_version") != EXPORT_CONFIG_SCHEMA_VERSION:
        raise ValueError(f"Unsupported three-state export config schema_version {payload.get('schema_version')!r}; expected 1.")
    if payload.get("application") != EXPORT_CONFIG_APPLICATION:
        raise ValueError("export config application must be 'three_mental_states'.")
    sources = _mapping(_required(payload, "sources"), field="sources")
    heads = _mapping(_required(sources, "heads"), field="sources.heads")
    if set(heads) != set(TASKS):
        raise ValueError(f"export config sources.heads must contain exactly {TASKS}; got {tuple(sorted(heads))}.")
    root = project_root()
    source_config = ThreeMentalStateSourceConfig(
        backbone_checkpoint=resolve_path(_string(sources, "backbone_checkpoint"), root),
        workload_head=resolve_path(_string(heads, "workload"), root),
        attention_head=resolve_path(_string(heads, "attention"), root),
        emotion_head=resolve_path(_string(heads, "emotion"), root),
    )
    runtime = _mapping(_required(payload, "runtime"), field="runtime")
    step = _positive(runtime, "step_sec")
    window = _positive(runtime, "window_sec")
    if step > window:
        raise ValueError("export config runtime.step_sec must not exceed runtime.window_sec.")
    aggregation = _string(runtime, "aggregation")
    if aggregation != "flatten":
        raise ValueError("export config runtime.aggregation must be 'flatten' for the three-state feature contract.")
    runtime_config = ThreeMentalStateRuntimeConfig(
        target_sample_rate_hz=_positive(runtime, "target_sample_rate_hz"),
        window_sec=window,
        step_sec=step,
        patch_sec=_positive(runtime, "patch_sec"),
        patch_stride_sec=_positive(runtime, "patch_stride_sec"),
        model_n_time_patches=_positive_int(runtime, "model_n_time_patches"),
        output_layer_idx=_positive_int(runtime, "output_layer_idx"),
        aggregation=aggregation,
    )
    package = _mapping(_required(payload, "package"), field="package")
    package_config = ThreeMentalStatePackageConfig(
        output_dir=resolve_path(_string(package, "output_dir"), root),
        package_id=_string(package, "id"),
        package_version=_string(package, "version"),
    )
    return ThreeMentalStateExportConfig(source_config, runtime_config, package_config, config_path)

