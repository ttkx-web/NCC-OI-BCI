from __future__ import annotations

"""Self-contained runtime packages for one shared 50M three-head predictor."""

import hashlib
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from bci_dayloop.inference.multi_head import (
    TASK_OUTPUT_DIMS,
    MultiHeadPredictor,
    _REQUIRED_SHARED_CONTRACT,
    _safe_torch_load,
)
from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.utils.config import dump_json, dump_yaml, load_yaml

from .loader import (
    _required_mapping,
    _resolve_package_file,
    _verify_sha256,
)


MULTI_HEAD_MODEL_TYPE = "model_50m_multi_head"
MULTI_HEAD_PACKAGE_SCHEMA_VERSION = 2
_TASKS = ("workload", "attention", "emotion")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_head_payload(task: str, path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    payload = _safe_torch_load(path)
    metadata = payload.get("metadata")
    state_dict = payload.get("head_state_dict")
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{task}: checkpoint metadata must be a mapping: {path}")
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"{task}: head_state_dict must be a mapping: {path}")
    normalized_state = {str(key): value for key, value in state_dict.items()}
    weight = normalized_state.get("linear.weight")
    bias = normalized_state.get("linear.bias")
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise KeyError(f"{task}: expected linear.weight and linear.bias: {path}")
    if weight.ndim != 2 or bias.ndim != 1 or tuple(bias.shape) != (weight.shape[0],):
        raise ValueError(f"{task}: invalid Linear tensor shapes: {path}")
    return dict(metadata), normalized_state


def _validate_head_for_export(
    *, task: str, path: Path, config: Model50MConfig
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    metadata, state_dict = _required_head_payload(task, path)
    for field, expected in _REQUIRED_SHARED_CONTRACT.items():
        actual = metadata.get(field)
        if actual != expected:
            raise ValueError(
                f"task={task}; field={field}; expected={expected!r}; "
                f"actual={actual!r}; checkpoint={path}."
            )
    if metadata.get("head_type", "linear") != "linear":
        raise ValueError(
            f"task={task}; field=head_type; expected='linear'; "
            f"actual={metadata.get('head_type')!r}; checkpoint={path}."
        )
    weight = state_dict["linear.weight"]
    output_dim, input_dim = (int(value) for value in weight.shape)
    if input_dim != config.classifier_input_dim:
        raise ValueError(
            f"task={task}; field=input_dim; expected={config.classifier_input_dim}; "
            f"actual={input_dim}; checkpoint={path}."
        )
    if output_dim != TASK_OUTPUT_DIMS[task]:
        raise ValueError(
            f"task={task}; field=output_dim; expected={TASK_OUTPUT_DIMS[task]}; "
            f"actual={output_dim}; checkpoint={path}."
        )
    class_names = metadata.get("class_names")
    if not isinstance(class_names, (list, tuple)) or len(class_names) != output_dim:
        raise ValueError(
            f"task={task}; field=class_names; expected {output_dim} names; "
            f"actual={class_names!r}; checkpoint={path}."
        )
    if int(metadata.get("num_classes", -1)) != output_dim:
        raise ValueError(
            f"task={task}; field=num_classes; expected={output_dim}; "
            f"actual={metadata.get('num_classes')!r}; checkpoint={path}."
        )
    return metadata, state_dict


def _runtime_head_payload(
    *, metadata: Mapping[str, Any], state_dict: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    retained = {
        key: metadata[key]
        for key in (
            *tuple(_REQUIRED_SHARED_CONTRACT),
            "head_type",
            "num_classes",
            "class_names",
            "channel_template",
            "backbone_sha256",
            "label_mapping",
        )
        if key in metadata
    }
    return {
        "format_version": 1,
        "head_state_dict": {
            key: value.detach().cpu()
            for key, value in state_dict.items()
        },
        "metadata": retained,
    }


def _preprocessing_payload(config: Model50MConfig) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "canonicalizer": {"target_unit": "uV"},
        "transform": {
            "type": "model_50m",
            "filter_enabled": bool(config.filter_enabled),
            "filter_low_hz": float(config.filter_low_hz),
            "filter_high_hz": float(config.filter_high_hz),
            "filter_order": int(config.filter_order),
            "reference_mode": config.reference_mode,
            "zscore_enabled": bool(config.zscore_enabled),
            "zscore_eps": float(config.zscore_eps),
            "missing_channel_fill_value": float(config.missing_channel_fill_value),
            "window_tolerance_seconds": float(config.window_tolerance_seconds),
        },
    }


def export_50m_multi_head_runtime_package(
    *,
    output_dir: str | Path,
    config: Model50MConfig,
    workload_head: str | Path,
    attention_head: str | Path,
    emotion_head: str | Path,
    package_id: str = "50m-three-mental-states",
    package_version: str = "1",
    step_sec: float = 2.0,
    overwrite: bool = False,
) -> Path:
    """Export one backbone and three validated runtime-only Linear heads."""
    package_dir = Path(output_dir).expanduser().resolve()
    backbone = Path(config.checkpoint_path).expanduser().resolve()
    if not backbone.is_file():
        raise FileNotFoundError(f"backbone checkpoint was not found: {backbone}")
    if config.classifier_input_dim != _REQUIRED_SHARED_CONTRACT["feature_dim"]:
        raise ValueError(
            "shared feature dim must be 65536, got "
            f"{config.classifier_input_dim}."
        )
    if step_sec <= 0 or step_sec > config.window_seconds:
        raise ValueError("step_sec must be positive and no greater than window_seconds.")
    source_paths = {
        "workload": Path(workload_head).expanduser().resolve(),
        "attention": Path(attention_head).expanduser().resolve(),
        "emotion": Path(emotion_head).expanduser().resolve(),
    }
    for task, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{task} head checkpoint was not found: {path}")
    validated = {
        task: _validate_head_for_export(task=task, path=path, config=config)
        for task, path in source_paths.items()
    }
    source_backbone_hash = _sha256_file(backbone)
    head_backbone_hashes = {
        str(metadata.get("backbone_sha256"))
        for metadata, _state in validated.values()
    }
    if head_backbone_hashes != {source_backbone_hash}:
        raise ValueError(
            "heads do not all reference the supplied backbone: "
            f"expected={source_backbone_hash}; actual={sorted(head_backbone_hashes)}."
        )
    if package_dir.exists() and not overwrite:
        raise FileExistsError(f"Runtime package directory already exists: {package_dir}")
    package_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{package_dir.name}.tmp-", dir=package_dir.parent))
    try:
        packaged_backbone = temporary / "backbone.pt"
        heads_dir = temporary / "heads"
        heads_dir.mkdir()
        shutil.copy2(backbone, packaged_backbone)
        head_manifest: dict[str, Any] = {}
        for task, (metadata, state_dict) in validated.items():
            head_path = heads_dir / f"{task}.pt"
            torch.save(_runtime_head_payload(metadata=metadata, state_dict=state_dict), head_path)
            output_dim = int(state_dict["linear.weight"].shape[0])
            head_manifest[task] = {
                "checkpoint": f"heads/{task}.pt",
                "head_type": "linear",
                "input_dim": int(state_dict["linear.weight"].shape[1]),
                "output_dim": output_dim,
                "class_names": [str(name) for name in metadata["class_names"]],
                "sha256": _sha256_file(head_path),
            }
        preprocessing_path = temporary / "preprocessing.yaml"
        dump_yaml(_preprocessing_payload(config), preprocessing_path)
        package_payload = {
            "schema_version": MULTI_HEAD_PACKAGE_SCHEMA_VERSION,
            "package": {
                "id": str(package_id),
                "version": str(package_version),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "prediction_mode": "multi_head",
            },
            "model": {
                "type": MULTI_HEAD_MODEL_TYPE,
                "family": "50m",
                "name": "50m-three-mental-states",
                "tasks": list(_TASKS),
                "d_model": int(config.d_model),
                "n_heads": int(config.n_heads),
                "depth": int(config.depth),
                "mlp_ratio": float(config.mlp_ratio),
                "dropout": float(config.dropout),
                "model_n_time_patches": int(config.model_n_time_patches),
                "patch_seconds": float(config.patch_seconds),
                "patch_stride_seconds": float(config.patch_stride_seconds),
            },
            "files": {
                "backbone": "backbone.pt",
                "heads": {task: value["checkpoint"] for task, value in head_manifest.items()},
                "preprocessing": "preprocessing.yaml",
                "metrics": "metrics.json",
                "sha256": {
                    "backbone": _sha256_file(packaged_backbone),
                    "heads": {task: value["sha256"] for task, value in head_manifest.items()},
                },
            },
            "input_contract": {
                "channel_names": list(config.standard_channels),
                "sample_rate": float(config.target_sample_rate),
                "window_sec": float(config.window_seconds),
                "num_samples": int(config.target_num_points),
                "input_unit": "uV",
                "strict_window_duration": bool(config.strict_window_duration),
            },
            "feature_contract": {
                "embedding_layer": 9,
                "output_layer_idx": int(config.output_layer_idx),
                "aggregation": config.aggregation,
                "feature_dim": int(config.classifier_input_dim),
                "num_tokens": int(config.num_tokens),
                "channel_mapping": "STANDARD_64_CHANNELS+zero_fill+channel_valid_mask",
            },
            "heads": head_manifest,
            "runtime": {"step_sec": float(step_sec)},
            "provenance": {
                "source_backbone_sha256": source_backbone_hash,
                "source_head_sha256": {task: _sha256_file(path) for task, path in source_paths.items()},
            },
        }
        dump_yaml(package_payload, temporary / "package.yaml")
        dump_json({"schema_version": 1, "export_smoke_test": None}, temporary / "metrics.json")
        if package_dir.exists():
            shutil.rmtree(package_dir)
        temporary.replace(package_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return package_dir


def _config_from_package(
    *, package_path: Path, payload: Mapping[str, Any], device: str
) -> tuple[Model50MConfig, dict[str, Any]]:
    model = _required_mapping(dict(payload), "model", source=package_path / "package.yaml")
    files = _required_mapping(dict(payload), "files", source=package_path / "package.yaml")
    contract = _required_mapping(dict(payload), "input_contract", source=package_path / "package.yaml")
    preprocessing_path = _resolve_package_file(
        package_path, str(files["preprocessing"]), logical_name="preprocessing config"
    )
    preprocessing = load_yaml(preprocessing_path)
    transform = _required_mapping(preprocessing, "transform", source=preprocessing_path)
    if transform.get("type") != "model_50m":
        raise ValueError("multi-head package requires preprocessing transform 'model_50m'.")
    channels = tuple(str(name) for name in contract["channel_names"])
    backbone = _resolve_package_file(
        package_path, str(files["backbone"]), logical_name="backbone"
    )
    config = Model50MConfig(
        checkpoint_path=backbone,
        device=device,
        target_sample_rate=float(contract["sample_rate"]),
        window_seconds=float(contract["window_sec"]),
        n_channels=len(channels),
        standard_channels=channels,
        strict_window_duration=bool(contract.get("strict_window_duration", True)),
        window_tolerance_seconds=float(transform.get("window_tolerance_seconds", 0.02)),
        patch_seconds=float(model["patch_seconds"]),
        patch_stride_seconds=float(model["patch_stride_seconds"]),
        filter_enabled=bool(transform["filter_enabled"]),
        filter_low_hz=float(transform["filter_low_hz"]),
        filter_high_hz=float(transform["filter_high_hz"]),
        filter_order=int(transform["filter_order"]),
        reference_mode=str(transform["reference_mode"]),
        zscore_enabled=bool(transform["zscore_enabled"]),
        zscore_eps=float(transform["zscore_eps"]),
        missing_channel_fill_value=float(transform["missing_channel_fill_value"]),
        d_model=int(model["d_model"]),
        n_heads=int(model["n_heads"]),
        depth=int(model["depth"]),
        mlp_ratio=float(model["mlp_ratio"]),
        dropout=float(model["dropout"]),
        model_n_time_patches=int(model["model_n_time_patches"]),
        output_layer_idx=int(_required_mapping(dict(payload), "feature_contract", source=package_path / "package.yaml")["output_layer_idx"]),
        aggregation=str(_required_mapping(dict(payload), "feature_contract", source=package_path / "package.yaml")["aggregation"]),
        num_classes=3,
        head_type="linear",
    )
    return config, files


def load_multi_head_runtime_package(
    package_path: str | Path, *, device: str = "cpu", verify_hashes: bool = True
) -> MultiHeadPredictor:
    """Load a self-contained package as the existing RawWindowPredictor."""
    package = Path(package_path).expanduser().resolve()
    manifest_path = package / "package.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"package.yaml was not found: {manifest_path}")
    payload = load_yaml(manifest_path)
    if int(payload.get("schema_version", -1)) != MULTI_HEAD_PACKAGE_SCHEMA_VERSION:
        raise ValueError("Unsupported multi-head runtime package schema version.")
    model = _required_mapping(payload, "model", source=manifest_path)
    if model.get("type") != MULTI_HEAD_MODEL_TYPE:
        raise ValueError(f"Expected model.type={MULTI_HEAD_MODEL_TYPE!r}, got {model.get('type')!r}.")
    if tuple(model.get("tasks", ())) != _TASKS:
        raise ValueError(f"Multi-head tasks must be {_TASKS}, got {model.get('tasks')!r}.")
    config, files = _config_from_package(package_path=package, payload=payload, device=device)
    feature = _required_mapping(payload, "feature_contract", source=manifest_path)
    for field, actual, expected in (
        ("feature_dim", int(feature.get("feature_dim", -1)), config.classifier_input_dim),
        ("num_tokens", int(feature.get("num_tokens", -1)), config.num_tokens),
        ("embedding_layer", int(feature.get("embedding_layer", -1)), 9),
    ):
        if actual != expected:
            raise ValueError(f"feature_contract.{field}: expected={expected}, actual={actual}.")
    heads_manifest = _required_mapping(payload, "heads", source=manifest_path)
    files_heads = _required_mapping(files, "heads", source=manifest_path)
    hashes = _required_mapping(files, "sha256", source=manifest_path)
    backbone = _resolve_package_file(package, str(files["backbone"]), logical_name="backbone")
    if verify_hashes:
        _verify_sha256(path=backbone, expected=hashes.get("backbone"), logical_name="backbone")
    paths: dict[str, Path] = {}
    for task in _TASKS:
        if task not in heads_manifest or task not in files_heads:
            raise ValueError(f"Multi-head package is missing required {task} head.")
        entry = _required_mapping(heads_manifest, task, source=manifest_path)
        path = _resolve_package_file(package, str(files_heads[task]), logical_name=f"{task} head")
        if verify_hashes:
            head_hashes = _required_mapping(hashes, "heads", source=manifest_path)
            _verify_sha256(path=path, expected=head_hashes.get(task), logical_name=f"{task} head")
        metadata, state_dict = _required_head_payload(task, path)
        actual_output = int(state_dict["linear.weight"].shape[0])
        actual_input = int(state_dict["linear.weight"].shape[1])
        if actual_input != int(entry.get("input_dim", -1)):
            raise ValueError(f"{task}: manifest input_dim={entry.get('input_dim')}, actual={actual_input}.")
        if actual_output != int(entry.get("output_dim", -1)):
            raise ValueError(f"{task}: manifest output_dim={entry.get('output_dim')}, actual={actual_output}.")
        manifest_classes = tuple(str(name) for name in entry.get("class_names", ()))
        metadata_classes = tuple(str(name) for name in metadata.get("class_names", ()))
        if manifest_classes != metadata_classes:
            raise ValueError(f"{task}: manifest class_names do not match head checkpoint metadata.")
        paths[task] = path
    return MultiHeadPredictor.from_config_and_checkpoints(
        config=config,
        workload_head=paths["workload"],
        attention_head=paths["attention"],
        emotion_head=paths["emotion"],
    )
