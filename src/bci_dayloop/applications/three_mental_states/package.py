"""Export and load a self-contained shared-50M multi-head Runtime Package."""
from __future__ import annotations
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import torch
from bci_dayloop.applications.three_mental_states.contract import (
    AWAKENING_CLASS_NAMES,
    AWAKENING_TASK,
    KNOWN_TASK_OUTPUT_DIMS,
    SHARED_FEATURE_CONTRACT,
    TASKS,
)
from bci_dayloop.applications.three_mental_states.predictor import (
    ThreeMentalStatePredictor,
    _metadata_backbone_sha256,
)
from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.packages.common import required_mapping, resolve_package_file, safe_torch_load, sha256_file, verify_sha256
from bci_dayloop.utils.config import dump_json, dump_yaml, load_yaml

MULTI_HEAD_MODEL_TYPE = "model_50m_multi_head"
MULTI_HEAD_PACKAGE_SCHEMA_VERSION = 2

def _head_payload(task: str, path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    payload = safe_torch_load(path); metadata, state = payload.get("metadata"), payload.get("head_state_dict")
    if not isinstance(metadata, Mapping): raise TypeError(f"{task}: checkpoint metadata must be a mapping: {path}")
    if not isinstance(state, Mapping): raise TypeError(f"{task}: head_state_dict must be a mapping: {path}")
    normalized = {str(key): value for key, value in state.items()}; weight, bias = normalized.get("linear.weight"), normalized.get("linear.bias")
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor): raise KeyError(f"{task}: expected linear.weight and linear.bias: {path}")
    if weight.ndim != 2 or bias.ndim != 1 or tuple(bias.shape) != (weight.shape[0],): raise ValueError(f"{task}: invalid Linear tensor shapes: {path}")
    return dict(metadata), normalized

def _validate_export_head(task: str, path: Path, config: Model50MConfig) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    metadata, state = _head_payload(task, path)
    for field, expected in SHARED_FEATURE_CONTRACT.items():
        actual = metadata.get(field)
        if actual is None and task == AWAKENING_TASK:
            # The Awakening trainer recorded these facts in its nested frozen-
            # backbone contract rather than duplicating every legacy flat key.
            if field in {"embedding_layer_resolved", "embedding_layer_internal_index"}:
                actual = expected
            elif field == "backbone_adaptation" and metadata.get("training_parameters", {}).get("backbone_frozen") is True:
                actual = "frozen"
            elif field == "freeze_backbone" and metadata.get("training_parameters", {}).get("backbone_frozen") is True:
                actual = True
        if actual != expected: raise ValueError(f"task={task}; field={field}; expected={expected!r}; actual={metadata.get(field)!r}; checkpoint={path}.")
    weight = state["linear.weight"]; output_dim, input_dim = map(int, weight.shape)
    if metadata.get("head_type", "linear") != "linear": raise ValueError(f"task={task}; field=head_type; expected='linear'; actual={metadata.get('head_type')!r}; checkpoint={path}.")
    if input_dim != config.classifier_input_dim: raise ValueError(f"task={task}; field=input_dim; expected={config.classifier_input_dim}; actual={input_dim}; checkpoint={path}.")
    expected_output_dim = KNOWN_TASK_OUTPUT_DIMS.get(task, int(metadata.get("num_classes", -1)))
    if output_dim != expected_output_dim: raise ValueError(f"task={task}; field=output_dim; expected={expected_output_dim}; actual={output_dim}; checkpoint={path}.")
    if not isinstance(metadata.get("class_names"), (list, tuple)) or len(metadata["class_names"]) != output_dim: raise ValueError(f"task={task}; field=class_names; expected {output_dim} names; actual={metadata.get('class_names')!r}; checkpoint={path}.")
    if int(metadata.get("num_classes", -1)) != output_dim: raise ValueError(f"task={task}; field=num_classes; expected={output_dim}; actual={metadata.get('num_classes')!r}; checkpoint={path}.")
    if tuple(metadata.get("channel_template", ())) != config.standard_channels:
        raise ValueError(f"task={task}; field=channel_template does not match the runtime channel profile; checkpoint={path}.")
    if task == AWAKENING_TASK:
        required = {
            "task_id": AWAKENING_TASK,
            "head_type": "linear",
            "aggregation": "flatten",
            "feature_dim": 65_536,
            "num_classes": 2,
            "positive_class_index": 1,
            "window_seconds": 2.0,
            "target_sample_rate": 100.0,
        }
        for field, expected in required.items():
            if metadata.get(field) != expected:
                raise ValueError(f"task=awakening; field={field}; expected={expected!r}; actual={metadata.get(field)!r}; checkpoint={path}.")
        if tuple(metadata.get("class_names", ())) != AWAKENING_CLASS_NAMES:
            raise ValueError(f"task=awakening; class_names must be {AWAKENING_CLASS_NAMES}; checkpoint={path}.")
        if tuple(metadata.get("class_order", ())) != AWAKENING_CLASS_NAMES:
            raise ValueError(f"task=awakening; class_order must be {AWAKENING_CLASS_NAMES}; checkpoint={path}.")
        preprocessing = metadata.get("preprocessing")
        if not isinstance(preprocessing, Mapping) or not preprocessing.get("version"):
            raise ValueError(f"task=awakening; preprocessing.version is required; checkpoint={path}.")
        for field, expected in {
            "window_seconds": 2.0,
            "target_sample_rate": 100.0,
            "aggregation": "flatten",
            "feature_dim": 65_536,
            "num_tokens": 128,
        }.items():
            if preprocessing.get(field) != expected:
                raise ValueError(f"task=awakening; preprocessing.{field} expected={expected!r}; actual={preprocessing.get(field)!r}; checkpoint={path}.")
    return metadata, state

def _preprocessing(config: Model50MConfig) -> dict[str, Any]:
    return {"schema_version": 1, "version": "model50m_runtime_preprocessing_v1", "canonicalizer": {"target_unit": "uV"}, "transform": {"type": "model_50m", "filter_enabled": bool(config.filter_enabled), "filter_low_hz": float(config.filter_low_hz), "filter_high_hz": float(config.filter_high_hz), "filter_order": int(config.filter_order), "reference_mode": config.reference_mode, "zscore_enabled": bool(config.zscore_enabled), "zscore_eps": float(config.zscore_eps), "missing_channel_fill_value": float(config.missing_channel_fill_value), "window_tolerance_seconds": float(config.window_tolerance_seconds)}}

def export_50m_multi_head_runtime_package(*, output_dir: str | Path, config: Model50MConfig, workload_head: str | Path, attention_head: str | Path, emotion_head: str | Path, awakening_head: str | Path | None = None, package_id: str = "50m-three-mental-states", package_version: str = "1", step_sec: float = 2.0, overwrite: bool = False) -> Path:
    package, backbone = Path(output_dir).expanduser().resolve(), Path(config.checkpoint_path).expanduser().resolve()
    if not backbone.is_file(): raise FileNotFoundError(f"backbone checkpoint was not found: {backbone}")
    if config.classifier_input_dim != SHARED_FEATURE_CONTRACT["feature_dim"]: raise ValueError(f"shared feature dim must be 65536, got {config.classifier_input_dim}.")
    if step_sec <= 0 or step_sec > config.window_seconds: raise ValueError("step_sec must be positive and no greater than window_seconds.")
    sources = {"workload": Path(workload_head).expanduser().resolve(), "attention": Path(attention_head).expanduser().resolve(), "emotion": Path(emotion_head).expanduser().resolve()}
    if awakening_head is not None:
        sources[AWAKENING_TASK] = Path(awakening_head).expanduser().resolve()
    for task, path in sources.items():
        if not path.is_file(): raise FileNotFoundError(f"{task} head checkpoint was not found: {path}")
    validated = {task: _validate_export_head(task, path, config) for task, path in sources.items()}; source_hash = sha256_file(backbone)
    referenced_hashes = {
        _metadata_backbone_sha256(task, metadata)
        for task, (metadata, _state) in validated.items()
    }
    if referenced_hashes != {source_hash}: raise ValueError("heads do not all reference the supplied backbone: expected=" + source_hash + ".")
    if package.exists() and not overwrite: raise FileExistsError(f"Runtime package directory already exists: {package}")
    package.parent.mkdir(parents=True, exist_ok=True); temporary = Path(tempfile.mkdtemp(prefix=f".{package.name}.tmp-", dir=package.parent))
    try:
        shutil.copy2(backbone, temporary / "backbone.pt"); (temporary / "heads").mkdir(); manifest_heads: dict[str, Any] = {}
        for task in sources:
            metadata, state = validated[task]
            path = temporary / "heads" / f"{task}.pt"
            retained = {key: metadata[key] for key in (*SHARED_FEATURE_CONTRACT, "task_id", "head_type", "num_classes", "class_names", "class_order", "channel_template", "backbone_sha256", "backbone_checkpoint_sha256", "label_mapping", "positive_class_index", "preprocessing", "preprocessing_hash", "channel_profile") if key in metadata}
            # Package-local normalized contract keys let all heads use one strict
            # runtime loader while provenance retains the checkpoint's real keys.
            retained.update(SHARED_FEATURE_CONTRACT)
            retained["backbone_sha256"] = source_hash
            torch.save({"format_version": 1, "head_state_dict": {key: value.detach().cpu() for key, value in state.items()}, "metadata": retained}, path)
            entry = {"checkpoint": f"heads/{task}.pt", "head_type": "linear", "input_dim": int(state["linear.weight"].shape[1]), "output_dim": int(state["linear.weight"].shape[0]), "class_names": [str(value) for value in metadata["class_names"]], "sha256": sha256_file(path)}
            if task == AWAKENING_TASK:
                entry["positive_class_index"] = 1
            manifest_heads[task] = entry
        preprocessing_path = temporary / "preprocessing.yaml"
        preprocessing_payload = _preprocessing(config)
        dump_yaml(preprocessing_payload, preprocessing_path)
        preprocessing_hash = sha256_file(preprocessing_path)
        model_name = "50m-three-states-plus-awakening" if AWAKENING_TASK in sources else "50m-three-mental-states"
        payload = {"schema_version": 2, "package": {"id": str(package_id), "version": str(package_version), "created_at_utc": datetime.now(timezone.utc).isoformat(), "prediction_mode": "multi_head"}, "model": {"type": MULTI_HEAD_MODEL_TYPE, "family": "50m", "name": model_name, "tasks": list(sources), "d_model": int(config.d_model), "n_heads": int(config.n_heads), "depth": int(config.depth), "mlp_ratio": float(config.mlp_ratio), "dropout": float(config.dropout), "model_n_time_patches": int(config.model_n_time_patches), "patch_seconds": float(config.patch_seconds), "patch_stride_seconds": float(config.patch_stride_seconds)}, "files": {"backbone": "backbone.pt", "heads": {task: value["checkpoint"] for task, value in manifest_heads.items()}, "preprocessing": "preprocessing.yaml", "metrics": "metrics.json", "sha256": {"backbone": sha256_file(temporary / "backbone.pt"), "heads": {task: value["sha256"] for task, value in manifest_heads.items()}, "preprocessing": preprocessing_hash}}, "input_contract": {"channel_names": list(config.standard_channels), "target_channel_profile": "STANDARD_64_CHANNELS", "sample_rate": float(config.target_sample_rate), "window_sec": float(config.window_seconds), "num_samples": int(config.target_num_points), "input_unit": "uV", "strict_window_duration": bool(config.strict_window_duration)}, "feature_contract": {"embedding_layer": 9, "output_layer_idx": int(config.output_layer_idx), "aggregation": config.aggregation, "feature_dim": int(config.classifier_input_dim), "num_tokens": int(config.num_tokens), "channel_mapping": "STANDARD_64_CHANNELS+zero_fill+channel_valid_mask"}, "preprocessing_contract": {"version": preprocessing_payload["version"], "sha256": preprocessing_hash}, "heads": manifest_heads, "runtime": {"step_sec": float(step_sec)}, "provenance": {"source_backbone_sha256": source_hash, "source_head_sha256": {task: sha256_file(path) for task, path in sources.items()}, "source_head_preprocessing_hash": {task: metadata.get("preprocessing_hash") for task, (metadata, _state) in validated.items()}, "source_head_preprocessing_version": {task: (metadata.get("preprocessing") or {}).get("version") if isinstance(metadata.get("preprocessing"), Mapping) else None for task, (metadata, _state) in validated.items()}}}
        dump_yaml(payload, temporary / "package.yaml"); dump_json({"schema_version": 1, "export_smoke_test": None}, temporary / "metrics.json")
        if package.exists(): shutil.rmtree(package)
        temporary.replace(package)
    except Exception:
        if temporary.exists(): shutil.rmtree(temporary)
        raise
    return package

def load_multi_head_runtime_package(package_path: str | Path, *, device: str = "cpu", verify_hashes: bool = True) -> ThreeMentalStatePredictor:
    package = Path(package_path).expanduser().resolve(); manifest = package / "package.yaml"
    if not manifest.is_file(): raise FileNotFoundError(f"package.yaml was not found: {manifest}")
    payload = load_yaml(manifest)
    if int(payload.get("schema_version", -1)) != 2: raise ValueError("Unsupported multi-head runtime package schema version.")
    model, files, contract, feature = (required_mapping(payload, key, source=manifest) for key in ("model", "files", "input_contract", "feature_contract"))
    if model.get("type") != MULTI_HEAD_MODEL_TYPE: raise ValueError(f"Expected model.type={MULTI_HEAD_MODEL_TYPE!r}, got {model.get('type')!r}.")
    raw_tasks = model.get("tasks")
    legacy_heads = payload.get("heads")
    if raw_tasks is None and isinstance(legacy_heads, Mapping) and set(legacy_heads) == set(TASKS):
        raw_tasks = list(TASKS)
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("Multi-head model.tasks must be a non-empty list.")
    if any(not isinstance(task, str) for task in raw_tasks):
        raise ValueError("Multi-head task_id values must be strings.")
    tasks = tuple(task.strip() for task in raw_tasks)
    if any(not task for task in tasks):
        raise ValueError("Multi-head task_id values must be non-empty strings.")
    if len(set(tasks)) != len(tasks):
        raise ValueError(f"Multi-head task_id values must be unique, got {tasks}.")
    preprocessing_path = resolve_package_file(package, str(files["preprocessing"]), logical_name="preprocessing config")
    preprocessing = load_yaml(preprocessing_path); transform = required_mapping(preprocessing, "transform", source=package / "preprocessing.yaml")
    if transform.get("type") != "model_50m": raise ValueError("multi-head package requires preprocessing transform 'model_50m'.")
    backbone = resolve_package_file(package, str(files["backbone"]), logical_name="backbone")
    config = Model50MConfig(checkpoint_path=backbone, device=device, target_sample_rate=float(contract["sample_rate"]), window_seconds=float(contract["window_sec"]), n_channels=len(contract["channel_names"]), standard_channels=tuple(str(item) for item in contract["channel_names"]), strict_window_duration=bool(contract.get("strict_window_duration", True)), window_tolerance_seconds=float(transform.get("window_tolerance_seconds", .02)), patch_seconds=float(model["patch_seconds"]), patch_stride_seconds=float(model["patch_stride_seconds"]), filter_enabled=bool(transform["filter_enabled"]), filter_low_hz=float(transform["filter_low_hz"]), filter_high_hz=float(transform["filter_high_hz"]), filter_order=int(transform["filter_order"]), reference_mode=str(transform["reference_mode"]), zscore_enabled=bool(transform["zscore_enabled"]), zscore_eps=float(transform["zscore_eps"]), missing_channel_fill_value=float(transform["missing_channel_fill_value"]), d_model=int(model["d_model"]), n_heads=int(model["n_heads"]), depth=int(model["depth"]), mlp_ratio=float(model["mlp_ratio"]), dropout=float(model["dropout"]), model_n_time_patches=int(model["model_n_time_patches"]), output_layer_idx=int(feature["output_layer_idx"]), aggregation=str(feature["aggregation"]), num_classes=3, head_type="linear")
    for field, actual, expected in (("feature_dim", int(feature.get("feature_dim", -1)), config.classifier_input_dim), ("num_tokens", int(feature.get("num_tokens", -1)), config.num_tokens), ("embedding_layer", int(feature.get("embedding_layer", -1)), 9)):
        if actual != expected: raise ValueError(f"feature_contract.{field}: expected={expected}, actual={actual}.")
    heads, file_heads, hashes = required_mapping(payload, "heads", source=manifest), required_mapping(files, "heads", source=manifest), required_mapping(files, "sha256", source=manifest)
    if set(heads) != set(tasks) or set(file_heads) != set(tasks):
        raise ValueError(
            f"Multi-head manifest tasks/heads/files must match exactly: tasks={tasks}, "
            f"heads={tuple(heads)}, files={tuple(file_heads)}."
        )
    if verify_hashes:
        verify_sha256(path=backbone, expected=hashes.get("backbone"), logical_name="backbone")
        if hashes.get("preprocessing") is not None:
            verify_sha256(path=preprocessing_path, expected=hashes.get("preprocessing"), logical_name="preprocessing config")
    paths: dict[str, Path] = {}
    for task in tasks:
        entry = required_mapping(heads, task, source=manifest); path = resolve_package_file(package, str(file_heads[task]), logical_name=f"{task} head")
        if verify_hashes: verify_sha256(path=path, expected=required_mapping(hashes, "heads", source=manifest).get(task), logical_name=f"{task} head")
        metadata, state = _head_payload(task, path)
        if int(state["linear.weight"].shape[1]) != int(entry.get("input_dim", -1)): raise ValueError(f"{task}: manifest input_dim={entry.get('input_dim')}, actual={int(state['linear.weight'].shape[1])}.")
        if int(state["linear.weight"].shape[0]) != int(entry.get("output_dim", -1)): raise ValueError(f"{task}: manifest output_dim={entry.get('output_dim')}, actual={int(state['linear.weight'].shape[0])}.")
        if entry.get("head_type") != "linear": raise ValueError(f"{task}: manifest head_type must be 'linear'.")
        if int(entry.get("input_dim", -1)) != config.classifier_input_dim: raise ValueError(f"{task}: feature dim does not match shared runtime contract.")
        if tuple(str(x) for x in entry.get("class_names", ())) != tuple(str(x) for x in metadata.get("class_names", ())): raise ValueError(f"{task}: manifest class_names do not match head checkpoint metadata.")
        if len(tuple(entry.get("class_names", ()))) != int(entry.get("output_dim", -1)):
            raise ValueError(f"{task}: class_names count does not match output_dim.")
        if task == AWAKENING_TASK:
            if tuple(entry.get("class_names", ())) != AWAKENING_CLASS_NAMES:
                raise ValueError(f"awakening: class_names must be {AWAKENING_CLASS_NAMES}.")
            if int(entry.get("positive_class_index", -1)) != 1:
                raise ValueError("awakening: positive_class_index must be 1.")
        paths[task] = path
    return ThreeMentalStatePredictor.from_config_and_head_checkpoints(
        config=config,
        head_checkpoints=paths,
    )
