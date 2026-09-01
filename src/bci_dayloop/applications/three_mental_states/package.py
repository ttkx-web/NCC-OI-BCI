"""Export and load the self-contained three-mental-state Runtime Package."""
from __future__ import annotations
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import torch
from bci_dayloop.applications.three_mental_states.contract import SHARED_FEATURE_CONTRACT, TASK_OUTPUT_DIMS, TASKS
from bci_dayloop.applications.three_mental_states.predictor import ThreeMentalStatePredictor
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
        if metadata.get(field) != expected: raise ValueError(f"task={task}; field={field}; expected={expected!r}; actual={metadata.get(field)!r}; checkpoint={path}.")
    weight = state["linear.weight"]; output_dim, input_dim = map(int, weight.shape)
    if metadata.get("head_type", "linear") != "linear": raise ValueError(f"task={task}; field=head_type; expected='linear'; actual={metadata.get('head_type')!r}; checkpoint={path}.")
    if input_dim != config.classifier_input_dim: raise ValueError(f"task={task}; field=input_dim; expected={config.classifier_input_dim}; actual={input_dim}; checkpoint={path}.")
    if output_dim != TASK_OUTPUT_DIMS[task]: raise ValueError(f"task={task}; field=output_dim; expected={TASK_OUTPUT_DIMS[task]}; actual={output_dim}; checkpoint={path}.")
    if not isinstance(metadata.get("class_names"), (list, tuple)) or len(metadata["class_names"]) != output_dim: raise ValueError(f"task={task}; field=class_names; expected {output_dim} names; actual={metadata.get('class_names')!r}; checkpoint={path}.")
    if int(metadata.get("num_classes", -1)) != output_dim: raise ValueError(f"task={task}; field=num_classes; expected={output_dim}; actual={metadata.get('num_classes')!r}; checkpoint={path}.")
    return metadata, state

def _preprocessing(config: Model50MConfig) -> dict[str, Any]:
    return {"schema_version": 1, "canonicalizer": {"target_unit": "uV"}, "transform": {"type": "model_50m", "filter_enabled": bool(config.filter_enabled), "filter_low_hz": float(config.filter_low_hz), "filter_high_hz": float(config.filter_high_hz), "filter_order": int(config.filter_order), "reference_mode": config.reference_mode, "zscore_enabled": bool(config.zscore_enabled), "zscore_eps": float(config.zscore_eps), "missing_channel_fill_value": float(config.missing_channel_fill_value), "window_tolerance_seconds": float(config.window_tolerance_seconds)}}

def export_50m_multi_head_runtime_package(*, output_dir: str | Path, config: Model50MConfig, workload_head: str | Path, attention_head: str | Path, emotion_head: str | Path, package_id: str = "50m-three-mental-states", package_version: str = "1", step_sec: float = 2.0, overwrite: bool = False) -> Path:
    package, backbone = Path(output_dir).expanduser().resolve(), Path(config.checkpoint_path).expanduser().resolve()
    if not backbone.is_file(): raise FileNotFoundError(f"backbone checkpoint was not found: {backbone}")
    if config.classifier_input_dim != SHARED_FEATURE_CONTRACT["feature_dim"]: raise ValueError(f"shared feature dim must be 65536, got {config.classifier_input_dim}.")
    if step_sec <= 0 or step_sec > config.window_seconds: raise ValueError("step_sec must be positive and no greater than window_seconds.")
    sources = {"workload": Path(workload_head).expanduser().resolve(), "attention": Path(attention_head).expanduser().resolve(), "emotion": Path(emotion_head).expanduser().resolve()}
    for task, path in sources.items():
        if not path.is_file(): raise FileNotFoundError(f"{task} head checkpoint was not found: {path}")
    validated = {task: _validate_export_head(task, path, config) for task, path in sources.items()}; source_hash = sha256_file(backbone)
    if {str(metadata.get("backbone_sha256")) for metadata, _ in validated.values()} != {source_hash}: raise ValueError("heads do not all reference the supplied backbone: expected=" + source_hash + ".")
    if package.exists() and not overwrite: raise FileExistsError(f"Runtime package directory already exists: {package}")
    package.parent.mkdir(parents=True, exist_ok=True); temporary = Path(tempfile.mkdtemp(prefix=f".{package.name}.tmp-", dir=package.parent))
    try:
        shutil.copy2(backbone, temporary / "backbone.pt"); (temporary / "heads").mkdir(); manifest_heads: dict[str, Any] = {}
        for task in TASKS:
            metadata, state = validated[task]; path = temporary / "heads" / f"{task}.pt"; retained = {key: metadata[key] for key in (*SHARED_FEATURE_CONTRACT, "head_type", "num_classes", "class_names", "channel_template", "backbone_sha256", "label_mapping") if key in metadata}; torch.save({"format_version": 1, "head_state_dict": {key: value.detach().cpu() for key, value in state.items()}, "metadata": retained}, path)
            manifest_heads[task] = {"checkpoint": f"heads/{task}.pt", "head_type": "linear", "input_dim": int(state["linear.weight"].shape[1]), "output_dim": int(state["linear.weight"].shape[0]), "class_names": [str(value) for value in metadata["class_names"]], "sha256": sha256_file(path)}
        dump_yaml(_preprocessing(config), temporary / "preprocessing.yaml")
        payload = {"schema_version": 2, "package": {"id": str(package_id), "version": str(package_version), "created_at_utc": datetime.now(timezone.utc).isoformat(), "prediction_mode": "multi_head"}, "model": {"type": MULTI_HEAD_MODEL_TYPE, "family": "50m", "name": "50m-three-mental-states", "tasks": list(TASKS), "d_model": int(config.d_model), "n_heads": int(config.n_heads), "depth": int(config.depth), "mlp_ratio": float(config.mlp_ratio), "dropout": float(config.dropout), "model_n_time_patches": int(config.model_n_time_patches), "patch_seconds": float(config.patch_seconds), "patch_stride_seconds": float(config.patch_stride_seconds)}, "files": {"backbone": "backbone.pt", "heads": {task: value["checkpoint"] for task, value in manifest_heads.items()}, "preprocessing": "preprocessing.yaml", "metrics": "metrics.json", "sha256": {"backbone": sha256_file(temporary / "backbone.pt"), "heads": {task: value["sha256"] for task, value in manifest_heads.items()}}}, "input_contract": {"channel_names": list(config.standard_channels), "sample_rate": float(config.target_sample_rate), "window_sec": float(config.window_seconds), "num_samples": int(config.target_num_points), "input_unit": "uV", "strict_window_duration": bool(config.strict_window_duration)}, "feature_contract": {"embedding_layer": 9, "output_layer_idx": int(config.output_layer_idx), "aggregation": config.aggregation, "feature_dim": int(config.classifier_input_dim), "num_tokens": int(config.num_tokens), "channel_mapping": "STANDARD_64_CHANNELS+zero_fill+channel_valid_mask"}, "heads": manifest_heads, "runtime": {"step_sec": float(step_sec)}, "provenance": {"source_backbone_sha256": source_hash, "source_head_sha256": {task: sha256_file(path) for task, path in sources.items()}}}
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
    if tuple(model.get("tasks", ())) != TASKS: raise ValueError(f"Multi-head tasks must be {TASKS}, got {model.get('tasks')!r}.")
    preprocessing = load_yaml(resolve_package_file(package, str(files["preprocessing"]), logical_name="preprocessing config")); transform = required_mapping(preprocessing, "transform", source=package / "preprocessing.yaml")
    if transform.get("type") != "model_50m": raise ValueError("multi-head package requires preprocessing transform 'model_50m'.")
    backbone = resolve_package_file(package, str(files["backbone"]), logical_name="backbone")
    config = Model50MConfig(checkpoint_path=backbone, device=device, target_sample_rate=float(contract["sample_rate"]), window_seconds=float(contract["window_sec"]), n_channels=len(contract["channel_names"]), standard_channels=tuple(str(item) for item in contract["channel_names"]), strict_window_duration=bool(contract.get("strict_window_duration", True)), window_tolerance_seconds=float(transform.get("window_tolerance_seconds", .02)), patch_seconds=float(model["patch_seconds"]), patch_stride_seconds=float(model["patch_stride_seconds"]), filter_enabled=bool(transform["filter_enabled"]), filter_low_hz=float(transform["filter_low_hz"]), filter_high_hz=float(transform["filter_high_hz"]), filter_order=int(transform["filter_order"]), reference_mode=str(transform["reference_mode"]), zscore_enabled=bool(transform["zscore_enabled"]), zscore_eps=float(transform["zscore_eps"]), missing_channel_fill_value=float(transform["missing_channel_fill_value"]), d_model=int(model["d_model"]), n_heads=int(model["n_heads"]), depth=int(model["depth"]), mlp_ratio=float(model["mlp_ratio"]), dropout=float(model["dropout"]), model_n_time_patches=int(model["model_n_time_patches"]), output_layer_idx=int(feature["output_layer_idx"]), aggregation=str(feature["aggregation"]), num_classes=3, head_type="linear")
    for field, actual, expected in (("feature_dim", int(feature.get("feature_dim", -1)), config.classifier_input_dim), ("num_tokens", int(feature.get("num_tokens", -1)), config.num_tokens), ("embedding_layer", int(feature.get("embedding_layer", -1)), 9)):
        if actual != expected: raise ValueError(f"feature_contract.{field}: expected={expected}, actual={actual}.")
    heads, file_heads, hashes = required_mapping(payload, "heads", source=manifest), required_mapping(files, "heads", source=manifest), required_mapping(files, "sha256", source=manifest)
    if verify_hashes: verify_sha256(path=backbone, expected=hashes.get("backbone"), logical_name="backbone")
    paths: dict[str, Path] = {}
    for task in TASKS:
        if task not in heads or task not in file_heads: raise ValueError(f"Multi-head package is missing required {task} head.")
        entry = required_mapping(heads, task, source=manifest); path = resolve_package_file(package, str(file_heads[task]), logical_name=f"{task} head")
        if verify_hashes: verify_sha256(path=path, expected=required_mapping(hashes, "heads", source=manifest).get(task), logical_name=f"{task} head")
        metadata, state = _head_payload(task, path)
        if int(state["linear.weight"].shape[1]) != int(entry.get("input_dim", -1)): raise ValueError(f"{task}: manifest input_dim={entry.get('input_dim')}, actual={int(state['linear.weight'].shape[1])}.")
        if int(state["linear.weight"].shape[0]) != int(entry.get("output_dim", -1)): raise ValueError(f"{task}: manifest output_dim={entry.get('output_dim')}, actual={int(state['linear.weight'].shape[0])}.")
        if tuple(str(x) for x in entry.get("class_names", ())) != tuple(str(x) for x in metadata.get("class_names", ())): raise ValueError(f"{task}: manifest class_names do not match head checkpoint metadata.")
        paths[task] = path
    return ThreeMentalStatePredictor.from_config_and_checkpoints(config=config, workload_head=paths["workload"], attention_head=paths["attention"], emotion_head=paths["emotion"])
