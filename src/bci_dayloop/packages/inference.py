"""Manifest-dispatched, public entry point for inference callers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from bci_dayloop.applications.three_mental_states.contract import TASKS
from bci_dayloop.applications.three_mental_states.package import (
    MULTI_HEAD_MODEL_TYPE,
    load_multi_head_runtime_package,
)
from bci_dayloop.packages.common import required_mapping
from bci_dayloop.packages.loader import load_runtime_package
from bci_dayloop.runtime.types import InputContract
from bci_dayloop.utils.config import load_yaml

if TYPE_CHECKING:
    from bci_dayloop.inference.predictor import PreparedPredictor, RawWindowPredictor

SINGLE_TASK_ID = "classification"
THREE_MENTAL_STATES_PREDICTION_MODE = "multi_head"
SUPPORTED_MODEL_TYPES = ("model_50m", "labram", "cbramod", MULTI_HEAD_MODEL_TYPE)


@dataclass(frozen=True, slots=True)
class InferenceTaskSpec:
    task_id: str
    class_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LoadedInferencePackage:
    package_path: Path
    model_type: str
    prediction_mode: str
    predictor: "PreparedPredictor | RawWindowPredictor"
    input_contract: InputContract
    window_sec: float
    step_sec: float
    tasks: tuple[InferenceTaskSpec, ...]


def _package_path(package_path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    package = Path(package_path).expanduser().resolve()
    if not package.is_dir():
        raise FileNotFoundError(f"Runtime package directory was not found: {package}")
    manifest = package / "package.yaml"
    if not manifest.is_file():
        raise FileNotFoundError(f"package.yaml was not found: {manifest}")
    return package, manifest, load_yaml(manifest)


def _multi_head_contract(payload: dict[str, Any], *, manifest: Path) -> tuple[InputContract, float, tuple[InferenceTaskSpec, ...]]:
    input_payload = required_mapping(payload, "input_contract", source=manifest)
    runtime = required_mapping(payload, "runtime", source=manifest)
    heads = required_mapping(payload, "heads", source=manifest)
    try:
        contract = InputContract(
            channel_names=tuple(str(value) for value in input_payload["channel_names"]),
            sample_rate=float(input_payload["sample_rate"]),
            window_sec=float(input_payload["window_sec"]),
            num_samples=int(input_payload["num_samples"]),
            input_unit=str(input_payload["input_unit"]),
            tensor_layout="BCT",
            strict_window_duration=bool(input_payload.get("strict_window_duration", True)),
            model_input_keys=("signal", "channel_valid_mask"),
        )
        step_sec = float(runtime["step_sec"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{manifest}: invalid three-state input/runtime contract.") from error
    if step_sec <= 0 or step_sec > contract.window_sec:
        raise ValueError(f"{manifest}: runtime.step_sec must be in (0, window_sec].")
    if set(heads) != set(TASKS):
        raise ValueError(f"{manifest}: three-state heads must contain exactly {TASKS}, got {tuple(sorted(heads))}.")
    tasks: list[InferenceTaskSpec] = []
    for task in TASKS:
        entry = required_mapping(heads, task, source=manifest)
        names = entry.get("class_names")
        if not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError(f"{manifest}: {task} head must have non-empty class_names.")
        tasks.append(InferenceTaskSpec(task, tuple(names)))
    return contract, step_sec, tuple(tasks)


def load_inference_package(
    package_path: str | Path, *, device: str = "cpu", verify_hashes: bool = True
) -> LoadedInferencePackage:
    """Load any supported inference package through its existing specific loader."""
    package, manifest, payload = _package_path(package_path)
    model = required_mapping(payload, "model", source=manifest)
    model_type = str(model.get("type", ""))
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model type {model_type!r}. Supported types: {', '.join(SUPPORTED_MODEL_TYPES)}.")
    if model_type == MULTI_HEAD_MODEL_TYPE:
        package_metadata = required_mapping(payload, "package", source=manifest)
        prediction_mode = str(package_metadata.get("prediction_mode", ""))
        if prediction_mode != THREE_MENTAL_STATES_PREDICTION_MODE:
            raise ValueError(f"{manifest}: expected package.prediction_mode='multi_head', got {prediction_mode!r}.")
        contract, step_sec, tasks = _multi_head_contract(payload, manifest=manifest)
        predictor = load_multi_head_runtime_package(package, device=device, verify_hashes=verify_hashes)
        return LoadedInferencePackage(package, model_type, prediction_mode, predictor, contract, contract.window_sec, step_sec, tasks)

    loaded = load_runtime_package(package, device=device, verify_hashes=verify_hashes)
    task_id = model.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        task_id = SINGLE_TASK_ID
    return LoadedInferencePackage(
        package, model_type, "classification", loaded.runtime_model,
        loaded.runtime_model.input_contract, loaded.window_sec, loaded.step_sec,
        (InferenceTaskSpec(task_id, tuple(loaded.class_names)),),
    )
