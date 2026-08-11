from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _bootstrap import ROOT

from bci_dayloop.data.hdf5_dataset import EEGHDF5
from bci_dayloop.models.cbramod.runtime import (
    CBraModRuntime,
    build_cbramod_runtime,
)
from bci_dayloop.runtime.types import RawEEGWindow
from bci_dayloop.utils.config import (
    dump_json,
    dump_yaml,
    load_yaml,
)


DEFAULT_COMMANDS = {
    "left_hand": "LEFT",
    "right_hand": "RIGHT",
    "feet": "FORWARD",
    "tongue": "STOP",
}


def resolve_repo_path(
    value: str | Path,
) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_json_mapping(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected JSON mapping in {path}."
        )

    return dict(payload)


def required_mapping(
    payload: Mapping[str, Any],
    key: str,
    *,
    source: Path,
) -> dict[str, Any]:
    value = payload.get(key)

    if not isinstance(value, Mapping):
        raise ValueError(
            f"{source}: field {key!r} must be a mapping."
        )

    return dict(value)

def read_training_source_trial_selection(
    *,
    report: Mapping[str, Any],
    source: Path,
) -> dict[str, str]:
    preprocessing = required_mapping(
        report,
        "preprocessing",
        source=source,
    )

    selection = preprocessing.get(
        "training_source_trial_selection"
    )

    if not isinstance(selection, Mapping):
        raise ValueError(
            f"{source}: preprocessing."
            "training_source_trial_selection is missing. "
            "Re-train with the variable-window training "
            "script before exporting this package."
        )

    policy = str(selection.get("policy", ""))
    anchor = str(selection.get("anchor", ""))

    if policy != (
        "one_contiguous_window_per_source_trial"
    ):
        raise ValueError(
            f"{source}: unsupported training source-trial "
            f"policy {policy!r}."
        )

    if anchor not in {"start", "center", "end"}:
        raise ValueError(
            f"{source}: unsupported training source-trial "
            f"anchor {anchor!r}."
        )

    return {
        "policy": policy,
        "anchor": anchor,
    }

def required_file(
    path: Path,
    *,
    name: str,
) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{name} was not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"{name} is not a file: {path}"
        )

    return path


def resolve_package_file(
    *,
    package_path: Path,
    relative_path: str,
    logical_name: str,
) -> Path:
    candidate = Path(relative_path)

    if candidate.is_absolute():
        raise ValueError(
            f"{logical_name} must use a package-relative path, "
            f"got {relative_path!r}."
        )

    resolved = (package_path / candidate).resolve()

    try:
        resolved.relative_to(package_path)
    except ValueError as error:
        raise ValueError(
            f"{logical_name} escapes package directory: "
            f"{relative_path!r}."
        ) from error

    return required_file(
        resolved,
        name=logical_name,
    )


def verify_hash(
    *,
    path: Path,
    expected_hash: str | None,
    logical_name: str,
) -> None:
    if expected_hash is None:
        return

    actual_hash = sha256_file(path)

    if actual_hash.lower() != str(
        expected_hash
    ).lower():
        raise ValueError(
            f"{logical_name} SHA-256 mismatch: "
            f"expected={expected_hash}, "
            f"actual={actual_hash}."
        )


def build_default_command_map(
    class_names: tuple[str, ...],
) -> dict[str, str]:
    return {
        class_name: DEFAULT_COMMANDS[class_name]
        for class_name in class_names
        if class_name in DEFAULT_COMMANDS
    }


def build_label_map(
    class_names: tuple[str, ...],
) -> dict[str, str]:
    return {
        str(index): class_name
        for index, class_name in enumerate(
            class_names
        )
    }


def validate_command_map(
    *,
    command_map: Mapping[str, str],
    class_names: tuple[str, ...],
) -> dict[str, str]:
    normalized = {
        str(key): str(value)
        for key, value in command_map.items()
    }

    unknown_classes = set(normalized) - set(class_names)

    if unknown_classes:
        raise ValueError(
            "command_map contains unknown classes: "
            f"{sorted(unknown_classes)}."
        )

    return normalized


def runtime_kwargs_from_training_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """
    将 train_cbramod_population_head.py 保存的训练记录转换为
    build_cbramod_runtime() 的模型配置。
    """

    preprocessing = required_mapping(
        report,
        "preprocessing",
        source=Path("training_report.json"),
    )

    return {
        "target_sample_rate": float(
            preprocessing["target_sample_rate"]
        ),
        "window_seconds": float(
            preprocessing["window_seconds"]
        ),
        "n_channels": len(
            preprocessing["standard_channels"]
        ),
        "time_segments": int(
            preprocessing["time_segments"]
        ),
        "points_per_patch": int(
            preprocessing["points_per_patch"]
        ),
        "input_unit": str(
            preprocessing["input_unit"]
        ),
        "strict_window_duration": bool(
            preprocessing.get(
                "strict_window_duration",
                True,
            )
        ),
        "window_tolerance_seconds": float(
            preprocessing.get(
                "window_tolerance_seconds",
                0.02,
            )
        ),
        "filter_enabled": bool(
            preprocessing.get(
                "filter_enabled",
                False,
            )
        ),
        "filter_low_hz": float(
            preprocessing.get(
                "filter_low_hz",
                0.1,
            )
        ),
        "filter_high_hz": float(
            preprocessing.get(
                "filter_high_hz",
                75.0,
            )
        ),
        "filter_order": int(
            preprocessing.get(
                "filter_order",
                4,
            )
        ),
        "reference_mode": str(
            preprocessing.get(
                "reference_mode",
                "none",
            )
        ),
        "normalization": str(
            preprocessing.get(
                "normalization",
                "none",
            )
        ),
        "head_type": str(
            report.get(
                "head_type",
                "official_mlp",
            )
        ),
    }


def extract_first_trial_window(
    *,
    dataset: EEGHDF5,
    session_name: str,
    window_seconds: float,
    anchor: str,
) -> RawEEGWindow:
    """
    从一个真实 HDF5 源 trial 中，按训练时相同的规则选择
    一个完整目标窗口，用于 export smoke test。

    仅用于离线 source/package 一致性验证；
    Runtime Replay 和真实设备会直接提供完整滑窗，
    不会再次执行该裁剪。
    """
    loaded = dataset.load(session_name)

    data = np.asarray(
        loaded["data"],
        dtype=np.float32,
    )

    if data.ndim != 3:
        raise ValueError(
            "Expected HDF5 data shape [N, C, T], got "
            f"{data.shape}."
        )

    if len(data) == 0:
        raise ValueError(
            f"Session {session_name!r} is empty."
        )

    if anchor not in {"start", "center", "end"}:
        raise ValueError(
            f"Unsupported direct-trial anchor: {anchor!r}."
        )

    metadata = dataset.metadata
    sample_rate = float(metadata.sample_rate)

    source_trial = data[0]
    source_samples = int(source_trial.shape[-1])
    target_samples = int(
        round(window_seconds * sample_rate)
    )

    if target_samples <= 0:
        raise ValueError(
            "Target smoke-test window has no samples."
        )

    if target_samples > source_samples:
        raise ValueError(
            f"Source trial is only "
            f"{source_samples / sample_rate:.3f}s, but "
            f"package requires {window_seconds:.3f}s. "
            "The export smoke test does not pad, concatenate, "
            "or cross source-trial boundaries."
        )

    if anchor == "start":
        start_sample = 0
    elif anchor == "center":
        start_sample = (
            source_samples - target_samples
        ) // 2
    else:  # anchor == "end"
        start_sample = source_samples - target_samples

    end_sample = start_sample + target_samples

    selected_trial = np.ascontiguousarray(
        source_trial[:, start_sample:end_sample]
    )

    return RawEEGWindow(
        data=selected_trial,
        channel_names=list(
            metadata.channel_names
        ),
        sample_rate=sample_rate,
        unit=str(metadata.unit),
        layout="CT",
        trial_id=str(
            int(loaded["trial_ids"][0])
        ),
        window_id="export_smoke_test",
        label=int(loaded["labels"][0]),
        metadata={
            "session": session_name,
            "source": (
                "export_cbramod_model_package"
            ),
            "training_source_trial_selection": {
                "policy": (
                    "one_contiguous_window_per_source_trial"
                ),
                "anchor": anchor,
                "source_samples": source_samples,
                "source_seconds": (
                    source_samples / sample_rate
                ),
                "selected_start_sample": start_sample,
                "selected_end_sample_exclusive": (
                    end_sample
                ),
                "selected_start_seconds": (
                    start_sample / sample_rate
                ),
                "selected_end_seconds": (
                    end_sample / sample_rate
                ),
                "selected_samples": target_samples,
                "selected_seconds": (
                    target_samples / sample_rate
                ),
            },
        },
    )


def predict_probabilities(
    *,
    runtime: CBraModRuntime,
    raw_window: RawEEGWindow,
) -> np.ndarray:
    output = runtime.runtime_model.predict(
        raw_window,
        return_features=False,
    )

    probabilities = (
        output.probabilities
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )

    expected_shape = (
        1,
        len(runtime.class_names),
    )

    if tuple(probabilities.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected probability shape. Expected "
            f"{expected_shape}, got "
            f"{tuple(probabilities.shape)}."
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            "Runtime prediction contains NaN or Inf."
        )

    if not np.allclose(
        probabilities.sum(axis=-1),
        1.0,
        atol=1e-5,
    ):
        raise RuntimeError(
            "Runtime probabilities do not sum to 1: "
            f"{probabilities.sum(axis=-1).tolist()}."
        )

    return probabilities


def build_package_payload(
    *,
    class_names: tuple[str, ...],
    command_map: Mapping[str, str],
    runtime_kwargs: Mapping[str, Any],
    package_id: str,
    package_version: str,
    dataset_name: str,
    step_sec: float,
    confidence_threshold: float,
    backbone_path: Path,
    classifier_path: Path,
    metrics: Mapping[str, Any],
    training_report_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """
    Returns:
        package.yaml payload
        preprocessing.yaml payload
        metrics.json payload
    """

    input_contract = {
        "channel_names": list(
            runtime_kwargs["standard_channels"]
        )
        if "standard_channels" in runtime_kwargs
        else None,
        "sample_rate": float(
            runtime_kwargs["target_sample_rate"]
        ),
        "window_sec": float(
            runtime_kwargs["window_seconds"]
        ),
        "num_samples": int(
            round(
                float(runtime_kwargs["target_sample_rate"])
                * float(runtime_kwargs["window_seconds"])
            )
        ),
        "input_unit": str(
            runtime_kwargs["input_unit"]
        ),
        "tensor_layout": "BCTP",
        "strict_window_duration": bool(
            runtime_kwargs["strict_window_duration"]
        ),
        "model_input_keys": ["signal"],
    }

    # 训练报告的 preprocessing 中包含正式使用的通道顺序。
    if input_contract["channel_names"] is None:
        raise RuntimeError(
            "CBraMod export requires standard_channels in the "
            "training report preprocessing manifest."
        )

    preprocessing_payload = {
        "schema_version": 1,
        "canonicalizer": {
            "target_unit": str(
                runtime_kwargs["input_unit"]
            ),
        },
        "transform": {
            "type": "cbramod",
            "target_sample_rate": float(
                runtime_kwargs["target_sample_rate"]
            ),
            "window_seconds": float(
                runtime_kwargs["window_seconds"]
            ),
            "n_channels": int(
                runtime_kwargs["n_channels"]
            ),
            "standard_channels": list(
                runtime_kwargs["standard_channels"]
            ),
            "time_segments": int(
                runtime_kwargs["time_segments"]
            ),
            "points_per_patch": int(
                runtime_kwargs["points_per_patch"]
            ),
            "strict_window_duration": bool(
                runtime_kwargs[
                    "strict_window_duration"
                ]
            ),
            "window_tolerance_seconds": float(
                runtime_kwargs[
                    "window_tolerance_seconds"
                ]
            ),
            "filter_enabled": bool(
                runtime_kwargs["filter_enabled"]
            ),
            "filter_low_hz": float(
                runtime_kwargs["filter_low_hz"]
            ),
            "filter_high_hz": float(
                runtime_kwargs["filter_high_hz"]
            ),
            "filter_order": int(
                runtime_kwargs["filter_order"]
            ),
            "reference_mode": str(
                runtime_kwargs["reference_mode"]
            ),
            "normalization": str(
                runtime_kwargs["normalization"]
            ),
        },
    }

    package_payload = {
        "schema_version": 2,
        "package": {
            "id": package_id,
            "version": package_version,
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "is_test_head": False,
            "warning_message": None,
        },
        "model": {
            "type": "cbramod",
            "name": "cbramod-frozen-head",
            "architecture": "cbramod_iclr2025",
            "task": "motor_imagery",
            "dataset": dataset_name,
            "num_classes": len(class_names),
            "class_names": list(class_names),
            "head_type": str(
                runtime_kwargs["head_type"]
            ),
            "freeze_encoder": True,
            "backbone_output_dim": 200,
            "classifier_type": (
                "official_mlp_frozen_backbone"
                if runtime_kwargs["head_type"]
                == "official_mlp"
                else "linear_frozen_backbone"
            ),
        },
        "files": {
            "backbone": "backbone.pt",
            "classifier": "classifier.pt",
            "preprocessing": "preprocessing.yaml",
            "metrics": "metrics.json",
            "label_map": "label_map.json",
            "sha256": {
                "backbone": sha256_file(
                    backbone_path
                ),
                "classifier": sha256_file(
                    classifier_path
                ),
            },
        },
        "input_contract": input_contract,
        "runtime": {
            "step_sec": float(step_sec),
            "confidence_threshold": float(
                confidence_threshold
            ),
            "command_map": {
                str(key): str(value)
                for key, value in command_map.items()
            },
        },
        "adaptation": {
            "offline": {
                "type": "none",
                "subject_id": None,
                "head_type": "population",
            },
            "online": {
                "type": "none",
            },
        },
        "provenance": {
            "source_model": "CBraMod",
            "source_backbone_filename": (
                backbone_path.name
            ),
            "source_backbone_sha256": sha256_file(
                backbone_path
            ),
            "source_classifier_filename": (
                classifier_path.name
            ),
            "source_classifier_sha256": sha256_file(
                classifier_path
            ),
            "source_training_report": (
                training_report_path.name
            ),
        },
    }

    metrics_payload = dict(metrics)
    metrics_payload.update(
        {
            "model_name": "cbramod-frozen-head",
            "classifier_type": (
                package_payload["model"][
                    "classifier_type"
                ]
            ),
            "backbone_frozen": True,
        }
    )

    return (
        package_payload,
        preprocessing_payload,
        metrics_payload,
    )


def load_package_runtime_for_smoke_test(
    *,
    package_path: Path,
    device: str,
    verify_hashes: bool,
) -> CBraModRuntime:
    """
    当前仅供本导出脚本内部验证。

    后续应将同样逻辑迁入 packages/loader.py 的
    _load_cbramod_package()，使 replay_offline.py 和 Streamlit
    能通过统一 load_runtime_package() 加载。
    """

    package_yaml_path = required_file(
        package_path / "package.yaml",
        name="package.yaml",
    )

    package_payload = load_yaml(package_yaml_path)

    if int(
        package_payload.get(
            "schema_version",
            -1,
        )
    ) != 2:
        raise ValueError(
            "Unsupported package schema. Expected 2."
        )

    model = required_mapping(
        package_payload,
        "model",
        source=package_yaml_path,
    )

    files = required_mapping(
        package_payload,
        "files",
        source=package_yaml_path,
    )

    contract = required_mapping(
        package_payload,
        "input_contract",
        source=package_yaml_path,
    )

    if model.get("type") != "cbramod":
        raise ValueError(
            "Expected package model.type='cbramod', got "
            f"{model.get('type')!r}."
        )

    backbone_path = resolve_package_file(
        package_path=package_path,
        relative_path=str(files["backbone"]),
        logical_name="package backbone",
    )

    classifier_path = resolve_package_file(
        package_path=package_path,
        relative_path=str(files["classifier"]),
        logical_name="package classifier",
    )

    preprocessing_path = resolve_package_file(
        package_path=package_path,
        relative_path=str(files["preprocessing"]),
        logical_name="package preprocessing config",
    )

    hashes = files.get("sha256", {})

    if not isinstance(hashes, Mapping):
        raise ValueError(
            "package files.sha256 must be a mapping."
        )

    if verify_hashes:
        verify_hash(
            path=backbone_path,
            expected_hash=hashes.get("backbone"),
            logical_name="package backbone",
        )

        verify_hash(
            path=classifier_path,
            expected_hash=hashes.get("classifier"),
            logical_name="package classifier",
        )

    preprocessing = load_yaml(preprocessing_path)

    transform = required_mapping(
        preprocessing,
        "transform",
        source=preprocessing_path,
    )

    if transform.get("type") != "cbramod":
        raise ValueError(
            "Expected preprocessing transform type "
            f"'cbramod', got {transform.get('type')!r}."
        )

    class_names = tuple(
        str(name)
        for name in model["class_names"]
    )

    if len(class_names) != int(
        model["num_classes"]
    ):
        raise ValueError(
            "model.class_names length does not match "
            "model.num_classes."
        )

    standard_channels = tuple(
        str(name)
        for name in transform["standard_channels"]
    )

    if tuple(
        str(name)
        for name in contract["channel_names"]
    ) != standard_channels:
        raise ValueError(
            "package input_contract.channel_names does not "
            "match preprocessing transform standard_channels."
        )

    return build_cbramod_runtime(
        checkpoint_path=backbone_path,
        classifier_path=classifier_path,
        class_names=class_names,
        device=device,

        target_sample_rate=float(
            contract["sample_rate"]
        ),
        window_seconds=float(
            contract["window_sec"]
        ),
        n_channels=int(transform["n_channels"]),
        time_segments=int(
            transform["time_segments"]
        ),
        points_per_patch=int(
            transform["points_per_patch"]
        ),
        input_unit=str(contract["input_unit"]),
        strict_window_duration=bool(
            contract.get(
                "strict_window_duration",
                True,
            )
        ),
        window_tolerance_seconds=float(
            transform.get(
                "window_tolerance_seconds",
                0.02,
            )
        ),
        filter_enabled=bool(
            transform.get(
                "filter_enabled",
                False,
            )
        ),
        filter_low_hz=float(
            transform.get(
                "filter_low_hz",
                0.1,
            )
        ),
        filter_high_hz=float(
            transform.get(
                "filter_high_hz",
                75.0,
            )
        ),
        filter_order=int(
            transform.get(
                "filter_order",
                4,
            )
        ),
        reference_mode=str(
            transform.get(
                "reference_mode",
                "none",
            )
        ),
        normalization=str(
            transform.get(
                "normalization",
                "none",
            )
        ),
        head_type=str(model["head_type"]),
    )


def replace_directory_atomically(
    *,
    temporary_path: Path,
    output_path: Path,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output package already exists: {output_path}. "
            "Pass --overwrite to replace it."
        )

    backup_path = output_path.with_name(
        f"{output_path.name}.backup-"
        f"{int(time.time())}"
    )

    if output_path.exists():
        output_path.replace(backup_path)

    try:
        temporary_path.replace(output_path)
    except Exception:
        if (
            backup_path.exists()
            and not output_path.exists()
        ):
            backup_path.replace(output_path)

        raise
    else:
        if backup_path.exists():
            shutil.rmtree(backup_path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a self-contained CBRaMod Runtime "
            "Model Package."
        )
    )

    parser.add_argument(
        "--data",
        default=(
            "data/processed/bnci2014_001/"
            "subject_01.h5"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/backbones/cbramod/"
            "pretrained_weights.pth"
        ),
    )

    parser.add_argument(
        "--classifier",
        default=(
            "checkpoints/heads/stage1/"
            "bnci2014_001/subject_01/"
            "cbramod/4s_flatten/head.pt"
        ),
    )

    parser.add_argument(
        "--training-report",
        default=(
            "runs/stage1/bnci2014_001/"
            "subject_01/cbramod/4s_flatten/"
            "training_report.json"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "model_packages/stage1/"
            "bnci2014_001/subject_01/"
            "cbramod/4s_flatten/v1"
        ),
    )

    parser.add_argument(
        "--session",
        default="1test",
        help=(
            "HDF5 session used for post-export "
            "smoke testing."
        ),
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "mps"),
    )

    parser.add_argument(
        "--step-sec",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.55,
    )

    parser.add_argument(
        "--command-map-json",
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
    )

    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    if args.step_sec <= 0:
        raise ValueError(
            "--step-sec must be positive."
        )

    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError(
            "--confidence-threshold must be in [0, 1]."
        )

    data_path = resolve_repo_path(args.data)
    checkpoint_path = resolve_repo_path(args.checkpoint)
    classifier_path = resolve_repo_path(args.classifier)
    training_report_path = resolve_repo_path(
        args.training_report
    )
    output_path = resolve_repo_path(args.output)

    command_map_path = (
        resolve_repo_path(args.command_map_json)
        if args.command_map_json is not None
        else None
    )

    for name, path in (
        ("HDF5 data", data_path),
        ("CBraMod backbone checkpoint", checkpoint_path),
        ("CBraMod classifier checkpoint", classifier_path),
        ("CBraMod training report", training_report_path),
    ):
        required_file(path, name=name)

    if command_map_path is not None:
        required_file(
            command_map_path,
            name="command-map JSON",
        )

    dataset = EEGHDF5(data_path)
    metadata = dataset.metadata

    class_names = tuple(
        str(name)
        for name in metadata.class_names
    )

    if len(class_names) != 4:
        raise ValueError(
            "CBraMod population package currently expects "
            "four BNCI2014_001 classes, got "
            f"{list(class_names)}."
        )

    report = load_json_mapping(training_report_path)

    report_class_names = tuple(
        str(name)
        for name in report.get(
            "class_names",
            [],
        )
    )

    if report_class_names != class_names:
        raise ValueError(
            "Training report class_names does not match "
            "HDF5 metadata class_names: "
            f"report={report_class_names}, "
            f"dataset={class_names}."
        )

    runtime_kwargs = runtime_kwargs_from_training_report(
        report
    )

    # 补回标准通道顺序；它是模型输入契约的一部分。
    preprocessing_manifest = required_mapping(
        report,
        "preprocessing",
        source=training_report_path,
    )

    runtime_kwargs["standard_channels"] = tuple(
        str(name)
        for name in preprocessing_manifest[
            "standard_channels"
        ]
    )

    training_source_trial_selection = (
        read_training_source_trial_selection(
            report=report,
            source=training_report_path,
        )
    )

    window_tag = (
        f"{float(runtime_kwargs['window_seconds']):g}s"
    )

    if len(runtime_kwargs["standard_channels"]) != int(
        runtime_kwargs["n_channels"]
    ):
        raise ValueError(
            "Training report standard_channels length does "
            "not match n_channels."
        )

    command_map = (
        load_json_mapping(command_map_path)
        if command_map_path is not None
        else build_default_command_map(class_names)
    )

    command_map = validate_command_map(
        command_map=command_map,
        class_names=class_names,
    )

    final_test = required_mapping(
        report,
        "target_final_test",
        source=training_report_path,
    )

    package_metrics = {
        "training_report_filename": (
            training_report_path.name
        ),
        "training_report_sha256": sha256_file(
            training_report_path
        ),
        "best_epoch": report.get("best_epoch"),
        "population_validation": report.get(
            "population_validation",
            {},
        ),
        "target_final_test": final_test,
        "preprocessing_hash": report.get(
            "preprocessing_hash"
        ),
        "backbone_sha256": report.get(
            "backbone_sha256"
        ),
        "classifier_source_sha256": sha256_file(
            classifier_path
        ),
        "training_source_trial_selection": (
            training_source_trial_selection
        ),
    }

    print("=" * 72)
    print("Export CBRaMod Runtime Model Package")
    print("=" * 72)
    print("data:", data_path)
    print("backbone:", checkpoint_path)
    print("classifier:", classifier_path)
    print("training report:", training_report_path)
    print("output:", output_path)
    print("class_names:", class_names)
    print("command_map:", command_map)
    print()

    # 导出前直接从训练产物加载一次，先确认源权重本身可用。
    source_runtime = build_cbramod_runtime(
        checkpoint_path=checkpoint_path,
        classifier_path=classifier_path,
        class_names=class_names,
        device=args.device,
        **{
            key: value
            for key, value in runtime_kwargs.items()
            if key != "standard_channels"
        },
    )

    raw_window = extract_first_trial_window(
        dataset=dataset,
        session_name=args.session,
        window_seconds=float(
            runtime_kwargs["window_seconds"]
        ),
        anchor=training_source_trial_selection[
            "anchor"
        ],
    )

    source_probabilities = predict_probabilities(
        runtime=source_runtime,
        raw_window=raw_window,
    )

    temporary_path = output_path.with_name(
        f".{output_path.name}.tmp-{os.getpid()}"
    )

    if temporary_path.exists():
        shutil.rmtree(temporary_path)

    temporary_path.mkdir(
        parents=True,
        exist_ok=False,
    )

    try:
        packaged_backbone = (
            temporary_path / "backbone.pt"
        )
        packaged_classifier = (
            temporary_path / "classifier.pt"
        )

        shutil.copy2(
            checkpoint_path,
            packaged_backbone,
        )

        shutil.copy2(
            classifier_path,
            packaged_classifier,
        )

        (
            package_payload,
            preprocessing_payload,
            metrics_payload,
        ) = build_package_payload(
            class_names=class_names,
            command_map=command_map,
            runtime_kwargs=runtime_kwargs,
            package_id=(
                "cbramod_bnci2014_001_"
                f"subject_{int(report['target_subject']):02d}_"
                f"population_{window_tag}_frozen_head"
            ),
            package_version=output_path.name,
            dataset_name=str(metadata.dataset_name),
            step_sec=args.step_sec,
            confidence_threshold=(
                args.confidence_threshold
            ),
            backbone_path=packaged_backbone,
            classifier_path=packaged_classifier,
            metrics=package_metrics,
            training_report_path=training_report_path,
        )

        dump_yaml(
            preprocessing_payload,
            temporary_path / "preprocessing.yaml",
        )

        dump_json(
            build_label_map(class_names),
            temporary_path / "label_map.json",
        )

        dump_json(
            metrics_payload,
            temporary_path / "metrics.json",
        )

        dump_yaml(
            package_payload,
            temporary_path / "package.yaml",
        )

        smoke_result: dict[str, Any] | None = None

        if not args.skip_smoke_test:
            package_runtime = (
                load_package_runtime_for_smoke_test(
                    package_path=temporary_path,
                    device=args.device,
                    verify_hashes=True,
                )
            )

            package_probabilities = predict_probabilities(
                runtime=package_runtime,
                raw_window=raw_window,
            )

            if not np.allclose(
                source_probabilities,
                package_probabilities,
                rtol=1e-5,
                atol=1e-6,
            ):
                raise RuntimeError(
                    "Source RuntimeModel and package RuntimeModel "
                    "produced inconsistent probabilities. "
                    f"source={source_probabilities.tolist()}, "
                    f"package={package_probabilities.tolist()}."
                )

            prediction = int(
                package_probabilities[0].argmax()
            )

            smoke_result = {
                "status": "passed",
                "probability_shape": list(
                    package_probabilities.shape
                ),
                "prediction": prediction,
                "prediction_name": class_names[
                    prediction
                ],
                "confidence": float(
                    package_probabilities[0, prediction]
                ),
                "probabilities": (
                    package_probabilities[0].tolist()
                ),
                "warning": (
                    "This is an export integrity smoke test, "
                    "not a formal accuracy evaluation."
                ),
            }

            dump_json(
                smoke_result,
                temporary_path
                / "export_smoke_test.json",
            )

        replace_directory_atomically(
            temporary_path=temporary_path,
            output_path=output_path,
            overwrite=args.overwrite,
        )

    except Exception:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)

        raise

    print()
    print("CBraMod Runtime Model Package exported.")
    print("package:", output_path)

    if args.skip_smoke_test:
        print("smoke test: skipped")
    else:
        print("smoke test: passed")

    print(
        "next step: add a cbramod branch to "
        "bci_dayloop.packages.loader.load_runtime_package()."
    )


if __name__ == "__main__":
    main()