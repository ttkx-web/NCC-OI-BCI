from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bci_dayloop.models.model_50m.adapter import (
    Model50MAdapter,
)
from bci_dayloop.models.model_50m.backend import (
    Model50MBackend,
)
from bci_dayloop.models.model_50m.config import (
    Model50MConfig,
    STANDARD_64_CHANNELS,
)
from bci_dayloop.preprocessing.canonical import (
    SignalCanonicalizer,
)
from bci_dayloop.preprocessing.model_50m import (
    Model50MInputTransform,
)
from bci_dayloop.runtime.model import RuntimeModel
from bci_dayloop.utils.config import load_yaml
from bci_dayloop.packages.common import (
    required_mapping as _required_mapping,
    resolve_package_file as _resolve_package_file,
    sha256_file as _sha256_file,
    verify_sha256 as _verify_sha256,
)

import torch
from torch import nn

from bci_dayloop.data.preprocessing import (
    PreprocessingConfig,
)
from bci_dayloop.models.labram_backend import (
    LaBraMBackend,
)
from bci_dayloop.models.labram_linear import (
    LaBraMLinearAdapter,
)
from bci_dayloop.preprocessing.labram import (
    LaBraMInputTransform,
)
from bci_dayloop.models.model_1b.runtime import (
    Model1BRuntime,
    build_1b_runtime,
)
from bci_dayloop.training.model_1b.population import (
    load_1b_head_checkpoint,
)


@dataclass(frozen=True, slots=True)
class LoadedRuntimePackage:
    runtime_model: RuntimeModel

    package_path: Path
    model_type: str
    model_name: str

    class_names: tuple[str, ...]
    command_map: dict[str, str]

    step_sec: float
    confidence_threshold: float

    is_test_head: bool
    warning_message: str | None

    metrics: dict[str, Any]
    package_metadata: dict[str, Any]

    @property
    def window_sec(self) -> float:
        return float(
            self.runtime_model
            .input_contract
            .window_sec
        )

    @property
    def target_sample_rate(self) -> float:
        return float(
            self.runtime_model
            .input_contract
            .sample_rate
        )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(
            f"JSON root must be a mapping: {path}"
        )

    return dict(payload)


def _validate_runtime_contract(
    *,
    runtime_model: RuntimeModel,
    contract: dict[str, Any],
) -> None:
    actual = runtime_model.input_contract

    expected_channels = tuple(
        str(name)
        for name in contract["channel_names"]
    )

    if actual.channel_names != expected_channels:
        raise ValueError(
            "Input contract channel order mismatch."
        )

    if not math.isclose(
        actual.sample_rate,
        float(contract["sample_rate"]),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "Input contract sample_rate mismatch."
        )

    if not math.isclose(
        actual.window_sec,
        float(contract["window_sec"]),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "Input contract window_sec mismatch."
        )

    if actual.num_samples != int(
        contract["num_samples"]
    ):
        raise ValueError(
            "Input contract num_samples mismatch."
        )

    if actual.input_unit != str(
        contract["input_unit"]
    ):
        raise ValueError(
            "Input contract input_unit mismatch."
        )

    expected_keys = tuple(
        str(key)
        for key in contract.get(
            "model_input_keys",
            ["signal"],
        )
    )

    if actual.model_input_keys != expected_keys:
        raise ValueError(
            "Input contract model_input_keys mismatch."
        )

def _load_labram_package(
    *,
    package_path: Path,
    package_payload: dict[str, Any],
    device: str,
    verify_hashes: bool,
) -> LoadedRuntimePackage:
    package_yaml = package_path / "package.yaml"

    model = _required_mapping(
        package_payload,
        "model",
        source=package_yaml,
    )

    files = _required_mapping(
        package_payload,
        "files",
        source=package_yaml,
    )

    contract = _required_mapping(
        package_payload,
        "input_contract",
        source=package_yaml,
    )

    runtime_config = _required_mapping(
        package_payload,
        "runtime",
        source=package_yaml,
    )

    package_metadata = _required_mapping(
        package_payload,
        "package",
        source=package_yaml,
    )

    backbone_path = _resolve_package_file(
        package_path,
        str(files["backbone"]),
        logical_name="backbone",
    )

    classifier_path = _resolve_package_file(
        package_path,
        str(files["classifier"]),
        logical_name="classifier",
    )

    preprocessing_path = _resolve_package_file(
        package_path,
        str(files["preprocessing"]),
        logical_name="preprocessing config",
    )

    metrics_path = _resolve_package_file(
        package_path,
        str(files["metrics"]),
        logical_name="metrics",
    )

    hashes = files.get("sha256", {})

    if not isinstance(hashes, dict):
        raise ValueError(
            "files.sha256 must be a mapping."
        )

    if verify_hashes:
        _verify_sha256(
            path=backbone_path,
            expected=hashes.get("backbone"),
            logical_name="backbone",
        )

        _verify_sha256(
            path=classifier_path,
            expected=hashes.get("classifier"),
            logical_name="classifier",
        )

    preprocessing = load_yaml(
        preprocessing_path
    )

    canonicalizer_config = _required_mapping(
        preprocessing,
        "canonicalizer",
        source=preprocessing_path,
    )

    transform_config = _required_mapping(
        preprocessing,
        "transform",
        source=preprocessing_path,
    )

    if transform_config.get("type") != "labram":
        raise ValueError(
            "Expected preprocessing transform "
            f"'labram', got "
            f"{transform_config.get('type')!r}."
        )

    bandpass = transform_config.get(
        "bandpass_hz",
        [0.1, 75.0],
    )

    if (
        not isinstance(bandpass, list)
        or len(bandpass) != 2
    ):
        raise ValueError(
            "transform.bandpass_hz must contain "
            "two values."
        )

    preprocessing_config = (
        PreprocessingConfig(
            bandpass_hz=(
                float(bandpass[0]),
                float(bandpass[1]),
            ),
            notch_hz=float(
                transform_config.get(
                    "notch_hz",
                    50.0,
                )
            ),
            target_sample_rate=float(
                transform_config[
                    "target_sample_rate"
                ]
            ),
            output_unit="uV",
            zscore_epsilon=float(
                transform_config.get(
                    "zscore_epsilon",
                    1e-6,
                )
            ),
            patch_samples=int(
                transform_config.get(
                    "patch_samples",
                    200,
                )
            ),
        )
    )

    class_names = tuple(
        str(name)
        for name in model["class_names"]
    )

    channel_names = tuple(
        str(name)
        for name in contract["channel_names"]
    )

    num_classes = int(
        model["num_classes"]
    )

    n_patches = int(
        model["n_patches"]
    )

    adapter = LaBraMLinearAdapter(
        channel_names=list(
            channel_names
        ),
        n_classes=num_classes,
        checkpoint=backbone_path,
        device=device,
        amp=bool(
            model.get("amp", True)
        ),
        freeze_encoder=bool(
            model.get(
                "freeze_encoder",
                True,
            )
        ),
        embedding_batch_size=int(
            model.get(
                "embedding_batch_size",
                4,
            )
        ),
        random_init=False,
        n_patches=n_patches,
    )

    head_payload = torch.load(
        classifier_path,
        map_location="cpu",
        weights_only=True,
    )

    embedding_dim = int(
        head_payload["embedding_dim"]
    )

    if embedding_dim != adapter.embedding_dim:
        raise ValueError(
            "LaBraM classifier embedding dimension "
            "does not match the Encoder: "
            f"classifier={embedding_dim}, "
            f"encoder={adapter.embedding_dim}."
        )

    if int(
        head_payload["n_classes"]
    ) != num_classes:
        raise ValueError(
            "LaBraM classifier class count mismatch."
        )

    adapter.head = nn.Linear(
        embedding_dim,
        num_classes,
    ).to(adapter.device)

    adapter.head.load_state_dict(
        head_payload["state_dict"]
    )

    adapter.head.eval()

    input_transform = LaBraMInputTransform(
        channel_names=channel_names,
        preprocessing_config=(
            preprocessing_config
        ),
        n_patches=n_patches,
        strict_window_duration=bool(
            contract.get(
                "strict_window_duration",
                True,
            )
        ),
    )

    runtime_model = RuntimeModel(
        canonicalizer=SignalCanonicalizer(
            target_unit=str(
                canonicalizer_config.get(
                    "target_unit",
                    "uV",
                )
            )
        ),
        input_transform=input_transform,
        backend=LaBraMBackend(adapter),
    )

    _validate_runtime_contract(
        runtime_model=runtime_model,
        contract=contract,
    )

    command_map = runtime_config.get(
        "command_map",
        {},
    )

    if not isinstance(command_map, dict):
        raise ValueError(
            "runtime.command_map must be a mapping."
        )

    return LoadedRuntimePackage(
        runtime_model=runtime_model,
        package_path=package_path,
        model_type="labram",
        model_name=str(
            model.get(
                "name",
                "labram-linear",
            )
        ),
        class_names=class_names,
        command_map={
            str(key): str(value)
            for key, value
            in command_map.items()
        },
        step_sec=float(
            runtime_config.get(
                "step_sec",
                0.5,
            )
        ),
        confidence_threshold=float(
            runtime_config.get(
                "confidence_threshold",
                0.55,
            )
        ),
        is_test_head=bool(
            package_metadata.get(
                "is_test_head",
                False,
            )
        ),
        warning_message=(
            package_metadata.get(
                "warning_message"
            )
        ),
        metrics=_load_json(
            metrics_path
        ),
        package_metadata=package_payload,
    )

def _load_50m_package(
    *,
    package_path: Path,
    package_payload: dict[str, Any],
    device: str,
    verify_hashes: bool,
) -> LoadedRuntimePackage:
    model = _required_mapping(
        package_payload,
        "model",
        source=package_path / "package.yaml",
    )

    files = _required_mapping(
        package_payload,
        "files",
        source=package_path / "package.yaml",
    )

    contract = _required_mapping(
        package_payload,
        "input_contract",
        source=package_path / "package.yaml",
    )

    runtime_config = _required_mapping(
        package_payload,
        "runtime",
        source=package_path / "package.yaml",
    )

    package_metadata = _required_mapping(
        package_payload,
        "package",
        source=package_path / "package.yaml",
    )

    backbone_path = _resolve_package_file(
        package_path,
        str(files["backbone"]),
        logical_name="backbone",
    )

    classifier_path = _resolve_package_file(
        package_path,
        str(files["classifier"]),
        logical_name="classifier",
    )

    preprocessing_path = _resolve_package_file(
        package_path,
        str(files["preprocessing"]),
        logical_name="preprocessing config",
    )

    metrics_path = _resolve_package_file(
        package_path,
        str(files["metrics"]),
        logical_name="metrics",
    )

    hashes = files.get("sha256", {})

    if hashes is None:
        hashes = {}

    if not isinstance(hashes, dict):
        raise ValueError(
            "files.sha256 must be a mapping."
        )

    if verify_hashes:
        _verify_sha256(
            path=backbone_path,
            expected=hashes.get("backbone"),
            logical_name="backbone",
        )
        _verify_sha256(
            path=classifier_path,
            expected=hashes.get("classifier"),
            logical_name="classifier",
        )

    preprocessing = load_yaml(
        preprocessing_path
    )

    canonicalizer_config = _required_mapping(
        preprocessing,
        "canonicalizer",
        source=preprocessing_path,
    )

    transform = _required_mapping(
        preprocessing,
        "transform",
        source=preprocessing_path,
    )

    if transform.get("type") != "model_50m":
        raise ValueError(
            "Expected preprocessing transform "
            f"'model_50m', got "
            f"{transform.get('type')!r}."
        )

    class_names = tuple(
        str(name)
        for name in model["class_names"]
    )

    num_classes = int(model["num_classes"])

    if len(class_names) != num_classes:
        raise ValueError(
            "model.class_names length does not match "
            "model.num_classes."
        )

    channel_names = tuple(
        str(name)
        for name in contract["channel_names"]
    )

    config = Model50MConfig(
        checkpoint_path=backbone_path,
        classifier_path=classifier_path,
        device=device,

        target_sample_rate=float(
            contract["sample_rate"]
        ),
        window_seconds=float(
            contract["window_sec"]
        ),
        n_channels=len(channel_names),
        standard_channels=channel_names,

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

        patch_seconds=float(
            model["patch_seconds"]
        ),
        patch_stride_seconds=float(
            model["patch_stride_seconds"]
        ),

        filter_enabled=bool(
            transform.get(
                "filter_enabled",
                True,
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
        zscore_enabled=bool(
            transform.get(
                "zscore_enabled",
                True,
            )
        ),
        zscore_eps=float(
            transform.get(
                "zscore_eps",
                1e-8,
            )
        ),
        missing_channel_fill_value=float(
            transform.get(
                "missing_channel_fill_value",
                0.0,
            )
        ),

        d_model=int(model["d_model"]),
        n_heads=int(model["n_heads"]),
        depth=int(model["depth"]),
        mlp_ratio=float(
            model["mlp_ratio"]
        ),
        dropout=float(
            model["dropout"]
        ),

        model_n_time_patches=int(
            model["model_n_time_patches"]
        ),
        output_layer_idx=int(
            model["output_layer_idx"]
        ),
        aggregation=str(
            model["aggregation"]
        ),
        num_classes=num_classes,
    )

    adapter = Model50MAdapter(
        config=config,
        class_names=class_names,
        strict_head_metadata=True,
    )

    backend = Model50MBackend(
        adapter=adapter
    )

    input_transform = Model50MInputTransform(
        config=config
    )

    canonicalizer = SignalCanonicalizer(
        target_unit=str(
            canonicalizer_config.get(
                "target_unit",
                contract["input_unit"],
            )
        )
    )

    runtime_model = RuntimeModel(
        canonicalizer=canonicalizer,
        input_transform=input_transform,
        backend=backend,
    )

    _validate_runtime_contract(
        runtime_model=runtime_model,
        contract=contract,
    )

    command_map_raw = runtime_config.get(
        "command_map",
        {},
    )

    if not isinstance(command_map_raw, dict):
        raise ValueError(
            "runtime.command_map must be a mapping."
        )

    step_sec = float(
        runtime_config.get(
            "step_sec",
            0.5,
        )
    )

    if step_sec <= 0:
        raise ValueError(
            "runtime.step_sec must be positive."
        )

    confidence_threshold = float(
        runtime_config.get(
            "confidence_threshold",
            0.55,
        )
    )

    return LoadedRuntimePackage(
        runtime_model=runtime_model,
        package_path=package_path,
        model_type="model_50m",
        model_name=str(
            model.get(
                "name",
                "50m-linear",
            )
        ),
        class_names=class_names,
        command_map={
            str(key): str(value)
            for key, value
            in command_map_raw.items()
        },
        step_sec=step_sec,
        confidence_threshold=confidence_threshold,
        is_test_head=bool(
            package_metadata.get(
                "is_test_head",
                False,
            )
        ),
        warning_message=(
            package_metadata.get(
                "warning_message"
            )
        ),
        metrics=_load_json(metrics_path),
        package_metadata=package_payload,
    )


def _load_1b_package(
    *,
    package_path: Path,
    package_payload: dict[str, Any],
    device: str,
    verify_hashes: bool,
) -> LoadedRuntimePackage:
    package_yaml = package_path / "package.yaml"
    model = _required_mapping(package_payload, "model", source=package_yaml)
    files = _required_mapping(package_payload, "files", source=package_yaml)
    contract = _required_mapping(package_payload, "input_contract", source=package_yaml)
    runtime_config = _required_mapping(package_payload, "runtime", source=package_yaml)
    package_metadata = _required_mapping(package_payload, "package", source=package_yaml)
    backbone_path = _resolve_package_file(package_path, str(files["backbone"]), logical_name="backbone")
    head_path = _resolve_package_file(package_path, str(files["head"]), logical_name="1B linear head")
    preprocessing_path = _resolve_package_file(package_path, str(files["preprocessing"]), logical_name="preprocessing config")
    metrics_path = _resolve_package_file(package_path, str(files["metrics"]), logical_name="metrics")
    hashes = files.get("sha256", {})
    if not isinstance(hashes, dict):
        raise ValueError("files.sha256 must be a mapping")
    if verify_hashes:
        _verify_sha256(path=backbone_path, expected=hashes.get("backbone"), logical_name="backbone")
        _verify_sha256(path=head_path, expected=hashes.get("head"), logical_name="1B linear head")

    class_names = tuple(str(name) for name in model.get("class_names", ()))
    num_classes = int(model.get("num_classes", 0))
    if len(class_names) != num_classes or len(set(class_names)) != len(class_names):
        raise ValueError("model.class_names must be unique and match model.num_classes")
    expected_mapping = {str(index): name for index, name in enumerate(class_names)}
    if model.get("label_mapping") != expected_mapping:
        raise ValueError("model.label_mapping does not match model.class_names order")
    if model.get("head_type") != "linear" or model.get("aggregation") != "flatten":
        raise ValueError("1B Runtime Package requires a linear flatten head")
    window_seconds = float(model.get("window_seconds", -1))
    num_time_patches = int(model.get("num_time_patches", -1))
    if not 1.0 <= window_seconds <= 10.0 or window_seconds != float(int(window_seconds)):
        raise ValueError("1B Package window_seconds must be an integer in [1, 10]")
    if num_time_patches != int(window_seconds):
        raise ValueError("1B Package num_time_patches must equal window_seconds")
    expected_tokens = 64 * num_time_patches
    expected_input_dim = expected_tokens * 2048
    if int(model.get("token_count", -1)) != expected_tokens:
        raise ValueError("1B Package token_count does not match window_seconds")
    if int(model.get("token_length", -1)) != 100:
        raise ValueError("1B Package token_length must be 100")
    if int(model.get("classifier_input_dim", -1)) != expected_input_dim:
        raise ValueError("1B Package classifier_input_dim does not match window_seconds")
    expected_architecture = {
        "d_model": 2048, "n_heads": 16, "depth": 20,
        "output_layer_idx": 19, "model_n_time_patches": 10,
    }
    if {key: model.get(key) for key in expected_architecture} != expected_architecture:
        raise ValueError("1B Package architecture does not match the formal checkpoint")
    if float(contract.get("window_sec", -1)) != window_seconds:
        raise ValueError("input_contract.window_sec does not match model.window_seconds")
    if int(contract.get("num_samples", -1)) != int(window_seconds * 100):
        raise ValueError("input_contract.num_samples does not match 1B window_seconds")
    contract_channels = tuple(str(name) for name in contract.get("channel_names", ()))
    model_channels = tuple(str(name) for name in model.get("standard_channels", ()))
    if len(contract_channels) != 64 or len(set(contract_channels)) != 64:
        raise ValueError("1B Package input_contract must contain exactly 64 unique channels")
    if model_channels != contract_channels:
        raise ValueError("1B Package model/input_contract channel order mismatch")
    if contract_channels != STANDARD_64_CHANNELS:
        raise ValueError("1B Package channel order does not match the verified standard 64-channel contract")

    preprocessing = load_yaml(preprocessing_path)
    canonicalizer_config = _required_mapping(preprocessing, "canonicalizer", source=preprocessing_path)
    transform = _required_mapping(preprocessing, "transform", source=preprocessing_path)
    if transform.get("type") != "model_1b":
        raise ValueError("Expected preprocessing transform type 'model_1b'")
    if float(transform.get("window_seconds", -1)) != window_seconds:
        raise ValueError("preprocessing window_seconds does not match Package model")
    if int(transform.get("num_time_patches", -1)) != num_time_patches:
        raise ValueError("preprocessing num_time_patches does not match Package model")

    backbone_sha = _sha256_file(backbone_path)
    head, head_metadata = load_1b_head_checkpoint(
        head_path, window_seconds=window_seconds, class_names=class_names,
        backbone_sha256=backbone_sha, device="cpu",
    )
    if int(head_metadata["classifier_input_dim"]) != expected_input_dim:
        raise ValueError("1B head metadata classifier_input_dim differs from package.yaml")
    runtime_model: Model1BRuntime = build_1b_runtime(
        backbone_checkpoint=backbone_path, head=head, class_names=class_names, device=device,
        window_seconds=window_seconds, target_sample_rate=float(contract["sample_rate"]),
        patch_seconds=float(model["patch_seconds"]), patch_stride_seconds=float(model["patch_stride_seconds"]),
        filter_enabled=bool(transform["filter_enabled"]), filter_low_hz=float(transform["filter_low_hz"]),
        filter_high_hz=float(transform["filter_high_hz"]), filter_order=int(transform["filter_order"]),
        reference_mode=str(transform["reference_mode"]), zscore_enabled=bool(transform["zscore_enabled"]),
        zscore_eps=float(transform["zscore_eps"]), missing_channel_fill_value=float(transform["missing_channel_fill_value"]),
        strict_window_duration=bool(contract.get("strict_window_duration", True)),
        window_tolerance_seconds=float(transform.get("window_tolerance_seconds", 0.02)),
    )
    _validate_runtime_contract(runtime_model=runtime_model, contract=contract)
    command_map_raw = runtime_config.get("command_map", {})
    if not isinstance(command_map_raw, dict):
        raise ValueError("runtime.command_map must be a mapping")
    step_sec = float(runtime_config.get("step_sec", 0.5))
    confidence_threshold = float(runtime_config.get("confidence_threshold", 0.55))
    if step_sec <= 0 or not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("invalid 1B runtime step_sec or confidence_threshold")
    return LoadedRuntimePackage(
        runtime_model=runtime_model, package_path=package_path, model_type="model_1b",
        model_name="1b-frozen-linear", class_names=class_names,
        command_map={str(key): str(value) for key, value in command_map_raw.items()},
        step_sec=step_sec, confidence_threshold=confidence_threshold,
        is_test_head=bool(package_metadata.get("is_test_head", False)),
        warning_message=package_metadata.get("warning_message"), metrics=_load_json(metrics_path),
        package_metadata=package_payload,
    )


def _load_cbramod_package(
    *,
    package_path: Path,
    package_payload: dict[str, Any],
    device: str,
    verify_hashes: bool,
) -> LoadedRuntimePackage:
    """Load a schema-v2 CBRaMod frozen-head Runtime Package."""

    # Keep these imports local so existing LaBraM/50M users are not forced
    # to vendor CBRaMod until they actually load a CBRaMod package.
    from bci_dayloop.models.cbramod.backend import CBraModBackend
    from bci_dayloop.models.cbramod.backbone import CBraModBackbone
    from bci_dayloop.models.cbramod.classifier import (
        build_cbramod_classifier,
    )
    from bci_dayloop.models.cbramod.config import CBraModConfig
    from bci_dayloop.models.cbramod.preprocessing import (
        CBraModPipelinePreprocessor,
    )
    from bci_dayloop.models.cbramod.runtime import (
        load_cbramod_classifier_checkpoint,
    )

    package_yaml = package_path / "package.yaml"

    model = _required_mapping(
        package_payload,
        "model",
        source=package_yaml,
    )
    files = _required_mapping(
        package_payload,
        "files",
        source=package_yaml,
    )
    contract = _required_mapping(
        package_payload,
        "input_contract",
        source=package_yaml,
    )
    runtime_config = _required_mapping(
        package_payload,
        "runtime",
        source=package_yaml,
    )
    package_metadata = _required_mapping(
        package_payload,
        "package",
        source=package_yaml,
    )

    if model.get("name") != "cbramod-frozen-head":
        raise ValueError(
            "Expected CBRaMod package model.name "
            "'cbramod-frozen-head', got "
            f"{model.get('name')!r}."
        )

    backbone_path = _resolve_package_file(
        package_path,
        str(files["backbone"]),
        logical_name="backbone",
    )
    classifier_path = _resolve_package_file(
        package_path,
        str(files["classifier"]),
        logical_name="classifier",
    )
    preprocessing_path = _resolve_package_file(
        package_path,
        str(files["preprocessing"]),
        logical_name="preprocessing config",
    )
    metrics_path = _resolve_package_file(
        package_path,
        str(files["metrics"]),
        logical_name="metrics",
    )

    hashes = files.get("sha256", {})
    if hashes is None:
        hashes = {}
    if not isinstance(hashes, dict):
        raise ValueError("files.sha256 must be a mapping.")

    if verify_hashes:
        _verify_sha256(
            path=backbone_path,
            expected=hashes.get("backbone"),
            logical_name="backbone",
        )
        _verify_sha256(
            path=classifier_path,
            expected=hashes.get("classifier"),
            logical_name="classifier",
        )

    preprocessing = load_yaml(preprocessing_path)
    canonicalizer_config = _required_mapping(
        preprocessing,
        "canonicalizer",
        source=preprocessing_path,
    )
    transform = _required_mapping(
        preprocessing,
        "transform",
        source=preprocessing_path,
    )

    if transform.get("type") != "cbramod":
        raise ValueError(
            "Expected preprocessing transform 'cbramod', got "
            f"{transform.get('type')!r}."
        )

    class_names = tuple(
        str(name)
        for name in model["class_names"]
    )
    num_classes = int(model["num_classes"])
    if len(class_names) != num_classes:
        raise ValueError(
            "model.class_names length does not match "
            "model.num_classes."
        )

    channel_names = tuple(
        str(name)
        for name in contract["channel_names"]
    )
    transform_channels = tuple(
        str(name)
        for name in transform["standard_channels"]
    )
    if channel_names != transform_channels:
        raise ValueError(
            "CBRaMod input_contract.channel_names does not "
            "match preprocessing transform standard_channels."
        )

    # 旧模型包使用 allow_missing_channels；它无法表达新的
    # spherical_spline 完整协议，不能静默迁移。
    if (
        "missing_channel_policy" not in transform
        and bool(
            transform.get(
                "allow_missing_channels",
                False,
            )
        )
    ):
        raise ValueError(
            "Legacy CBraMod package uses "
            "allow_missing_channels=true. Re-export it with "
            "missing_channel_policy, min_observed_channels, "
            "and spline_alpha."
        )

    n_channels = int(transform["n_channels"])
    policy_value = transform.get("missing_channel_policy")
    missing_channel_policy = (
        "error" if policy_value is None else str(policy_value)
    )
    min_observed_value = transform.get("min_observed_channels")
    if min_observed_value is None:
        if missing_channel_policy == "spherical_spline":
            raise ValueError(
                "CBRaMod spherical_spline package must explicitly declare "
                "transform.min_observed_channels."
            )
        min_observed_channels = n_channels
    else:
        min_observed_channels = int(min_observed_value)
    spline_alpha_value = transform.get("spline_alpha")
    if spline_alpha_value is None:
        if missing_channel_policy == "spherical_spline":
            raise ValueError(
                "CBRaMod spherical_spline package must explicitly declare "
                "transform.spline_alpha."
            )
        spline_alpha = 1e-5
    else:
        spline_alpha = float(spline_alpha_value)

    completion_metadata = runtime_config.get("channel_completion")
    if (
        missing_channel_policy == "spherical_spline"
        and completion_metadata is None
    ):
        raise ValueError(
            "CBRaMod spherical_spline package must declare "
            "runtime.channel_completion."
        )
    if completion_metadata is not None:
        if not isinstance(completion_metadata, dict):
            raise ValueError(
                "runtime.channel_completion must be a mapping."
            )
        expected_completion_values = {
            "missing_channel_policy": missing_channel_policy,
            "min_observed_channels": min_observed_channels,
            "spline_alpha": spline_alpha,
        }
        for key, expected_value in expected_completion_values.items():
            if completion_metadata.get(key) != expected_value:
                raise ValueError(
                    "CBRaMod runtime.channel_completion does not match "
                    f"preprocessing transform for {key}."
                )
        observed_names = tuple(
            str(name)
            for name in completion_metadata.get(
                "observed_channel_names", ()
            )
        )
        missing_names = tuple(
            str(name)
            for name in completion_metadata.get("missing_expected", ())
        )
        if len(observed_names) != len(set(observed_names)):
            raise ValueError(
                "CBRaMod runtime observed channels contain duplicates."
            )
        if len(missing_names) != len(set(missing_names)):
            raise ValueError(
                "CBRaMod runtime missing channels contain duplicates."
            )
        if set(observed_names).intersection(missing_names):
            raise ValueError(
                "CBRaMod runtime observed and missing channels overlap."
            )
        if tuple(
            name for name in channel_names if name in observed_names
        ) != observed_names or tuple(
            name for name in channel_names if name in missing_names
        ) != missing_names:
            raise ValueError(
                "CBRaMod runtime channel completion order does not match "
                "the package target montage."
            )
        if set(observed_names).union(missing_names) != set(channel_names):
            raise ValueError(
                "CBRaMod runtime observed/missing channels do not partition "
                "the package target montage."
            )
        if int(completion_metadata.get("observed_required", -1)) != len(
            observed_names
        ):
            raise ValueError(
                "CBRaMod runtime observed_required does not match its "
                "observed channel list."
            )
        if (
            missing_channel_policy == "spherical_spline"
            and completion_metadata.get("channel_completion_source")
            != "shared_runtime_preprocessor"
        ):
            raise ValueError(
                "CBRaMod spherical-spline package must use the shared "
                "runtime preprocessor."
            )
        if completion_metadata.get(
            "completion_matrix_sha256"
        ) != transform.get("completion_matrix_sha256"):
            raise ValueError(
                "CBRaMod completion matrix SHA-256 differs between runtime "
                "and preprocessing metadata."
            )
        provenance = package_payload.get("provenance", {})
        if not isinstance(provenance, dict):
            raise ValueError("CBRaMod package provenance must be a mapping.")
        provenance_sha256 = provenance.get("completion_matrix_sha256")
        if (
            provenance_sha256 is not None
            and provenance_sha256
            != transform.get("completion_matrix_sha256")
        ):
            raise ValueError(
                "CBRaMod completion matrix SHA-256 differs between package "
                "provenance and preprocessing metadata."
            )

    config = CBraModConfig(
        checkpoint_path=backbone_path,
        classifier_path=classifier_path,
        device=device,

        target_sample_rate=float(contract["sample_rate"]),
        window_seconds=float(contract["window_sec"]),
        n_channels=n_channels,
        standard_channels=channel_names,
        time_segments=int(transform["time_segments"]),
        points_per_patch=int(transform["points_per_patch"]),
        input_unit=str(contract["input_unit"]),

        strict_window_duration=bool(
            contract.get("strict_window_duration", True)
        ),
        window_tolerance_seconds=float(
            transform.get("window_tolerance_seconds", 0.02)
        ),
        filter_enabled=bool(
            transform.get("filter_enabled", False)
        ),
        filter_low_hz=float(
            transform.get("filter_low_hz", 0.1)
        ),
        filter_high_hz=float(
            transform.get("filter_high_hz", 75.0)
        ),
        filter_order=int(transform.get("filter_order", 4)),
        reference_mode=str(
            transform.get("reference_mode", "none")
        ),
        normalization=str(
            transform.get("normalization", "none")
        ),
        zscore_eps=float(transform.get("zscore_eps", 1e-8)),
        missing_channel_policy=missing_channel_policy,
        min_observed_channels=min_observed_channels,
        spline_alpha=spline_alpha,

        num_classes=num_classes,
        backbone_output_dim=int(
            model.get("backbone_output_dim", 200)
        ),
        head_type=str(model["head_type"]),
        head_hidden_dim_1=int(
            model.get("head_hidden_dim_1", 800)
        ),
        head_hidden_dim_2=int(
            model.get("head_hidden_dim_2", 200)
        ),
        head_dropout=float(model.get("head_dropout", 0.1)),
    )

    backbone = CBraModBackbone(config)
    classifier = build_cbramod_classifier(config).to(
        backbone.device
    )
    classifier_report = load_cbramod_classifier_checkpoint(
        classifier,
        classifier_path,
        config=config,
        class_names=class_names,
        strict_metadata=True,
    )

    if missing_channel_policy == "spherical_spline":
        classifier_metadata = classifier_report.metadata
        assert isinstance(completion_metadata, dict)
        provenance = package_payload.get("provenance", {})
        if not isinstance(provenance, dict):
            raise ValueError("CBRaMod package provenance must be a mapping.")
        required_classifier_metadata = {
            "deployment_profile": completion_metadata.get(
                "deployment_profile"
            ),
            "training_channel_source_count": provenance.get(
                "training_channel_source_count"
            ),
            "observed_channel_count": len(
                completion_metadata["observed_channel_names"]
            ),
            "observed_channel_names": completion_metadata[
                "observed_channel_names"
            ],
            "simulated_missing_channels": completion_metadata[
                "missing_expected"
            ],
            "missing_channel_policy": missing_channel_policy,
            "min_observed_channels": min_observed_channels,
            "spline_alpha": spline_alpha,
            "completion_matrix_sha256": transform.get(
                "completion_matrix_sha256"
            ),
            "channel_completion_source": "shared_runtime_preprocessor",
        }
        for key, expected_value in required_classifier_metadata.items():
            if classifier_metadata.get(key) != expected_value:
                raise ValueError(
                    "CBRaMod classifier metadata does not match the "
                    f"spherical-spline package for {key}."
                )

    runtime_model = RuntimeModel(
        canonicalizer=SignalCanonicalizer(
            target_unit=str(
                canonicalizer_config.get(
                    "target_unit",
                    contract["input_unit"],
                )
            )
        ),
        input_transform=CBraModPipelinePreprocessor(config),
        backend=CBraModBackend(
            backbone=backbone,
            classifier=classifier,
            config=config,
        ),
    )

    _validate_runtime_contract(
        runtime_model=runtime_model,
        contract=contract,
    )

    command_map_raw = runtime_config.get("command_map", {})
    if not isinstance(command_map_raw, dict):
        raise ValueError(
            "runtime.command_map must be a mapping."
        )

    step_sec = float(runtime_config.get("step_sec", 0.5))
    if step_sec <= 0:
        raise ValueError("runtime.step_sec must be positive.")

    confidence_threshold = float(
        runtime_config.get("confidence_threshold", 0.55)
    )
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError(
            "runtime.confidence_threshold must be in [0, 1]."
        )

    return LoadedRuntimePackage(
        runtime_model=runtime_model,
        package_path=package_path,
        model_type="cbramod",
        model_name="cbramod-frozen-head",
        class_names=class_names,
        command_map={
            str(key): str(value)
            for key, value in command_map_raw.items()
        },
        step_sec=step_sec,
        confidence_threshold=confidence_threshold,
        is_test_head=bool(
            package_metadata.get("is_test_head", False)
        ),
        warning_message=package_metadata.get(
            "warning_message"
        ),
        metrics=_load_json(metrics_path),
        package_metadata=package_payload,
    )


def load_runtime_package(
    package_path: str | Path,
    *,
    device: str = "cpu",
    verify_hashes: bool = True,
) -> LoadedRuntimePackage:
    package = Path(
        package_path
    ).expanduser().resolve()

    if not package.is_dir():
        raise FileNotFoundError(
            f"Runtime package directory was not found: "
            f"{package}"
        )

    package_yaml = package / "package.yaml"

    if not package_yaml.is_file():
        raise FileNotFoundError(
            f"package.yaml was not found: "
            f"{package_yaml}"
        )

    payload = load_yaml(package_yaml)

    schema_version = int(
        payload.get(
            "schema_version",
            -1,
        )
    )

    if schema_version != 2:
        raise ValueError(
            "Unsupported runtime package schema: "
            f"{schema_version}. Expected 2."
        )

    model = _required_mapping(
        payload,
        "model",
        source=package_yaml,
    )

    model_type = str(
        model.get("type", "")
    )

    if model_type == "model_50m":
        return _load_50m_package(
            package_path=package,
            package_payload=payload,
            device=device,
            verify_hashes=verify_hashes,
        )

    if model_type == "model_1b":
        return _load_1b_package(
            package_path=package,
            package_payload=payload,
            device=device,
            verify_hashes=verify_hashes,
        )

    if model_type == "labram":
        return _load_labram_package(
            package_path=package,
            package_payload=payload,
            device=device,
            verify_hashes=verify_hashes,
        )

    if model_type == "cbramod":
        return _load_cbramod_package(
            package_path=package,
            package_payload=payload,
            device=device,
            verify_hashes=verify_hashes,
        )

    raise ValueError(
        f"Unsupported model type {model_type!r}. "
        "Currently available: model_50m, model_1b, labram, cbramod."
    )
