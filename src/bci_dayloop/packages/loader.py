from __future__ import annotations

import hashlib
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
)
from bci_dayloop.preprocessing.canonical import (
    SignalCanonicalizer,
)
from bci_dayloop.preprocessing.model_50m import (
    Model50MInputTransform,
)
from bci_dayloop.runtime.model import RuntimeModel
from bci_dayloop.utils.config import load_yaml


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


def _required_mapping(
    payload: dict[str, Any],
    key: str,
    *,
    source: Path,
) -> dict[str, Any]:
    value = payload.get(key)

    if not isinstance(value, dict):
        raise ValueError(
            f"{source} field {key!r} "
            "must be a mapping."
        )

    return dict(value)


def _resolve_package_file(
    package_path: Path,
    value: str,
    *,
    logical_name: str,
) -> Path:
    relative_path = Path(value)

    if relative_path.is_absolute():
        raise ValueError(
            f"{logical_name} must use a package-relative "
            f"path, got {value!r}."
        )

    resolved = (
        package_path / relative_path
    ).resolve()

    # 防止 "../../outside.pt" 逃出 Package。
    try:
        resolved.relative_to(package_path)
    except ValueError as error:
        raise ValueError(
            f"{logical_name} escapes package directory: "
            f"{value!r}."
        ) from error

    if not resolved.is_file():
        raise FileNotFoundError(
            f"{logical_name} was not found: {resolved}"
        )

    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _verify_sha256(
    *,
    path: Path,
    expected: str | None,
    logical_name: str,
) -> None:
    if not expected:
        return

    actual = _sha256_file(path)

    if actual.lower() != str(expected).lower():
        raise ValueError(
            f"{logical_name} SHA256 mismatch: "
            f"expected={expected}, actual={actual}."
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

    raise ValueError(
        f"Unsupported model type {model_type!r}. "
        "Currently available: model_50m."
    )