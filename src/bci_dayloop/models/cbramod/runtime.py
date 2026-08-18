from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from bci_dayloop.models.base import ModelBackend
from bci_dayloop.preprocessing.canonical import SignalCanonicalizer
from bci_dayloop.runtime.model import RuntimeModel
from bci_dayloop.runtime.types import (
    ModelOutput,
    RawEEGWindow,
)

from .backend import CBraModBackend
from .backbone import CBraModBackbone
from .classifier import (
    CBraModClassifier,
    build_cbramod_classifier,
)
from .config import BCICIV2A_22_CHANNELS, CBraModConfig
from .preprocessing import CBraModPipelinePreprocessor


class EEGMetadataLike(Protocol):
    """Runtime Package 所需的最小数据元信息。"""

    channel_names: Sequence[str]
    sample_rate: float
    unit: str
    class_names: Sequence[str]


@dataclass(frozen=True, slots=True)
class CBraModClassifierLoadReport:
    checkpoint_path: Path
    metadata: dict[str, Any]
    strict_metadata: bool


@dataclass(frozen=True, slots=True)
class CBraModRuntimePrediction:
    """从原始 EEG 窗口完成一次 RuntimeModel 推理的结果。"""

    prediction: int
    confidence: float
    probabilities: tuple[float, ...]

    preprocessing_trace: tuple[str, ...]
    preprocessing_diagnostics: dict[str, object]
    model_diagnostics: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "probabilities": list(self.probabilities),
            "preprocessing_trace": list(
                self.preprocessing_trace
            ),
            "preprocessing_diagnostics": dict(
                self.preprocessing_diagnostics
            ),
            "model_diagnostics": dict(
                self.model_diagnostics
            ),
        }


@dataclass(slots=True)
class CBraModRuntime:
    """
    CBRaMod 的统一 Runtime 封装。

    对实时/Replay 链路应使用：

        runtime.runtime_model

    而不是旧的 BaseModelAdapter。
    """

    config: CBraModConfig
    class_names: tuple[str, ...]

    backbone: CBraModBackbone
    classifier: CBraModClassifier
    backend: CBraModBackend

    canonicalizer: SignalCanonicalizer
    preprocessor: CBraModPipelinePreprocessor
    runtime_model: RuntimeModel

    classifier_load_report: CBraModClassifierLoadReport

    def predict_raw_window(
        self,
        raw_window: RawEEGWindow,
        *,
        return_features: bool = False,
    ) -> CBraModRuntimePrediction:
        output = self.runtime_model.predict(
            raw_window,
            return_features=return_features,
        )

        return CBraModRuntimePrediction(
            prediction=output.predicted_class,
            confidence=output.confidence,
            probabilities=tuple(
                float(value)
                for value in output.probabilities[0]
                .detach()
                .cpu()
                .tolist()
            ),
            preprocessing_trace=tuple(
                self.runtime_model
                .prepare(raw_window)
                .preprocessing_trace
            ),
            preprocessing_diagnostics=dict(
                self.preprocessor.last_diagnostics.to_dict()
                if self.preprocessor.last_diagnostics is not None
                else {}
            ),
            model_diagnostics=dict(output.diagnostics),
        )

    def health_check(self) -> dict[str, object]:
        """
        只验证整条链路可运行，不验证准确率。
        """

        rng = np.random.default_rng(seed=42)

        raw_signal = rng.standard_normal(
            (
                self.config.n_channels,
                self.config.num_samples,
            ),
            dtype=np.float32,
        )

        raw_window = RawEEGWindow(
            data=raw_signal,
            channel_names=list(
                self.config.standard_channels
            ),
            sample_rate=self.config.target_sample_rate,
            unit=self.config.input_unit,
            layout="CT",
            window_id="health_check",
        )

        output = self.runtime_model.predict(raw_window)

        expected_probability_shape = (
            1,
            self.config.num_classes,
        )

        if tuple(output.probabilities.shape) != (
            expected_probability_shape
        ):
            raise RuntimeError(
                "CBraMod health check returned an unexpected "
                "probability shape. Expected "
                f"{expected_probability_shape}, got "
                f"{tuple(output.probabilities.shape)}."
            )

        return {
            "status": "ok",
            "model_name": "cbramod-frozen-head",
            "device": str(self.backend.device),
            "class_names": list(self.class_names),
            "backbone_checkpoint": str(
                self.config.checkpoint_path
            ),
            "classifier_checkpoint": str(
                self.config.classifier_path
            ),
            "input_contract": {
                "channel_names": list(
                    self.runtime_model
                    .input_contract
                    .channel_names
                ),
                "sample_rate": (
                    self.runtime_model
                    .input_contract
                    .sample_rate
                ),
                "window_sec": (
                    self.runtime_model
                    .input_contract
                    .window_sec
                ),
                "num_samples": (
                    self.runtime_model
                    .input_contract
                    .num_samples
                ),
                "tensor_layout": (
                    self.runtime_model
                    .input_contract
                    .tensor_layout
                ),
            },
            "probability_shape": list(
                output.probabilities.shape
            ),
        }


def _required_file(
    value: str | Path,
    *,
    name: str,
) -> Path:
    path = Path(value).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"{name} was not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"{name} is not a file: {path}"
        )

    return path


def checkpoint_sha256(
    path: str | Path,
) -> str | None:
    """计算 checkpoint SHA-256，供 package manifest 使用。"""

    resolved = Path(path)

    if not resolved.is_file():
        return None

    digest = hashlib.sha256()

    with resolved.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _extract_classifier_state_dict(
    payload: object,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """
    支持以下分类头 checkpoint 格式：

    1. 推荐格式：
       {
         "state_dict": ...,
         "model_name": "cbramod-frozen-head",
         "head_type": "official_mlp",
         ...
       }

    2. 纯 state_dict：
       {"head.0.weight": tensor, ...}
    """

    if not isinstance(payload, Mapping):
        raise TypeError(
            "CBraMod classifier checkpoint must be a mapping, "
            f"got {type(payload).__name__}."
        )

    metadata = {
        str(key): value
        for key, value in payload.items()
        if key not in {
            "state_dict",
            "classifier_state_dict",
            "classifier",
        }
    }

    candidate: object = payload

    for key in (
        "state_dict",
        "classifier_state_dict",
        "classifier",
    ):
        nested = payload.get(key)

        if isinstance(nested, Mapping):
            candidate = nested
            break

    if not isinstance(candidate, Mapping):
        raise TypeError(
            "CBraMod classifier checkpoint does not contain "
            "a valid state_dict."
        )

    state_dict: dict[str, torch.Tensor] = {}

    for key, value in candidate.items():
        if not isinstance(key, str):
            raise TypeError(
                "CBraMod classifier state_dict contains a "
                f"non-string key: {key!r}."
            )

        if not isinstance(value, torch.Tensor):
            raise TypeError(
                "CBraMod classifier state_dict contains a "
                "non-tensor value for "
                f"{key!r}: {type(value).__name__}."
            )

        state_dict[key] = value

    if not state_dict:
        raise ValueError(
            "CBraMod classifier state_dict is empty."
        )

    for prefix in ("module.", "classifier."):
        keys = tuple(state_dict)

        if keys and all(
            key.startswith(prefix)
            for key in keys
        ):
            state_dict = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
            }

    return state_dict, metadata


def _validate_classifier_metadata(
    metadata: Mapping[str, Any],
    *,
    config: CBraModConfig,
    class_names: Sequence[str],
    strict: bool,
) -> None:
    expected: dict[str, object] = {
        "model_name": "cbramod-frozen-head",
        "head_type": config.head_type,
        "num_classes": config.num_classes,
        "classifier_input_dim": (
            config.classifier_input_dim
        ),
        "feature_shape": list(
            config.expected_unbatched_shape[:-1]
            + (config.backbone_output_dim,)
        ),
        "class_names": list(class_names),
    }

    missing = [
        key
        for key in expected
        if key not in metadata
    ]

    if strict and missing:
        raise ValueError(
            "CBraMod classifier checkpoint is missing required "
            f"metadata fields: {missing}."
        )

    for key, expected_value in expected.items():
        if key not in metadata:
            continue

        actual_value = metadata[key]

        if actual_value != expected_value:
            raise ValueError(
                "CBraMod classifier checkpoint metadata "
                f"mismatch for {key}: "
                f"checkpoint={actual_value!r}, "
                f"runtime={expected_value!r}."
            )


def load_cbramod_classifier_checkpoint(
    classifier: CBraModClassifier,
    checkpoint_path: str | Path,
    *,
    config: CBraModConfig,
    class_names: Sequence[str],
    strict_metadata: bool = True,
) -> CBraModClassifierLoadReport:
    """加载并严格校验 CBRaMod 下游分类头。"""

    path = _required_file(
        checkpoint_path,
        name="CBraMod classifier checkpoint",
    )

    payload: Any = torch.load(
        path,
        map_location="cpu",
    )

    state_dict, metadata = _extract_classifier_state_dict(
        payload
    )

    _validate_classifier_metadata(
        metadata,
        config=config,
        class_names=class_names,
        strict=strict_metadata,
    )

    classifier.load_state_dict(
        state_dict,
        strict=True,
    )

    classifier.eval()

    return CBraModClassifierLoadReport(
        checkpoint_path=path,
        metadata=dict(metadata),
        strict_metadata=bool(strict_metadata),
    )


def save_cbramod_classifier_checkpoint(
    classifier: CBraModClassifier,
    checkpoint_path: str | Path,
    *,
    config: CBraModConfig,
    class_names: Sequence[str],
    extra_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """
    保存 CBRaMod 分类头。

    后续 train_cbramod_population_head.py 必须使用这个函数，
    保证 Runtime Package 可以严格加载训练产物。
    """

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "format_version": 1,
        "model_name": "cbramod-frozen-head",
        "head_type": config.head_type,
        "num_classes": config.num_classes,
        "classifier_input_dim": (
            config.classifier_input_dim
        ),
        "feature_shape": list(
            config.expected_unbatched_shape[:-1]
            + (config.backbone_output_dim,)
        ),
        "class_names": list(class_names),
        "state_dict": classifier.state_dict(),
    }

    if extra_metadata is not None:
        overlap = set(payload).intersection(
            extra_metadata
        )

        if overlap:
            raise ValueError(
                "extra_metadata must not overwrite required "
                "CBraMod classifier metadata: "
                f"{sorted(overlap)}."
            )

        payload.update(dict(extra_metadata))

    torch.save(payload, path)

    return path


def build_cbramod_runtime(
    *,
    checkpoint_path: str | Path,
    classifier_path: str | Path,
    class_names: Sequence[str],
    device: str = "cpu",

    target_sample_rate: float = 200.0,
    window_seconds: float = 4.0,
    n_channels: int = 22,
    standard_channels: Sequence[str] = BCICIV2A_22_CHANNELS,
    time_segments: int = 4,
    points_per_patch: int = 200,

    input_unit: str = "uV",
    strict_window_duration: bool = True,
    window_tolerance_seconds: float = 0.02,

    filter_enabled: bool = False,
    filter_low_hz: float = 0.1,
    filter_high_hz: float = 75.0,
    filter_order: int = 4,
    reference_mode: str = "none",
    normalization: str = "none",
    missing_channel_policy: str = "error",
    min_observed_channels: int | None = None,
    spline_alpha: float = 1e-5,

    head_type: str = "official_mlp",
    head_hidden_dim_1: int = 800,
    head_hidden_dim_2: int = 200,
    head_dropout: float = 0.1,

    strict_classifier_metadata: bool = True,
) -> CBraModRuntime:
    """
    构建 CBRaMod 统一 RuntimeModel。

    注意：
    这里不接收 source channel_names、source sample_rate 或 source unit。
    它们来自每一个 RawEEGWindow，并由 SignalCanonicalizer 和
    CBraModPipelinePreprocessor 在运行时处理。
    """

    resolved_backbone_path = _required_file(
        checkpoint_path,
        name="CBraMod backbone checkpoint",
    )

    resolved_classifier_path = _required_file(
        classifier_path,
        name="CBraMod classifier checkpoint",
    )

    normalized_class_names = tuple(
        str(name)
        for name in class_names
    )

    if len(normalized_class_names) <= 1:
        raise ValueError(
            "class_names must contain at least two classes."
        )

    if len(set(normalized_class_names)) != len(
        normalized_class_names
    ):
        raise ValueError(
            "class_names contains duplicates."
        )

    config = CBraModConfig(
        checkpoint_path=resolved_backbone_path,
        classifier_path=resolved_classifier_path,
        device=device,

        target_sample_rate=target_sample_rate,
        window_seconds=window_seconds,
        n_channels=n_channels,
        standard_channels=tuple(
            str(name) for name in standard_channels
        ),
        time_segments=time_segments,
        points_per_patch=points_per_patch,

        input_unit=input_unit,
        strict_window_duration=strict_window_duration,
        window_tolerance_seconds=(
            window_tolerance_seconds
        ),

        filter_enabled=filter_enabled,
        filter_low_hz=filter_low_hz,
        filter_high_hz=filter_high_hz,
        filter_order=filter_order,
        reference_mode=reference_mode,
        normalization=normalization,
        missing_channel_policy=missing_channel_policy,
        min_observed_channels=min_observed_channels,
        spline_alpha=spline_alpha,

        num_classes=len(normalized_class_names),
        head_type=head_type,
        head_hidden_dim_1=head_hidden_dim_1,
        head_hidden_dim_2=head_hidden_dim_2,
        head_dropout=head_dropout,
    )

    backbone = CBraModBackbone(config)

    classifier = build_cbramod_classifier(config).to(
        backbone.device
    )

    classifier_report = load_cbramod_classifier_checkpoint(
        classifier,
        resolved_classifier_path,
        config=config,
        class_names=normalized_class_names,
        strict_metadata=strict_classifier_metadata,
    )

    backend = CBraModBackend(
        backbone=backbone,
        classifier=classifier,
        config=config,
    )

    canonicalizer = SignalCanonicalizer(
        target_unit=config.input_unit
    )

    preprocessor = CBraModPipelinePreprocessor(config)

    runtime_model = RuntimeModel(
        canonicalizer=canonicalizer,
        input_transform=preprocessor,
        backend=backend,
    )

    return CBraModRuntime(
        config=config,
        class_names=normalized_class_names,
        backbone=backbone,
        classifier=classifier,
        backend=backend,
        canonicalizer=canonicalizer,
        preprocessor=preprocessor,
        runtime_model=runtime_model,
        classifier_load_report=classifier_report,
    )


def build_cbramod_runtime_from_metadata(
    *,
    checkpoint_path: str | Path,
    classifier_path: str | Path,
    metadata: EEGMetadataLike,
    device: str = "cpu",
    **kwargs: Any,
) -> CBraModRuntime:
    """
    从 HDF5 metadata 构建 Runtime。

    metadata 的 channel_names、sample_rate 和 unit 不被写死到 config；
    它们将在每个 RawEEGWindow 中运行时检查和转换。
    """

    required_attributes = (
        "channel_names",
        "sample_rate",
        "unit",
        "class_names",
    )

    missing_attributes = [
        name
        for name in required_attributes
        if not hasattr(metadata, name)
    ]

    if missing_attributes:
        raise AttributeError(
            "metadata is missing required attributes: "
            f"{missing_attributes}."
        )

    if not metadata.channel_names:
        raise ValueError(
            "metadata.channel_names cannot be empty."
        )

    if float(metadata.sample_rate) <= 0:
        raise ValueError(
            "metadata.sample_rate must be positive."
        )

    if not str(metadata.unit).strip():
        raise ValueError(
            "metadata.unit cannot be empty."
        )

    return build_cbramod_runtime(
        checkpoint_path=checkpoint_path,
        classifier_path=classifier_path,
        class_names=metadata.class_names,
        device=device,
        **kwargs,
    )
