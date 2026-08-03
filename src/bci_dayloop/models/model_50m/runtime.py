from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from bci_dayloop.models.base import add_batch_dimension

from .adapter import Model50MAdapter
from .config import Model50MConfig
from .pipeline_preprocessor import Model50MPipelinePreprocessor


class EEGMetadataLike(Protocol):
    """
    HDF5 metadata 需要提供的最小字段。

    不依赖具体 Metadata 类，只要对象具有这些属性即可。
    """

    channel_names: Sequence[str]
    sample_rate: float
    unit: str
    class_names: Sequence[str]


@dataclass(frozen=True, slots=True)
class Model50MRuntimePrediction:
    """从原始 EEG 窗口得到的单窗口预测结果。"""

    prediction: int
    confidence: float
    probabilities: tuple[float, ...]

    preprocessing_ms: float | None
    adapter_timing: dict[str, float] | None

    mapped_channel_count: int | None
    missing_channel_count: int | None
    unknown_channel_names: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "probabilities": list(self.probabilities),
            "preprocessing_ms": self.preprocessing_ms,
            "adapter_timing": self.adapter_timing,
            "mapped_channel_count": self.mapped_channel_count,
            "missing_channel_count": self.missing_channel_count,
            "unknown_channel_names": list(self.unknown_channel_names),
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class Model50MRuntime:
    """
    50M 在 BCI Pipeline 中的运行时对象。

    对外保留：
        adapter:
            可直接调用 predict_proba()。

        preprocessor:
            可直接调用 transform(raw_window)。

    同时提供 predict_raw_window()，方便离线测试或不经过
    SlidingWindowDecoder 时直接完成完整推理。
    """

    config: Model50MConfig
    adapter: Model50MAdapter
    preprocessor: Model50MPipelinePreprocessor
    class_names: tuple[str, ...]

    def predict_raw_window(
        self,
        raw_window: np.ndarray,
    ) -> Model50MRuntimePrediction:
        """
        从 Pipeline 原始 EEG 窗口完成一次完整预测。

        Args:
            raw_window:
                [C, T] 原始 EEG。

                当前阶段必须为真实 10 秒窗口。
                例如原始采样率 250 Hz 时，T 应为 2500。

        Returns:
            Model50MRuntimePrediction
        """
        model_input = self.preprocessor.transform(
            raw_window,
            self.preprocessor.sample_rate,
            self.preprocessor.input_unit,
        )
        probabilities_batch = self.adapter.predict_proba(add_batch_dimension(model_input))

        if probabilities_batch.shape != (
            1,
            self.config.num_classes,
        ):
            raise RuntimeError(
                "Unexpected probability shape: "
                f"expected {(1, self.config.num_classes)}, "
                f"got {probabilities_batch.shape}."
            )

        probabilities = probabilities_batch[0]

        prediction = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction])

        adapter_timing = (
            self.adapter.last_timing.to_dict()
            if self.adapter.last_timing is not None
            else None
        )

        # 当前 PipelinePreprocessor 没有强制记录单独耗时，
        # 因此暂时返回 None。后续可以在其 transform() 内增加计时。
        preprocessing_ms = None

        diagnostics = self.preprocessor.last_diagnostics
        return Model50MRuntimePrediction(
            prediction=prediction,
            confidence=confidence,
            probabilities=tuple(
                float(value) for value in probabilities
            ),
            preprocessing_ms=preprocessing_ms,
            adapter_timing=adapter_timing,
            mapped_channel_count=diagnostics.mapped_channel_count if diagnostics else None,
            missing_channel_count=diagnostics.missing_channel_count if diagnostics else None,
            unknown_channel_names=diagnostics.unknown_channel_names if diagnostics else (),
            notes=diagnostics.notes if diagnostics else (),
        )

    def health_check(self) -> dict[str, Any]:
        """
        检查模型、分类头和运行时配置是否可以正常使用。

        不检查分类准确率，因为当前 test_linear_head.pt
        只是流程测试头。
        """
        adapter_report = self.adapter.health_check()

        return {
            "status": "ok",
            "model": adapter_report,
            "runtime": {
                "class_names": list(self.class_names),
                "raw_sample_rate": (
                    self.preprocessor.sample_rate
                ),
                "raw_channel_count": len(
                    self.preprocessor.channel_names
                ),
                "raw_unit": self.preprocessor.input_unit,
                "target_sample_rate": (
                    self.config.target_sample_rate
                ),
                "window_seconds": (
                    self.config.window_seconds
                ),
                "target_shape": [
                    self.config.n_channels,
                    self.config.target_num_points,
                ],
                "num_tokens": self.config.num_tokens,
                "classifier_input_dim": (
                    self.config.classifier_input_dim
                ),
            },
        }


def _resolve_required_file(
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


def build_50m_runtime(
    *,
    checkpoint_path: str | Path,
    classifier_path: str | Path,
    channel_names: Sequence[str],
    sample_rate: float,
    input_unit: str,
    class_names: Sequence[str],
    device: str = "cpu",

    # 当前阶段保持 50M 原始输入配置。
    target_sample_rate: float = 100.0,
    window_seconds: float = 10.0,
    patch_seconds: float = 1.0,
    patch_stride_seconds: float = 1.0,

    # 当前暂定预处理。
    filter_enabled: bool = True,
    filter_low_hz: float = 0.1,
    filter_high_hz: float = 75.0,
    reference_mode: str = "none",

    # Backbone 与分类配置。
    output_layer_idx: int = 8,
    aggregation: str = "flatten",

    strict_window_duration: bool = True,
    strict_head_metadata: bool = True,
    model_n_time_patches: int = 10,
) -> Model50MRuntime:
    """
    构建 50M Pipeline 运行时。

    这是 B 同事接入 Replay 时建议使用的唯一构建入口。

    Returns:
        Model50MRuntime，其中包含：

        runtime.adapter
        runtime.preprocessor
        runtime.config

    示例：
        runtime = build_50m_runtime(...)

        decoder = SlidingWindowDecoder(
            runtime.adapter,
            runtime.preprocessor,
            ...
        )
    """
    checkpoint_path = _resolve_required_file(
        checkpoint_path,
        name="50M backbone checkpoint",
    )

    classifier_path = _resolve_required_file(
        classifier_path,
        name="50M classifier checkpoint",
    )

    normalized_channel_names = tuple(
        str(name) for name in channel_names
    )
    normalized_class_names = tuple(
        str(name) for name in class_names
    )

    if not normalized_channel_names:
        raise ValueError(
            "channel_names cannot be empty."
        )

    if not normalized_class_names:
        raise ValueError(
            "class_names cannot be empty."
        )

    if sample_rate <= 0:
        raise ValueError(
            f"sample_rate must be positive, got {sample_rate}."
        )

    config = Model50MConfig(
        checkpoint_path=checkpoint_path,
        classifier_path=classifier_path,
        device=device,

        target_sample_rate=target_sample_rate,
        window_seconds=window_seconds,
        patch_seconds=patch_seconds,
        patch_stride_seconds=patch_stride_seconds,

        filter_enabled=filter_enabled,
        filter_low_hz=filter_low_hz,
        filter_high_hz=filter_high_hz,
        reference_mode=reference_mode,

        strict_window_duration=strict_window_duration,

        output_layer_idx=output_layer_idx,
        aggregation=aggregation,
        num_classes=len(normalized_class_names),
        model_n_time_patches=model_n_time_patches,
    )

    adapter = Model50MAdapter(
        config=config,
        class_names=normalized_class_names,
        strict_head_metadata=strict_head_metadata,
    )

    preprocessor = Model50MPipelinePreprocessor(
        config=config,
        channel_names=normalized_channel_names,
        sample_rate=float(sample_rate),
        input_unit=str(input_unit),
    )

    return Model50MRuntime(
        config=config,
        adapter=adapter,
        preprocessor=preprocessor,
        class_names=normalized_class_names,
    )


def build_50m_runtime_from_metadata(
    *,
    checkpoint_path: str | Path,
    classifier_path: str | Path,
    metadata: EEGMetadataLike,
    device: str = "cpu",

    target_sample_rate: float,
    window_seconds: float,
    model_n_time_patches: int,
    patch_seconds: float,
    patch_stride_seconds: float,

    filter_enabled: bool = True,
    filter_low_hz: float = 0.1,
    filter_high_hz: float = 75.0,
    reference_mode: str = "none",

    output_layer_idx: int,
    aggregation: str,
    strict_head_metadata: bool = True,

) -> Model50MRuntime:
    """
    使用 EEGHDF5 metadata 直接构建运行时。

    metadata 至少要包含：
        channel_names
        sample_rate
        unit
        class_names
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

    return build_50m_runtime(
    checkpoint_path=checkpoint_path,
    classifier_path=classifier_path,
    channel_names=metadata.channel_names,
    sample_rate=float(metadata.sample_rate),
    input_unit=str(metadata.unit),
    class_names=metadata.class_names,
    device=device,

    target_sample_rate=target_sample_rate,
    window_seconds=window_seconds,
    model_n_time_patches=model_n_time_patches,
    patch_seconds=patch_seconds,
    patch_stride_seconds=patch_stride_seconds,

    filter_enabled=filter_enabled,
    filter_low_hz=filter_low_hz,
    filter_high_hz=filter_high_hz,
    reference_mode=reference_mode,

    output_layer_idx=output_layer_idx,
    aggregation=aggregation,
    strict_window_duration=True,
    strict_head_metadata=strict_head_metadata,
)
