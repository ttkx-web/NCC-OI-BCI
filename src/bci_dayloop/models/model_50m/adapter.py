from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .backbone import Model50MBackbone
from .classifier import (
    ClassifierLoadReport,
    Model50MClassifier,
    load_classifier_checkpoint,
)
from .config import Model50MConfig
from .preprocessing import (
    Model50MPreprocessor,
    PreprocessResult,
)
from .tokenization import (
    Model50MBatchedInput,
    Model50MTokenizer,
    stack_model50m_tokens,
)


@dataclass(frozen=True, slots=True)
class AdapterTiming:
    """最近一次推理各阶段耗时，单位为毫秒。"""

    input_conversion_ms: float
    tokenization_ms: float
    backbone_ms: float
    aggregation_ms: float
    classifier_ms: float
    total_ms: float

    def to_dict(self) -> dict[str, float]:
        return {
            "input_conversion_ms": self.input_conversion_ms,
            "tokenization_ms": self.tokenization_ms,
            "backbone_ms": self.backbone_ms,
            "aggregation_ms": self.aggregation_ms,
            "classifier_ms": self.classifier_ms,
            "total_ms": self.total_ms,
        }


@dataclass(frozen=True, slots=True)
class RawPredictionResult:
    """
    单个原始 EEG 窗口的完整预测结果。

    主要供 Smoke Test 或后续不经过 SlidingWindowDecoder 时使用。
    """

    prediction: int
    confidence: float
    probabilities: tuple[float, ...]
    timing: AdapterTiming

    mapped_channel_count: int
    missing_channel_count: int
    unknown_channel_names: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "probabilities": list(self.probabilities),
            "timing": self.timing.to_dict(),
            "mapped_channel_count": self.mapped_channel_count,
            "missing_channel_count": self.missing_channel_count,
            "unknown_channel_names": list(self.unknown_channel_names),
            "notes": list(self.notes),
        }


class Model50MAdapter:
    """
    50M 模型与当前 BCI Pipeline 之间的适配层。

    当前主要接口：

        predict_proba(X) -> np.ndarray [B, num_classes]

    可接受的 X 形状：

        [64, 1000]
        [B, 64, 1000]
        [64, 10, 100]
        [B, 64, 10, 100]

    其中：
        64   = 固定通道数
        1000 = 10 秒 × 100 Hz
        10   = 每个通道的时间 Patch 数
        100  = 每个 Patch 的点数

    当前 SlidingWindowDecoder 调用：

        model_input = preprocessor.transform(...)
        probabilities = model.predict_proba(
            model_input[None, ...]
        )[0]

    因此，只要 50M 专用预处理返回 [64, 1000]，
    当前 Decoder 不需要修改 predict_proba 调用方式。
    """

    def __init__(
        self,
        config: Model50MConfig,
        *,
        class_names: Sequence[str] | None = None,
        strict_head_metadata: bool = True,
    ) -> None:
        if config.classifier_path is None:
            raise ValueError(
                "Model50MAdapter requires config.classifier_path. "
                "For the current flow test, set it to "
                "'checkpoints/50m/test_linear_head.pt'."
            )

        self.config = config
        self.class_names = (
            tuple(str(name) for name in class_names)
            if class_names is not None
            else tuple(str(index) for index in range(config.num_classes))
        )

        if len(self.class_names) != config.num_classes:
            raise ValueError(
                "class_names length does not match num_classes: "
                f"class_names={len(self.class_names)}, "
                f"num_classes={config.num_classes}."
            )

        # 原始 EEG 直接推理时使用。
        self.preprocessor = Model50MPreprocessor(config)
        self.tokenizer = Model50MTokenizer(config)

        # 加载 Backbone。
        self.backbone = Model50MBackbone(
            config=config,
            load_checkpoint=True,
            freeze=True,
        )

        # 创建分类结构。
        self.classifier = Model50MClassifier(
            config=config,
            backbone=self.backbone,
        )

        # 加载 test_linear_head.pt 或正式分类头。
        self.classifier_load_report: ClassifierLoadReport = (
            load_classifier_checkpoint(
                classifier=self.classifier,
                checkpoint_path=config.classifier_path,
                strict_metadata=strict_head_metadata,
            )
        )

        self.classifier.eval()

        self.last_timing: AdapterTiming | None = None
        self._warned_about_inferred_mask = False

    @property
    def device(self) -> torch.device:
        return self.classifier.device

    @property
    def num_classes(self) -> int:
        return self.config.num_classes

    @property
    def expected_input_shape(self) -> tuple[int, int]:
        """预处理后的单窗口形状。"""
        return (
            self.config.n_channels,
            self.config.target_num_points,
        )

    @property
    def model_name(self) -> str:
        return "50M"

    def _normalize_input_shape(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        将不同输入形式统一成：

            [B, 64, 1000]
        """
        array = np.asarray(X)

        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(
                f"X must be numeric, got dtype={array.dtype}."
            )

        if not np.isfinite(array).all():
            raise ValueError("X contains NaN or Inf.")

        # --------------------------------------------------------------
        # 单个整段 EEG：[64, 1000]
        # --------------------------------------------------------------
        if array.ndim == 2:
            array = array[None, ...]

        # --------------------------------------------------------------
        # 单个 Patch 表示：[64, 10, 100]
        # 当前 Decoder 外面通常还会增加 batch 维，所以这种形式主要用于
        # 独立测试。
        # --------------------------------------------------------------
        elif array.ndim == 3:
            expected_patch_shape = (
                self.config.n_channels,
                self.config.num_time_patches,
                self.config.patch_num_points,
            )

            if tuple(array.shape) == expected_patch_shape:
                array = array.reshape(
                    self.config.n_channels,
                    self.config.target_num_points,
                )
                array = array[None, ...]

            # 否则认为已经是 [B, 64, 1000]。
            elif (
                array.shape[1] == self.config.n_channels
                and array.shape[2] == self.config.target_num_points
            ):
                pass

            else:
                raise ValueError(
                    "Unsupported 3-D EEG input shape. Expected either "
                    f"{expected_patch_shape} or "
                    f"[B, {self.config.n_channels}, "
                    f"{self.config.target_num_points}], "
                    f"got {array.shape}."
                )

        # --------------------------------------------------------------
        # 批量 Patch：[B, 64, 10, 100]
        # --------------------------------------------------------------
        elif array.ndim == 4:
            expected_tail_shape = (
                self.config.n_channels,
                self.config.num_time_patches,
                self.config.patch_num_points,
            )

            if tuple(array.shape[1:]) != expected_tail_shape:
                raise ValueError(
                    "Unsupported 4-D EEG input shape. Expected "
                    f"[B, {expected_tail_shape[0]}, "
                    f"{expected_tail_shape[1]}, "
                    f"{expected_tail_shape[2]}], "
                    f"got {array.shape}."
                )

            array = array.reshape(
                array.shape[0],
                self.config.n_channels,
                self.config.target_num_points,
            )

        else:
            raise ValueError(
                "EEG input must have 2, 3 or 4 dimensions, "
                f"got shape {array.shape}."
            )

        expected_tail = (
            self.config.n_channels,
            self.config.target_num_points,
        )

        if tuple(array.shape[1:]) != expected_tail:
            actual_seconds = (
                array.shape[-1] / self.config.target_sample_rate
                if array.ndim == 3
                else None
            )

            raise ValueError(
                "50M input shape mismatch. "
                f"Expected [B, {expected_tail[0]}, "
                f"{expected_tail[1]}] "
                f"({self.config.window_seconds:.1f} seconds at "
                f"{self.config.target_sample_rate:.1f} Hz), "
                f"got {array.shape}. "
                f"Actual time length is approximately "
                f"{actual_seconds!r} seconds. "
                "Do not pad a 4-second window to 10 seconds; "
                "set Replay window_sec to 10.0."
            )

        return np.ascontiguousarray(
            array,
            dtype=np.float32,
        )

    def _normalize_channel_masks(
        self,
        signals: np.ndarray,
        channel_valid_masks: np.ndarray | None,
    ) -> np.ndarray:
        """
        返回 [B, 64] 的通道有效性 Mask。

        推荐显式传入由 Model50MPreprocessor 产生的 Mask。

        当前 Pipeline 的 predict_proba(X) 只传 X，因此在没有 Mask 时，
        暂时根据“整条通道是否全为 0”进行推断。
        """
        batch_size = signals.shape[0]

        if channel_valid_masks is None:
            if not self._warned_about_inferred_mask:
                warnings.warn(
                    "channel_valid_masks was not provided. "
                    "Model50MAdapter will infer missing channels from "
                    "all-zero channel signals. This is acceptable for "
                    "the current flow test, but the formal Pipeline "
                    "should pass the mask produced by "
                    "Model50MPreprocessor.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self._warned_about_inferred_mask = True

            masks = np.any(
                np.abs(signals) > 1e-7,
                axis=-1,
            ).astype(np.float32)

        else:
            masks = np.asarray(channel_valid_masks)

            if masks.ndim == 1:
                masks = masks[None, ...]

            expected_shape = (
                batch_size,
                self.config.n_channels,
            )

            if tuple(masks.shape) != expected_shape:
                raise ValueError(
                    "channel_valid_masks shape mismatch: "
                    f"expected {expected_shape}, "
                    f"got {masks.shape}."
                )

            if not np.isfinite(masks).all():
                raise ValueError(
                    "channel_valid_masks contains NaN or Inf."
                )

            masks = (masks > 0.5).astype(np.float32)

        valid_counts = masks.sum(axis=1)

        if np.any(valid_counts <= 0):
            bad_indices = np.where(valid_counts <= 0)[0].tolist()

            raise ValueError(
                "At least one EEG sample contains no valid channels. "
                f"Invalid batch indices: {bad_indices}."
            )

        # 缺失通道必须保持全 0。
        for batch_index in range(batch_size):
            invalid_channels = masks[batch_index] < 0.5

            if invalid_channels.any() and not np.allclose(
                signals[batch_index, invalid_channels],
                0.0,
                atol=1e-7,
                rtol=0.0,
            ):
                raise ValueError(
                    "A channel marked invalid contains non-zero values. "
                    f"Batch index: {batch_index}."
                )

        return masks.astype(np.float32, copy=False)

    def _build_model_batch(
        self,
        X: np.ndarray,
        channel_valid_masks: np.ndarray | None = None,
    ) -> tuple[Model50MBatchedInput, float, float]:
        """
        将 Pipeline 输入转换为 Model50MBatchedInput。

        Returns:
            model_batch
            input_conversion_ms
            tokenization_ms
        """
        conversion_start = time.perf_counter()

        signals = self._normalize_input_shape(X)

        masks = self._normalize_channel_masks(
            signals=signals,
            channel_valid_masks=channel_valid_masks,
        )

        conversion_ms = (
            time.perf_counter() - conversion_start
        ) * 1000.0

        tokenization_start = time.perf_counter()

        tokenized_samples = [
            self.tokenizer.tokenize(
                signal=signals[index],
                channel_valid_mask=masks[index],
            )
            for index in range(signals.shape[0])
        ]

        model_batch = stack_model50m_tokens(
            tokenized_samples,
            device=self.device,
        )

        tokenization_ms = (
            time.perf_counter() - tokenization_start
        ) * 1000.0

        return model_batch, conversion_ms, tokenization_ms

    @torch.no_grad()
    def predict_proba(
        self,
        X: np.ndarray,
        channel_valid_masks: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        当前 Pipeline 兼容接口。

        Args:
            X:
                [B,64,1000]、[64,1000]、
                [B,64,10,100] 或 [64,10,100]。

            channel_valid_masks:
                可选，[B,64] 或 [64]。

        Returns:
            probabilities:
                np.float32，[B, num_classes]。
        """
        total_start = time.perf_counter()

        model_batch, conversion_ms, tokenization_ms = (
            self._build_model_batch(
                X=X,
                channel_valid_masks=channel_valid_masks,
            )
        )

        output = self.classifier.predict_batch(model_batch)

        total_ms = (
            time.perf_counter() - total_start
        ) * 1000.0

        self.last_timing = AdapterTiming(
            input_conversion_ms=float(conversion_ms),
            tokenization_ms=float(tokenization_ms),
            backbone_ms=float(output.backbone_ms),
            aggregation_ms=float(output.aggregation_ms),
            classifier_ms=float(output.classifier_ms),
            total_ms=float(total_ms),
        )

        probabilities = (
            output.probabilities
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

        expected_shape = (
            model_batch.batch_size,
            self.config.num_classes,
        )

        if probabilities.shape != expected_shape:
            raise RuntimeError(
                "Unexpected probability shape: "
                f"expected {expected_shape}, "
                f"got {probabilities.shape}."
            )

        if not np.isfinite(probabilities).all():
            raise RuntimeError(
                "Model50MAdapter produced NaN or Inf probabilities."
            )

        return probabilities

    @torch.no_grad()
    def predict(
        self,
        X: np.ndarray,
        channel_valid_masks: np.ndarray | None = None,
    ) -> np.ndarray:
        """返回类别编号，[B]。"""
        probabilities = self.predict_proba(
            X=X,
            channel_valid_masks=channel_valid_masks,
        )

        return probabilities.argmax(axis=-1).astype(
            np.int64,
            copy=False,
        )

    @torch.no_grad()
    def extract_embeddings(
        self,
        X: np.ndarray,
        channel_valid_masks: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        返回 Flatten/Mean 之后、分类头之前的特征。

        当前默认 flatten：

            [B, 327680]
        """
        model_batch, _, _ = self._build_model_batch(
            X=X,
            channel_valid_masks=channel_valid_masks,
        )

        features = self.classifier.extract_features(
            model_batch
        )

        return (
            features.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

    @torch.no_grad()
    def predict_raw(
        self,
        *,
        signal: np.ndarray,
        channel_names: Sequence[str],
        original_sample_rate: float,
        input_unit: str,
    ) -> RawPredictionResult:
        """
        直接从原始 EEG [C,T] 完成预处理和预测。

        当前 SlidingWindowDecoder 暂时不会调用这个方法；
        它主要用于独立 Smoke Test 和后续重构。
        """
        total_start = time.perf_counter()

        preprocess_start = time.perf_counter()

        preprocessed: PreprocessResult = self.preprocessor(
            signal=signal,
            channel_names=channel_names,
            original_sample_rate=original_sample_rate,
            input_unit=input_unit,
        )

        preprocess_ms = (
            time.perf_counter() - preprocess_start
        ) * 1000.0

        probabilities = self.predict_proba(
            X=preprocessed.signal[None, ...],
            channel_valid_masks=(
                preprocessed.channel_valid_mask[None, ...]
            ),
        )[0]

        prediction = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction])

        base_timing = self.last_timing

        if base_timing is None:
            raise RuntimeError(
                "Adapter timing was not recorded."
            )

        total_ms = (
            time.perf_counter() - total_start
        ) * 1000.0

        timing = AdapterTiming(
            input_conversion_ms=(
                preprocess_ms
                + base_timing.input_conversion_ms
            ),
            tokenization_ms=base_timing.tokenization_ms,
            backbone_ms=base_timing.backbone_ms,
            aggregation_ms=base_timing.aggregation_ms,
            classifier_ms=base_timing.classifier_ms,
            total_ms=float(total_ms),
        )

        self.last_timing = timing

        return RawPredictionResult(
            prediction=prediction,
            confidence=confidence,
            probabilities=tuple(
                float(value) for value in probabilities
            ),
            timing=timing,
            mapped_channel_count=(
                preprocessed.mapped_channel_count
            ),
            missing_channel_count=(
                preprocessed.missing_channel_count
            ),
            unknown_channel_names=(
                preprocessed.unknown_channel_names
            ),
            notes=preprocessed.notes,
        )

    def health_check(self) -> dict[str, Any]:
        """
        使用随机预处理后数据检查完整链路。

        test_linear_head.pt 为随机分类头，因此仅检查运行是否成功。
        """
        rng = np.random.default_rng(seed=42)

        signal = rng.standard_normal(
            (
                1,
                self.config.n_channels,
                self.config.target_num_points,
            ),
            dtype=np.float32,
        )

        masks = np.ones(
            (1, self.config.n_channels),
            dtype=np.float32,
        )

        probabilities = self.predict_proba(
            X=signal,
            channel_valid_masks=masks,
        )

        return {
            "status": "ok",
            "model_name": self.model_name,
            "device": str(self.device),
            "checkpoint_path": str(
                self.config.checkpoint_path
            ),
            "classifier_path": str(
                self.config.classifier_path
            ),
            "input_shape": tuple(signal.shape),
            "probability_shape": tuple(
                probabilities.shape
            ),
            "probability_sum": float(
                probabilities[0].sum()
            ),
            "prediction": int(
                probabilities[0].argmax()
            ),
            "confidence": float(
                probabilities[0].max()
            ),
            "last_timing": (
                self.last_timing.to_dict()
                if self.last_timing is not None
                else None
            ),
            "warning": (
                "The current test_linear_head.pt is not trained. "
                "Prediction and confidence have no business meaning."
            ),
        }


def build_model50m_adapter(
    config: Model50MConfig,
    *,
    class_names: Sequence[str] | None = None,
    strict_head_metadata: bool = True,
) -> Model50MAdapter:
    """统一构建入口。"""
    return Model50MAdapter(
        config=config,
        class_names=class_names,
        strict_head_metadata=strict_head_metadata,
    )