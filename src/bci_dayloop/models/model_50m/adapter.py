from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from bci_dayloop.models.base import BaseModelAdapter, ModelInput
from bci_dayloop.utils.config import dump_json, dump_yaml, load_yaml

from .backbone import Model50MBackbone
from .classifier import (
    ClassifierLoadReport,
    Model50MClassifier,
    load_classifier_checkpoint,
    save_classifier_checkpoint,
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


class Model50MAdapter(BaseModelAdapter):
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

    model_name = "50m-linear"

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

                "'checkpoints/heads/stage05/"
                "bnci2014_001/subject_01/"
                "10s_flatten/head.pt'."
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
        X: ModelInput,
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
        if isinstance(X, dict):
            if "signal" not in X:
                raise ValueError("50M model input dictionary is missing required key 'signal'")
            if "channel_valid_mask" not in X:
                raise ValueError("50M model input dictionary is missing required key 'channel_valid_mask'")
            if channel_valid_masks is not None:
                raise ValueError("Pass channel_valid_mask either in the model input dictionary or as a keyword, not both")
            signals = X["signal"]
            channel_valid_masks = X["channel_valid_mask"]
        elif isinstance(X, np.ndarray):
            signals = X
        else:
            raise TypeError("50M model input must be numpy.ndarray or dict[str, numpy.ndarray]")

        total_start = time.perf_counter()

        model_batch, conversion_ms, tokenization_ms = (
            self._build_model_batch(
                X=signals,
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

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        del X, y, kwargs
        raise NotImplementedError(
            "Model50MAdapter currently supports loading an existing classifier head and inference only; "
            "this does not mean a classifier head has been trained. A formal 50M classifier training script "
            "must provide fit() in a later delivery."
        )

    def update(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        del X, y, kwargs
        raise NotImplementedError(
            "Model50MAdapter currently supports loading an existing classifier head and inference only; "
            "this does not mean a classifier head has been trained. A formal 50M classifier training script "
            "must provide update() in a later delivery."
        )

    @staticmethod
    def _checkpoint_sha256(path: Path) -> str | None:
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _required_package_file(package: Path, name: str) -> Path:
        path = package / name
        if not path.is_file():
            raise FileNotFoundError(f"50M model package file was not found: {path.resolve()}")
        return path

    @staticmethod
    def _resolve_checkpoint_reference(package: Path, value: str | Path) -> Path:
        reference = Path(value).expanduser()
        candidates = [reference] if reference.is_absolute() else [package / reference, reference]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"50M backbone checkpoint was not found: {candidates[0].resolve()}")

    def save(
        self,
        path: str | Path,
        *,
        preprocessing: dict[str, Any] | None = None,
        label_map: dict[int | str, str] | None = None,
        command_map: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Path:
        del kwargs
        package = Path(path)
        package.mkdir(parents=True, exist_ok=True)
        classifier_path = save_classifier_checkpoint(
            self.classifier,
            package / "classifier.pt",
            extra_metadata={"class_names": list(self.class_names)},
        )
        model_payload = {
            "name": self.model_name,
            "num_classes": int(self.config.num_classes),
            "aggregation": self.config.aggregation,
            "output_layer_idx": int(self.config.output_layer_idx),
            "window_seconds": float(self.config.window_seconds),
            "target_sample_rate": float(self.config.target_sample_rate),
            "patch_seconds": float(self.config.patch_seconds),
            "patch_stride_seconds": float(self.config.patch_stride_seconds),
            "n_channels": int(self.config.n_channels),
            "d_model": int(self.config.d_model),
            "n_heads": int(self.config.n_heads),
            "depth": int(self.config.depth),
            "mlp_ratio": float(self.config.mlp_ratio),
            "dropout": float(self.config.dropout),
            "class_names": list(self.class_names),
            "num_time_patches": int(
                self.config.num_time_patches
            ),
            "model_n_time_patches": int(
                self.config.model_n_time_patches
            ),
        }
        preprocessing_payload = {
            "filter_enabled": bool(self.config.filter_enabled),
            "filter_low_hz": float(self.config.filter_low_hz),
            "filter_high_hz": float(self.config.filter_high_hz),
            "filter_order": int(self.config.filter_order),
            "reference_mode": self.config.reference_mode,
            "zscore_enabled": bool(self.config.zscore_enabled),
            "zscore_eps": float(self.config.zscore_eps),
            "missing_channel_fill_value": float(self.config.missing_channel_fill_value),
            "strict_window_duration": bool(self.config.strict_window_duration),
            "window_tolerance_seconds": float(self.config.window_tolerance_seconds),
        }
        preprocessing_payload.update(preprocessing or {})
        dump_yaml(model_payload, package / "model.yaml")
        dump_yaml(preprocessing_payload, package / "preprocessing.yaml")
        saved_label_map = label_map or {str(index): name for index, name in enumerate(self.class_names)}
        dump_json({str(key): value for key, value in saved_label_map.items()}, package / "label_map.json")
        dump_json(command_map or {}, package / "command_map.json")
        checkpoint = Path(self.config.checkpoint_path).expanduser().resolve()
        dump_json(
            {
                "backbone": "50M",
                "checkpoint_path": str(self.config.checkpoint_path),
                "checkpoint_path_absolute": str(checkpoint),
                "checkpoint_sha256": self._checkpoint_sha256(checkpoint),
                "classifier_path": classifier_path.name,
            },
            package / "base_model.json",
        )
        return package

    def load(self, path: str | Path) -> "Model50MAdapter":
        package = Path(path)
        model = load_yaml(self._required_package_file(package, "model.yaml"))
        if model.get("name") != self.model_name:
            raise ValueError(f"Expected 50M model package name '{self.model_name}', got {model.get('name')!r}")
        for key, actual, expected in (
            ("num_classes", model.get("num_classes"), self.config.num_classes),
            ("aggregation", model.get("aggregation"), self.config.aggregation),
            ("output_layer_idx", model.get("output_layer_idx"), self.config.output_layer_idx),
        ):
            if actual != expected:
                raise ValueError(f"50M model package metadata mismatch for {key}: package={actual!r}, adapter={expected!r}")
        report = load_classifier_checkpoint(
            self.classifier,
            self._required_package_file(package, "classifier.pt"),
            strict_metadata=True,
        )
        saved_classes = report.metadata.get("class_names")
        if saved_classes is not None and tuple(saved_classes) != self.class_names:
            raise ValueError("50M classifier metadata class_names does not match the current adapter")
        self.classifier_load_report = report
        return self

    @classmethod
    def from_package(cls, path: str | Path, device: str = "cpu") -> "Model50MAdapter":
        package = Path(path).expanduser().resolve()
        model_path = cls._required_package_file(package, "model.yaml")
        preprocessing_path = cls._required_package_file(package, "preprocessing.yaml")
        label_path = cls._required_package_file(package, "label_map.json")
        cls._required_package_file(package, "command_map.json")
        base_path = cls._required_package_file(package, "base_model.json")
        classifier_path = cls._required_package_file(package, "classifier.pt")
        model = load_yaml(model_path)
        if model.get("name") != cls.model_name:
            raise ValueError(f"Expected 50M model package name '{cls.model_name}', got {model.get('name')!r}")
        preprocessing = load_yaml(preprocessing_path)
        with label_path.open("r", encoding="utf-8") as handle:
            label_map = json.load(handle)
        with base_path.open("r", encoding="utf-8") as handle:
            base_model = json.load(handle)
        checkpoint_value = base_model.get("checkpoint_path_absolute") or base_model.get("checkpoint_path")
        if checkpoint_value is None:
            raise ValueError(f"50M model package base_model.json is missing checkpoint_path: {base_path}")
        checkpoint_path = cls._resolve_checkpoint_reference(package, checkpoint_value)
        class_names = tuple(str(label_map[str(index)]) for index in range(int(model["num_classes"])))
        config = Model50MConfig(
            checkpoint_path=checkpoint_path,
            classifier_path=classifier_path,
            device=device,
            target_sample_rate=float(model["target_sample_rate"]),
            window_seconds=float(model["window_seconds"]),
            patch_seconds=float(model["patch_seconds"]),
            patch_stride_seconds=float(model["patch_stride_seconds"]),
            n_channels=int(model["n_channels"]),
            filter_enabled=bool(preprocessing.get("filter_enabled", True)),
            filter_low_hz=float(preprocessing.get("filter_low_hz", 0.1)),
            filter_high_hz=float(preprocessing.get("filter_high_hz", 75.0)),
            filter_order=int(preprocessing.get("filter_order", 4)),
            reference_mode=preprocessing.get("reference_mode", "none"),
            zscore_enabled=bool(preprocessing.get("zscore_enabled", True)),
            zscore_eps=float(preprocessing.get("zscore_eps", 1e-8)),
            missing_channel_fill_value=float(preprocessing.get("missing_channel_fill_value", 0.0)),
            strict_window_duration=bool(preprocessing.get("strict_window_duration", True)),
            window_tolerance_seconds=float(preprocessing.get("window_tolerance_seconds", 0.02)),
            d_model=int(model["d_model"]),
            n_heads=int(model["n_heads"]),
            depth=int(model["depth"]),
            mlp_ratio=float(model["mlp_ratio"]),
            dropout=float(model["dropout"]),
            output_layer_idx=int(model["output_layer_idx"]),
            aggregation=model["aggregation"],
            num_classes=int(model["num_classes"]),
            model_n_time_patches=int(
                model.get("model_n_time_patches", 10)
            ),
        )
        return cls(config=config, class_names=class_names, strict_head_metadata=True)

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
