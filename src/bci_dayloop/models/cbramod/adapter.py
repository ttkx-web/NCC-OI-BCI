from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bci_dayloop.models.base import (
    BaseModelAdapter,
    ModelInput,
)
from bci_dayloop.utils.config import (
    dump_json,
    dump_yaml,
    load_yaml,
)

from .backend import CBraModBackend
from .backbone import CBraModBackbone
from .classifier import (
    CBraModClassifier,
    build_cbramod_classifier,
)
from .config import CBraModConfig
from .runtime import (
    CBraModClassifierLoadReport,
    checkpoint_sha256,
    load_cbramod_classifier_checkpoint,
    save_cbramod_classifier_checkpoint,
)


@dataclass(frozen=True, slots=True)
class CBraModAdapterTiming:
    """Adapter 最近一次批量推理耗时，单位为毫秒。"""

    input_conversion_ms: float
    model_ms: float
    total_ms: float

    def to_dict(self) -> dict[str, float]:
        return {
            "input_conversion_ms": (
                self.input_conversion_ms
            ),
            "model_ms": self.model_ms,
            "total_ms": self.total_ms,
        }


class CBraModAdapter(BaseModelAdapter):
    """
    兼容旧 BaseModelAdapter / ModelFactory 的 CBRaMod 适配层。

    这个类只接受已经完成 CBRaMod 专属预处理的输入：

        [22, 4, 200]
        [B, 22, 4, 200]
        {"signal": 上述任意一种 ndarray}

    它不接受原始 [C, T] EEG，也不执行通道映射、重采样或 patchify。
    原始 EEG Runtime 推理必须使用：

        CBraModRuntime.runtime_model
    """

    model_name = "cbramod-frozen-head"

    def __init__(
        self,
        config: CBraModConfig,
        *,
        class_names: Sequence[str] | None = None,
        strict_head_metadata: bool = True,
    ) -> None:
        if config.classifier_path is None:
            raise ValueError(
                "CBraModAdapter requires config.classifier_path."
            )

        self.config = config

        self.class_names = (
            tuple(str(name) for name in class_names)
            if class_names is not None
            else tuple(
                str(index)
                for index in range(config.num_classes)
            )
        )

        if len(self.class_names) != config.num_classes:
            raise ValueError(
                "class_names length does not match num_classes: "
                f"class_names={len(self.class_names)}, "
                f"num_classes={config.num_classes}."
            )

        if len(set(self.class_names)) != len(
            self.class_names
        ):
            raise ValueError(
                "class_names contains duplicates."
            )

        self.backbone = CBraModBackbone(config)

        self.classifier: CBraModClassifier = (
            build_cbramod_classifier(config).to(
                self.backbone.device
            )
        )

        self.classifier_load_report = (
            load_cbramod_classifier_checkpoint(
                self.classifier,
                config.classifier_path,
                config=config,
                class_names=self.class_names,
                strict_metadata=strict_head_metadata,
            )
        )

        self.backend = CBraModBackend(
            backbone=self.backbone,
            classifier=self.classifier,
            config=self.config,
        )

        self.last_timing: CBraModAdapterTiming | None = None

    @property
    def device(self) -> torch.device:
        return self.backend.device

    @property
    def num_classes(self) -> int:
        return self.config.num_classes

    @property
    def expected_input_shape(
        self,
    ) -> tuple[int, int, int]:
        """单个预处理输入的 [C, S, P]。"""
        return self.config.expected_unbatched_shape

    def _extract_signal(
        self,
        X: ModelInput,
    ) -> np.ndarray:
        if isinstance(X, np.ndarray):
            signal = X

        elif isinstance(X, dict):
            if "signal" not in X:
                raise ValueError(
                    "CBraMod model input dictionary is missing "
                    "required key 'signal'."
                )

            signal = X["signal"]

        else:
            raise TypeError(
                "CBraMod model input must be numpy.ndarray or "
                "dict[str, numpy.ndarray], got "
                f"{type(X).__name__}."
            )

        if not isinstance(signal, np.ndarray):
            raise TypeError(
                "CBraMod signal must be numpy.ndarray, got "
                f"{type(signal).__name__}."
            )

        return signal

    def _normalize_input_shape(
        self,
        X: ModelInput,
    ) -> np.ndarray:
        signal = np.asarray(self._extract_signal(X))

        if not np.issubdtype(signal.dtype, np.number):
            raise TypeError(
                "CBraMod input must be numeric, got "
                f"dtype={signal.dtype}."
            )

        if not np.isfinite(signal).all():
            raise ValueError(
                "CBraMod input contains NaN or Inf."
            )

        expected_tail = self.expected_input_shape

        if signal.ndim == 3:
            if tuple(signal.shape) != expected_tail:
                raise ValueError(
                    "CBraMod single-window input shape mismatch. "
                    "Expected "
                    f"{expected_tail}, got "
                    f"{tuple(signal.shape)}."
                )

            signal = signal[None, ...]

        elif signal.ndim == 4:
            if tuple(signal.shape[1:]) != expected_tail:
                raise ValueError(
                    "CBraMod batched input shape mismatch. "
                    "Expected "
                    f"[B, {expected_tail[0]}, "
                    f"{expected_tail[1]}, "
                    f"{expected_tail[2]}], got "
                    f"{tuple(signal.shape)}."
                )

        else:
            raise ValueError(
                "CBraMod input must have shape [C, S, P] or "
                "[B, C, S, P]. Raw [C, T] input is not accepted "
                "by CBraModAdapter; use RuntimeModel instead. "
                f"Got {tuple(signal.shape)}."
            )

        if signal.shape[0] <= 0:
            raise ValueError(
                "CBraMod input batch size must be positive."
            )

        return np.ascontiguousarray(
            signal,
            dtype=np.float32,
        )

    @torch.no_grad()
    def predict_proba(
        self,
        X: ModelInput,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        返回 shape [B, num_classes] 的 float32 概率。

        注意：RuntimeModel 当前一次预测一个窗口，因此这里按样本循环，
        保持旧 Adapter 的批量 predict_proba 接口兼容性。
        """

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))

            raise TypeError(
                "CBraModAdapter.predict_proba got unexpected "
                f"keyword arguments: {unexpected}."
            )

        total_started = time.perf_counter()

        conversion_started = time.perf_counter()

        signal = self._normalize_input_shape(X)

        signal_tensor = torch.from_numpy(signal)

        conversion_ms = (
            time.perf_counter() - conversion_started
        ) * 1000.0

        model_started = time.perf_counter()

        probabilities: list[np.ndarray] = []

        for batch_index in range(signal_tensor.shape[0]):
            output = self.backend.predict_tensor(
                {"signal": signal_tensor[batch_index:batch_index + 1]},
                return_features=False,
            )

            probabilities.append(
                output.probabilities
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )

        result = np.concatenate(
            probabilities,
            axis=0,
        ).astype(np.float32, copy=False)

        model_ms = (
            time.perf_counter() - model_started
        ) * 1000.0

        total_ms = (
            time.perf_counter() - total_started
        ) * 1000.0

        expected_shape = (
            signal.shape[0],
            self.num_classes,
        )

        if tuple(result.shape) != expected_shape:
            raise RuntimeError(
                "CBraModAdapter returned an unexpected "
                "probability shape. Expected "
                f"{expected_shape}, got "
                f"{tuple(result.shape)}."
            )

        if not np.isfinite(result).all():
            raise RuntimeError(
                "CBraModAdapter produced NaN or Inf "
                "probabilities."
            )

        self.last_timing = CBraModAdapterTiming(
            input_conversion_ms=float(conversion_ms),
            model_ms=float(model_ms),
            total_ms=float(total_ms),
        )

        return result

    @torch.no_grad()
    def extract_embeddings(
        self,
        X: ModelInput,
    ) -> np.ndarray:
        """
        返回分类头前的 CBRaMod encoder 输出：

            [B, 22, 4, 200]
        """

        signal = self._normalize_input_shape(X)

        features = self.backend.encode_tensor(
            {
                "signal": torch.from_numpy(signal),
            }
        )

        return (
            features.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del X, y, kwargs

        raise NotImplementedError(
            "CBraModAdapter is for package loading and inference. "
            "Use train_cbramod_population_head.py to train the "
            "frozen-backbone classification head."
        )

    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del X, y, kwargs

        raise NotImplementedError(
            "cbramod-frozen-head does not support online updates. "
            "NeuroOnline must be implemented as a separate "
            "explicit experiment."
        )

    @staticmethod
    def _required_package_file(
        package: Path,
        name: str,
    ) -> Path:
        path = package / name

        if not path.is_file():
            raise FileNotFoundError(
                "CBraMod model package file was not found: "
                f"{path.resolve()}"
            )

        return path

    @staticmethod
    def _resolve_checkpoint_reference(
        package: Path,
        value: str | Path,
    ) -> Path:
        reference = Path(value).expanduser()

        candidates = (
            [reference]
            if reference.is_absolute()
            else [
                package / reference,
                reference,
            ]
        )

        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()

        raise FileNotFoundError(
            "CBraMod backbone checkpoint was not found. "
            f"First attempted path: {candidates[0].resolve()}"
        )

    def save(
        self,
        path: str | Path,
        *,
        preprocessing: dict[str, Any] | None = None,
        label_map: dict[int | str, str] | None = None,
        command_map: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Path:
        """
        导出旧 ModelFactory 兼容的 CBRaMod package。

        Runtime Package loader 后续应直接从该目录读取：
        - model.yaml
        - preprocessing.yaml
        - classifier.pt
        - label_map.json
        - command_map.json
        - base_model.json
        """

        del kwargs

        package = Path(path)
        package.mkdir(parents=True, exist_ok=True)

        classifier_path = save_cbramod_classifier_checkpoint(
            self.classifier,
            package / "classifier.pt",
            config=self.config,
            class_names=self.class_names,
        )

        model_payload = {
            "name": self.model_name,
            "architecture": "cbramod_iclr2025",
            "num_classes": int(self.config.num_classes),
            "class_names": list(self.class_names),
            "head_type": self.config.head_type,
            "head_hidden_dim_1": (
                self.config.head_hidden_dim_1
            ),
            "head_hidden_dim_2": (
                self.config.head_hidden_dim_2
            ),
            "head_dropout": self.config.head_dropout,
            "classifier_input_dim": (
                self.config.classifier_input_dim
            ),
            "backbone_output_dim": (
                self.config.backbone_output_dim
            ),
            "n_channels": self.config.n_channels,
            "standard_channels": list(
                self.config.standard_channels
            ),
            "time_segments": self.config.time_segments,
            "points_per_patch": (
                self.config.points_per_patch
            ),
            "window_seconds": (
                self.config.window_seconds
            ),
            "target_sample_rate": (
                self.config.target_sample_rate
            ),
            "input_unit": self.config.input_unit,
        }

        preprocessing_payload = {
            "strict_window_duration": (
                self.config.strict_window_duration
            ),
            "window_tolerance_seconds": (
                self.config.window_tolerance_seconds
            ),
            "filter_enabled": self.config.filter_enabled,
            "filter_low_hz": self.config.filter_low_hz,
            "filter_high_hz": self.config.filter_high_hz,
            "filter_order": self.config.filter_order,
            "reference_mode": self.config.reference_mode,
            "normalization": self.config.normalization,
            "zscore_eps": self.config.zscore_eps,
            "missing_channel_policy": (
                self.config.missing_channel_policy
            ),
            "min_observed_channels": (
                self.config.min_observed_channels
            ),
            "spline_alpha": self.config.spline_alpha,
        }

        preprocessing_payload.update(preprocessing or {})

        dump_yaml(
            model_payload,
            package / "model.yaml",
        )

        dump_yaml(
            preprocessing_payload,
            package / "preprocessing.yaml",
        )

        saved_label_map = label_map or {
            str(index): class_name
            for index, class_name in enumerate(
                self.class_names
            )
        }

        dump_json(
            {
                str(key): value
                for key, value in saved_label_map.items()
            },
            package / "label_map.json",
        )

        dump_json(
            command_map or {},
            package / "command_map.json",
        )

        backbone_path = Path(
            self.config.checkpoint_path
        ).expanduser().resolve()

        dump_json(
            {
                "backbone": "CBraMod",
                "checkpoint_path": str(
                    self.config.checkpoint_path
                ),
                "checkpoint_path_absolute": str(
                    backbone_path
                ),
                "checkpoint_sha256": checkpoint_sha256(
                    backbone_path
                ),
                "classifier_path": classifier_path.name,
                "classifier_sha256": checkpoint_sha256(
                    classifier_path
                ),
            },
            package / "base_model.json",
        )

        return package

    def load(
        self,
        path: str | Path,
    ) -> "CBraModAdapter":
        """从已有 package 覆盖加载分类头。"""

        package = Path(path)

        model = load_yaml(
            self._required_package_file(
                package,
                "model.yaml",
            )
        )

        if model.get("name") != self.model_name:
            raise ValueError(
                "Expected CBraMod model package name "
                f"{self.model_name!r}, got "
                f"{model.get('name')!r}."
            )

        expected_values = {
            "num_classes": self.config.num_classes,
            "head_type": self.config.head_type,
            "classifier_input_dim": (
                self.config.classifier_input_dim
            ),
            "n_channels": self.config.n_channels,
            "time_segments": self.config.time_segments,
            "points_per_patch": (
                self.config.points_per_patch
            ),
        }

        for key, expected_value in expected_values.items():
            actual_value = model.get(key)

            if actual_value != expected_value:
                raise ValueError(
                    "CBraMod package metadata mismatch for "
                    f"{key}: package={actual_value!r}, "
                    f"adapter={expected_value!r}."
                )

        self.classifier_load_report = (
            load_cbramod_classifier_checkpoint(
                self.classifier,
                self._required_package_file(
                    package,
                    "classifier.pt",
                ),
                config=self.config,
                class_names=self.class_names,
                strict_metadata=True,
            )
        )

        return self

    @classmethod
    def from_package(
        cls,
        path: str | Path,
        device: str = "cpu",
    ) -> "CBraModAdapter":
        """从 CBRaMod Model Package 创建旧接口 Adapter。"""

        package = Path(path).expanduser().resolve()

        model_path = cls._required_package_file(
            package,
            "model.yaml",
        )

        preprocessing_path = cls._required_package_file(
            package,
            "preprocessing.yaml",
        )

        label_path = cls._required_package_file(
            package,
            "label_map.json",
        )

        cls._required_package_file(
            package,
            "command_map.json",
        )

        base_path = cls._required_package_file(
            package,
            "base_model.json",
        )

        classifier_path = cls._required_package_file(
            package,
            "classifier.pt",
        )

        model = load_yaml(model_path)

        if model.get("name") != cls.model_name:
            raise ValueError(
                "Expected CBraMod model package name "
                f"{cls.model_name!r}, got "
                f"{model.get('name')!r}."
            )

        preprocessing = load_yaml(preprocessing_path)

        with label_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            label_map = json.load(handle)

        with base_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            base_model = json.load(handle)

        checkpoint_value = (
            base_model.get("checkpoint_path_absolute")
            or base_model.get("checkpoint_path")
        )

        if checkpoint_value is None:
            raise ValueError(
                "CBraMod base_model.json is missing "
                f"checkpoint_path: {base_path}"
            )

        checkpoint_path = cls._resolve_checkpoint_reference(
            package,
            checkpoint_value,
        )

        num_classes = int(model["num_classes"])

        try:
            class_names = tuple(
                str(label_map[str(index)])
                for index in range(num_classes)
            )
        except KeyError as error:
            raise ValueError(
                "CBraMod label_map.json must use continuous "
                "numeric keys from 0 to num_classes - 1."
            ) from error

        saved_class_names = tuple(
            str(name)
            for name in model.get(
                "class_names",
                class_names,
            )
        )

        if saved_class_names != class_names:
            raise ValueError(
                "CBraMod model.yaml class_names does not match "
                "label_map.json order."
            )

        config = CBraModConfig(
            checkpoint_path=checkpoint_path,
            classifier_path=classifier_path,
            device=device,

            target_sample_rate=float(
                model["target_sample_rate"]
            ),
            window_seconds=float(
                model["window_seconds"]
            ),
            n_channels=int(model["n_channels"]),
            standard_channels=tuple(
                str(name)
                for name in model["standard_channels"]
            ),
            time_segments=int(model["time_segments"]),
            points_per_patch=int(
                model["points_per_patch"]
            ),
            input_unit=str(model["input_unit"]),

            strict_window_duration=bool(
                preprocessing.get(
                    "strict_window_duration",
                    True,
                )
            ),
            window_tolerance_seconds=float(
                preprocessing.get(
                    "window_tolerance_seconds",
                    0.02,
                )
            ),
            filter_enabled=bool(
                preprocessing.get(
                    "filter_enabled",
                    False,
                )
            ),
            filter_low_hz=float(
                preprocessing.get(
                    "filter_low_hz",
                    0.1,
                )
            ),
            filter_high_hz=float(
                preprocessing.get(
                    "filter_high_hz",
                    75.0,
                )
            ),
            filter_order=int(
                preprocessing.get(
                    "filter_order",
                    4,
                )
            ),
            reference_mode=str(
                preprocessing.get(
                    "reference_mode",
                    "none",
                )
            ),
            normalization=str(
                preprocessing.get(
                    "normalization",
                    "none",
                )
            ),
            zscore_eps=float(
                preprocessing.get(
                    "zscore_eps",
                    1e-8,
                )
            ),
            missing_channel_policy=str(
                preprocessing.get(
                    "missing_channel_policy",
                    "error",
                )
            ),
            min_observed_channels=int(
                preprocessing.get(
                    "min_observed_channels",
                    2,
                )
            ),
            spline_alpha=float(
                preprocessing.get(
                    "spline_alpha",
                    1e-5,
                )
            ),

            num_classes=num_classes,
            backbone_output_dim=int(
                model["backbone_output_dim"]
            ),
            head_type=str(model["head_type"]),
            head_hidden_dim_1=int(
                model["head_hidden_dim_1"]
            ),
            head_hidden_dim_2=int(
                model["head_hidden_dim_2"]
            ),
            head_dropout=float(
                model["head_dropout"]
            ),
        )

        return cls(
            config=config,
            class_names=class_names,
            strict_head_metadata=True,
        )