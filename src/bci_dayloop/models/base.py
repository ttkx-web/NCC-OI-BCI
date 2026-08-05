from __future__ import annotations

import math

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

import torch

from bci_dayloop.runtime.types import (
    CanonicalEEGWindow,
    InputContract,
    PreparedModelInput,
    ModelOutput,
    ModelTensor,
)

ModelInput: TypeAlias = np.ndarray | dict[str, np.ndarray]

def add_batch_dimension(value: ModelInput) -> ModelInput:
    """Return a batched copy of the input container without changing its contents."""
    if isinstance(value, np.ndarray):
        return value[None, ...]
    if isinstance(value, dict):
        if not value:
            raise ValueError("Cannot add a batch dimension to an empty model input dictionary")
        batched: dict[str, np.ndarray] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Model input dictionary keys must be strings")
            if not isinstance(item, np.ndarray):
                raise TypeError(f"Model input '{key}' must be a numpy.ndarray, got {type(item).__name__}")
            batched[key] = item[None, ...]
        return batched
    raise TypeError(
        f"Unsupported model input type {type(value).__name__}; expected numpy.ndarray or dict[str, numpy.ndarray]"
    )

class ModelInputTransform(ABC):
    """
    CanonicalEEGWindow 到模型专属输入之间的统一接口。

    子类负责：
    - 模型专属滤波
    - 重采样
    - 通道映射与补齐
    - 固定窗口长度
    - 训练时一致的标准化
    - Tensor 和 mask 构造
    """

    @property
    @abstractmethod
    def input_contract(self) -> InputContract:
        """返回模型处理完成后的目标输入契约."""

    def __call__(
        self,
        window: CanonicalEEGWindow,
    ) -> PreparedModelInput:
        self.validate_canonical_window(window)
        prepared = self.transform(window)
        self.validate_prepared_input(prepared)
        return prepared

    def validate_canonical_window(
        self,
        window: CanonicalEEGWindow,
    ) -> None:
        data = np.asarray(window.data)

        if data.ndim != 2:
            raise ValueError(
                "Canonical EEG data must have shape [C,T], "
                f"got {data.shape}."
            )

        if data.shape[0] != len(window.channel_names):
            raise ValueError(
                "Canonical EEG channel count mismatch: "
                f"data={data.shape[0]}, "
                f"names={len(window.channel_names)}."
            )

        if data.shape[0] == 0:
            raise ValueError(
                "Canonical EEG contains no channels."
            )

        if data.shape[1] <= 1:
            raise ValueError(
                "Canonical EEG contains too few "
                f"time points: {data.shape[1]}."
            )

        if not np.issubdtype(data.dtype, np.number):
            raise TypeError(
                "Canonical EEG must be numeric, "
                f"got dtype={data.dtype}."
            )

        if not np.isfinite(data).all():
            raise ValueError(
                "Canonical EEG contains NaN or Inf."
            )

        if (
            not math.isfinite(window.sample_rate)
            or window.sample_rate <= 0
        ):
            raise ValueError(
                "Canonical EEG sample_rate must be "
                f"positive, got {window.sample_rate}."
            )

        if (
            window.unit.strip().lower()
            != self.input_contract.input_unit.strip().lower()
        ):
            raise ValueError(
                "Canonical EEG unit does not match "
                "the transform contract: "
                f"window={window.unit!r}, "
                f"contract="
                f"{self.input_contract.input_unit!r}."
            )

    def validate_prepared_input(
        self,
        prepared: PreparedModelInput,
    ) -> None:
        model_input = prepared.model_input

        if isinstance(model_input, dict):
            missing_keys = (
                set(self.input_contract.model_input_keys)
                - set(model_input)
            )

            if missing_keys:
                raise ValueError(
                    "Prepared model input is missing keys: "
                    f"{sorted(missing_keys)}."
                )

            for key, tensor in model_input.items():
                if not tensor.is_floating_point():
                    raise TypeError(
                        f"Prepared Tensor {key!r} must be "
                        f"floating point, got {tensor.dtype}."
                    )

                if not tensor.isfinite().all():
                    raise ValueError(
                        f"Prepared Tensor {key!r} "
                        "contains NaN or Inf."
                    )

        else:
            if not model_input.is_floating_point():
                raise TypeError(
                    "Prepared model Tensor must be "
                    f"floating point, got "
                    f"{model_input.dtype}."
                )

            if not model_input.isfinite().all():
                raise ValueError(
                    "Prepared model Tensor contains "
                    "NaN or Inf."
                )

    @abstractmethod
    def transform(
        self,
        window: CanonicalEEGWindow,
    ) -> PreparedModelInput:
        """
        将一个规范化 EEG 窗口转换为模型专属输入。

        调用方应优先使用 transform_instance(window)，
        或直接使用 transform_instance(window)；
        Runtime 中推荐通过 __call__ 触发统一验证。
        """

class ModelPreprocessor(ABC):
    @abstractmethod
    def transform(
        self,
        window: np.ndarray,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError


class BaseModelAdapter(ABC):
    """
    旧模型适配接口。

    在 Replay、LaBraM、50M 全部迁移到 RuntimeModel 之前保留。
    这里应优先恢复你重构前已有的完整实现，而不是只写空壳。
    """

    @abstractmethod
    def predict_proba(
        self,
        window: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        window: Any,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError


class ModelBackend(ABC):
    """
    模型计算后端的统一接口。

    职责：
    - 接收已经完成模型专属预处理的输入；
    - 执行 backbone 和分类头计算；
    - 返回统一 ModelOutput；
    - 暴露后续个体化需要训练的参数。

    不负责：
    - 原始 EEG 单位转换；
    - 滤波与重采样；
    - 通道映射和重排；
    - 窗口切分；
    - 模型专属归一化。
    """

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """模型当前所在设备。"""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """分类类别数量。"""
        raise NotImplementedError

    @abstractmethod
    def predict_tensor(
        self,
        model_input: ModelTensor,
        return_features: bool = False,
    ) -> ModelOutput:
        """
        对一个已经预处理好的窗口执行预测。

        Args:
            model_input:
                单个 Tensor，或者由多个 Tensor 组成的字典。

                例如 50M：
                    {
                        "signal": Tensor[B, C, T],
                        "channel_valid_mask": Tensor[B, C],
                    }

            return_features:
                是否在 ModelOutput 中返回分类头之前的特征。

        Returns:
            统一的 ModelOutput。
        """
        raise NotImplementedError

    @abstractmethod
    def encode_tensor(
        self,
        model_input: ModelTensor,
    ) -> torch.Tensor:
        """
        提取分类头之前的特征。

        主要供以下流程使用：
        - 训练个体分类头；
        - Rest-Tune；
        - NeuroOnline；
        - 特征分析。
        """
        raise NotImplementedError

    @abstractmethod
    def get_trainable_parameters(
        self,
        scope: str,
    ) -> list[torch.nn.Parameter]:
        """
        返回指定训练范围的参数。

        推荐支持：
            head
            backbone
            full
        """
        raise NotImplementedError
