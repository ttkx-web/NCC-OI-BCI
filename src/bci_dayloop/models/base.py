from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import torch

from bci_dayloop.runtime.types import (
    ModelOutput,
    ModelTensor,
)


# ============================================================
# 旧 Pipeline 接口
# ============================================================

ModelInput: TypeAlias = (
    np.ndarray
    | dict[str, np.ndarray]
)


def add_batch_dimension(
    value: ModelInput,
) -> ModelInput:
    """为单窗口模型输入增加 batch 维度。"""

    if isinstance(value, np.ndarray):
        return value[None, ...]

    if isinstance(value, dict):
        if not value:
            raise ValueError(
                "Cannot add a batch dimension to an "
                "empty model input dictionary."
            )

        batched: dict[str, np.ndarray] = {}

        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "Model input dictionary keys must "
                    "be strings."
                )

            if not isinstance(item, np.ndarray):
                raise TypeError(
                    f"Model input {key!r} must be "
                    "a numpy.ndarray, got "
                    f"{type(item).__name__}."
                )

            batched[key] = item[None, ...]

        return batched

    raise TypeError(
        "Unsupported model input type "
        f"{type(value).__name__}."
    )


class ModelPreprocessor(ABC):
    """旧 SlidingWindowDecoder 使用的预处理接口。"""

    @abstractmethod
    def transform(
        self,
        window: np.ndarray,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError


class BaseModelAdapter(ABC):
    """
    旧 Pipeline 模型接口。

    在 Replay、LaBraM 和 50M 全部迁移到 RuntimeModel
    之前继续保留。
    """

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(
        self,
        X: ModelInput,
        **kwargs: Any,
    ) -> np.ndarray:
        raise NotImplementedError

    def predict(
        self,
        X: ModelInput,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        默认根据 predict_proba 返回类别。

        LaBraM 不需要单独实现 predict；
        50M 可以继续使用自己的覆盖实现。
        """

        probabilities = self.predict_proba(
            X,
            **kwargs,
        )

        return np.asarray(
            probabilities
        ).argmax(axis=-1).astype(
            np.int64,
            copy=False,
        )

    @abstractmethod
    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def save(
        self,
        path: str | Path,
        **kwargs: Any,
    ) -> Path:
        raise NotImplementedError

    @abstractmethod
    def load(
        self,
        path: str | Path,
    ) -> "BaseModelAdapter":
        raise NotImplementedError


# ============================================================
# 新 Runtime 接口
# ============================================================

class ModelBackend(ABC):
    """
    新 Runtime 的纯模型计算后端。

    不负责滤波、重采样、通道映射和归一化。
    """

    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_classes(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def predict_tensor(
        self,
        model_input: ModelTensor,
        return_features: bool = False,
    ) -> ModelOutput:
        raise NotImplementedError

    @abstractmethod
    def encode_tensor(
        self,
        model_input: ModelTensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_trainable_parameters(
        self,
        scope: str,
    ) -> list[torch.nn.Parameter]:
        raise NotImplementedError