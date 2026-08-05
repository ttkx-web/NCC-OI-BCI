from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import numpy as np

import torch

from bci_dayloop.runtime.types import ModelOutput


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


class ModelPreprocessor(Protocol):
    """Prepares one EEG window for any registered model adapter."""

    def transform(
        self,
        samples: np.ndarray,
        sample_rate: float,
        input_unit: str,
        *,
        reshape: bool = True,
    ) -> ModelInput: ...


class BaseModelAdapter(ABC):
    """Stable interface shared by trainable and online decoder backends."""

    model_name: str

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        """Fit the adapter and return training/validation metrics."""

    @abstractmethod
    def predict_proba(self, X: ModelInput) -> np.ndarray:
        """Return class probabilities with shape [N, classes]."""

    @abstractmethod
    def save(self, path: str | Path, **kwargs: Any) -> Path:
        """Save an independently reloadable model package."""

    @abstractmethod
    def load(self, path: str | Path) -> "BaseModelAdapter":
        """Load a model package into this adapter."""

    @abstractmethod
    def update(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        """Update the trainable portion from newly labelled windows."""



class ModelBackend(ABC):
    @abstractmethod
    def predict_tensor(
        self,
        model_input: torch.Tensor,
        return_features: bool = False,
    ) -> ModelOutput:
        """只执行模型前向，不处理原始 EEG。"""

    @abstractmethod
    def encode_tensor(
        self,
        model_input: torch.Tensor,
    ) -> torch.Tensor:
        """返回 backbone 特征。"""

    @abstractmethod
    def get_trainable_parameters(
        self,
        scope: str,
    ) -> list[torch.nn.Parameter]:
        """供个体化、Rest-Tune 和 NeuroOnline 使用。"""
