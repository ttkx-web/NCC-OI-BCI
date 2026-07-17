from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseModelAdapter(ABC):
    """Stable interface shared by trainable and online decoder backends."""

    model_name: str

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        """Fit the adapter and return training/validation metrics."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
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

