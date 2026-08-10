from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from bci_dayloop.runtime.adaptation_types import (
    AdaptationContext,
)
from bci_dayloop.runtime.model import RuntimeModel
from bci_dayloop.runtime.types import RawEEGWindow


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    raw_window: RawEEGWindow
    label: int | None = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class OfflineAdaptationResult:
    strategy_name: str
    applied: bool
    model_revision: str
    metrics: dict[str, Any] = field(
        default_factory=dict
    )
    artifacts: dict[str, str] = field(
        default_factory=dict
    )


class OfflineAdaptationStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def adapt(
        self,
        *,
        runtime_model: RuntimeModel,
        samples: Iterable[CalibrationSample],
        context: AdaptationContext,
    ) -> OfflineAdaptationResult:
        raise NotImplementedError