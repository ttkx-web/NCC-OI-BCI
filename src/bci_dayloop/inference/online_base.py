from __future__ import annotations

from abc import ABC, abstractmethod

from bci_dayloop.runtime.adaptation_types import (
    AdaptationContext,
    FeedbackEvent,
    OnlineObservation,
    OnlineUpdateResult,
)
from bci_dayloop.runtime.model import RuntimeModel


class OnlineAdaptationStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def initialize(
        self,
        *,
        runtime_model: RuntimeModel,
        context: AdaptationContext,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def observe(
        self,
        observation: OnlineObservation,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def submit_feedback(
        self,
        feedback: FeedbackEvent,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def maybe_update(
        self,
        *,
        runtime_model: RuntimeModel,
    ) -> OnlineUpdateResult:
        raise NotImplementedError

    @abstractmethod
    def state_dict(
        self,
    ) -> dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(
        self,
        state: dict[str, object],
    ) -> None:
        raise NotImplementedError