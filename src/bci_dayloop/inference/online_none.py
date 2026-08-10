from __future__ import annotations

from bci_dayloop.inference.online_base import (
    OnlineAdaptationStrategy,
)
from bci_dayloop.runtime.adaptation_types import (
    AdaptationContext,
    FeedbackEvent,
    OnlineObservation,
    OnlineUpdateResult,
)
from bci_dayloop.runtime.model import RuntimeModel


class NoOnlineAdaptation(
    OnlineAdaptationStrategy
):
    def __init__(self) -> None:
        self._model_revision = "base"

    @property
    def name(self) -> str:
        return "none"

    def initialize(
        self,
        *,
        runtime_model: RuntimeModel,
        context: AdaptationContext,
    ) -> None:
        del runtime_model
        del context

    def observe(
        self,
        observation: OnlineObservation,
    ) -> None:
        del observation

    def submit_feedback(
        self,
        feedback: FeedbackEvent,
    ) -> None:
        del feedback

    def maybe_update(
        self,
        *,
        runtime_model: RuntimeModel,
    ) -> OnlineUpdateResult:
        del runtime_model

        return OnlineUpdateResult(
            strategy_name=self.name,
            applied=False,
            update_step=0,
            model_revision=(
                self._model_revision
            ),
            reason="online adaptation disabled",
        )

    def state_dict(
        self,
    ) -> dict[str, object]:
        return {
            "model_revision": (
                self._model_revision
            )
        }

    def load_state_dict(
        self,
        state: dict[str, object],
    ) -> None:
        self._model_revision = str(
            state.get(
                "model_revision",
                "base",
            )
        )