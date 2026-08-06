from __future__ import annotations

from collections.abc import Iterable

from bci_dayloop.personalization.offline_base import (
    CalibrationSample,
    OfflineAdaptationResult,
    OfflineAdaptationStrategy,
)
from bci_dayloop.runtime.adaptation_types import (
    AdaptationContext,
)
from bci_dayloop.runtime.model import RuntimeModel


class NoOfflineAdaptation(
    OfflineAdaptationStrategy
):
    @property
    def name(self) -> str:
        return "none"

    def adapt(
        self,
        *,
        runtime_model: RuntimeModel,
        samples: Iterable[CalibrationSample],
        context: AdaptationContext,
    ) -> OfflineAdaptationResult:
        del runtime_model
        del samples
        del context

        return OfflineAdaptationResult(
            strategy_name=self.name,
            applied=False,
            model_revision="base",
        )