from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bci_dayloop.runtime.types import (
    ModelOutput,
    PreparedModelInput,
)


@dataclass(frozen=True, slots=True)
class AdaptationContext:
    run_id: str
    subject_id: str | None = None
    session: str | None = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class OnlineObservation:
    observation_id: str
    prepared_input: PreparedModelInput
    output: ModelOutput
    timestamp_sec: float
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class FeedbackEvent:
    observation_id: str
    label: int | None = None
    reward: float | None = None
    timestamp_sec: float | None = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class OnlineUpdateResult:
    strategy_name: str
    applied: bool
    update_step: int
    model_revision: str
    samples_used: int = 0
    latency_ms: float = 0.0
    reason: str | None = None
    metrics: dict[str, Any] = field(
        default_factory=dict
    )