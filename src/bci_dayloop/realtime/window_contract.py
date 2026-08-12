"""Approved Stage 2B realtime window durations shared by package consumers."""

from __future__ import annotations

import math
from typing import Protocol


APPROVED_REALTIME_WINDOW_SECONDS: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)
REALTIME_STEP_SECONDS = 0.5
NEURACLE_SOURCE_SAMPLING_RATE = 1000.0


class _WindowContract(Protocol):
    window_sec: float
    num_samples: int


def validate_approved_realtime_window_contract(
    contract: _WindowContract,
    *,
    sampling_rate: float,
) -> float:
    """Return an approved duration or fail before a source is connected.

    This deliberately validates metadata only.  It never pads, crops, or
    changes a signal, and is shared by the package policy, bridge, and probe.
    """
    window_sec = float(contract.window_sec)
    if not math.isfinite(window_sec) or window_sec not in APPROVED_REALTIME_WINDOW_SECONDS:
        raise ValueError(
            "Runtime Package window_sec is BLOCKED: only 1.0, 2.0, 3.0, or 4.0 seconds are approved"
        )
    if int(contract.num_samples) != round(window_sec * sampling_rate):
        raise ValueError(
            "Runtime Package num_samples does not match its approved window_sec and sampling_rate"
        )
    return window_sec


def approved_source_sample_count(window_sec: float) -> int:
    """Return the exact sample count for the approved 1000 Hz source."""
    if (
        not math.isfinite(window_sec)
        or window_sec not in APPROVED_REALTIME_WINDOW_SECONDS
    ):
        raise ValueError(
            "Realtime source window is BLOCKED: only 1.0, 2.0, 3.0, or 4.0 seconds are approved"
        )
    return round(window_sec * NEURACLE_SOURCE_SAMPLING_RATE)
