"""Fail-closed Stage 2B bridge from a verified realtime window to Runtime.prepare."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

import numpy as np

from bci_dayloop.realtime.channel_units import (
    EEG_UNIT,
    VENDOR_CONFIRMED,
    normalize_channel_type,
)
from bci_dayloop.runtime.types import PreparedModelInput, RawEEGWindow

from .contracts import RealtimeWindow
from .runtime_mapping import (
    APPROVED_NEURACLE_59_TO_STANDARD64,
    ApprovedRealtimeMappingPolicy,
)
from .runtime_policy import (
    Model50MRealtimePolicy,
    RealtimeModelPolicy,
    RuntimePrepareOnly,
)


@dataclass(frozen=True, slots=True)
class RealtimePreparedWindow:
    """Metadata-only prepare result; the tensor is omitted from summaries."""

    window_id: int
    continuous_segment_id: int | str
    source_shape: tuple[int, int]
    prepared_signal_shape: tuple[int, ...] | None
    valid_channel_count: int | None
    missing_target_channels: tuple[str, ...]
    ignored_source_channels: tuple[str, ...]
    prepare_latency_ms: float | None
    marker_summary: tuple[tuple[str, int | str | None], ...]
    model_input_safe: bool
    failure_reason: str | None
    realtime_policy_id: str
    policy_metadata: dict[str, object] = field(default_factory=dict)
    prepared_input: PreparedModelInput | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def to_summary(self) -> dict[str, object]:
        return {
            "window_id": self.window_id,
            "continuous_segment_id": self.continuous_segment_id,
            "source_shape": list(self.source_shape),
            "prepared_signal_shape": (
                list(self.prepared_signal_shape)
                if self.prepared_signal_shape
                else None
            ),
            "valid_channel_count": self.valid_channel_count,
            "missing_target_channels": list(self.missing_target_channels),
            "ignored_source_channels": list(self.ignored_source_channels),
            "prepare_latency_ms": self.prepare_latency_ms,
            "marker_summary": [
                {"event_type": kind, "code": code}
                for kind, code in self.marker_summary
            ],
            "model_input_safe": self.model_input_safe,
            "failure_reason": self.failure_reason,
            "realtime_policy_id": self.realtime_policy_id,
            "policy_metadata": dict(self.policy_metadata),
        }


class RealtimeRuntimeBridge:
    """Common source gate -> model policy -> Runtime.prepare -> policy gate.

    This class intentionally exposes no prediction path and never calls a
    classifier head or backend inference API.
    """

    def __init__(
        self,
        runtime_model: RuntimePrepareOnly,
        *,
        policy: (
            RealtimeModelPolicy
            | ApprovedRealtimeMappingPolicy
            | None
        ) = None,
    ) -> None:
        self.runtime_model = runtime_model
        if policy is None:
            self.policy: RealtimeModelPolicy = Model50MRealtimePolicy()
        elif isinstance(policy, ApprovedRealtimeMappingPolicy):
            self.policy = Model50MRealtimePolicy(mapping=policy)
        else:
            self.policy = policy

    def prepare(self, window: RealtimeWindow) -> RealtimePreparedWindow:
        marker_summary = tuple(
            (marker.event_type, marker.code)
            for marker in window.markers
        )
        segment_id = window.metadata.get("continuous_segment_id")
        source_shape = tuple(int(value) for value in window.samples.shape)
        failure = _source_failure(window)
        if failure:
            return self._failure(
                window,
                source_shape,
                segment_id,
                marker_summary,
                failure,
            )

        try:
            selected_data, selected_names = self.policy.select_source(window)
        except Exception as exc:
            return self._failure(
                window,
                source_shape,
                segment_id,
                marker_summary,
                "realtime policy source selection failed: "
                f"{type(exc).__name__}: {exc}",
            )

        raw_window = RawEEGWindow(
            data=selected_data,
            channel_names=list(selected_names),
            sample_rate=window.sampling_rate,
            unit=window.unit,
            layout="CT",
            window_id=str(window.window_id),
            metadata={
                "source": "stage2b_realtime_window",
                "continuous_segment_id": segment_id,
                "provenance": dict(window.metadata),
                "realtime_policy_id": self.policy.policy_id,
                "marker_summary": tuple(
                    {"event_type": kind, "code": code}
                    for kind, code in marker_summary
                ),
            },
        )
        started = time.perf_counter()
        try:
            prepared = self.runtime_model.prepare(raw_window)
        except Exception as exc:
            return self._failure(
                window,
                source_shape,
                segment_id,
                marker_summary,
                "RuntimeModel.prepare failed: "
                f"{type(exc).__name__}: {exc}",
                latency=(time.perf_counter() - started) * 1000.0,
            )
        latency = (time.perf_counter() - started) * 1000.0
        try:
            validation = self.policy.validate_prepared(
                prepared,
                self.runtime_model,
            )
        except Exception as exc:
            return self._failure(
                window,
                source_shape,
                segment_id,
                marker_summary,
                "prepared-input validation failed: "
                f"{type(exc).__name__}: {exc}",
                latency,
            )
        if validation.failure_reason:
            return self._failure(
                window,
                source_shape,
                segment_id,
                marker_summary,
                validation.failure_reason,
                latency,
                validation.signal_shape,
                validation.valid_channel_count,
            )
        return RealtimePreparedWindow(
            window_id=window.window_id,
            continuous_segment_id=segment_id,  # source gate guarantees int/str
            source_shape=source_shape,
            prepared_signal_shape=validation.signal_shape,
            valid_channel_count=validation.valid_channel_count,
            missing_target_channels=self.policy.missing_target_channels,
            ignored_source_channels=self.policy.ignored_source_channels,
            prepare_latency_ms=latency,
            marker_summary=marker_summary,
            model_input_safe=True,
            failure_reason=None,
            realtime_policy_id=self.policy.policy_id,
            policy_metadata=dict(validation.policy_metadata or {}),
            prepared_input=prepared,
        )

    def _failure(
        self,
        window: RealtimeWindow,
        source_shape: tuple[int, int],
        segment_id: object,
        marker_summary: tuple[tuple[str, int | str | None], ...],
        reason: str,
        latency: float | None = None,
        signal_shape: tuple[int, ...] | None = None,
        valid_count: int | None = None,
    ) -> RealtimePreparedWindow:
        return RealtimePreparedWindow(
            window_id=window.window_id,
            continuous_segment_id=(
                segment_id
                if isinstance(segment_id, (int, str))
                else "missing"
            ),
            source_shape=source_shape,
            prepared_signal_shape=signal_shape,
            valid_channel_count=valid_count,
            missing_target_channels=self.policy.missing_target_channels,
            ignored_source_channels=self.policy.ignored_source_channels,
            prepare_latency_ms=latency,
            marker_summary=marker_summary,
            model_input_safe=False,
            failure_reason=reason,
            realtime_policy_id=self.policy.policy_id,
            policy_metadata={},
        )


def _source_failure(window: RealtimeWindow) -> str | None:
    approved = APPROVED_NEURACLE_59_TO_STANDARD64
    if window.samples.shape != (59, 4000):
        return "source samples must have shape [59, 4000]"
    if window.channel_names != approved.source_channel_names:
        return "source channel_names do not match the approved ordered policy"
    if window.sampling_rate != 1000.0 or not math.isclose(
        window.samples.shape[1] / window.sampling_rate,
        4.0,
        abs_tol=1e-9,
    ):
        return "source must be 4.0 seconds at 1000 Hz"
    if window.timestamps.shape != (4000,) or not np.all(
        np.diff(window.timestamps) > 0
    ):
        return "source timestamps must be strictly increasing with length 4000"
    if window.unit != EEG_UNIT or window.metadata.get("source_unit") != EEG_UNIT:
        return "source unit and source_unit provenance must be uV"
    if window.metadata.get("unit_evidence_level") != VENDOR_CONFIRMED:
        return "source unit_evidence_level must be vendor_confirmed"
    if window.metadata.get("model_safe") is not True:
        return "source model_safe must be true"
    if not isinstance(window.metadata.get("continuous_segment_id"), (int, str)):
        return "source continuous_segment_id is required"
    types = _sequence(window, "channel_types")
    units = _sequence(window, "channel_units")
    if types is None or units is None or len(types) != 59 or len(units) != 59:
        return "source channel provenance lengths must match 59 channels"
    if any(normalize_channel_type(value) != "eeg" for value in types):
        return "source channel_types must all be EEG"
    if any(value != EEG_UNIT for value in units):
        return "source channel_units must all be uV"
    if not np.isfinite(window.samples).all():
        return "source samples contain NaN or Inf"
    return None


def _sequence(
    window: RealtimeWindow,
    name: str,
) -> tuple[object, ...] | None:
    value = window.metadata.get(name)
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return None
