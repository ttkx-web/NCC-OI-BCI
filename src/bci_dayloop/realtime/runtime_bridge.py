"""Fail-closed Stage 2B bridge from a verified realtime window to Runtime.prepare."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from collections.abc import Mapping
from typing import Protocol

import numpy as np
import torch

from bci_dayloop.realtime.channel_units import EEG_UNIT, VENDOR_CONFIRMED, normalize_channel_type
from bci_dayloop.runtime.types import InputContract, PreparedModelInput, RawEEGWindow

from .contracts import RealtimeWindow
from .runtime_mapping import APPROVED_NEURACLE_59_TO_STANDARD64, ApprovedRealtimeMappingPolicy


class RuntimePrepareOnly(Protocol):
    @property
    def input_contract(self) -> InputContract: ...
    def prepare(self, raw_window: RawEEGWindow) -> PreparedModelInput: ...


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
    prepared_input: PreparedModelInput | None = field(default=None, repr=False, compare=False)

    def to_summary(self) -> dict[str, object]:
        return {
            "window_id": self.window_id,
            "continuous_segment_id": self.continuous_segment_id,
            "source_shape": list(self.source_shape),
            "prepared_signal_shape": list(self.prepared_signal_shape) if self.prepared_signal_shape else None,
            "valid_channel_count": self.valid_channel_count,
            "missing_target_channels": list(self.missing_target_channels),
            "ignored_source_channels": list(self.ignored_source_channels),
            "prepare_latency_ms": self.prepare_latency_ms,
            "marker_summary": [{"event_type": kind, "code": code} for kind, code in self.marker_summary],
            "model_input_safe": self.model_input_safe,
            "failure_reason": self.failure_reason,
        }


class RealtimeRuntimeBridge:
    """Source gate -> RawEEGWindow -> Runtime.prepare -> prepared-input gate.

    This class intentionally exposes no prediction path and never calls a
    classifier head or backend inference API.
    """

    def __init__(self, runtime_model: RuntimePrepareOnly, *, policy: ApprovedRealtimeMappingPolicy = APPROVED_NEURACLE_59_TO_STANDARD64) -> None:
        self.runtime_model = runtime_model
        self.policy = policy

    def prepare(self, window: RealtimeWindow) -> RealtimePreparedWindow:
        marker_summary = tuple((marker.event_type, marker.code) for marker in window.markers)
        segment_id = window.metadata.get("continuous_segment_id")
        source_shape = tuple(int(value) for value in window.samples.shape)
        failure = _source_failure(window, self.policy)
        if failure:
            return self._failure(window, source_shape, segment_id, marker_summary, failure)

        raw_window = RawEEGWindow(
            data=window.samples,
            channel_names=list(window.channel_names),
            sample_rate=window.sampling_rate,
            unit=window.unit,
            layout="CT",
            window_id=str(window.window_id),
            metadata={
                "source": "stage2b_realtime_window",
                "continuous_segment_id": segment_id,
                "provenance": dict(window.metadata),
                "marker_summary": tuple({"event_type": kind, "code": code} for kind, code in marker_summary),
            },
        )
        started = time.perf_counter()
        try:
            prepared = self.runtime_model.prepare(raw_window)
        except Exception as exc:
            return self._failure(
                window, source_shape, segment_id, marker_summary,
                f"RuntimeModel.prepare failed: {type(exc).__name__}: {exc}",
                latency=(time.perf_counter() - started) * 1000.0,
            )
        latency = (time.perf_counter() - started) * 1000.0
        try:
            failure, signal_shape, valid_count = _prepared_failure(
                prepared,
                self.runtime_model.input_contract,
                self.runtime_model,
                self.policy,
            )
        except Exception as exc:
            return self._failure(
                window,
                source_shape,
                segment_id,
                marker_summary,
                f"prepared-input validation failed: {type(exc).__name__}: {exc}",
                latency,
            )
        if failure:
            return self._failure(window, source_shape, segment_id, marker_summary, failure, latency, signal_shape, valid_count)
        return RealtimePreparedWindow(
            window_id=window.window_id,
            continuous_segment_id=segment_id,  # source gate guarantees int/str
            source_shape=source_shape,
            prepared_signal_shape=signal_shape,
            valid_channel_count=valid_count,
            missing_target_channels=self.policy.expected_missing_target_channels,
            ignored_source_channels=self.policy.expected_ignored_source_channels,
            prepare_latency_ms=latency,
            marker_summary=marker_summary,
            model_input_safe=True,
            failure_reason=None,
            prepared_input=prepared,
        )

    def _failure(self, window: RealtimeWindow, source_shape: tuple[int, int], segment_id: object, marker_summary: tuple[tuple[str, int | str | None], ...], reason: str, latency: float | None = None, signal_shape: tuple[int, ...] | None = None, valid_count: int | None = None) -> RealtimePreparedWindow:
        return RealtimePreparedWindow(
            window_id=window.window_id,
            continuous_segment_id=segment_id if isinstance(segment_id, (int, str)) else "missing",
            source_shape=source_shape,
            prepared_signal_shape=signal_shape,
            valid_channel_count=valid_count,
            missing_target_channels=self.policy.expected_missing_target_channels,
            ignored_source_channels=self.policy.expected_ignored_source_channels,
            prepare_latency_ms=latency,
            marker_summary=marker_summary,
            model_input_safe=False,
            failure_reason=reason,
        )


def _source_failure(window: RealtimeWindow, policy: ApprovedRealtimeMappingPolicy) -> str | None:
    if window.samples.shape != (59, 4000):
        return "source samples must have shape [59, 4000]"
    if window.channel_names != policy.source_channel_names:
        return "source channel_names do not match the approved ordered policy"
    if window.sampling_rate != 1000.0 or not math.isclose(window.samples.shape[1] / window.sampling_rate, 4.0, abs_tol=1e-9):
        return "source must be 4.0 seconds at 1000 Hz"
    if window.timestamps.shape != (4000,) or not np.all(np.diff(window.timestamps) > 0):
        return "source timestamps must be strictly increasing with length 4000"
    if window.unit != EEG_UNIT or window.metadata.get("source_unit") != EEG_UNIT:
        return "source unit and source_unit provenance must be uV"
    if window.metadata.get("unit_evidence_level") != VENDOR_CONFIRMED:
        return "source unit_evidence_level must be vendor_confirmed"
    if window.metadata.get("model_safe") is not True:
        return "source model_safe must be true"
    if not isinstance(window.metadata.get("continuous_segment_id"), (int, str)):
        return "source continuous_segment_id is required"
    types, units = _sequence(window, "channel_types"), _sequence(window, "channel_units")
    if types is None or units is None or len(types) != 59 or len(units) != 59:
        return "source channel provenance lengths must match 59 channels"
    if any(normalize_channel_type(value) != "eeg" for value in types):
        return "source channel_types must all be EEG"
    if any(value != EEG_UNIT for value in units):
        return "source channel_units must all be uV"
    if not np.isfinite(window.samples).all():
        return "source samples contain NaN or Inf"
    return None


def _sequence(window: RealtimeWindow, name: str) -> tuple[object, ...] | None:
    value = window.metadata.get(name)
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return None


def _prepared_failure(prepared: PreparedModelInput, contract: InputContract, runtime_model: RuntimePrepareOnly, policy: ApprovedRealtimeMappingPolicy) -> tuple[str | None, tuple[int, ...] | None, int | None]:
    if not isinstance(prepared, PreparedModelInput):
        return "RuntimeModel.prepare must return PreparedModelInput", None, None
    contract_failure = _runtime_contract_failure(contract, runtime_model, policy)
    if contract_failure:
        return contract_failure, None, None
    if not isinstance(prepared.model_input, dict) or not set(contract.model_input_keys).issubset(prepared.model_input):
        return "prepared model_input is missing Runtime Package keys", None, None
    signal, mask = prepared.model_input["signal"], prepared.model_input["channel_valid_mask"]
    if not isinstance(signal, torch.Tensor) or not isinstance(mask, torch.Tensor):
        return "prepared signal and channel_valid_mask must be tensors", None, None
    signal_shape = tuple(int(value) for value in signal.shape)
    if signal.dtype != torch.float32 or signal_shape != (1, 64, 400) or not torch.isfinite(signal).all().item():
        return "prepared signal must be finite float32 with shape [1, 64, 400]", signal_shape, None
    if tuple(mask.shape) != (1, 64) or not (mask.dtype == torch.bool or torch.all((mask == 0) | (mask == 1)).item()):
        return "prepared channel_valid_mask must be [1, 64] containing only 0/1 or bool", signal_shape, None
    mask_bool = mask.to(dtype=torch.bool)[0]
    valid_count = int(mask_bool.sum().item())
    if valid_count != 57:
        return "prepared channel_valid_mask valid count must be 57", signal_shape, valid_count
    if tuple(index for index, valid in enumerate(mask_bool.tolist()) if not valid) != policy.target_missing_indices:
        return "prepared channel_valid_mask false positions do not match the approved policy", signal_shape, valid_count
    diagnostics = prepared.diagnostics
    if not isinstance(diagnostics, Mapping):
        return "prepared diagnostics must be a mapping", signal_shape, valid_count
    if int(diagnostics.get("mapped_channel_count", -1)) != policy.expected_valid_channels:
        return "prepared mapped_channel_count must be 57", signal_shape, valid_count
    if int(diagnostics.get("missing_channel_count", -1)) != len(policy.expected_missing_target_channels):
        return "prepared missing_channel_count must be 7", signal_shape, valid_count
    if tuple(str(value) for value in diagnostics.get("unknown_channel_names", ())) != policy.expected_ignored_source_channels:
        return "prepared unknown source channels do not match the approved policy", signal_shape, valid_count
    for name in ("duplicate_channel_count", "padded_points", "cropped_points"):
        if int(diagnostics.get(name, -1)) != 0:
            return f"prepared {name} must be 0", signal_shape, valid_count
    return None, signal_shape, valid_count


def _runtime_contract_failure(contract: InputContract, runtime_model: RuntimePrepareOnly, policy: ApprovedRealtimeMappingPolicy) -> str | None:
    if contract.channel_names != policy.target_channel_names:
        return "Runtime Package channel order does not match the approved policy"
    if contract.sample_rate != 100.0 or contract.window_sec != 4.0 or contract.num_samples != 400:
        return "Runtime Package must declare 64 channels at 100 Hz for 4.0 seconds / 400 samples"
    if contract.input_unit != EEG_UNIT or contract.model_input_keys != ("signal", "channel_valid_mask"):
        return "Runtime Package input unit or model input keys do not match the approved policy"
    config = getattr(getattr(runtime_model, "input_transform", None), "config", None)
    if config is None or getattr(config, "output_layer_idx", None) != 8 or getattr(config, "aggregation", None) != "flatten":
        return "Runtime Package must use output_layer_idx=8 and aggregation=flatten"
    return None
