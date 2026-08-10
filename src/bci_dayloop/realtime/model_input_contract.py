"""Fail-closed audit of a realtime window before any model preprocessing.

This module intentionally performs no channel mapping, resampling, filtering,
normalization, padding, interpolation, or inference.  It only reports whether
an already model-shaped :class:`RealtimeWindow` exactly satisfies the declared
four-second 50M boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np

from bci_dayloop.models.model_50m.config import CHANNEL_ALIASES, STANDARD_64_CHANNELS

from .channel_units import EEG_UNIT, VENDOR_CONFIRMED, normalize_channel_type
from .contracts import RealtimeWindow


@dataclass(frozen=True, slots=True)
class RealtimeModelInputContract:
    """The strict, pre-inference boundary for a four-second 50M window."""

    channel_names: tuple[str, ...] = STANDARD_64_CHANNELS
    sampling_rate: float = 100.0
    window_seconds: float = 4.0
    unit: str = EEG_UNIT
    unit_evidence_level: str = VENDOR_CONFIRMED

    def __post_init__(self) -> None:
        if not self.channel_names:
            raise ValueError("channel_names cannot be empty")
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("channel_names must be unique")
        if not math.isfinite(self.sampling_rate) or self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive and finite")
        if not math.isfinite(self.window_seconds) or self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive and finite")
        if not self.unit:
            raise ValueError("unit must be explicit")

    @property
    def window_samples(self) -> int:
        return round(self.sampling_rate * self.window_seconds)


@dataclass(frozen=True, slots=True)
class RealtimeModelInputAudit:
    """A non-mutating comparison result; ``model_input_safe`` is fail-closed."""

    model_input_safe: bool
    failure_reasons: tuple[str, ...]
    matched_channels: tuple[str, ...]
    missing_model_channels: tuple[str, ...]
    unexpected_realtime_channels: tuple[str, ...]
    order_differences: tuple[str, ...]
    alias_differences: tuple[str, ...]
    duplicate_channel_names: tuple[str, ...]
    case_differences: tuple[str, ...]


_STANDARD_BY_CASEFOLD = {name.casefold(): name for name in STANDARD_64_CHANNELS}


def audit_realtime_window_model_input(
    window: RealtimeWindow,
    *,
    contract: RealtimeModelInputContract | None = None,
) -> RealtimeModelInputAudit:
    """Return an exact-contract audit without altering ``window`` or its samples.

    A canonical name is used only to describe differences.  It never authorizes
    reordering, alias substitution, dropping, averaging, or synthesizing data.
    """
    active_contract = contract or RealtimeModelInputContract()
    reasons: list[str] = []

    if window.sampling_rate != active_contract.sampling_rate:
        reasons.append(
            "sampling_rate mismatch: "
            f"expected {active_contract.sampling_rate:g}, got {window.sampling_rate:g}"
        )
    if window.samples.shape[1] != active_contract.window_samples:
        reasons.append(
            "window sample count mismatch: "
            f"expected {active_contract.window_samples}, got {window.samples.shape[1]}"
        )
    if not math.isclose(
        window.samples.shape[1] / window.sampling_rate,
        active_contract.window_seconds,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        reasons.append(
            "window duration mismatch: "
            f"expected {active_contract.window_seconds:g}s, "
            f"got {window.samples.shape[1] / window.sampling_rate:g}s"
        )
    if window.unit != active_contract.unit:
        reasons.append(f"unit mismatch: expected {active_contract.unit}, got {window.unit}")
    if window.timestamps.size > 1 and not np.all(np.diff(window.timestamps) > 0):
        reasons.append("window timestamps must be strictly increasing")

    comparison = _compare_channel_names(window.channel_names, active_contract.channel_names)
    if comparison.missing_model_channels:
        reasons.append("missing required model channels")
    if comparison.unexpected_realtime_channels:
        reasons.append("unexpected realtime channels")
    if comparison.duplicate_channel_names:
        reasons.append("duplicate realtime channel names")
    if comparison.order_differences:
        reasons.append("channel order differs from the model contract")

    _check_unit_metadata(window.metadata, len(window.channel_names), active_contract, reasons)

    return RealtimeModelInputAudit(
        model_input_safe=not reasons,
        failure_reasons=tuple(reasons),
        matched_channels=comparison.matched_channels,
        missing_model_channels=comparison.missing_model_channels,
        unexpected_realtime_channels=comparison.unexpected_realtime_channels,
        order_differences=comparison.order_differences,
        alias_differences=comparison.alias_differences,
        duplicate_channel_names=comparison.duplicate_channel_names,
        case_differences=comparison.case_differences,
    )


@dataclass(frozen=True, slots=True)
class _ChannelComparison:
    matched_channels: tuple[str, ...]
    missing_model_channels: tuple[str, ...]
    unexpected_realtime_channels: tuple[str, ...]
    order_differences: tuple[str, ...]
    alias_differences: tuple[str, ...]
    duplicate_channel_names: tuple[str, ...]
    case_differences: tuple[str, ...]


def _compare_channel_names(
    realtime_names: Sequence[str], expected_names: Sequence[str]
) -> _ChannelComparison:
    matched: list[str] = []
    unexpected: list[str] = []
    aliases: list[str] = []
    cases: list[str] = []
    seen: dict[str, str] = {}
    duplicates: list[str] = []

    for name in realtime_names:
        source = str(name)
        normalized = source.strip()
        comparison_key = normalized.casefold()
        if comparison_key in seen:
            duplicates.append(source)
        else:
            seen[comparison_key] = source

        canonical = _canonical_model_name(normalized)
        if canonical is None:
            unexpected.append(source)
            continue
        if canonical not in matched:
            matched.append(canonical)
        if normalized != canonical and normalized.casefold() == canonical.casefold():
            cases.append(f"{source} -> {canonical}")
        elif normalized.upper() in CHANNEL_ALIASES and normalized != canonical:
            aliases.append(f"{source} -> {canonical}")

    expected = tuple(expected_names)
    expected_set = set(expected)
    missing = tuple(name for name in expected if name not in matched)
    order = tuple(
        f"index {index}: expected {expected_name}, got {actual_name}"
        for index, (actual_name, expected_name) in enumerate(zip(realtime_names, expected))
        if actual_name != expected_name
    )
    if len(realtime_names) != len(expected):
        order += (
            f"channel count: expected {len(expected)}, got {len(realtime_names)}",
        )

    return _ChannelComparison(
        matched_channels=tuple(name for name in matched if name in expected_set),
        missing_model_channels=missing,
        unexpected_realtime_channels=tuple(unexpected),
        order_differences=order,
        alias_differences=tuple(aliases),
        duplicate_channel_names=tuple(duplicates),
        case_differences=tuple(cases),
    )


def _canonical_model_name(name: str) -> str | None:
    alias = CHANNEL_ALIASES.get(name.upper())
    if alias is not None:
        return alias
    return _STANDARD_BY_CASEFOLD.get(name.casefold())


def _check_unit_metadata(
    metadata: Mapping[str, object],
    channel_count: int,
    contract: RealtimeModelInputContract,
    reasons: list[str],
) -> None:
    if metadata.get("model_safe") is not True:
        reasons.append("window metadata model_safe must be true")
    if metadata.get("unit_evidence_level") != contract.unit_evidence_level:
        reasons.append(
            "window unit evidence mismatch: "
            f"expected {contract.unit_evidence_level}"
        )

    channel_types = _metadata_sequence(metadata, "channel_types", reasons)
    channel_units = _metadata_sequence(metadata, "channel_units", reasons)
    if channel_types is None or channel_units is None:
        return
    if len(channel_types) != channel_count or len(channel_units) != channel_count:
        reasons.append("channel metadata lengths must match window channel count")
        return
    if any(normalize_channel_type(channel_type) != "eeg" for channel_type in channel_types):
        reasons.append("window contains a non-EEG channel type")
    if any(channel_unit != contract.unit for channel_unit in channel_units):
        reasons.append("window channel_units must all be uV")


def _metadata_sequence(
    metadata: Mapping[str, object], name: str, reasons: list[str]
) -> tuple[object, ...] | None:
    value = metadata.get(name)
    if value is None or isinstance(value, (str, bytes)):
        reasons.append(f"window metadata missing sequence field: {name}")
        return None
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError:
        reasons.append(f"window metadata field is not a sequence: {name}")
        return None
