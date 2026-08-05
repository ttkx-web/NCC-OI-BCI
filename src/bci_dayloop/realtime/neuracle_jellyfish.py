"""License-safe Adapter for an authorized Neuracle JellyFish packet backend.

This module deliberately does not implement the proprietary TCP framing or binary
packet parser.  A separately authorized backend supplies decoded META and packet
objects; the Adapter owns NCC-OI-BCI's contracts, validation, timestamps, and
anonymous diagnostics.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
import hashlib
import math
import time
from typing import Protocol, runtime_checkable

import numpy as np

from .contracts import EEGChunk, EventMarker


class NeuracleConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    AWAITING_METADATA = "awaiting_metadata"
    READY = "ready"
    STREAMING = "streaming"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"
    FAILED = "failed"


class NeuracleSourceError(RuntimeError):
    """Base error for a rejected Neuracle realtime source operation."""


class NeuracleProtocolUnavailableError(NeuracleSourceError):
    """Raised until a separately authorized protocol backend is supplied."""


class NeuraclePacketError(NeuracleSourceError):
    """Raised when a decoded packet violates the realtime source contract."""


@dataclass(frozen=True)
class NeuracleJellyFishConfig:
    host: str = "127.0.0.1"
    port: int = 8712
    ready_timeout_sec: float = 15.0
    socket_timeout_sec: float = 1.0
    reconnect_attempts: int = 3
    reconnect_initial_backoff_sec: float = 0.25
    reconnect_max_backoff_sec: float = 2.0
    expected_sampling_rate: float | None = None
    expected_channel_names: tuple[str, ...] | None = None
    raw_unit: str = "unknown"

    def __post_init__(self) -> None:
        if not self.host.strip() or not (1 <= self.port <= 65535):
            raise ValueError("host must be non-empty and port must be in 1..65535")
        for name in (
            "ready_timeout_sec",
            "socket_timeout_sec",
            "reconnect_initial_backoff_sec",
            "reconnect_max_backoff_sec",
        ):
            if not math.isfinite(float(getattr(self, name))) or float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.reconnect_attempts < 0:
            raise ValueError("reconnect_attempts must be non-negative")
        if self.expected_sampling_rate is not None and (
            not math.isfinite(self.expected_sampling_rate) or self.expected_sampling_rate <= 0
        ):
            raise ValueError("expected_sampling_rate must be positive and finite")
        if not self.raw_unit.strip():
            raise ValueError("raw_unit must be explicitly declared")


@runtime_checkable
class JellyFishBackend(Protocol):
    """Authorized backend boundary; it may use TCP, a vendor SDK, or another transport."""

    def connect(self, host: str, port: int, socket_timeout_sec: float) -> None: ...

    def wait_for_metadata(self, timeout_sec: float) -> Mapping[str, object]: ...

    def read_packet(self) -> Mapping[str, object] | None: ...

    def close(self) -> None: ...


BackendFactory = Callable[[], JellyFishBackend]


def _unavailable_backend() -> JellyFishBackend:
    raise NeuracleProtocolUnavailableError(
        "No authorized Neuracle JellyFish protocol backend is installed; "
        "the proprietary parser was intentionally not copied."
    )


class NeuracleJellyFishSource:
    """RealtimeEEGSource-compatible Adapter for decoded JellyFish META/data packets."""

    def __init__(
        self,
        config: NeuracleJellyFishConfig = NeuracleJellyFishConfig(),
        *,
        backend_factory: BackendFactory = _unavailable_backend,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._backend_factory = backend_factory
        self._sleep = sleep
        self._monotonic = monotonic
        self._backend: JellyFishBackend | None = None
        self._state = NeuracleConnectionState.DISCONNECTED
        self._metadata: dict[str, object] | None = None
        self._events: deque[EventMarker] = deque()
        self._next_sequence_id = 0
        self._last_raw_start: int | None = None
        self._last_raw_end: int | None = None
        self._last_packet_key: object | None = None
        self._last_packet_monotonic: float | None = None
        self._received_packets = 0
        self._malformed_packets = 0
        self._missing_packets = 0
        self._duplicate_packets = 0
        self._out_of_order_packets = 0
        self._reconnect_count = 0
        self._last_error: str | None = None

    def connect(self) -> None:
        self._connect_with_retries(reconnecting=False)

    def reconnect(self) -> None:
        self.disconnect()
        self._reconnect_count += 1
        self._connect_with_retries(reconnecting=True)

    def disconnect(self) -> None:
        backend, self._backend = self._backend, None
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:  # pragma: no cover - defensive vendor boundary
                self._last_error = f"backend close failed: {exc}"
        self._metadata = None
        self._events.clear()
        self._last_raw_start = None
        self._last_raw_end = None
        self._last_packet_key = None
        self._last_packet_monotonic = None
        self._state = NeuracleConnectionState.STOPPED

    def read_chunk(self) -> EEGChunk | None:
        if self._backend is None or self._metadata is None:
            raise NeuracleSourceError("JellyFish source is not ready")
        try:
            packet = self._backend.read_packet()
        except Exception as exc:
            self._fail(f"packet receive failed: {exc}")
            raise NeuracleSourceError(self._last_error) from exc
        if packet is None:
            return None
        try:
            chunk = self._packet_to_chunk(packet)
        except NeuraclePacketError as exc:
            self._malformed_packets += 1
            self._last_error = str(exc)
            raise
        self._received_packets += 1
        self._state = NeuracleConnectionState.STREAMING
        self._last_packet_monotonic = self._monotonic()
        return chunk

    def read_event(self) -> EventMarker | None:
        """Return parsed JellyFish Trigger events without merging external marker sources."""
        return self._events.popleft() if self._events else None

    def health(self) -> Mapping[str, object]:
        metadata = self._metadata or {}
        age = (
            None
            if self._last_packet_monotonic is None
            else max(0.0, self._monotonic() - self._last_packet_monotonic)
        )
        return {
            "state": self._state.value,
            "connected": self._backend is not None,
            "metadata_ready": self._metadata is not None,
            "channel_count": metadata.get("forwarded_channel_count"),
            "sampling_rate": metadata.get("sampling_rate"),
            "received_packets": self._received_packets,
            "malformed_packets": self._malformed_packets,
            "missing_packets": self._missing_packets,
            "duplicate_packets": self._duplicate_packets,
            "out_of_order_packets": self._out_of_order_packets,
            "reconnect_count": self._reconnect_count,
            "last_error": self._last_error,
            "last_packet_age_sec": age,
            "unit_evidence_level": "realtime_unverified",
            "model_safe": False,
        }

    @property
    def metadata(self) -> Mapping[str, object] | None:
        return None if self._metadata is None else dict(self._metadata)

    def _connect_with_retries(self, *, reconnecting: bool) -> None:
        self._state = (
            NeuracleConnectionState.RECONNECTING if reconnecting else NeuracleConnectionState.CONNECTING
        )
        self._last_error = None
        for attempt in range(self.config.reconnect_attempts + 1):
            backend: JellyFishBackend | None = None
            try:
                backend = self._backend_factory()
                backend.connect(self.config.host, self.config.port, self.config.socket_timeout_sec)
                self._backend = backend
                self._state = NeuracleConnectionState.AWAITING_METADATA
                self._metadata = self._normalize_metadata(
                    backend.wait_for_metadata(self.config.ready_timeout_sec)
                )
                self._state = NeuracleConnectionState.READY
                return
            except Exception as exc:
                self._last_error = str(exc)
                self._backend = None
                if backend is not None:
                    try:
                        backend.close()
                    except Exception:
                        pass
                if attempt == self.config.reconnect_attempts:
                    self._state = NeuracleConnectionState.FAILED
                    raise NeuracleSourceError(
                        f"JellyFish connection failed after {attempt + 1} attempt(s): {exc}"
                    ) from exc
                backoff = min(
                    self.config.reconnect_max_backoff_sec,
                    self.config.reconnect_initial_backoff_sec * (2**attempt),
                )
                self._sleep(backoff)

    def _normalize_metadata(self, raw: Mapping[str, object]) -> dict[str, object]:
        channel_names = tuple(_required_sequence(raw, "channel_names", "channelNames"))
        channel_types = tuple(_required_sequence(raw, "channel_types", "channelTypes"))
        sample_rates = tuple(
            float(value) for value in _required_sequence(raw, "sample_rates", "sampleRates")
        )
        count = int(_value(raw, "forwarded_channel_count", "channelCount", default=len(channel_names)))
        if count != len(channel_names) or len(channel_types) != count or len(sample_rates) != count:
            raise NeuracleSourceError("META channel fields do not match forwarded_channel_count")
        if not all(math.isfinite(rate) and rate > 0 for rate in sample_rates):
            raise NeuracleSourceError("META sample_rates must be positive and finite")
        sampling_rate = sample_rates[0]
        if any(rate != sampling_rate for rate in sample_rates):
            raise NeuracleSourceError("META has mixed channel sample rates; Adapter cannot form one EEGChunk")
        if self.config.expected_sampling_rate is not None and sampling_rate != self.config.expected_sampling_rate:
            raise NeuracleSourceError(
                f"META sampling_rate {sampling_rate} does not match expected {self.config.expected_sampling_rate}"
            )
        if (
            self.config.expected_channel_names is not None
            and channel_names != self.config.expected_channel_names
        ):
            raise NeuracleSourceError("META channel_names do not exactly match expected_channel_names")
        return {
            "module_name": str(_value(raw, "module_name", "moduleName", default="unknown")),
            "module_type": str(_value(raw, "module_type", "moduleType", default="unknown")),
            "channel_names": channel_names,
            "channel_types": channel_types,
            "sample_rates": sample_rates,
            "sampling_rate": sampling_rate,
            "data_count_per_channel": tuple(
                _required_sequence(raw, "data_count_per_channel", "dataCountPerChannel")
            ),
            "max_digital": tuple(_required_sequence(raw, "max_digital", "maxDigital")),
            "min_digital": tuple(_required_sequence(raw, "min_digital", "minDigital")),
            "max_physical": tuple(_required_sequence(raw, "max_physical", "maxPhysical")),
            "min_physical": tuple(_required_sequence(raw, "min_physical", "minPhysical")),
            "gain": tuple(_required_sequence(raw, "gain")),
            "forwarded_channel_count": count,
            "serial_number_hash": _hash_if_present(_value(raw, "serial_number", "serialNumber")),
        }

    def _packet_to_chunk(self, packet: Mapping[str, object]) -> EEGChunk:
        assert self._metadata is not None
        try:
            raw_start = int(_value(packet, "start_timestamp", "startTimeStamp"))
            raw_length = int(_value(packet, "timestamp_length", "timeStampLength"))
            samples = np.asarray(packet["samples"], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise NeuraclePacketError("malformed packet: missing timestamp or samples") from exc
        if raw_length <= 0 or samples.ndim != 2 or samples.shape[1] == 0:
            raise NeuraclePacketError("malformed packet: invalid timestamp_length or sample shape")
        channel_count = int(self._metadata["forwarded_channel_count"])
        if samples.shape[0] != channel_count:
            self._fail("packet channel count changed from META")
            raise NeuraclePacketError(self._last_error)
        packet_rate = _value(packet, "sampling_rate", "sampleRate", default=self._metadata["sampling_rate"])
        if float(packet_rate) != float(self._metadata["sampling_rate"]):
            self._fail("packet sampling rate changed from META")
            raise NeuraclePacketError(self._last_error)
        packet_key = packet.get("packet_id", raw_start)
        if self._last_packet_key == packet_key:
            self._duplicate_packets += 1
            raise NeuraclePacketError("duplicate JellyFish packet")
        if self._last_raw_start is not None and raw_start < self._last_raw_start:
            self._out_of_order_packets += 1
            raise NeuraclePacketError("out-of-order JellyFish packet")
        timestamp_gap = self._last_raw_end is not None and raw_start > self._last_raw_end
        if timestamp_gap:
            self._missing_packets += 1
        sampling_rate = float(self._metadata["sampling_rate"])
        expected_length = round(samples.shape[1] * 1000.0 / sampling_rate)
        if abs(raw_length - expected_length) > 1:
            raise NeuraclePacketError("packet timestamp_length is inconsistent with sample count and rate")
        raw_timestamps = raw_start + np.arange(samples.shape[1], dtype=np.float64) * (1000.0 / sampling_rate)
        if self._last_raw_end is not None and raw_timestamps[0] < self._last_raw_end:
            self._out_of_order_packets += 1
            raise NeuraclePacketError("JellyFish device timestamps moved backwards")
        host_received_at = self._monotonic()
        self._emit_triggers(packet, host_received_at)
        sequence_id = self._next_sequence_id
        self._next_sequence_id += 1
        self._last_packet_key = packet_key
        self._last_raw_start = raw_start
        self._last_raw_end = raw_start + raw_length
        return EEGChunk(
            samples=samples,
            channel_names=self._metadata["channel_names"],  # type: ignore[arg-type]
            sampling_rate=sampling_rate,
            unit=self.config.raw_unit,
            timestamps=raw_timestamps / 1000.0,
            sequence_id=sequence_id,
            device_id=None,
            received_at=host_received_at,
            metadata={
                "channel_types": self._metadata["channel_types"],
                "module_name": self._metadata["module_name"],
                "module_type": self._metadata["module_type"],
                "raw_start_timestamp": raw_start,
                "raw_timestamp_length": raw_length,
                "raw_trigger_count": _value(packet, "trigger_count", "triggerCount", default=0),
                "raw_module_count": _value(packet, "module_count", "moduleCount", default=1),
                "host_received_at_monotonic": host_received_at,
                "packet_count": self._received_packets + 1,
                "source_packet_start": packet_key,
                "source_packet_end": packet_key,
                "serial_number_hash": self._metadata["serial_number_hash"],
                "unit_evidence_level": "realtime_unverified",
                "model_safe": False,
                "timestamp_gap": timestamp_gap,
            },
        )

    def _emit_triggers(self, packet: Mapping[str, object], received_at: float) -> None:
        raw_triggers = list(packet.get("triggers", ()))
        trigger_modules = packet.get("trigger_modules", ())
        if trigger_modules is None:
            trigger_modules = ()
        for module in trigger_modules:  # type: ignore[union-attr]
            if isinstance(module, Mapping):
                raw_triggers.extend(module.get("triggers", ()))
        for trigger in raw_triggers:
            if not isinstance(trigger, Mapping) or "code" not in trigger:
                raise NeuraclePacketError("malformed JellyFish trigger")
            raw_timestamp = int(
                _value(
                    trigger,
                    "raw_timestamp",
                    "rawTimeStamp",
                    default=_value(packet, "start_timestamp", "startTimeStamp"),
                )
            )
            self._events.append(
                EventMarker(
                    timestamp=raw_timestamp / 1000.0,
                    event_type="trigger",
                    code=trigger["code"],  # type: ignore[arg-type]
                    metadata={
                        "source": "jellyfish_trigger",
                        "received_at": received_at,
                        "raw_device_timestamp": raw_timestamp,
                    },
                )
            )

    def _fail(self, message: str) -> None:
        self._last_error = message
        self._state = NeuracleConnectionState.FAILED


def _value(raw: Mapping[str, object], *names: str, default: object = None) -> object:
    for name in names:
        if name in raw:
            return raw[name]
    return default


def _required_sequence(raw: Mapping[str, object], *names: str) -> tuple[object, ...]:
    value = _value(raw, *names)
    if isinstance(value, (str, bytes)) or value is None:
        raise NeuracleSourceError(f"META missing sequence field: {names[0]}")
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise NeuracleSourceError(f"META field must be a sequence: {names[0]}") from exc


def _hash_if_present(value: object) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]
