"""Stable, privacy-safe wrapper around the authorized Neuracle DataServerThread."""

from __future__ import annotations

import hashlib
import time
from typing import Callable, Mapping

import numpy as np

from .neuracle_api import DataServerThread


class NeuracleBackendError(RuntimeError):
    """Raised when the vendor forwarder cannot meet the stable backend contract."""


class NeuracleJellyFishBackend:
    """Wrap DataServerThread without exposing its private or sensitive fields."""

    def __init__(
        self,
        *,
        vendor_sample_rate: int = 1000,
        update_queue_max_packets: int = 256,
        server_factory: Callable[..., DataServerThread] = DataServerThread,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.vendor_sample_rate = vendor_sample_rate
        self.update_queue_max_packets = update_queue_max_packets
        self._server_factory = server_factory
        self._monotonic = monotonic
        self._sleep = sleep
        self._server: DataServerThread | None = None
        self._metadata: dict[str, object] | None = None
        self._started = False

    def connect(self, host: str, port: int, socket_timeout_sec: float = 1.0) -> None:
        del socket_timeout_sec  # The authorized vendor class manages its socket mode internally.
        self.stop()
        server = self._server_factory(
            sample_rate=self.vendor_sample_rate,
            update_queue_max_packets=self.update_queue_max_packets,
        )
        if server.connect(hostname=host, port=port):
            server.stop()
            raise NeuracleBackendError("JellyFish forwarder connection failed")
        self._server = server

    def wait_metadata(self, timeout_sec: float) -> Mapping[str, object]:
        server = self._require_server()
        deadline = self._monotonic() + timeout_sec
        while not server.isReady():
            if self._monotonic() >= deadline:
                raise NeuracleBackendError("Timed out waiting for JellyFish META")
            self._sleep(0.01)
        self._metadata = self._sanitize_metadata(server)
        return dict(self._metadata)

    def start(self) -> None:
        server = self._require_server()
        if self._metadata is None:
            raise NeuracleBackendError("Cannot start JellyFish backend before META is ready")
        server.start()
        self._started = True

    def read_packet_or_update(self) -> Mapping[str, object] | None:
        server = self._require_server()
        if not self._started:
            raise NeuracleBackendError("JellyFish backend has not been started")
        if server.updateQueueOverflow:
            raise NeuracleBackendError("Neuracle vendor update queue overflow")
        item = server.getUpdatePacket()
        if item is None:
            return None
        metadata = self._metadata or self._sanitize_metadata(server)
        samples = np.asarray(item["samples"], dtype=np.float32)
        return {
            "packet_id": item["packetCountEnd"],
            "source_packet_start": item["packetCountStart"],
            "source_packet_end": item["packetCountEnd"],
            "start_timestamp": item["startTimeStamp"],
            "timestamp_length": item["timeStampLength"],
            "samples": samples,
            "sampling_rate": metadata["sample_rates"][0],
            "triggers": tuple(_triggers_from_samples(item, metadata)),
            "host_received_at_monotonic": item["hostReceivedAtMonotonic"],
        }

    def stop(self) -> None:
        server, self._server = self._server, None
        self._metadata = None
        self._started = False
        if server is not None:
            server.stop()

    def metadata(self) -> Mapping[str, object] | None:
        return None if self._metadata is None else dict(self._metadata)

    def health(self) -> Mapping[str, object]:
        server = self._server
        return {
            "connected": server is not None,
            "metadata_ready": self._metadata is not None,
            "started": self._started,
            "queue_overflow": bool(server.updateQueueOverflow) if server is not None else False,
            "queued_packets": len(server.updateQueue) if server is not None else 0,
            "read_thread_alive": bool(server.readThread and server.readThread.is_alive()) if server is not None else False,
            "resolve_thread_alive": bool(server.resolveThread and server.resolveThread.is_alive()) if server is not None else False,
        }

    def _require_server(self) -> DataServerThread:
        if self._server is None:
            raise NeuracleBackendError("JellyFish backend is not connected")
        return self._server

    @staticmethod
    def _sanitize_metadata(server: DataServerThread) -> dict[str, object]:
        serial = getattr(server, "serialNumber", None)
        return {
            "module_name": str(getattr(server, "moduleName", "unknown")),
            "module_type": str(getattr(server, "moduleType", "unknown")),
            "anonymized_serial_hash": _hash_serial(serial),
            "channel_names": tuple(getattr(server, "channelNames", ())),
            "channel_types": tuple(getattr(server, "channelTypes", ())),
            "sample_rates": tuple(float(value) for value in getattr(server, "srates", ())),
            "data_count_per_channel": tuple(getattr(server, "dataCountPerChannel", ())),
            "digital_max": tuple(getattr(server, "maxDigital", ())),
            "digital_min": tuple(getattr(server, "minDigital", ())),
            "physical_max": tuple(getattr(server, "maxPhysical", ())),
            "physical_min": tuple(getattr(server, "minPhysical", ())),
            "gain": tuple(getattr(server, "gain", ())),
            "channel_count": int(getattr(server, "n_chan", 0)),
            "forwarded_channel_count": int(getattr(server, "n_chan", 0)),
        }


def _triggers_from_samples(item: Mapping[str, object], metadata: Mapping[str, object]) -> list[dict[str, object]]:
    samples = np.asarray(item["samples"])
    channel_types = tuple(str(value).lower() for value in metadata["channel_types"])
    sample_rate = float(tuple(metadata["sample_rates"])[0])
    start = int(item["startTimeStamp"])
    triggers: list[dict[str, object]] = []
    for index, channel_type in enumerate(channel_types):
        if "trigger" not in channel_type and "stim" not in channel_type and "event" not in channel_type:
            continue
        for sample_index in np.flatnonzero(samples[index] != 0):
            triggers.append(
                {
                    "code": int(samples[index, sample_index]),
                    "raw_timestamp": round(start + sample_index * 1000.0 / sample_rate),
                }
            )
    return triggers


def _hash_serial(value: object) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]
