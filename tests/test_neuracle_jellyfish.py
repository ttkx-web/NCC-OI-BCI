from collections import deque

import numpy as np
import pytest

from bci_dayloop.realtime.logging import RealtimeRunLogger
from bci_dayloop.realtime.neuracle_jellyfish import (
    NeuracleJellyFishConfig,
    NeuracleJellyFishSource,
    NeuraclePacketError,
    NeuracleSourceError,
)


class FakeBackend:
    def __init__(self, metadata: dict[str, object], packets: tuple[dict[str, object], ...] = (), *, wait_error: Exception | None = None, connect_error: Exception | None = None) -> None:
        self._metadata = metadata
        self.packets = deque(packets)
        self.wait_error = wait_error
        self.connect_error = connect_error
        self.closed = False
        self.started = False
        self.connect_calls = 0

    def connect(self, _host: str, _port: int, _timeout: float) -> None:
        self.connect_calls += 1
        if self.connect_error is not None:
            raise self.connect_error

    def wait_metadata(self, _timeout: float) -> dict[str, object]:
        if self.wait_error is not None:
            raise self.wait_error
        return self._metadata

    def start(self) -> None:
        self.started = True

    def read_packet_or_update(self) -> dict[str, object] | None:
        return self.packets.popleft() if self.packets else None

    def stop(self) -> None:
        self.closed = True

    def metadata(self) -> dict[str, object] | None:
        return self._metadata

    def health(self) -> dict[str, object]:
        return {"started": self.started}


def _metadata(*, channel_count: int = 4, sampling_rate: float = 250.0) -> dict[str, object]:
    names = tuple(["C3", "ECG", "HEOG", "TRG", *[f"EEG{index}" for index in range(channel_count - 4)]])
    types = tuple(["EEG", "ECG", "EOG", "Trigger", *["EEG"] * (channel_count - 4)])
    return {
        "person_name": "must-not-leak",
        "module_name": "JellyFish",
        "module_type": "Neuracle",
        "serial_number": "full-secret-serial",
        "channel_names": names,
        "channel_types": types,
        "sample_rates": (sampling_rate,) * channel_count,
        "data_count_per_channel": (4,) * channel_count,
        "max_digital": (8388607,) * channel_count,
        "min_digital": (-8388608,) * channel_count,
        "max_physical": (375.0,) * channel_count,
        "min_physical": (-375.0,) * channel_count,
        "gain": (1,) * channel_count,
        "forwarded_channel_count": channel_count,
    }


def _packet(
    packet_id: int,
    start_timestamp: int,
    *,
    channel_count: int = 4,
    sampling_rate: float = 250.0,
    triggers: tuple[dict[str, object], ...] = (),
    trigger_modules: tuple[dict[str, object], ...] = (),
) -> dict[str, object]:
    return {
        "packet_id": packet_id,
        "start_timestamp": start_timestamp,
        "timestamp_length": 16,
        "samples": np.arange(channel_count * 4, dtype=np.float32).reshape(channel_count, 4),
        "sampling_rate": sampling_rate,
        "triggers": triggers,
        "trigger_modules": trigger_modules,
    }


def _source(
    packets: tuple[dict[str, object], ...] = (), *, metadata: dict[str, object] | None = None, **config: object
) -> tuple[NeuracleJellyFishSource, FakeBackend]:
    backend = FakeBackend(_metadata() if metadata is None else metadata, packets)
    source = NeuracleJellyFishSource(
        NeuracleJellyFishConfig(reconnect_attempts=0, **config),
        backend_factory=lambda: backend,
        sleep=lambda _seconds: None,
        monotonic=lambda: 10.0,
    )
    return source, backend


def test_meta_preserves_all_forwarded_channels_and_anonymizes_sensitive_fields() -> None:
    metadata = _metadata(channel_count=70)
    source, _ = _source((_packet(1, 1000, channel_count=70),), metadata=metadata)
    source.connect()
    chunk = source.read_chunk()
    assert chunk is not None
    assert chunk.samples.shape == (70, 4)
    assert chunk.channel_names == tuple(metadata["channel_names"])
    assert chunk.metadata["channel_types"] == tuple(metadata["channel_types"])
    assert {"EEG", "ECG", "EOG", "Trigger"}.issubset(set(chunk.metadata["channel_types"]))
    assert "person_name" not in chunk.metadata
    assert "serial_number" not in chunk.metadata
    assert len(chunk.metadata["serial_number_hash"] or "") == 12
    assert source.metadata is not None
    assert source.metadata["max_digital"] == tuple(metadata["max_digital"])
    assert source.metadata["gain"] == tuple(metadata["gain"])
    assert chunk.metadata["source_packet_start"] == 1


def test_vendor_camel_case_meta_and_packet_fields_are_accepted_at_the_adapter_boundary() -> None:
    raw = _metadata()
    camel_meta = {
        "moduleName": raw["module_name"],
        "moduleType": raw["module_type"],
        "serialNumber": raw["serial_number"],
        "channelNames": raw["channel_names"],
        "channelTypes": raw["channel_types"],
        "sampleRates": raw["sample_rates"],
        "dataCountPerChannel": raw["data_count_per_channel"],
        "maxDigital": raw["max_digital"],
        "minDigital": raw["min_digital"],
        "maxPhysical": raw["max_physical"],
        "minPhysical": raw["min_physical"],
        "gain": raw["gain"],
        "channelCount": raw["forwarded_channel_count"],
    }
    source, _ = _source(
        (
            {
                "packet_id": 1,
                "startTimeStamp": 1000,
                "timeStampLength": 16,
                "triggerCount": 0,
                "moduleCount": 1,
                "samples": np.zeros((4, 4), dtype=np.float32),
            },
        ),
        metadata=camel_meta,
    )
    source.connect()
    chunk = source.read_chunk()
    assert chunk is not None
    assert chunk.metadata["raw_trigger_count"] == 0
    assert chunk.metadata["raw_module_count"] == 1


def test_device_timestamps_are_continuous_not_restarted_and_sequence_increases() -> None:
    source, _ = _source((_packet(1, 1000), _packet(2, 1016)))
    source.connect()
    first = source.read_chunk()
    second = source.read_chunk()
    assert first is not None and second is not None
    assert first.sequence_id == 0
    assert second.sequence_id == 1
    assert first.timestamps.tolist() == pytest.approx([1.0, 1.004, 1.008, 1.012])
    assert second.timestamps[0] == pytest.approx(1.016)
    assert second.timestamps[0] > first.timestamps[-1]
    assert second.metadata["timestamp_gap"] is False


def test_packet_gap_duplicate_and_out_of_order_are_detected() -> None:
    gap_source, _ = _source((_packet(1, 1000), _packet(2, 1032)))
    gap_source.connect()
    gap_source.read_chunk()
    gap = gap_source.read_chunk()
    assert gap is not None and gap.metadata["timestamp_gap"] is True
    assert gap_source.health()["missing_packets"] == 1

    duplicate_source, _ = _source((_packet(1, 1000), _packet(1, 1000)))
    duplicate_source.connect()
    duplicate_source.read_chunk()
    with pytest.raises(NeuraclePacketError, match="duplicate"):
        duplicate_source.read_chunk()
    assert duplicate_source.health()["duplicate_packets"] == 1

    ordered_source, _ = _source((_packet(2, 1016), _packet(1, 1000)))
    ordered_source.connect()
    ordered_source.read_chunk()
    with pytest.raises(NeuraclePacketError, match="out-of-order"):
        ordered_source.read_chunk()
    assert ordered_source.health()["out_of_order_packets"] == 1


@pytest.mark.parametrize(
    "packet, message",
    [
        ({"start_timestamp": 1000}, "malformed"),
        (_packet(1, 1000, channel_count=3), "channel count changed"),
        (_packet(1, 1000, sampling_rate=500.0), "sampling rate changed"),
    ],
)
def test_malformed_or_changed_packet_is_recorded_and_rejected(
    packet: dict[str, object], message: str
) -> None:
    source, _ = _source((packet,))
    source.connect()
    with pytest.raises(NeuraclePacketError, match=message):
        source.read_chunk()
    health = source.health()
    assert health["malformed_packets"] == 1
    assert health["last_error"] is not None


def test_expected_meta_constraints_and_ready_timeout_fail_before_streaming() -> None:
    rate_source, _ = _source(metadata=_metadata(sampling_rate=250.0), expected_sampling_rate=500.0)
    with pytest.raises(NeuracleSourceError, match="does not match expected"):
        rate_source.connect()
    names_source, _ = _source(expected_channel_names=("different",))
    with pytest.raises(NeuracleSourceError, match="channel_names"):
        names_source.connect()
    timeout_backend = FakeBackend(_metadata(), wait_error=TimeoutError("META timeout"))
    timeout_source = NeuracleJellyFishSource(
        NeuracleJellyFishConfig(reconnect_attempts=0),
        backend_factory=lambda: timeout_backend,
        sleep=lambda _seconds: None,
    )
    with pytest.raises(NeuracleSourceError, match="META timeout"):
        timeout_source.connect()
    assert timeout_source.health()["state"] == "failed"


def test_finite_reconnect_disconnect_and_reconnect_clear_old_packets() -> None:
    attempts: list[FakeBackend] = []

    def failing_factory() -> FakeBackend:
        backend = FakeBackend(_metadata(), connect_error=ConnectionError("no forwarder"))
        attempts.append(backend)
        return backend

    failing = NeuracleJellyFishSource(
        NeuracleJellyFishConfig(reconnect_attempts=2, reconnect_initial_backoff_sec=0),
        backend_factory=failing_factory,
        sleep=lambda _seconds: None,
    )
    with pytest.raises(NeuracleSourceError, match="after 3 attempt"):
        failing.connect()
    assert len(attempts) == 3
    assert failing.health()["state"] == "failed"

    first = FakeBackend(_metadata(), (_packet(1, 1000),))
    second = FakeBackend(_metadata(), (_packet(1, 1000),))
    backends = deque((first, second))
    source = NeuracleJellyFishSource(
        NeuracleJellyFishConfig(reconnect_attempts=0),
        backend_factory=lambda: backends.popleft(),
        sleep=lambda _seconds: None,
    )
    source.connect()
    assert source.read_chunk() is not None
    source.disconnect()
    assert first.closed is True
    source.reconnect()
    after_reconnect = source.read_chunk()
    assert after_reconnect is not None
    assert after_reconnect.sequence_id == 1
    assert source.health()["reconnect_count"] == 1


def test_bulk_and_per_module_triggers_become_event_markers_and_unit_is_blocked() -> None:
    packet = _packet(
        1,
        1000,
        triggers=({"code": 10, "raw_timestamp": 1008},),
        trigger_modules=({"triggers": ({"code": 20, "raw_timestamp": 1012},)},),
    )
    source, _ = _source((packet,))
    source.connect()
    chunk = source.read_chunk()
    first = source.read_event()
    second = source.read_event()
    assert chunk is not None
    assert [first.code if first else None, second.code if second else None] == [10, 20]
    assert first is not None and first.metadata["source"] == "jellyfish_trigger"
    assert first.metadata["raw_device_timestamp"] == 1008
    assert first.metadata["received_at"] == 10.0
    assert chunk.unit == "unknown"
    assert chunk.metadata["unit_evidence_level"] == "realtime_unverified"
    assert chunk.metadata["model_safe"] is False


def test_operational_log_omits_person_and_full_serial_number(tmp_path) -> None:
    source, _ = _source((_packet(1, 1000),))
    source.connect()
    chunk = source.read_chunk()
    assert chunk is not None
    logger = RealtimeRunLogger(tmp_path)
    logger.log_chunk(chunk)
    content = (tmp_path / "chunks.jsonl").read_text(encoding="utf-8")
    assert "must-not-leak" not in content
    assert "full-secret-serial" not in content
