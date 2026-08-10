from collections import deque
import threading

import numpy as np
import pytest

from bci_dayloop.vendor.neuracle.backend import NeuracleJellyFishBackend
from bci_dayloop.vendor.neuracle.neuracle_api import DataServerThread


class FakeVendorServer:
    def __init__(self, *, sample_rate: int, update_queue_max_packets: int) -> None:
        self.sample_rate = sample_rate
        self.update_queue_max_packets = update_queue_max_packets
        self.moduleName = "JellyFish"
        self.moduleType = "Neuracle"
        self.serialNumber = 123456
        self.personName = "not-for-export"
        self.channelNames = ["C3", "ECG", "HEOG", "TRG"]
        self.channelTypes = ["EEG", "ECG", "EOG", "Trigger"]
        self.srates = [250, 250, 250, 250]
        self.dataCountPerChannel = [4, 4, 4, 4]
        self.maxDigital = [1, 1, 1, 1]
        self.minDigital = [-1, -1, -1, -1]
        self.maxPhysical = [375.0] * 4
        self.minPhysical = [-375.0] * 4
        self.gain = [1] * 4
        self.n_chan = 4
        self.updateQueueOverflow = False
        self.updateQueue = deque(
            (
                {
                    "samples": np.array(
                        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 7, 0, 0]],
                        dtype=np.float32,
                    ),
                    "startTimeStamp": 1000,
                    "timeStampLength": 16,
                    "packetCountStart": 11,
                    "packetCountEnd": 11,
                    "hostReceivedAtMonotonic": 12.5,
                },
            )
        )
        self.connected = False
        self.started = False
        self.stopped = False
        self.readThread = None
        self.resolveThread = None

    def connect(self, *, hostname: str, port: int) -> bool:
        self.connected = hostname == "127.0.0.1" and port == 8712
        return not self.connected

    def isReady(self) -> bool:
        return True

    def start(self) -> None:
        self.started = True

    def getUpdatePacket(self) -> dict[str, object] | None:
        return self.updateQueue.popleft() if self.updateQueue else None

    def stop(self) -> None:
        self.stopped = True


def test_vendor_backend_imports_and_sanitizes_meta_without_person_or_serial() -> None:
    holder: list[FakeVendorServer] = []

    def factory(**kwargs: object) -> FakeVendorServer:
        server = FakeVendorServer(**kwargs)  # type: ignore[arg-type]
        holder.append(server)
        return server

    backend = NeuracleJellyFishBackend(server_factory=factory, sleep=lambda _seconds: None)
    backend.connect("127.0.0.1", 8712)
    metadata = backend.wait_metadata(0.1)

    assert metadata["channel_names"] == ("C3", "ECG", "HEOG", "TRG")
    assert metadata["channel_types"] == ("EEG", "ECG", "EOG", "Trigger")
    assert metadata["channel_count"] == 4
    assert "personName" not in metadata
    assert "serialNumber" not in metadata
    assert len(metadata["anonymized_serial_hash"] or "") == 12
    assert holder[0].connected is True


def test_vendor_backend_preserves_packet_timestamps_and_trigger_records() -> None:
    backend = NeuracleJellyFishBackend(
        server_factory=lambda **kwargs: FakeVendorServer(**kwargs),  # type: ignore[arg-type]
        sleep=lambda _seconds: None,
    )
    backend.connect("127.0.0.1", 8712)
    backend.wait_metadata(0.1)
    backend.start()
    packet = backend.read_packet_or_update()

    assert packet is not None
    assert packet["start_timestamp"] == 1000
    assert packet["timestamp_length"] == 16
    assert packet["source_packet_start"] == 11
    assert packet["samples"].shape == (4, 4)
    assert packet["triggers"] == ({"code": 7, "raw_timestamp": 1004},)


def test_vendor_packet_queue_overflow_is_explicit_and_stop_joins_threads() -> None:
    server = DataServerThread(update_queue_max_packets=1)
    packet = {
        "startTimeStamp": 1000,
        "timeStampLength": 4,
        "packetCountStart": 7,
        "packetCountEnd": 9,
    }
    server.appendUpdatePacket(np.zeros((1, 1)), packet)
    queued = server.getUpdatePacket()
    assert queued is not None
    assert queued["packetCountStart"] == 7
    assert queued["packetCountEnd"] == 9
    server.appendUpdatePacket(np.zeros((1, 1)), packet)
    with pytest.raises(RuntimeError, match="queue overflow"):
        server.appendUpdatePacket(np.zeros((1, 1)), packet)
    assert server.updateQueueOverflow is True

    threads = [threading.Thread(target=lambda: None), threading.Thread(target=lambda: None)]
    for thread in threads:
        thread.start()
        thread.join()
    server.readThread, server.resolveThread = threads
    server.stop()
    assert all(not thread.is_alive() for thread in threads)
