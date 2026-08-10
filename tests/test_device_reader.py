from pathlib import Path

import numpy as np
import pytest

from bci_dayloop.data.device_reader import DeviceReader, DeviceReaderFactory
from bci_dayloop.data.neuracle_bdf import NeuracleBDFReader
from bci_dayloop.data.records import RawEEGRecord, UnitEvidence


def _unit_evidence() -> UnitEvidence:
    return UnitEvidence("uV", None, "vendor_confirmed")


class DummyReader:
    reader_name = "dummy"

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def load(
        self,
        path: str | Path,
        *,
        subject_id: str | None = None,
        session_id: str | None = None,
        device_id: str | None = None,
    ) -> RawEEGRecord:
        return RawEEGRecord(
            eeg=np.zeros((1, 2)),
            channel_names=("C3",),
            sampling_rate=250.0,
            unit_evidence=UnitEvidence("uV", None, "vendor_confirmed"),
            subject_id=subject_id,
            session_id=session_id,
            device_id=device_id,
            source_path=str(path),
        )


@pytest.fixture(autouse=True)
def clear_device_reader_registry() -> None:
    DeviceReaderFactory._registry.clear()
    yield
    DeviceReaderFactory._registry.clear()


def test_dummy_reader_satisfies_protocol() -> None:
    assert isinstance(DummyReader(), DeviceReader)


def test_register_and_create_reader() -> None:
    DeviceReaderFactory.register("dummy", DummyReader)

    reader = DeviceReaderFactory.create("dummy")

    assert isinstance(reader, DummyReader)


def test_create_passes_kwargs_to_builder_unchanged() -> None:
    received: dict[str, object] = {}

    def builder(**kwargs: object) -> DummyReader:
        received.update(kwargs)
        return DummyReader(**kwargs)

    DeviceReaderFactory.register("dummy", builder)

    reader = DeviceReaderFactory.create("dummy", setting="value", retries=3)

    assert isinstance(reader, DummyReader)
    assert received == {"setting": "value", "retries": 3}
    assert reader.kwargs == received


def test_list_readers_is_sorted() -> None:
    DeviceReaderFactory.register("zeta", DummyReader)
    DeviceReaderFactory.register("alpha", DummyReader)

    assert DeviceReaderFactory.list_readers() == ["alpha", "zeta"]


def test_reader_names_are_trimmed_and_lowercased() -> None:
    DeviceReaderFactory.register("  DuMmY  ", DummyReader)

    assert DeviceReaderFactory.list_readers() == ["dummy"]
    assert isinstance(DeviceReaderFactory.create(" DUMMY "), DummyReader)


@pytest.mark.parametrize("name", ["", "   ", "\t"])
def test_empty_reader_name_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        DeviceReaderFactory.register(name, DummyReader)


def test_duplicate_reader_registration_is_rejected() -> None:
    DeviceReaderFactory.register("dummy", DummyReader)

    with pytest.raises(ValueError, match="already registered"):
        DeviceReaderFactory.register(" DUMMY ", DummyReader)


def test_unknown_reader_lists_available_names() -> None:
    DeviceReaderFactory.register("alpha", DummyReader)
    DeviceReaderFactory.register("beta", DummyReader)

    with pytest.raises(ValueError, match="Available readers: alpha, beta"):
        DeviceReaderFactory.create("missing")


def test_dummy_reader_load_returns_raw_eeg_record() -> None:
    DeviceReaderFactory.register("dummy", DummyReader)
    reader = DeviceReaderFactory.create("dummy")

    record = reader.load(
        "recording.ndf",
        subject_id="sub-001",
        session_id="ses-01",
        device_id="device-anonymous",
    )

    assert isinstance(record, RawEEGRecord)
    assert record.subject_id == "sub-001"
    assert record.session_id == "ses-01"
    assert record.device_id == "device-anonymous"


def test_builtin_neuracle_reader_is_registered_explicitly_and_idempotently() -> None:
    from bci_dayloop.data.device_reader import register_builtin_readers

    register_builtin_readers()
    register_builtin_readers()

    assert "neuracle-bdf" in DeviceReaderFactory.list_readers()
    reader = DeviceReaderFactory.create("neuracle-bdf", unit_evidence=_unit_evidence())
    assert isinstance(reader, NeuracleBDFReader)


def test_data_package_exports_public_device_reading_api() -> None:
    from bci_dayloop.data import (
        DeviceReader as PublicDeviceReader,
        DeviceReaderFactory as PublicDeviceReaderFactory,
        EEGEvent as PublicEEGEvent,
        NeuracleBDFReader as PublicNeuracleBDFReader,
        RawEEGRecord as PublicRawEEGRecord,
        UnitEvidence as PublicUnitEvidence,
        align_events_with_csv,
        annotations_to_events,
        parse_neuracle_marker,
        register_builtin_readers,
    )

    assert PublicUnitEvidence is UnitEvidence
    assert PublicEEGEvent.__name__ == "EEGEvent"
    assert PublicRawEEGRecord is RawEEGRecord
    assert PublicDeviceReader is DeviceReader
    assert PublicDeviceReaderFactory is DeviceReaderFactory
    assert PublicNeuracleBDFReader is NeuracleBDFReader
    assert callable(register_builtin_readers)
    assert callable(parse_neuracle_marker)
    assert callable(annotations_to_events)
    assert callable(align_events_with_csv)
