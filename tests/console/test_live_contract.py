from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.config import Settings
from app.schemas.runs import LiveCreate, RunState
from app.services.live_service import LiveRuntimeController
from app.services.run_service import RunRecord


def _record() -> RunRecord:
    return RunRecord(
        id="run_live_test",
        request=LiveCreate(model_id="model_test", source="neuracle_jellyfish", compute_device="cpu"),
        created_at=0.0,
        run_type="live",
    )


def _package() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="model_50m",
        window_sec=4.0,
        target_sample_rate=100.0,
        runtime_model=SimpleNamespace(input_contract=SimpleNamespace(channel_names=("C3", "C4"))),
    )


def test_live_fatal_block_invalidates_prediction_and_never_emits_eeg_payload() -> None:
    record = _record()
    controller = LiveRuntimeController(record, SimpleNamespace())
    controller._block(_package(), code="PACKET_GAP", message="packet gap blocked")
    assert record.state is RunState.FAILED
    events = record.broker.history
    assert events[-1]["type"] == "state"
    assert any(event["type"] == "input_contract" and event["payload"]["safe"] is False for event in events)
    assert any(event["type"] == "error" and event["payload"]["fatal"] is True for event in events)
    assert not any(event["type"] == "prediction" for event in events)
    text = str(events).lower()
    for forbidden in ("samples", "raw_eeg", "waveform", "waveform_preview"):
        assert forbidden not in text


def test_live_health_failure_is_fail_closed() -> None:
    disconnected = LiveRuntimeController._health_failure({"connected": False})
    packet_gap = LiveRuntimeController._health_failure({"connected": True, "missing_packets": 1})
    assert disconnected is not None and disconnected[0] == "DEVICE_DISCONNECTED"
    assert packet_gap is not None and packet_gap[0] == "PACKET_GAP"


def test_live_controller_builds_source_from_configured_endpoint(tmp_path: Path) -> None:
    configured = Settings(
        repository_root=tmp_path,
        neuracle_jellyfish_host="jellyfish.internal",
        neuracle_jellyfish_port=18712,
    )
    controller = LiveRuntimeController(
        _record(),
        SimpleNamespace(),
        console_settings=configured,
    )

    source = controller._create_source()

    assert source.config.host == "jellyfish.internal"
    assert source.config.port == 18712


def test_device_health_removes_endpoint_and_backend_error_details() -> None:
    secret = "jellyfish.internal:18712"
    source = SimpleNamespace(health=lambda: {
        "connected": False,
        "missing_packets": 0,
        "last_error": f"connection refused at {secret}",
        "endpoint": secret,
        "serial_number": "device-secret",
    })
    controller = LiveRuntimeController(_record(), SimpleNamespace(), source_factory=lambda: source)
    controller._source = source

    health = controller._source_health()

    assert health == {"connected": False, "missing_packets": 0}
    assert secret not in str(health)
    assert "device-secret" not in str(health)


def test_unreachable_source_is_fatal_and_never_emits_prediction() -> None:
    class UnreachableSource:
        def connect(self) -> None:
            raise ConnectionError("private-host:18712")

    record = _record()
    controller = LiveRuntimeController(
        record,
        SimpleNamespace(),
        source_factory=UnreachableSource,
    )

    assert controller._connect_source(_package()) is False

    events = record.broker.history
    assert record.state is RunState.FAILED
    assert any(event["type"] == "input_contract" and event["payload"]["safe"] is False for event in events)
    assert any(
        event["type"] == "error"
        and event["payload"]["code"] == "DEVICE_UNREACHABLE"
        and event["payload"]["fatal"] is True
        for event in events
    )
    assert not any(event["type"] == "prediction" for event in events)
    assert "private-host" not in str(events)
