from __future__ import annotations

from types import SimpleNamespace

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


def test_live_fatal_block_invalidates_prediction_and_never_emits_eeg_payload() -> None:
    record = _record()
    package = SimpleNamespace(
        window_sec=4.0,
        target_sample_rate=100.0,
        runtime_model=SimpleNamespace(input_contract=SimpleNamespace(channel_names=("C3", "C4"))),
    )
    controller = LiveRuntimeController(record, SimpleNamespace())
    controller._block(package, code="PACKET_GAP", message="检测到数据包缺失，预测已阻断")
    assert record.state is RunState.FAILED
    events = record.broker.history
    assert events[-1]["type"] == "state"
    assert any(event["type"] == "input_contract" and event["payload"]["safe"] is False for event in events)
    assert any(event["type"] == "error" and event["payload"]["fatal"] is True for event in events)
    text = str(events).lower()
    for forbidden in ("samples", "raw_eeg", "waveform", "waveform_preview"):
        assert forbidden not in text


def test_live_health_failure_is_fail_closed() -> None:
    assert LiveRuntimeController._health_failure({"connected": False}) == (
        "DEVICE_DISCONNECTED", "设备连接已断开，预测已阻断",
    )
    assert LiveRuntimeController._health_failure({"connected": True, "missing_packets": 1}) == (
        "PACKET_GAP", "检测到数据包缺失，预测已阻断",
    )
