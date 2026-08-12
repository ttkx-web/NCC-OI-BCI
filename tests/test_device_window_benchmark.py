from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bci_dayloop.benchmarking.core import RuntimeBenchmarkCore
from bci_dayloop.benchmarking.reporting import (
    BenchmarkCandidate,
    build_candidate_summary,
    flatten_window_record,
)
from bci_dayloop.benchmarking.windows import BenchmarkWindow, DeviceWindowProvider
from bci_dayloop.runtime.types import (
    CanonicalEEGWindow,
    InputContract,
    ModelOutput,
    PreparedModelInput,
    RawEEGWindow,
)


@dataclass
class _FakeSource:
    chunks: list[object]
    connected: bool = False
    disconnected: bool = False

    def connect(self) -> None:
        self.connected = True

    def read_chunk(self) -> object | None:
        return self.chunks.pop(0) if self.chunks else None

    def read_event(self) -> None:
        return None

    def disconnect(self) -> None:
        self.connected = False
        self.disconnected = True

    def health(self) -> dict[str, object]:
        return {
            "received_packets": 1,
            "missing_packets": 0,
            "duplicate_packets": 0,
            "out_of_order_packets": 0,
            "malformed_packets": 0,
            "reconnect_count": 0,
        }


class _FakePipeline:
    accepted_eeg_sample_count = 4000
    expected_windows = 1
    emitted_windows = 1
    failed_windows = 0
    timestamp_gap_count = 0
    buffer_overflow_count = 0

    def __init__(self, window: object) -> None:
        self.window = window
        self.closed = False

    def process(self, chunk: object, markers: object) -> tuple[object, ...]:
        return (SimpleNamespace(status="emitted", window=self.window, reason=None),)

    def close(self) -> None:
        self.closed = True


class _FakeBridge:
    def __init__(self, raw_window: RawEEGWindow) -> None:
        self.raw_window = raw_window
        self.validated = 0

    def to_raw_window(self, window: object) -> RawEEGWindow:
        return self.raw_window

    def validate_prepared_window(self, window: object, prepared: object) -> object:
        self.validated += 1
        return SimpleNamespace(model_input_safe=True, failure_reason=None)


class _PreparedRuntime:
    input_contract = InputContract(
        channel_names=("Cz",),
        sample_rate=1000.0,
        window_sec=4.0,
        num_samples=4000,
        input_unit="uV",
        tensor_layout="BCT",
        model_input_keys=("signal",),
        strict_window_duration=True,
    )

    def __init__(self) -> None:
        self.predicted = False

    def prepare(self, raw_window: RawEEGWindow) -> PreparedModelInput:
        return PreparedModelInput(
            model_input={"signal": torch.ones((1, 1, 4))},
            canonical_window=CanonicalEEGWindow(
                data=np.ones((1, 4000), dtype=np.float32),
                channel_names=["Cz"],
                sample_rate=1000.0,
                unit="uV",
            ),
            preprocessing_trace=[],
        )

    def predict_prepared(self, prepared: PreparedModelInput) -> ModelOutput:
        self.predicted = True
        return ModelOutput(
            logits=torch.tensor([[1.0]], dtype=torch.float32),
            predicted_class=0,
            confidence=1.0,
            probabilities=torch.tensor([[1.0]], dtype=torch.float32),
        )


def _raw_window() -> RawEEGWindow:
    return RawEEGWindow(
        data=np.ones((1, 4000), dtype=np.float32),
        channel_names=["Cz"],
        sample_rate=1000.0,
        unit="uV",
        layout="CT",
    )


def test_device_provider_preserves_perf_timestamps_and_releases_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    import bci_dayloop.benchmarking.windows as windows

    monkeypatch.setattr(windows, "select_verified_eeg_channels", lambda chunk: chunk)
    realtime_window = SimpleNamespace(
        window_id=7,
        source_sequence_end=3,
        start_sample_index=500,
        end_sample_index=4500,
    )
    source = _FakeSource(chunks=[SimpleNamespace(sequence_id=3)])
    pipeline = _FakePipeline(realtime_window)
    bridge = _FakeBridge(_raw_window())
    provider = DeviceWindowProvider(
        source=source,  # type: ignore[arg-type]
        pipeline=pipeline,  # type: ignore[arg-type]
        bridge=bridge,  # type: ignore[arg-type]
        duration_sec=1.0,
        maximum_windows=1,
    )

    item = next(iter(provider))

    assert item.source_mode == "device"
    assert item.window_ready_at_monotonic is not None
    assert item.last_sample_received_at_monotonic is not None
    assert item.last_sample_received_at_monotonic <= item.window_ready_at_monotonic
    provider.close()
    assert source.disconnected is True
    assert pipeline.closed is True
    assert provider.source_integrity["gap_count"] == 0
    assert provider.source_integrity["received_samples"] == 4000


def test_core_fail_closed_validator_prevents_prediction() -> None:
    runtime = _PreparedRuntime()
    item = BenchmarkWindow(
        raw_window=_raw_window(),
        source_mode="device",
        sequence_index=1,
        window_id="device:1",
        source_start_sample=0,
        source_end_sample_exclusive=4000,
        prepare_validator=lambda prepared: (_ for _ in ()).throw(ValueError("blocked")),
    )

    with pytest.raises(ValueError, match="blocked"):
        RuntimeBenchmarkCore(runtime_model=runtime, device="cpu")._run_one(item)

    assert runtime.predicted is False


def test_live_summary_deadline_excludes_warmup_and_preserves_max() -> None:
    runtime = _PreparedRuntime()
    item = BenchmarkWindow(
        raw_window=_raw_window(),
        source_mode="device",
        sequence_index=1,
        window_id="device:1",
        source_start_sample=0,
        source_end_sample_exclusive=4000,
        window_ready_at_monotonic=0.0,
        last_sample_received_at_monotonic=0.0,
    )
    records = RuntimeBenchmarkCore(runtime_model=runtime, device="cpu").run(
        provider=[item, item, item],  # type: ignore[arg-type]
        warmup_windows=1,
        measured_windows=2,
    )
    candidate = BenchmarkCandidate(
        candidate_id="fake",
        model_name="Fake",
        model_type="fake",
        package_path="model_packages/fake",
        package_sha256=None,
        window_sec=4.0,
        step_sec=0.5,
        device="cpu",
        source_mode="device",
        warmup_windows=1,
        measured_windows=2,
    )
    summary = build_candidate_summary(
        candidate=candidate,
        records=records,
        deadline_ms=500.0,
        expected_windows=2,
        failed_windows=0,
        source_integrity={"gap_count": 0},
        status="PASS",
    )

    assert summary.num_records == 2
    assert summary.completed_windows == 2
    assert summary.deadline_miss_count is not None
    assert summary.compute_total_ms["max"] >= summary.compute_total_ms["p95"]
    assert flatten_window_record(candidate=candidate, record=records[0], deadline_ms=500.0)["deadline_missed"] in {True, False}
