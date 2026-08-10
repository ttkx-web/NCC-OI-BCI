from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest
import torch

from bci_dayloop.realtime.contracts import EventMarker, RealtimeWindow
from bci_dayloop.realtime.runtime_bridge import RealtimeRuntimeBridge
from bci_dayloop.realtime.runtime_mapping import APPROVED_NEURACLE_59_TO_STANDARD64
from bci_dayloop.runtime.types import CanonicalEEGWindow, InputContract, PreparedModelInput, RawEEGWindow


POLICY = APPROVED_NEURACLE_59_TO_STANDARD64


@dataclass
class _Config:
    output_layer_idx: int = 8
    aggregation: str = "flatten"


@dataclass
class _Transform:
    config: _Config


class FakeRuntime:
    def __init__(self, prepared: PreparedModelInput | None = None, *, contract: InputContract | None = None) -> None:
        self.input_contract = contract or _contract()
        self.input_transform = _Transform(_Config())
        self.prepared = prepared or _prepared()
        self.received: RawEEGWindow | None = None
        self.predict_called = False
        self.predict_prepared_called = False

    def prepare(self, raw_window: RawEEGWindow) -> PreparedModelInput:
        self.received = raw_window
        return self.prepared

    def predict(self, *_args: object, **_kwargs: object) -> None:
        self.predict_called = True
        raise AssertionError("Bridge must never call predict")

    def predict_prepared(self, *_args: object, **_kwargs: object) -> None:
        self.predict_prepared_called = True
        raise AssertionError("Bridge must never call predict_prepared")


def _contract(**changes: object) -> InputContract:
    values: dict[str, object] = {
        "channel_names": POLICY.target_channel_names,
        "sample_rate": 100.0,
        "window_sec": 4.0,
        "num_samples": 400,
        "input_unit": "uV",
        "tensor_layout": "BCT",
        "model_input_keys": ("signal", "channel_valid_mask"),
    }
    values.update(changes)
    return InputContract(**values)  # type: ignore[arg-type]


def _prepared(**changes: object) -> PreparedModelInput:
    signal = torch.arange(64 * 400, dtype=torch.float32).reshape(1, 64, 400)
    mask = torch.ones((1, 64), dtype=torch.bool)
    mask[0, list(POLICY.target_missing_indices)] = False
    canonical = CanonicalEEGWindow(
        data=np.zeros((64, 400), dtype=np.float32),
        channel_names=list(POLICY.target_channel_names),
        sample_rate=100.0,
        unit="uV",
    )
    values: dict[str, object] = {
        "model_input": {"signal": signal, "channel_valid_mask": mask},
        "canonical_window": canonical,
        "preprocessing_trace": ["unified-runtime"],
        "diagnostics": {
            "unknown_channel_names": ("PO5", "PO6"),
            "mapped_channel_count": 57,
            "missing_channel_count": 7,
            "duplicate_channel_count": 0,
            "padded_points": 0,
            "cropped_points": 0,
        },
    }
    values.update(changes)
    return PreparedModelInput(**values)  # type: ignore[arg-type]


def _window(**changes: object) -> RealtimeWindow:
    samples = np.arange(59 * 4000, dtype=np.float32).reshape(59, 4000)
    values: dict[str, object] = {
        "window_id": 7,
        "samples": samples,
        "channel_names": POLICY.source_channel_names,
        "sampling_rate": 1000.0,
        "unit": "uV",
        "timestamps": np.arange(4000, dtype=np.float64) / 1000.0,
        "start_sample_index": 500,
        "end_sample_index": 4500,
        "source_sequence_start": 11,
        "source_sequence_end": 14,
        "markers": (EventMarker(timestamp=0.5, event_type="trigger", code=1),),
        "metadata": {
            "source_unit": "uV",
            "unit_evidence_level": "vendor_confirmed",
            "model_safe": True,
            "channel_types": tuple("EEG" for _ in range(59)),
            "channel_units": tuple("uV" for _ in range(59)),
            "continuous_segment_id": 2,
        },
    }
    values.update(changes)
    return RealtimeWindow(**values)  # type: ignore[arg-type]


def _bridge(runtime: FakeRuntime | None = None) -> tuple[RealtimeRuntimeBridge, FakeRuntime]:
    instance = runtime or FakeRuntime()
    return RealtimeRuntimeBridge(instance), instance


def test_bridge_passes_approved_window_to_prepare_without_signal_transformation() -> None:
    bridge, runtime = _bridge()
    window = _window()

    result = bridge.prepare(window)

    assert result.model_input_safe is True
    assert result.failure_reason is None
    assert result.source_shape == (59, 4000)
    assert result.prepared_signal_shape == (1, 64, 400)
    assert result.valid_channel_count == 57
    assert result.marker_summary == (("trigger", 1),)
    assert runtime.received is not None
    assert runtime.received.data is window.samples
    assert runtime.received.channel_names == list(POLICY.source_channel_names)
    assert runtime.received.sample_rate == 1000.0
    assert runtime.received.unit == "uV"
    assert runtime.received.metadata["continuous_segment_id"] == 2
    assert runtime.received.metadata["provenance"]["channel_units"] == tuple("uV" for _ in range(59))
    assert runtime.received.metadata["marker_summary"] == ({"event_type": "trigger", "code": 1},)
    assert runtime.predict_called is False
    assert runtime.predict_prepared_called is False
    assert "prepared_input" not in result.to_summary()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"channel_names": POLICY.source_channel_names[::-1]}, "channel_names"),
        ({"samples": np.zeros((58, 4000), dtype=np.float32), "channel_names": POLICY.source_channel_names[:-1]}, "shape"),
        ({"samples": np.zeros((60, 4000), dtype=np.float32), "channel_names": POLICY.source_channel_names + ("X1",)}, "shape"),
        ({"sampling_rate": 500.0}, "1000 Hz"),
        ({"unit": "mV"}, "unit"),
    ],
)
def test_source_gate_rejects_invalid_identity_rate_or_unit(
    change: dict[str, object], message: str,
) -> None:
    bridge, runtime = _bridge()

    result = bridge.prepare(_window(**change))

    assert result.model_input_safe is False
    assert message in result.failure_reason
    assert runtime.received is None


@pytest.mark.parametrize(
    ("metadata_change", "message"),
    [
        ({"unit_evidence_level": "unknown"}, "unit_evidence_level"),
        ({"model_safe": False}, "model_safe"),
        ({"channel_units": tuple("unknown" for _ in range(59))}, "channel_units"),
        ({"channel_types": tuple("EOG" for _ in range(59))}, "channel_types"),
        ({"channel_units": ("uV",)}, "lengths"),
        ({"continuous_segment_id": None}, "continuous_segment_id"),
    ],
)
def test_source_gate_rejects_unapproved_or_incomplete_provenance(
    metadata_change: dict[str, object], message: str,
) -> None:
    bridge, runtime = _bridge()
    metadata = dict(_window().metadata)
    metadata.update(metadata_change)

    result = bridge.prepare(_window(metadata=metadata))

    assert result.model_input_safe is False
    assert message in result.failure_reason
    assert runtime.received is None


def test_source_gate_requires_strict_timestamp_increase() -> None:
    bridge, runtime = _bridge()
    timestamps = np.arange(4000, dtype=np.float64) / 1000.0
    timestamps[100] = timestamps[99]

    result = bridge.prepare(_window(timestamps=timestamps))

    assert result.model_input_safe is False
    assert "strictly increasing" in result.failure_reason
    assert runtime.received is None


def test_source_gate_rejects_nonfinite_samples_even_if_upstream_contract_is_bypassed() -> None:
    bridge, runtime = _bridge()
    window = _window()
    corrupt = window.samples.copy()
    corrupt.setflags(write=True)
    corrupt[0, 0] = np.nan
    object.__setattr__(window, "samples", corrupt)

    result = bridge.prepare(window)

    assert result.model_input_safe is False
    assert "NaN or Inf" in result.failure_reason
    assert runtime.received is None


@pytest.mark.parametrize(
    ("prepared", "message"),
    [
        (_prepared(model_input={"signal": torch.zeros((1, 64, 400), dtype=torch.float32)}), "missing Runtime Package keys"),
        (_prepared(model_input={"signal": torch.zeros((1, 64, 399), dtype=torch.float32), "channel_valid_mask": torch.ones((1, 64), dtype=torch.bool)}), "signal must be finite"),
        (_prepared(model_input={"signal": torch.full((1, 64, 400), float("nan"), dtype=torch.float32), "channel_valid_mask": torch.ones((1, 64), dtype=torch.bool)}), "signal must be finite"),
        (_prepared(diagnostics={"unknown_channel_names": ("OTHER",), "mapped_channel_count": 57, "missing_channel_count": 7, "duplicate_channel_count": 0, "padded_points": 0, "cropped_points": 0}), "unknown source channels"),
        (_prepared(diagnostics={"unknown_channel_names": ("PO5", "PO6"), "mapped_channel_count": 56, "missing_channel_count": 7, "duplicate_channel_count": 0, "padded_points": 0, "cropped_points": 0}), "mapped_channel_count"),
        (_prepared(diagnostics={"unknown_channel_names": ("PO5", "PO6"), "mapped_channel_count": 57, "missing_channel_count": 8, "duplicate_channel_count": 0, "padded_points": 0, "cropped_points": 0}), "missing_channel_count"),
        (_prepared(diagnostics={"unknown_channel_names": ("PO5", "PO6"), "mapped_channel_count": 57, "missing_channel_count": 7, "duplicate_channel_count": 1, "padded_points": 0, "cropped_points": 0}), "duplicate_channel_count"),
        (_prepared(diagnostics={"unknown_channel_names": ("PO5", "PO6"), "mapped_channel_count": 57, "missing_channel_count": 7, "duplicate_channel_count": 0, "padded_points": 1, "cropped_points": 0}), "padded_points"),
        (_prepared(diagnostics={"unknown_channel_names": ("PO5", "PO6"), "mapped_channel_count": 57, "missing_channel_count": 7, "duplicate_channel_count": 0, "padded_points": 0, "cropped_points": 1}), "cropped_points"),
    ],
)
def test_prepared_gate_rejects_unapproved_runtime_output(
    prepared: PreparedModelInput, message: str,
) -> None:
    bridge, _runtime = _bridge(FakeRuntime(prepared))

    result = bridge.prepare(_window())

    assert result.model_input_safe is False
    assert message in result.failure_reason


def test_prepared_gate_rejects_mask_at_an_unapproved_position() -> None:
    prepared = _prepared()
    mask = prepared.model_input["channel_valid_mask"].clone()
    mask[0, POLICY.target_missing_indices[0]] = True
    mask[0, 0] = False
    prepared = _prepared(model_input={"signal": prepared.model_input["signal"], "channel_valid_mask": mask})
    bridge, _runtime = _bridge(FakeRuntime(prepared))

    result = bridge.prepare(_window())

    assert result.model_input_safe is False
    assert "false positions" in result.failure_reason


def test_prepared_gate_rejects_runtime_contract_mismatch() -> None:
    bridge, _runtime = _bridge(FakeRuntime(contract=_contract(sample_rate=200.0, num_samples=800)))

    result = bridge.prepare(_window())

    assert result.model_input_safe is False
    assert "100 Hz" in result.failure_reason
