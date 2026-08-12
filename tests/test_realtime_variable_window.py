from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bci_dayloop.models.cbramod.config import BCICIV2A_22_CHANNELS
from bci_dayloop.realtime.contracts import RealtimeWindow
from bci_dayloop.realtime.runtime_bridge import RealtimeRuntimeBridge
from bci_dayloop.realtime.runtime_mapping import APPROVED_NEURACLE_59_TO_STANDARD64
from bci_dayloop.realtime.runtime_policy import RealtimeModelPolicyRegistry
from bci_dayloop.runtime.types import CanonicalEEGWindow, InputContract, PreparedModelInput, RawEEGWindow


SOURCE = APPROVED_NEURACLE_59_TO_STANDARD64
LIVE19 = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1",
    "Cz", "C2", "C4", "C6", "CP3", "CP1", "CP2", "CP4", "Pz", "POz",
)
CBR_MISSING = ("CPz", "P1", "P2")
COMPLETION_SHA = "a3f7918a08115e0d3bb33ffb5cbb8fc1a467313872a6bc28fb16d6bc0636bae5"


def _window(seconds: float, *, timestamps: np.ndarray | None = None) -> RealtimeWindow:
    samples = round(seconds * 1000)
    return RealtimeWindow(
        window_id=1,
        samples=np.zeros((59, samples), dtype=np.float32),
        channel_names=SOURCE.source_channel_names,
        sampling_rate=1000.0,
        unit="uV",
        timestamps=(
            np.arange(samples, dtype=np.float64) / 1000.0
            if timestamps is None else timestamps
        ),
        start_sample_index=0,
        end_sample_index=samples,
        source_sequence_start=0,
        source_sequence_end=0,
        metadata={
            "source_unit": "uV",
            "unit_evidence_level": "vendor_confirmed",
            "model_safe": True,
            "channel_types": tuple("EEG" for _ in range(59)),
            "channel_units": tuple("uV" for _ in range(59)),
            "continuous_segment_id": 1,
        },
    )


class _Runtime:
    def __init__(self, model_type: str, seconds: float) -> None:
        self.model_type = model_type
        self.seconds = seconds
        if model_type == "model_50m":
            channels = SOURCE.target_channel_names
            sample_rate = 100.0
            layout = "BCT"
            keys = ("signal", "channel_valid_mask")
            self.input_transform = SimpleNamespace(
                config=SimpleNamespace(output_layer_idx=8, aggregation="flatten")
            )
        elif model_type == "labram":
            channels = LIVE19
            sample_rate = 200.0
            layout = "BCTP"
            keys = ("signal",)
        else:
            channels = tuple(BCICIV2A_22_CHANNELS)
            sample_rate = 200.0
            layout = "BCTP"
            keys = ("signal",)
            self.input_transform = SimpleNamespace(
                config=SimpleNamespace(
                    missing_channel_policy="spherical_spline",
                    min_observed_channels=19,
                    spline_alpha=1e-5,
                )
            )
        self.input_contract = InputContract(
            channel_names=channels,
            sample_rate=sample_rate,
            window_sec=seconds,
            num_samples=round(seconds * sample_rate),
            input_unit="uV",
            tensor_layout=layout,
            strict_window_duration=(model_type != "model_50m"),
            model_input_keys=keys,
        )
        self.received: RawEEGWindow | None = None

    def prepare(self, raw_window: RawEEGWindow) -> PreparedModelInput:
        self.received = raw_window
        if self.model_type == "model_50m":
            mask = torch.ones((1, 64), dtype=torch.bool)
            mask[0, list(SOURCE.target_missing_indices)] = False
            model_input = {
                "signal": torch.zeros(
                    (1, 64, round(self.seconds * 100)), dtype=torch.float32
                ),
                "channel_valid_mask": mask,
            }
            diagnostics = {
                "unknown_channel_names": SOURCE.expected_ignored_source_channels,
                "mapped_channel_count": 57,
                "missing_channel_count": 7,
                "duplicate_channel_count": 0,
                "padded_points": 0,
                "cropped_points": 0,
            }
        elif self.model_type == "labram":
            assert raw_window.data.shape == (19, round(self.seconds * 1000))
            assert tuple(raw_window.channel_names) == LIVE19
            model_input = {
                "signal": torch.zeros(
                    (1, 19, int(self.seconds), 200), dtype=torch.float32
                )
            }
            diagnostics = {
                "source_channel_count": 19,
                "target_channel_count": 19,
                "missing_channel_names": (),
            }
        else:
            assert raw_window.data.shape == (59, round(self.seconds * 1000))
            model_input = {
                "signal": torch.zeros(
                    (1, 22, int(self.seconds), 200), dtype=torch.float32
                )
            }
            diagnostics = {
                "observed_channel_count": 19,
                "observed_channel_names": list(LIVE19),
                "missing_channel_names": ["CPZ", "P1", "P2"],
                "duplicate_channel_count": 0,
                "completion_policy": "spherical_spline",
                "completion_matrix_sha256": COMPLETION_SHA,
            }
        return PreparedModelInput(
            model_input=model_input,
            canonical_window=CanonicalEEGWindow(
                data=np.asarray(raw_window.data, dtype=np.float32),
                channel_names=list(raw_window.channel_names),
                sample_rate=raw_window.sample_rate,
                unit=raw_window.unit,
            ),
            preprocessing_trace=["synthetic-runtime"],
            diagnostics=diagnostics,
        )


def _package(model_type: str, seconds: float) -> object:
    runtime = _Runtime(model_type, seconds)
    metadata: dict[str, object] = {}
    if model_type == "cbramod":
        metadata = {
            "runtime": {"channel_completion": {
                "deployment_profile": "neuracle_live19_spline22",
                "observed_required": 19,
                "observed_channel_names": list(LIVE19),
                "missing_expected": list(CBR_MISSING),
                "missing_channel_policy": "spherical_spline",
                "min_observed_channels": 19,
                "spline_alpha": 1e-5,
                "channel_completion_source": "shared_runtime_preprocessor",
                "completion_matrix_sha256": COMPLETION_SHA,
            }},
            "provenance": {"completion_matrix_sha256": COMPLETION_SHA},
        }
    return SimpleNamespace(
        runtime_model=runtime,
        model_type=model_type,
        model_name=model_type,
        is_test_head=False,
        package_metadata=metadata,
    )


@pytest.mark.parametrize(
    ("model_type", "seconds", "expected_shape"),
    [
        ("model_50m", seconds, (1, 64, int(seconds * 100)))
        for seconds in (1.0, 2.0, 3.0, 4.0)
    ]
    + [
        ("labram", seconds, (1, 19, int(seconds), 200))
        for seconds in (1.0, 2.0, 3.0, 4.0)
    ]
    + [
        ("cbramod", seconds, (1, 22, int(seconds), 200))
        for seconds in (1.0, 2.0, 3.0, 4.0)
    ],
)
def test_all_approved_model_window_contracts_prepare_exactly(
    model_type: str, seconds: float, expected_shape: tuple[int, ...]
) -> None:
    package = _package(model_type, seconds)
    policy = RealtimeModelPolicyRegistry.create(package)
    result = RealtimeRuntimeBridge(package.runtime_model, policy=policy).prepare(
        _window(seconds)
    )

    assert result.model_input_safe is True
    assert result.prepared_signal_shape == expected_shape
    if model_type == "model_50m":
        assert result.valid_channel_count == 57
    if model_type == "cbramod":
        assert result.policy_metadata["missing_channel_names"] == list(CBR_MISSING)


@pytest.mark.parametrize("seconds", (0.5, 5.0))
def test_unapproved_package_windows_are_blocked_before_source_use(seconds: float) -> None:
    with pytest.raises(ValueError, match="BLOCKED"):
        RealtimeModelPolicyRegistry.create(_package("model_50m", seconds))


def test_source_sample_count_and_timestamp_gap_are_fail_closed() -> None:
    package = _package("model_50m", 2.0)
    policy = RealtimeModelPolicyRegistry.create(package)
    bridge = RealtimeRuntimeBridge(package.runtime_model, policy=policy)
    wrong_samples = bridge.prepare(_window(1.0))
    assert wrong_samples.model_input_safe is False
    assert "[59, 2000]" in str(wrong_samples.failure_reason)

    timestamps = np.arange(2000, dtype=np.float64) / 1000.0
    timestamps[1000:] += 0.1
    gap = bridge.prepare(_window(2.0, timestamps=timestamps))
    assert gap.model_input_safe is False
    assert "gap" in str(gap.failure_reason)


@pytest.mark.parametrize("diagnostic_name", ("padded_points", "cropped_points"))
def test_dynamic_50m_prepared_gate_rejects_padding_or_crop(
    diagnostic_name: str,
) -> None:
    package = _package("model_50m", 2.0)
    policy = RealtimeModelPolicyRegistry.create(package)
    bridge = RealtimeRuntimeBridge(package.runtime_model, policy=policy)
    window = _window(2.0)
    prepared = bridge.prepare(window).prepared_input
    assert prepared is not None
    diagnostics = dict(prepared.diagnostics)
    diagnostics[diagnostic_name] = 1

    result = bridge.validate_prepared_window(
        window, replace(prepared, diagnostics=diagnostics)
    )

    assert result.model_input_safe is False
    assert diagnostic_name in str(result.failure_reason)
