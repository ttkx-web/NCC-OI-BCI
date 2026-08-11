from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bci_dayloop.realtime.contracts import RealtimeWindow
from bci_dayloop.realtime.runtime_bridge import RealtimeRuntimeBridge
from bci_dayloop.realtime.runtime_mapping import (
    APPROVED_NEURACLE_59_TO_STANDARD64,
)
from bci_dayloop.realtime.runtime_policy import (
    CBraModRealtimePolicy,
    LaBraMRealtimePolicy,
    RealtimeModelPolicyRegistry,
)
from bci_dayloop.runtime.types import (
    CanonicalEEGWindow,
    InputContract,
    PreparedModelInput,
    RawEEGWindow,
)


SOURCE_POLICY = APPROVED_NEURACLE_59_TO_STANDARD64
LIVE19 = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1",
    "Cz", "C2", "C4", "C6", "CP3", "CP1", "CP2", "CP4", "Pz",
    "POz",
)
CBRAMOD22 = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1",
    "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
)
CBRAMOD_MISSING = ("CPz", "P1", "P2")
COMPLETION_SHA = (
    "a3f7918a08115e0d3bb33ffb5cbb8fc1a467313872a6bc28fb16d6bc0636bae5"
)


def _window() -> RealtimeWindow:
    samples = np.repeat(
        np.arange(59, dtype=np.float32)[:, None],
        4000,
        axis=1,
    )
    return RealtimeWindow(
        window_id=1,
        samples=samples,
        channel_names=SOURCE_POLICY.source_channel_names,
        sampling_rate=1000.0,
        unit="uV",
        timestamps=np.arange(4000, dtype=np.float64) / 1000.0,
        start_sample_index=0,
        end_sample_index=4000,
        source_sequence_start=0,
        source_sequence_end=39,
        metadata={
            "source_unit": "uV",
            "unit_evidence_level": "vendor_confirmed",
            "model_safe": True,
            "channel_types": tuple("EEG" for _ in range(59)),
            "channel_units": tuple("uV" for _ in range(59)),
            "continuous_segment_id": 1,
        },
    )


def _labram_contract(
    channel_names: tuple[str, ...] = LIVE19,
    **changes: object,
) -> InputContract:
    values: dict[str, object] = {
        "channel_names": channel_names,
        "sample_rate": 200.0,
        "window_sec": 4.0,
        "num_samples": 800,
        "input_unit": "uV",
        "tensor_layout": "BCTP",
        "strict_window_duration": True,
        "model_input_keys": ("signal",),
    }
    values.update(changes)
    return InputContract(**values)  # type: ignore[arg-type]


class FakeLaBraMRuntime:
    def __init__(
        self,
        *,
        contract: InputContract | None = None,
        extra_key: bool = False,
        wrong_shape: bool = False,
    ) -> None:
        self.input_contract = contract or _labram_contract()
        self.received: RawEEGWindow | None = None
        self.extra_key = extra_key
        self.wrong_shape = wrong_shape

    def prepare(self, raw_window: RawEEGWindow) -> PreparedModelInput:
        self.received = raw_window
        channels = len(self.input_contract.channel_names)
        shape = (
            (1, channels, 4, 199)
            if self.wrong_shape
            else (1, channels, 4, 200)
        )
        model_input = {
            "signal": torch.zeros(shape, dtype=torch.float32),
        }
        if self.extra_key:
            model_input["unexpected"] = torch.zeros((1,), dtype=torch.float32)
        return PreparedModelInput(
            model_input=model_input,
            canonical_window=CanonicalEEGWindow(
                data=np.asarray(raw_window.data, dtype=np.float32),
                channel_names=list(raw_window.channel_names),
                sample_rate=raw_window.sample_rate,
                unit=raw_window.unit,
            ),
            preprocessing_trace=["fake-labram"],
            diagnostics={
                "source_channel_count": channels,
                "target_channel_count": channels,
                "missing_channel_names": [],
            },
        )


class FakeCBraModRuntime:
    def __init__(
        self,
        *,
        missing_channel_policy: str = "spherical_spline",
        min_observed_channels: int = 19,
        completion_sha: str = COMPLETION_SHA,
    ) -> None:
        self.input_contract = InputContract(
            channel_names=CBRAMOD22,
            sample_rate=200.0,
            window_sec=4.0,
            num_samples=800,
            input_unit="uV",
            tensor_layout="BCTP",
            strict_window_duration=True,
            model_input_keys=("signal",),
        )
        self.input_transform = SimpleNamespace(
            config=SimpleNamespace(
                missing_channel_policy=missing_channel_policy,
                min_observed_channels=min_observed_channels,
                spline_alpha=1e-5,
            )
        )
        self.completion_sha = completion_sha
        self.received: RawEEGWindow | None = None

    def prepare(self, raw_window: RawEEGWindow) -> PreparedModelInput:
        self.received = raw_window
        return PreparedModelInput(
            model_input={
                "signal": torch.zeros((1, 22, 4, 200), dtype=torch.float32)
            },
            canonical_window=CanonicalEEGWindow(
                data=np.zeros((22, 800), dtype=np.float32),
                channel_names=list(CBRAMOD22),
                sample_rate=200.0,
                unit="uV",
            ),
            preprocessing_trace=["fake-cbramod-shared-preprocessor"],
            diagnostics={
                "observed_channel_count": 19,
                "observed_channel_names": list(LIVE19),
                "missing_channel_names": ["CPZ", "P1", "P2"],
                "duplicate_channel_count": 0,
                "completion_policy": "spherical_spline",
                "completion_matrix_sha256": self.completion_sha,
            },
        )


def _cbramod_metadata(
    *,
    policy: str = "spherical_spline",
    min_observed_channels: int = 19,
    completion_sha: str = COMPLETION_SHA,
    include_completion: bool = True,
) -> dict[str, object]:
    completion: dict[str, object] | None = None
    if include_completion:
        completion = {
            "deployment_profile": "neuracle_live19_spline22",
            "observed_required": 19,
            "observed_channel_names": list(LIVE19),
            "missing_expected": list(CBRAMOD_MISSING),
            "missing_channel_policy": policy,
            "min_observed_channels": min_observed_channels,
            "spline_alpha": 1e-5,
            "channel_completion_source": "shared_runtime_preprocessor",
            "completion_matrix_sha256": completion_sha,
        }
    return {
        "runtime": {"channel_completion": completion},
        "provenance": {"completion_matrix_sha256": completion_sha},
    }


def _package(
    runtime: object,
    *,
    model_type: str = "labram",
    is_test_head: bool = False,
    package_metadata: dict[str, object] | None = None,
) -> object:
    return SimpleNamespace(
        runtime_model=runtime,
        model_type=model_type,
        model_name="labram-linear",
        is_test_head=is_test_head,
        package_metadata=package_metadata or {},
    )


def test_labram_policy_selects_and_reorders_source_exactly() -> None:
    runtime = FakeLaBraMRuntime()
    policy = RealtimeModelPolicyRegistry.create(_package(runtime))
    result = RealtimeRuntimeBridge(runtime, policy=policy).prepare(_window())

    assert result.model_input_safe is True
    assert result.prepared_signal_shape == (1, 19, 4, 200)
    assert result.valid_channel_count == 19
    assert len(result.ignored_source_channels) == 40
    assert runtime.received is not None
    assert tuple(runtime.received.channel_names) == LIVE19
    assert runtime.received.data.shape == (19, 4000)
    expected_indices = [
        SOURCE_POLICY.source_channel_names.index(name)
        for name in LIVE19
    ]
    assert runtime.received.data[:, 0].tolist() == pytest.approx(expected_indices)


def test_labram_policy_rejects_missing_required_channel() -> None:
    runtime = FakeLaBraMRuntime(
        contract=_labram_contract(LIVE19 + ("CPz",))
    )
    with pytest.raises(ValueError, match="missing"):
        RealtimeModelPolicyRegistry.create(_package(runtime))


def test_labram_policy_rejects_duplicate_required_channel() -> None:
    duplicate = LIVE19[:-1] + (LIVE19[0],)
    with pytest.raises(ValueError, match="duplicate"):
        runtime = FakeLaBraMRuntime(contract=_labram_contract(duplicate))
        RealtimeModelPolicyRegistry.create(_package(runtime))


def test_labram_policy_rejects_alias_or_case_ambiguity() -> None:
    ambiguous = ("fz",) + LIVE19[1:]
    runtime = FakeLaBraMRuntime(contract=_labram_contract(ambiguous))
    with pytest.raises(ValueError, match="ambiguity"):
        RealtimeModelPolicyRegistry.create(_package(runtime))


@pytest.mark.parametrize(
    "contract",
    [
        _labram_contract(sample_rate=100.0, num_samples=400),
        _labram_contract(input_unit="mV"),
        _labram_contract(tensor_layout="BCT"),
        _labram_contract(model_input_keys=("signal", "other")),
    ],
)
def test_labram_policy_rejects_wrong_package_contract(
    contract: InputContract,
) -> None:
    runtime = FakeLaBraMRuntime(contract=contract)
    with pytest.raises(ValueError, match="LaBraM realtime package"):
        RealtimeModelPolicyRegistry.create(_package(runtime))


def test_labram_prepared_gate_rejects_extra_model_input_key() -> None:
    runtime = FakeLaBraMRuntime(extra_key=True)
    policy = LaBraMRealtimePolicy(_package(runtime))
    result = RealtimeRuntimeBridge(runtime, policy=policy).prepare(_window())
    assert result.model_input_safe is False
    assert "only" in str(result.failure_reason)


def test_labram_prepared_gate_rejects_wrong_shape() -> None:
    runtime = FakeLaBraMRuntime(wrong_shape=True)
    policy = LaBraMRealtimePolicy(_package(runtime))
    result = RealtimeRuntimeBridge(runtime, policy=policy).prepare(_window())
    assert result.model_input_safe is False
    assert "shape" in str(result.failure_reason)


def test_unknown_model_type_is_blocked_without_fallback() -> None:
    runtime = FakeLaBraMRuntime()
    with pytest.raises(ValueError, match="BLOCKED"):
        RealtimeModelPolicyRegistry.create(
            _package(runtime, model_type="unknown-model")
        )


def test_cbramod_registry_policy_preserves_source_and_validates_prepared() -> None:
    runtime = FakeCBraModRuntime()
    package = _package(
        runtime,
        model_type="cbramod",
        package_metadata=_cbramod_metadata(),
    )
    policy = RealtimeModelPolicyRegistry.create(package)
    result = RealtimeRuntimeBridge(runtime, policy=policy).prepare(_window())

    assert isinstance(policy, CBraModRealtimePolicy)
    assert policy.missing_target_channels == CBRAMOD_MISSING
    assert len(policy.ignored_source_channels) == 40
    assert result.model_input_safe is True
    assert result.prepared_signal_shape == (1, 22, 4, 200)
    assert result.valid_channel_count == 19
    assert result.policy_metadata == {
        "observed_channel_count": 19,
        "missing_channel_names": list(CBRAMOD_MISSING),
        "completion_policy": "spherical_spline",
        "completion_matrix_sha256": COMPLETION_SHA,
    }
    assert runtime.received is not None
    assert runtime.received.data.shape == (59, 4000)
    assert tuple(runtime.received.channel_names) == SOURCE_POLICY.source_channel_names


def test_cbramod_strict22_package_is_blocked() -> None:
    runtime = FakeCBraModRuntime(missing_channel_policy="error", min_observed_channels=22)
    with pytest.raises(ValueError, match="spherical_spline"):
        RealtimeModelPolicyRegistry.create(
            _package(
                runtime,
                model_type="cbramod",
                package_metadata=_cbramod_metadata(include_completion=False),
            )
        )


def test_cbramod_test_head_is_blocked() -> None:
    runtime = FakeCBraModRuntime()
    with pytest.raises(ValueError, match="test head"):
        RealtimeModelPolicyRegistry.create(
            _package(
                runtime,
                model_type="cbramod",
                is_test_head=True,
                package_metadata=_cbramod_metadata(),
            )
        )


@pytest.mark.parametrize(
    ("runtime", "metadata", "message"),
    [
        (
            FakeCBraModRuntime(missing_channel_policy="error", min_observed_channels=22),
            _cbramod_metadata(policy="error", min_observed_channels=22),
            "spherical_spline",
        ),
        (
            FakeCBraModRuntime(min_observed_channels=18),
            _cbramod_metadata(min_observed_channels=18),
            "min_observed_channels=19",
        ),
    ],
)
def test_cbramod_wrong_live_contract_is_blocked(
    runtime: FakeCBraModRuntime,
    metadata: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RealtimeModelPolicyRegistry.create(
            _package(
                runtime,
                model_type="cbramod",
                package_metadata=metadata,
            )
        )


def test_cbramod_prepared_completion_sha_must_match_package() -> None:
    runtime = FakeCBraModRuntime(completion_sha="wrong")
    policy = CBraModRealtimePolicy(
        _package(
            runtime,
            model_type="cbramod",
            package_metadata=_cbramod_metadata(),
        )
    )
    result = RealtimeRuntimeBridge(runtime, policy=policy).prepare(_window())
    assert result.model_input_safe is False
    assert "SHA" in str(result.failure_reason)


def test_registry_exposes_only_approved_model_types() -> None:
    assert RealtimeModelPolicyRegistry.list_model_types() == [
        "cbramod",
        "labram",
        "model_50m",
    ]
