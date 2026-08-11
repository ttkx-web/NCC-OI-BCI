from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bci_dayloop.realtime.contracts import EEGChunk, EventMarker
from bci_dayloop.realtime.runtime_mapping import APPROVED_NEURACLE_59_TO_STANDARD64
from bci_dayloop.runtime.types import CanonicalEEGWindow, InputContract, PreparedModelInput, RawEEGWindow
from scripts.probe_neuracle_runtime_inference import main


POLICY = APPROVED_NEURACLE_59_TO_STANDARD64
LIVE19_FOR_TEST = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1",
    "Cz", "C2", "C4", "C6", "CP3", "CP1", "CP2", "CP4", "Pz",
    "POz",
)
CBRAMOD22_FOR_TEST = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1",
    "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
)
CBRAMOD_COMPLETION_SHA = (
    "a3f7918a08115e0d3bb33ffb5cbb8fc1a467313872a6bc28fb16d6bc0636bae5"
)


class FakeInferenceSource:
    instances: list["FakeInferenceSource"] = []

    def __init__(self, config: object) -> None:
        self.config = config
        self.metadata = {
            "channel_types": tuple("EEG" for _ in range(59)) + ("Trigger",),
            "channel_units": tuple("uV" for _ in range(59)) + ("code",),
            "unit_evidence_level": "vendor_confirmed",
        }
        self.connected = False
        self.state = "ready"
        self.emitted = False
        self.events = [EventMarker(1.0, "trigger", code=4, metadata={"raw_device_timestamp": 1000})]
        type(self).instances.append(self)

    def connect(self) -> None:
        self.connected = True

    def read_chunk(self) -> EEGChunk | None:
        self.state = "streaming"
        if self.emitted:
            return None
        self.emitted = True
        return EEGChunk(
            samples=np.zeros((60, 4000), dtype=np.float32),
            channel_names=POLICY.source_channel_names + ("Trigger",),
            sampling_rate=1000.0,
            unit="mixed",
            timestamps=np.arange(4000, dtype=np.float64) / 1000.0,
            sequence_id=0,
            device_id=None,
            received_at=0.0,
            metadata={
                "channel_types": self.metadata["channel_types"],
                "channel_units": self.metadata["channel_units"],
                "unit_evidence_level": "vendor_confirmed",
                "model_safe": False,
                "raw_start_timestamp": 0,
            },
        )

    def read_event(self) -> EventMarker | None:
        return self.events.pop(0) if self.events else None

    def health(self) -> dict[str, object]:
        return {
            "state": self.state,
            "connected": self.connected,
            "metadata_ready": self.connected,
            "received_packets": 1 if self.emitted else 0,
            "missing_packets": 0,
            "duplicate_packets": 0,
            "out_of_order_packets": 0,
            "malformed_packets": 0,
            "reconnect_count": 0,
            "last_error": None,
        }

    def disconnect(self) -> None:
        self.connected = False
        self.state = "stopped"


class FakeRuntime:
    def __init__(self, *, prepared_is_safe: bool = True, prediction_error: bool = False) -> None:
        self.input_contract = InputContract(
            channel_names=POLICY.target_channel_names,
            sample_rate=100.0,
            window_sec=4.0,
            num_samples=400,
            input_unit="uV",
            tensor_layout="BCT",
            model_input_keys=("signal", "channel_valid_mask"),
        )
        self.input_transform = SimpleNamespace(
            config=SimpleNamespace(output_layer_idx=8, aggregation="flatten")
        )
        self.prepared_is_safe = prepared_is_safe
        self.prediction_error = prediction_error
        self.predict_calls = 0

    def prepare(self, raw_window: RawEEGWindow) -> PreparedModelInput:
        assert raw_window.data.shape == (59, 4000)
        mask = torch.ones((1, 64), dtype=torch.bool)
        mask[0, list(POLICY.target_missing_indices)] = False
        diagnostics = {
            "unknown_channel_names": ("PO5", "PO6"),
            "mapped_channel_count": 57,
            "missing_channel_count": 7,
            "duplicate_channel_count": 0,
            "padded_points": 0,
            "cropped_points": 0,
        }
        if not self.prepared_is_safe:
            diagnostics["padded_points"] = 1
        return PreparedModelInput(
            model_input={
                "signal": torch.zeros((1, 64, 400), dtype=torch.float32),
                "channel_valid_mask": mask,
            },
            canonical_window=CanonicalEEGWindow(
                data=np.zeros((64, 400), dtype=np.float32),
                channel_names=list(POLICY.target_channel_names),
                sample_rate=100.0,
                unit="uV",
            ),
            preprocessing_trace=["fake unified runtime"],
            diagnostics=diagnostics,
        )

    def predict_prepared(self, _prepared: PreparedModelInput) -> object:
        self.predict_calls += 1
        if self.prediction_error:
            raise RuntimeError("synthetic prediction failure")
        return SimpleNamespace(
            predicted_class=2,
            confidence=0.8,
            probabilities=torch.tensor([[0.05, 0.10, 0.80, 0.05]], dtype=torch.float32),
        )


class FakeLaBraMRuntime:
    def __init__(
        self,
        *,
        channel_names: tuple[str, ...] = LIVE19_FOR_TEST,
    ) -> None:
        self.input_contract = InputContract(
            channel_names=channel_names,
            sample_rate=200.0,
            window_sec=4.0,
            num_samples=800,
            input_unit="uV",
            tensor_layout="BCTP",
            model_input_keys=("signal",),
        )
        self.predict_calls = 0

    def prepare(self, raw_window: RawEEGWindow) -> PreparedModelInput:
        assert raw_window.data.shape == (
            len(self.input_contract.channel_names),
            4000,
        )
        assert tuple(raw_window.channel_names) == self.input_contract.channel_names
        channels = len(self.input_contract.channel_names)
        return PreparedModelInput(
            model_input={
                "signal": torch.zeros(
                    (1, channels, 4, 200),
                    dtype=torch.float32,
                )
            },
            canonical_window=CanonicalEEGWindow(
                data=np.zeros((channels, 4000), dtype=np.float32),
                channel_names=list(self.input_contract.channel_names),
                sample_rate=1000.0,
                unit="uV",
            ),
            preprocessing_trace=["fake labram runtime"],
            diagnostics={
                "source_channel_count": channels,
                "target_channel_count": channels,
                "missing_channel_names": [],
            },
        )

    def predict_prepared(self, _prepared: PreparedModelInput) -> object:
        self.predict_calls += 1
        return SimpleNamespace(
            predicted_class=1,
            confidence=0.7,
            probabilities=torch.tensor(
                [[0.1, 0.7, 0.1, 0.1]],
                dtype=torch.float32,
            ),
        )


class FakeCBraModRuntime:
    def __init__(self, *, min_observed_channels: int = 19) -> None:
        self.input_contract = InputContract(
            channel_names=CBRAMOD22_FOR_TEST,
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
                missing_channel_policy="spherical_spline",
                min_observed_channels=min_observed_channels,
                spline_alpha=1e-5,
            )
        )
        self.predict_calls = 0

    def prepare(self, raw_window: RawEEGWindow) -> PreparedModelInput:
        assert raw_window.data.shape == (59, 4000)
        assert tuple(raw_window.channel_names) == POLICY.source_channel_names
        return PreparedModelInput(
            model_input={
                "signal": torch.zeros((1, 22, 4, 200), dtype=torch.float32)
            },
            canonical_window=CanonicalEEGWindow(
                data=np.zeros((22, 800), dtype=np.float32),
                channel_names=list(CBRAMOD22_FOR_TEST),
                sample_rate=200.0,
                unit="uV",
            ),
            preprocessing_trace=["fake cbramod shared runtime preprocessor"],
            diagnostics={
                "observed_channel_count": 19,
                "observed_channel_names": list(LIVE19_FOR_TEST),
                "missing_channel_names": ["CPZ", "P1", "P2"],
                "duplicate_channel_count": 0,
                "completion_policy": "spherical_spline",
                "completion_matrix_sha256": CBRAMOD_COMPLETION_SHA,
            },
        )

    def predict_prepared(self, _prepared: PreparedModelInput) -> object:
        self.predict_calls += 1
        return SimpleNamespace(
            predicted_class=0,
            confidence=0.6,
            probabilities=torch.tensor(
                [[0.6, 0.1, 0.2, 0.1]], dtype=torch.float32
            ),
        )


def _cbramod_package_metadata(
    *, min_observed_channels: int = 19,
) -> dict[str, object]:
    return {
        "package": {"id": "approved_cbramod"},
        "runtime": {
            "channel_completion": {
                "deployment_profile": "neuracle_live19_spline22",
                "observed_required": 19,
                "observed_channel_names": list(LIVE19_FOR_TEST),
                "missing_expected": ["CPz", "P1", "P2"],
                "missing_channel_policy": "spherical_spline",
                "min_observed_channels": min_observed_channels,
                "spline_alpha": 1e-5,
                "channel_completion_source": "shared_runtime_preprocessor",
                "completion_matrix_sha256": CBRAMOD_COMPLETION_SHA,
            }
        },
        "provenance": {
            "completion_matrix_sha256": CBRAMOD_COMPLETION_SHA,
        },
    }


def _package(
    tmp_path: Path,
    runtime: object,
    *,
    test_head: bool = False,
    model_type: str = "model_50m",
) -> object:
    package_path = tmp_path / "package"
    package_path.mkdir()
    package_metadata = (
        _cbramod_package_metadata()
        if model_type == "cbramod"
        else {"package": {"id": f"approved_{model_type}"}}
    )
    return SimpleNamespace(
        runtime_model=runtime,
        package_path=package_path,
        model_type=model_type,
        model_name=(
            "50m-linear"
            if model_type == "model_50m"
            else (
                "labram-linear"
                if model_type == "labram"
                else "cbramod-frozen-head"
            )
        ),
        class_names=("left_hand", "right_hand", "feet", "tongue"),
        step_sec=0.5,
        is_test_head=test_head,
        package_metadata=package_metadata,
    )


def _run_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runtime: object,
    *,
    test_head: bool = False,
    model_type: str = "model_50m",
) -> tuple[int, dict[str, object]]:
    FakeInferenceSource.instances.clear()
    package = _package(
        tmp_path,
        runtime,
        test_head=test_head,
        model_type=model_type,
    )
    monkeypatch.setattr("scripts.probe_neuracle_runtime_inference.NeuracleJellyFishSource", FakeInferenceSource)
    monkeypatch.setattr("scripts.probe_neuracle_runtime_inference.load_runtime_package", lambda *_args, **_kwargs: package)
    output_dir = tmp_path / "out"
    result = main(
        [
            "--package", str(package.package_path),
            "--duration-sec", "0.01",
            "--output-dir", str(output_dir),
            "--no-save-waveform",
        ]
    )
    return result, json.loads((output_dir / "runtime_inference_summary.json").read_text(encoding="utf-8"))


def test_runtime_inference_probe_uses_bridge_then_predicts_sanitized_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    runtime = FakeRuntime()

    result, summary = _run_probe(monkeypatch, tmp_path, runtime)

    records = [
        json.loads(line)
        for line in (tmp_path / "out" / "runtime_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert result == 0
    assert summary["status"] == "passed"
    assert summary["is_test_head"] is False
    assert summary["received_packets"] == 1
    assert summary["received_samples"] == 4000
    assert summary["emitted_windows"] == 1
    assert summary["model_input_safe_count"] == 1
    assert summary["prediction_success_count"] == 1
    assert summary["prediction_failure_count"] == 0
    assert summary["model_type"] == "model_50m"
    assert summary["model_name"] == "50m-linear"
    assert summary["package_id"] == "approved_model_50m"
    assert summary["realtime_policy_id"] == (
        "model_50m_neuracle_59_to_standard64_v1"
    )
    assert summary["prepared_shape"] == [1, 64, 400]
    assert summary["final_health"]["state"] == "stopped"
    assert summary["final_health"]["connected"] is False
    assert len(records) == 1
    assert records[0]["source_shape"] == [59, 4000]
    assert records[0]["prepared_shape"] == [1, 64, 400]
    assert records[0]["valid_channel_count"] == 57
    assert records[0]["predicted_name"] == "feet"
    assert records[0]["probabilities"] == pytest.approx([0.05, 0.10, 0.80, 0.05])
    assert records[0]["marker_summary"] == [{"event_type": "trigger", "code": 4}]
    assert records[0]["model_type"] == "model_50m"
    assert records[0]["model_name"] == "50m-linear"
    assert records[0]["package_id"] == "approved_model_50m"
    assert records[0]["realtime_policy_id"] == (
        "model_50m_neuracle_59_to_standard64_v1"
    )
    serialized = json.dumps({"summary": summary, "records": records})
    assert '"samples":[' not in serialized
    assert "127.0.0.1" not in serialized
    assert "raw_device_timestamp" not in serialized
    assert runtime.predict_calls == 1


def test_prepare_gate_failure_never_calls_predict_prepared(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    runtime = FakeRuntime(prepared_is_safe=False)

    result, summary = _run_probe(monkeypatch, tmp_path, runtime)

    assert result == 2
    assert summary["failed_windows"] == 1
    assert summary["model_input_failure_count"] == 1
    assert summary["prediction_success_count"] == 0
    assert summary["prediction_failure_count"] == 0
    assert runtime.predict_calls == 0
    assert not (tmp_path / "out" / "runtime_predictions.jsonl").exists()


def test_prediction_exception_is_counted_and_does_not_persist_a_window_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    runtime = FakeRuntime(prediction_error=True)

    result, summary = _run_probe(monkeypatch, tmp_path, runtime)

    assert result == 2
    assert summary["failed_windows"] == 1
    assert summary["model_input_safe_count"] == 1
    assert summary["prediction_success_count"] == 0
    assert summary["prediction_failure_count"] == 1
    assert summary["last_error"] == "runtime_prediction_failed"
    assert runtime.predict_calls == 1
    assert not (tmp_path / "out" / "runtime_predictions.jsonl").exists()


def test_test_head_is_rejected_before_connecting_to_a_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    runtime = FakeRuntime()

    result, summary = _run_probe(monkeypatch, tmp_path, runtime, test_head=True)

    assert result == 2
    assert summary["is_test_head"] is True
    assert summary["last_error"] == "probe_failed"
    assert not FakeInferenceSource.instances


def test_labram_package_uses_registered_policy_and_predicts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = FakeLaBraMRuntime()

    result, summary = _run_probe(
        monkeypatch,
        tmp_path,
        runtime,
        model_type="labram",
    )

    records = [
        json.loads(line)
        for line in (
            tmp_path / "out" / "runtime_predictions.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert result == 0
    assert summary["status"] == "passed"
    assert summary["model_type"] == "labram"
    assert summary["model_name"] == "labram-linear"
    assert summary["package_id"] == "approved_labram"
    assert summary["realtime_policy_id"] == (
        "labram_package_required_channels_v1"
    )
    assert summary["prepared_shape"] == [1, 19, 4, 200]
    assert records[0]["prepared_shape"] == [1, 19, 4, 200]
    assert records[0]["valid_channel_count"] == 19
    assert records[0]["model_type"] == "labram"
    assert records[0]["predicted_name"] == "right_hand"
    serialized = json.dumps({"summary": summary, "records": records})
    assert '"samples":[' not in serialized
    assert "127.0.0.1" not in serialized
    assert "raw_device_timestamp" not in serialized
    assert runtime.predict_calls == 1


def test_labram_incompatible_contract_fails_before_device_connect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = FakeLaBraMRuntime(
        channel_names=LIVE19_FOR_TEST + ("CPz",)
    )

    result, summary = _run_probe(
        monkeypatch,
        tmp_path,
        runtime,
        model_type="labram",
    )

    assert result == 2
    assert summary["last_error"] == "realtime_policy_blocked"
    assert summary["compatibility_status"] == "blocked"
    assert "BLOCKED" in str(summary["compatibility_error"])
    assert not FakeInferenceSource.instances


def test_cbramod_package_uses_registered_policy_and_reports_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = FakeCBraModRuntime()
    result, summary = _run_probe(
        monkeypatch,
        tmp_path,
        runtime,
        model_type="cbramod",
    )
    records = [
        json.loads(line)
        for line in (
            tmp_path / "out" / "runtime_predictions.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert result == 0
    assert summary["compatibility_status"] == "passed"
    assert summary["model_type"] == "cbramod"
    assert summary["model_name"] == "cbramod-frozen-head"
    assert summary["package_id"] == "approved_cbramod"
    assert summary["realtime_policy_id"] == (
        "cbramod_neuracle_59_live19_spline22_v1"
    )
    assert summary["prepared_shape"] == [1, 22, 4, 200]
    assert summary["observed_channel_count"] == 19
    assert summary["missing_channel_names"] == ["CPz", "P1", "P2"]
    assert summary["completion_policy"] == "spherical_spline"
    assert summary["completion_matrix_sha256"] == CBRAMOD_COMPLETION_SHA
    assert records[0]["observed_channel_count"] == 19
    assert records[0]["missing_channel_names"] == ["CPz", "P1", "P2"]
    assert records[0]["completion_matrix_sha256"] == CBRAMOD_COMPLETION_SHA
    assert runtime.predict_calls == 1
    serialized = json.dumps({"summary": summary, "records": records})
    assert '"samples":[' not in serialized
    assert "127.0.0.1" not in serialized
    assert "raw_device_timestamp" not in serialized


def test_cbramod_incompatible_contract_fails_before_device_connect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    FakeInferenceSource.instances.clear()
    runtime = FakeCBraModRuntime(min_observed_channels=18)
    package = _package(tmp_path, runtime, model_type="cbramod")
    package.package_metadata = _cbramod_package_metadata(
        min_observed_channels=18
    )
    monkeypatch.setattr(
        "scripts.probe_neuracle_runtime_inference.NeuracleJellyFishSource",
        FakeInferenceSource,
    )
    monkeypatch.setattr(
        "scripts.probe_neuracle_runtime_inference.load_runtime_package",
        lambda *_args, **_kwargs: package,
    )
    output_dir = tmp_path / "out"
    result = main(
        [
            "--package", str(package.package_path),
            "--duration-sec", "0.01",
            "--output-dir", str(output_dir),
            "--no-save-waveform",
        ]
    )
    summary = json.loads(
        (output_dir / "runtime_inference_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert result == 2
    assert summary["compatibility_status"] == "blocked"
    assert summary["last_error"] == "realtime_policy_blocked"
    assert not FakeInferenceSource.instances


def test_unknown_model_type_fails_before_device_connect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime()

    result, summary = _run_probe(
        monkeypatch,
        tmp_path,
        runtime,
        model_type="unknown-model",
    )

    assert result == 2
    assert summary["last_error"] == "realtime_policy_blocked"
    assert summary["compatibility_status"] == "blocked"
    assert "BLOCKED" in str(summary["compatibility_error"])
    assert not FakeInferenceSource.instances
