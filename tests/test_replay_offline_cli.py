from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts import replay_offline as cli
from bci_dayloop.data.hdf5_dataset import HDF5Metadata
from bci_dayloop.inference.observability import LatencyBreakdown, PipelineRunStats
from bci_dayloop.inference.realtime import SlidingWindowDecoder
from bci_dayloop.inference.runtime_control import PipelineController, PipelineControllerSnapshot, PipelineState
from bci_dayloop.acquisition.base import AbstractAcquirer, AcquirerMetadata
from types import SimpleNamespace

import torch

from bci_dayloop.models.base import ModelBackend
from bci_dayloop.preprocessing.base import (
    ModelInputTransform,
)
from bci_dayloop.preprocessing.canonical import (
    SignalCanonicalizer,
)
from bci_dayloop.runtime.model import RuntimeModel
from bci_dayloop.runtime.types import (
    CanonicalEEGWindow,
    InputContract,
    ModelOutput,
    ModelTensor,
    PreparedModelInput,
)


def config() -> dict:
    return {
        "project": {"run_dir": "runs/from-yaml"},
        "data": {"output_hdf5": "data/from-yaml.h5"},
        "model": {"device": "cuda"},
        "replay": {
            "acquirer": "replay",
            "session": "1test",
            "speed": 2.0,
            "loop": False,
            "window_sec": 3.0,
            "step_sec": 1.0,
            "confidence_threshold": 0.6,
            "max_windows": 8,
            "jsonl_log": "runs/from-yaml/windows.jsonl",
            "summary_json": "runs/from-yaml/summary.json",
        },
    }


def test_cli_arguments_override_yaml_and_yaml_applies_when_unspecified():
    args = cli.build_parser().parse_args(
        [
            "--data", "data/from-cli.h5",
            "--model-package", "runs/from-cli/package",
            "--device", "cpu",
            "--max-windows", "4",
            "--window-sec", "4",
            "--step-sec", "0.5",
            "--replay-speed", "12",
            "--jsonl-log", "runs/from-cli/windows.jsonl",
            "--summary-json", "runs/from-cli/summary.json",
        ]
    )
    settings = cli.resolve_replay_settings(args, config())
    assert settings.device == "cpu"
    assert settings.window_sec == 4.0
    assert settings.step_sec == 0.5
    assert settings.replay_speed == 12.0
    assert settings.maximum_windows == 4
    assert settings.jsonl_log_path.name == "windows.jsonl"
    assert settings.summary_json_path.name == "summary.json"

    yaml_settings = cli.resolve_replay_settings(cli.build_parser().parse_args([]), config())
    assert yaml_settings.window_sec == 3.0
    assert yaml_settings.step_sec == 1.0
    assert yaml_settings.replay_speed == 2.0
    assert yaml_settings.maximum_windows == 8


def test_window_targets_and_no_jsonl_log():
    expected, target = cli.expected_and_target_windows(
        trial_count=2,
        samples_per_trial=10,
        sample_rate=2.0,
        window_sec=3.0,
        step_sec=1.0,
        maximum_windows=4,
    )
    assert (expected, target) == (8, 4)
    assert cli.expected_and_target_windows(
        trial_count=2,
        samples_per_trial=10,
        sample_rate=2.0,
        window_sec=3.0,
        step_sec=1.0,
        maximum_windows=99,
    ) == (8, 8)
    assert cli.expected_and_target_windows(
        trial_count=1,
        samples_per_trial=3,
        sample_rate=2.0,
        window_sec=2.0,
        step_sec=0.5,
        maximum_windows=9,
    ) == (0, 0)

    settings = cli.resolve_replay_settings(cli.build_parser().parse_args(["--no-jsonl-log"]), config())
    assert settings.jsonl_log_path is None


class IdentityInputTransform(ModelInputTransform):
    def __init__(
        self,
        *,
        channel_names: tuple[str, ...],
        sample_rate: float,
        window_sec: float,
    ) -> None:
        self._input_contract = InputContract(
            channel_names=channel_names,
            sample_rate=sample_rate,
            window_sec=window_sec,
            num_samples=int(
                round(sample_rate * window_sec)
            ),
            input_unit="uV",
            tensor_layout="BCT",
            strict_window_duration=True,
            model_input_keys=("signal",),
        )

    @property
    def input_contract(self) -> InputContract:
        return self._input_contract

    def transform(
        self,
        window: CanonicalEEGWindow,
    ) -> PreparedModelInput:
        signal = torch.from_numpy(
            np.ascontiguousarray(
                window.data,
                dtype=np.float32,
            )
        ).unsqueeze(0)

        return PreparedModelInput(
            model_input={
                "signal": signal,
            },
            canonical_window=window,
            preprocessing_trace=[
                *window.processing_history,
                "identity_input_transform",
            ],
        )


class FixedBackend(ModelBackend):
    def __init__(
        self,
        probabilities: tuple[float, ...] = (
            0.05,
            0.05,
            0.85,
            0.05,
        ),
    ) -> None:
        self._probabilities = torch.tensor(
            [probabilities],
            dtype=torch.float32,
        )

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def num_classes(self) -> int:
        return int(
            self._probabilities.shape[-1]
        )

    def predict_tensor(
        self,
        model_input: ModelTensor,
        return_features: bool = False,
    ) -> ModelOutput:
        if not isinstance(model_input, dict):
            raise TypeError(
                "Expected dictionary model input."
            )

        signal = model_input["signal"]

        probabilities = self._probabilities.clone()
        logits = torch.log(
            probabilities.clamp_min(1e-8)
        )

        confidence, prediction = probabilities.max(
            dim=-1
        )

        return ModelOutput(
            logits=logits,
            probabilities=probabilities,
            predicted_class=int(
                prediction[0].item()
            ),
            confidence=float(
                confidence[0].item()
            ),
            features=(
                signal
                if return_features
                else None
            ),
        )

    def encode_tensor(
        self,
        model_input: ModelTensor,
    ) -> torch.Tensor:
        if not isinstance(model_input, dict):
            raise TypeError(
                "Expected dictionary model input."
            )

        return model_input["signal"]

    def get_trainable_parameters(
        self,
        scope: str,
    ) -> list[torch.nn.Parameter]:
        return []


def build_fixed_runtime(
    *,
    sample_rate: float,
    window_sec: float,
    channel_names: tuple[str, ...],
) -> RuntimeModel:
    return RuntimeModel(
        canonicalizer=SignalCanonicalizer(
            target_unit="uV",
        ),
        input_transform=IdentityInputTransform(
            channel_names=channel_names,
            sample_rate=sample_rate,
            window_sec=window_sec,
        ),
        backend=FixedBackend(),
    )


def test_controller_builder_configures_jsonl_without_creating_it(tmp_path):
    settings = cli.resolve_replay_settings(
        cli.build_parser().parse_args(["--jsonl-log", str(tmp_path / "logs" / "windows.jsonl")]), config()
    )
    metadata = HDF5Metadata(20.0, ["C3", "C4"], ["left_hand", "right_hand", "feet", "tongue"], "uV", "test")
    runtime_model = build_fixed_runtime(
        sample_rate=20.0,
        window_sec=3.0,
        channel_names=("C3", "C4"),
    )

    runtime_package = SimpleNamespace(
        runtime_model=runtime_model,
        class_names=(
            "left_hand",
            "right_hand",
            "feet",
            "tongue",
        ),
        command_map={},
        window_sec=3.0,
        step_sec=1.0,
        confidence_threshold=0.6,
    )

    controller = cli.build_pipeline_controller(
        settings,
        runtime_package=runtime_package,
        metadata=metadata,
        stats=PipelineRunStats(),
        target_windows=4,
    )
    assert controller.max_windows == 4
    assert controller.decoder.jsonl_logger.path == settings.jsonl_log_path
    assert not settings.jsonl_log_path.exists()


def test_completed_and_failed_controller_snapshots_are_reported():
    settings = cli.resolve_replay_settings(cli.build_parser().parse_args([]), config())
    metadata = HDF5Metadata(20.0, ["C3", "C4"], ["left_hand", "right_hand"], "uV", "test")
    stats = PipelineRunStats()
    stats.set_expected_windows(2)
    stats.record_chunk()
    stats.record_success(LatencyBreakdown(1.0, 2.0, 3.0))
    completed = PipelineControllerSnapshot(PipelineState.COMPLETED, 1, 1.0, 2.0, 1, False, None, None)
    report = cli.build_report(
        settings,
        model_name="fixed",
        metadata=metadata,
        expected_windows=3,
        target_windows=2,
        stats_snapshot=stats.snapshot(),
        controller_snapshot=completed,
    )
    assert report.controller_state == "COMPLETED"
    assert report.expected_windows == 3 and report.target_windows == 2

    failed = PipelineControllerSnapshot(PipelineState.FAILED, 1, 1.0, 2.0, 1, False, "ValueError", "bad model")
    failed_report = cli.build_report(
        settings,
        model_name="fixed",
        metadata=metadata,
        expected_windows=3,
        target_windows=2,
        stats_snapshot=stats.snapshot(),
        controller_snapshot=failed,
    )
    assert failed_report.last_error_type == "ValueError"
    assert failed_report.last_error_message == "bad model"


def test_keyboard_interrupt_stop_helper_does_not_record_a_failure_window():
    class SlowAcquirer(AbstractAcquirer):
        def __init__(self):
            self.metadata = AcquirerMetadata("fake", 20.0, ["C3", "C4"], "uV")
            self.running = False
            self.current_label = 2
            self.current_trial_id = 1

        def start_stream(self):
            self.running = True

        def stop_stream(self):
            self.running = False

        def get_chunk(self, window_sec=None):
            return np.empty((2, 0), dtype=np.float32), np.empty(0)

        def get_new_samples(self):
            time.sleep(0.01)
            return np.ones((2, 10), dtype=np.float32), np.empty(0)

    stats = PipelineRunStats()
    runtime_model = build_fixed_runtime(
        sample_rate=20.0,
        window_sec=1.0,
        channel_names=("C3", "C4"),
    )

    decoder = SlidingWindowDecoder(
        runtime_model=runtime_model,
        class_names=(
            "left_hand",
            "right_hand",
            "feet",
            "tongue",
        ),
        channel_names=(
            "C3",
            "C4",
        ),
        sample_rate=20.0,
        input_unit="uV",
        window_sec=1.0,
        step_sec=0.5,
        run_stats=stats,
    )
    controller = PipelineController(decoder, SlowAcquirer)
    controller.start()
    time.sleep(0.02)

    assert cli.stop_after_keyboard_interrupt(controller) is None
    assert controller.snapshot().state == PipelineState.STOPPED
    assert stats.snapshot().failed_windows == 0
