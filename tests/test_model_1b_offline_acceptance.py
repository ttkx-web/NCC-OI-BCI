from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts import replay_offline as replay_cli
from scripts import run_window_latency_benchmark as benchmark_cli
from bci_dayloop.benchmarking.core import RuntimeBenchmarkCore
from bci_dayloop.benchmarking.windows import ReplayWindowProvider
from bci_dayloop.data.hdf5_dataset import HDF5Metadata, write_hdf5
from bci_dayloop.models.model_50m.config import STANDARD_64_CHANNELS
from bci_dayloop.packages.loader import LoadedRuntimePackage
from bci_dayloop.runtime.types import InputContract, ModelOutput, RawEEGWindow
from bci_dayloop.utils.config import load_yaml


CLASS_NAMES = ("left_hand", "right_hand", "feet", "tongue")


class _Fake1BRuntime:
    """A protocol-level 1B runtime stand-in; it deliberately is not RuntimeModel."""

    input_contract = InputContract(
        channel_names=STANDARD_64_CHANNELS,
        sample_rate=100.0,
        window_sec=4.0,
        num_samples=400,
        input_unit="uV",
        tensor_layout="BCT",
        model_input_keys=(
            "token_inputs",
            "token_channel_indices",
            "token_time_indices",
            "channel_valid_mask",
        ),
        strict_window_duration=True,
    )

    def __init__(self) -> None:
        self.prepared_windows = 0

    def prepare(self, raw_window: RawEEGWindow) -> object:
        assert raw_window.data.shape == (64, 400)
        self.prepared_windows += 1
        return SimpleNamespace(
            preprocessing_trace=("fake_1b_prepare",),
            diagnostics={"model_type": "model_1b"},
        )

    def predict_prepared(
        self,
        prepared: object,
        *,
        return_features: bool = False,
    ) -> ModelOutput:
        del prepared, return_features
        probabilities = torch.tensor(
            [[0.1, 0.2, 0.6, 0.1]],
            dtype=torch.float32,
        )
        return ModelOutput(
            logits=torch.log(probabilities),
            probabilities=probabilities,
            predicted_class=2,
            confidence=0.6,
            diagnostics={"model_type": "model_1b"},
        )


def _write_hdf5(path: Path) -> Path:
    data = np.zeros((3, 64, 400), dtype=np.float32)
    return write_hdf5(
        path,
        data,
        np.asarray([0, 1, 2], dtype=np.int64),
        np.ones(3, dtype=np.int64),
        ["1test", "1test", "1test"],
        np.asarray([1, 2, 3], dtype=np.int64),
        HDF5Metadata(
            sample_rate=100.0,
            channel_names=list(STANDARD_64_CHANNELS),
            class_names=list(CLASS_NAMES),
            unit="uV",
            dataset_name="synthetic_bnci",
        ),
    )


def _loaded_package(path: Path, runtime: _Fake1BRuntime) -> LoadedRuntimePackage:
    return LoadedRuntimePackage(
        runtime_model=runtime,  # type: ignore[arg-type]
        package_path=path,
        model_type="model_1b",
        model_name="1b-frozen-linear",
        class_names=CLASS_NAMES,
        command_map={"feet": "FORWARD"},
        step_sec=0.5,
        confidence_threshold=0.55,
        is_test_head=False,
        warning_message=None,
        metrics={},
        package_metadata={},
    )


def _write_package_descriptor(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "package.yaml").write_text(
        """schema_version: 2
package:
  id: synthetic-1b-4s
  version: '1'
model:
  type: model_1b
  name: 1b-frozen-linear
  class_names: [left_hand, right_hand, feet, tongue]
input_contract:
  window_sec: 4.0
runtime:
  step_sec: 0.5
""",
        encoding="utf-8",
    )


def test_1b_replay_and_benchmark_configs_are_static_4s_package_contracts() -> None:
    replay = load_yaml(ROOT / "configs/stage1/replay_1b_4s.yaml")
    assert replay["replay"]["window_sec"] == 4.0
    assert replay["replay"]["step_sec"] == 0.5
    assert replay["replay"]["model_package"] == (
        "model_packages/stage1_1b/bnci2014_001/subject_01/"
        "population/4s_flatten/v1"
    )
    assert replay["online"]["strategy"] == "none"

    one_b = load_yaml(ROOT / "configs/benchmarks/window_latency_1b_4s.yaml")
    pair = load_yaml(ROOT / "configs/benchmarks/window_latency_50m_1b_4s.yaml")
    assert one_b["benchmark"]["schedule"] == {
        "window_sec": 4.0,
        "step_sec": 0.5,
        "align_decision_endpoints": True,
        "warmup_windows": 20,
        "measured_windows": 200,
    }
    assert [item["id"] for item in pair["benchmark"]["candidates"]] == [
        "model_50m_4s",
        "model_1b_4s",
    ]
    assert "NEURACLE_JELLYFISH_HOST" not in (
        (ROOT / "configs/benchmarks/window_latency_1b_4s.yaml")
        .read_text(encoding="utf-8")
    )


def test_replay_offline_uses_the_common_loader_for_a_model_1b_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = _write_hdf5(tmp_path / "replay.h5")
    package_path = tmp_path / "package"
    package_path.mkdir()
    runtime = _Fake1BRuntime()
    loaded = _loaded_package(package_path, runtime)
    loaded_paths: list[Path] = []

    def fake_load(path: Path, **_: object) -> LoadedRuntimePackage:
        loaded_paths.append(Path(path))
        return loaded

    monkeypatch.setattr(replay_cli, "load_runtime_package", fake_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replay_offline.py",
            "--config", str(ROOT / "configs/stage1/replay_1b_4s.yaml"),
            "--data", str(data_path),
            "--model-package", str(package_path),
            "--device", "cpu",
            "--max-windows", "1",
            "--replay-speed", "1000000",
            "--no-jsonl-log",
            "--summary-json", str(tmp_path / "replay_summary.json"),
        ],
    )
    exit_code = replay_cli.main()

    assert exit_code == 0
    assert loaded_paths == [package_path]
    assert runtime.prepared_windows == 1


def test_model_1b_package_is_a_normal_benchmark_candidate_and_records_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = _write_hdf5(tmp_path / "benchmark.h5")
    package_path = tmp_path / "package"
    _write_package_descriptor(package_path)
    runtime = _Fake1BRuntime()
    loaded = _loaded_package(package_path, runtime)
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        f"""benchmark:
  mode: replay
  device: cpu
  data:
    path: {data_path}
    session: 1test
  schedule:
    step_sec: 0.5
    warmup_windows: 20
    measured_windows: 200
  output:
    root_dir: {tmp_path / 'default_runs'}
  candidates:
    - id: model_1b_4s
      package: {package_path}
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        benchmark_cli,
        "load_runtime_package",
        lambda *_args, **_kwargs: loaded,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_window_latency_benchmark.py",
            "--config", str(config_path),
            "--warmup-windows", "1",
            "--measured-windows", "2",
            "--output-root", str(tmp_path / "runs"),
            "--run-id", "model_1b_smoke",
        ],
    )

    assert benchmark_cli.main() == 0
    summary = load_yaml(tmp_path / "runs/model_1b_smoke/summary.json")
    candidate = summary["candidates"][0]
    assert candidate["candidate"]["model_type"] == "model_1b"
    assert candidate["candidate"]["warmup_windows"] == 1
    assert candidate["candidate"]["measured_windows"] == 2
    for field in (
        "preprocessing_ms",
        "inference_ms",
        "output_materialization_ms",
        "compute_total_ms",
    ):
        assert {"p50", "p95", "max"} <= set(candidate[field])

    records = RuntimeBenchmarkCore(
        runtime_model=runtime,  # type: ignore[arg-type]
        device="cpu",
    ).run(
        provider=ReplayWindowProvider(
            data_path=data_path,
            session="1test",
            window_sec=4.0,
            step_sec=0.5,
            maximum_windows=1,
        ),
        warmup_windows=0,
        measured_windows=1,
    )
    record = records[0]
    assert record.prediction == 2
    assert record.confidence == pytest.approx(0.6)
    assert record.probabilities == pytest.approx([0.1, 0.2, 0.6, 0.1])
    assert all(
        getattr(record, field) >= 0.0
        for field in (
            "preprocessing_ms",
            "inference_ms",
            "output_materialization_ms",
            "compute_total_ms",
        )
    )
