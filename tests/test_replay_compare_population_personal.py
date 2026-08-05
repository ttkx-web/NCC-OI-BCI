from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import replay_compare_population_personal as cli  # noqa: E402
from bci_dayloop.data.hdf5_dataset import HDF5Metadata  # noqa: E402
from bci_dayloop.models.runtime_package import ModelRuntimePackage  # noqa: E402


CLASS_NAMES = ("left_hand", "right_hand", "feet", "tongue")


def _metadata(*, sample_rate: float = 100.0) -> HDF5Metadata:
    return HDF5Metadata(
        sample_rate=sample_rate,
        channel_names=["C3", "Cz", "C4"],
        class_names=list(CLASS_NAMES),
        unit="uV",
        dataset_name="unit-test",
    )


def _session_data(
    *,
    samples_per_trial: int = 400,
    labels: tuple[int, ...] = (0, 1, 2, 3),
) -> dict[str, np.ndarray]:
    trials = np.stack(
        [
            np.full(
                (3, samples_per_trial),
                fill_value=index + 1,
                dtype=np.float32,
            )
            for index in range(len(labels))
        ]
    )
    return {
        "data": trials,
        "labels": np.asarray(labels, dtype=np.int64),
        "subject_ids": np.ones(len(labels), dtype=np.int64),
        "session_ids": np.asarray(["1test"] * len(labels)),
        "trial_ids": np.arange(100, 100 + len(labels), dtype=np.int64),
    }


def _write_contract_package(
    path: Path,
    *,
    window_seconds: float = 4.0,
    aggregation: str = "flatten",
) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    (path / "model.yaml").write_text(
        "name: 50m-linear\n"
        "num_classes: 4\n"
        f"aggregation: {aggregation}\n"
        "output_layer_idx: 8\n"
        f"window_seconds: {window_seconds}\n"
        "target_sample_rate: 100.0\n"
        "patch_seconds: 1.0\n"
        "patch_stride_seconds: 1.0\n"
        "n_channels: 64\n"
        "d_model: 512\n"
        "model_n_time_patches: 10\n"
        "class_names:\n"
        "  - left_hand\n"
        "  - right_hand\n"
        "  - feet\n"
        "  - tongue\n",
        encoding="utf-8",
    )
    (path / "preprocessing.yaml").write_text(
        "filter_enabled: true\n"
        "filter_low_hz: 0.1\n"
        "filter_high_hz: 45.0\n"
        "filter_order: 4\n"
        "reference_mode: none\n"
        "zscore_enabled: true\n"
        "zscore_eps: 1.0e-08\n"
        "missing_channel_fill_value: 0.0\n"
        "strict_window_duration: true\n"
        "window_tolerance_seconds: 0.02\n",
        encoding="utf-8",
    )
    return path


class FixedModel:
    def __init__(
        self,
        probabilities: tuple[float, ...],
        *,
        timing: dict[str, float] | None = None,
    ) -> None:
        self.probabilities = np.asarray(
            [probabilities],
            dtype=np.float32,
        )
        self.last_timing = (
            SimpleNamespace(to_dict=lambda: dict(timing))
            if timing is not None
            else None
        )
        self.received: list[Any] = []

    def predict_proba(self, model_input: Any) -> np.ndarray:
        self.received.append(model_input)
        return self.probabilities.copy()


class IdentityPreprocessor:
    def __init__(self) -> None:
        self.calls = 0

    def transform(
        self,
        signal: np.ndarray,
        sample_rate: float,
        input_unit: str,
        *,
        reshape: bool = True,
    ) -> dict[str, np.ndarray]:
        self.calls += 1
        assert sample_rate == 100.0
        assert input_unit == "uV"
        assert reshape is True
        return {
            "signal": np.asarray(signal, dtype=np.float32),
            "mask": np.ones(signal.shape[0], dtype=np.float32),
        }


def _runtime(
    package_path: Path,
    *,
    probabilities: tuple[float, ...],
    window_sec: float = 4.0,
    class_names: tuple[str, ...] = CLASS_NAMES,
    preprocessor: Any | None = None,
) -> ModelRuntimePackage:
    return ModelRuntimePackage(
        model=FixedModel(
            probabilities,
            timing={"backbone_ms": 2.0, "classifier_ms": 0.1},
        ),  # type: ignore[arg-type]
        preprocessor=preprocessor or IdentityPreprocessor(),  # type: ignore[arg-type]
        model_name="50m-linear",
        class_names=class_names,
        command_map={},
        label_map={str(i): name for i, name in enumerate(class_names)},
        window_sec=window_sec,
        step_sec=0.5,
        target_sample_rate=100.0,
        is_test_head=False,
        warning_message=None,
        package_path=package_path,
    )


def test_direct_trial_mode_builds_one_window_per_trial() -> None:
    metadata = _metadata()
    data = _session_data()

    mode, windows = cli.build_windows(
        data,
        metadata,
        window_sec=4.0,
        mode="direct_trial",
    )

    assert mode == "direct_trial"
    assert len(windows) == 4
    assert [window.window_id for window in windows] == [1, 2, 3, 4]
    assert [window.class_id for window in windows] == [0, 1, 2, 3]
    assert [window.class_name for window in windows] == list(CLASS_NAMES)
    assert [window.source_trial_ids for window in windows] == [
        (100,),
        (101,),
        (102,),
        (103,),
    ]
    assert all(window.signal.shape == (3, 400) for window in windows)
    assert all(window.construction == "direct_trial" for window in windows)


def test_auto_selects_direct_trial_for_matching_4s_trials() -> None:
    mode, windows = cli.build_windows(
        _session_data(samples_per_trial=400),
        _metadata(sample_rate=100.0),
        window_sec=4.0,
        mode="auto",
    )

    assert mode == "direct_trial"
    assert len(windows) == 4


def test_direct_trial_rejects_duration_mismatch() -> None:
    with pytest.raises(ValueError, match="direct_trial requires"):
        cli.build_windows(
            _session_data(samples_per_trial=300),
            _metadata(sample_rate=100.0),
            window_sec=4.0,
            mode="direct_trial",
        )


def test_same_label_concat_never_crosses_class_boundaries() -> None:
    data = _session_data(
        samples_per_trial=200,
        labels=(0, 1, 0, 1),
    )
    mode, windows = cli.build_windows(
        data,
        _metadata(sample_rate=100.0),
        window_sec=4.0,
        mode="same_label_concat",
    )

    assert mode == "same_label_concat"
    assert len(windows) == 2
    assert [window.class_id for window in windows] == [0, 1]
    assert windows[0].source_trial_ids == (100, 102)
    assert windows[1].source_trial_ids == (101, 103)
    assert np.all(windows[0].signal[:, :200] == 1.0)
    assert np.all(windows[0].signal[:, 200:] == 3.0)
    assert np.all(windows[1].signal[:, :200] == 2.0)
    assert np.all(windows[1].signal[:, 200:] == 4.0)


def test_session_validation_rejects_missing_or_invalid_arrays() -> None:
    metadata = _metadata()
    valid = _session_data()

    missing = dict(valid)
    del missing["trial_ids"]
    with pytest.raises(KeyError, match="missing fields"):
        cli._validate_session(missing, metadata)

    invalid_channels = dict(valid)
    invalid_channels["data"] = np.zeros((4, 2, 400), dtype=np.float32)
    with pytest.raises(ValueError, match="Channel dimension"):
        cli._validate_session(invalid_channels, metadata)

    invalid_labels = dict(valid)
    invalid_labels["labels"] = np.asarray([0, 1, 2, 9], dtype=np.int64)
    with pytest.raises(ValueError, match="outside the class range"):
        cli._validate_session(invalid_labels, metadata)


def test_compatible_packages_require_matching_contracts(
    tmp_path: Path,
) -> None:
    population_path = _write_contract_package(tmp_path / "population")
    personal_path = _write_contract_package(tmp_path / "personal")
    population = _runtime(
        population_path,
        probabilities=(0.7, 0.1, 0.1, 0.1),
    )
    personal = _runtime(
        personal_path,
        probabilities=(0.8, 0.1, 0.05, 0.05),
    )

    result = cli.validate_packages(
        population,
        personal,
        population_path,
        personal_path,
        allow_mismatch=False,
    )

    assert result["warnings"] == []
    assert (
        result["population_contract"]
        == result["personal_contract"]
    )


def test_contract_mismatch_is_error_or_warning_by_flag(
    tmp_path: Path,
) -> None:
    population_path = _write_contract_package(
        tmp_path / "population",
        aggregation="flatten",
    )
    personal_path = _write_contract_package(
        tmp_path / "personal",
        aggregation="mean",
    )
    population = _runtime(
        population_path,
        probabilities=(0.7, 0.1, 0.1, 0.1),
    )
    personal = _runtime(
        personal_path,
        probabilities=(0.8, 0.1, 0.05, 0.05),
    )

    with pytest.raises(ValueError, match="contracts are not identical"):
        cli.validate_packages(
            population,
            personal,
            population_path,
            personal_path,
            allow_mismatch=False,
        )

    result = cli.validate_packages(
        population,
        personal,
        population_path,
        personal_path,
        allow_mismatch=True,
    )
    assert result["warnings"] == [
        "Population and personal model/preprocessing contracts are not identical."
    ]


def test_runtime_level_mismatch_is_always_rejected(
    tmp_path: Path,
) -> None:
    population_path = _write_contract_package(tmp_path / "population")
    personal_path = _write_contract_package(tmp_path / "personal")
    population = _runtime(
        population_path,
        probabilities=(0.7, 0.1, 0.1, 0.1),
        window_sec=4.0,
    )
    personal = _runtime(
        personal_path,
        probabilities=(0.8, 0.1, 0.05, 0.05),
        window_sec=10.0,
    )

    with pytest.raises(ValueError, match="window_sec"):
        cli.validate_packages(
            population,
            personal,
            population_path,
            personal_path,
            allow_mismatch=True,
        )


def test_predict_returns_class_confidence_latency_and_adapter_timing(
    tmp_path: Path,
) -> None:
    package = _write_contract_package(tmp_path / "package")
    runtime = _runtime(
        package,
        probabilities=(0.05, 0.15, 0.75, 0.05),
    )
    model_input = {
        "signal": np.zeros((1, 3, 400), dtype=np.float32),
    }

    result = cli.predict(
        runtime,
        model_input,
        preprocessing_ms=1.25,
    )

    assert result.class_id == 2
    assert result.class_name == "feet"
    assert result.confidence == pytest.approx(0.75)
    assert result.probabilities == pytest.approx(
        (0.05, 0.15, 0.75, 0.05)
    )
    assert result.model_latency_ms >= 0.0
    assert result.total_latency_ms >= 1.25
    assert result.adapter_timing == {
        "backbone_ms": 2.0,
        "classifier_ms": 0.1,
    }


@pytest.mark.parametrize(
    "probabilities, expected_message",
    [
        ((0.2, 0.2), "Expected probability shape"),
        ((0.5, 0.5, 0.5, 0.0), "Invalid model probability vector"),
        ((float("nan"), 0.2, 0.3, 0.5), "Invalid model probability vector"),
    ],
)
def test_predict_rejects_invalid_probability_output(
    tmp_path: Path,
    probabilities: tuple[float, ...],
    expected_message: str,
) -> None:
    package = _write_contract_package(tmp_path / "package")
    runtime = _runtime(package, probabilities=probabilities)

    with pytest.raises(RuntimeError, match=expected_message):
        cli.predict(
            runtime,
            np.zeros((1, 3, 400), dtype=np.float32),
            preprocessing_ms=0.0,
        )


def test_metrics_and_latency_summary_are_computed_correctly() -> None:
    result = cli.metrics(
        labels=[0, 0, 1, 1],
        predictions=[0, 1, 1, 1],
        class_names=("left", "right"),
    )

    assert result["accuracy"] == pytest.approx(0.75)
    assert result["balanced_accuracy"] == pytest.approx(0.75)
    assert result["macro_f1"] == pytest.approx(
        ((2 / 3) + 0.8) / 2
    )
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]

    latency = cli.latency_summary([1.0, 2.0, 3.0, float("nan")])
    assert latency["count"] == 3
    assert latency["current_ms"] == 3.0
    assert latency["mean_ms"] == 2.0
    assert latency["p50_ms"] == 2.0
    assert latency["p95_ms"] == pytest.approx(2.9)


def test_save_csv_serializes_both_model_results(
    tmp_path: Path,
) -> None:
    result = cli.WindowResult(
        window_id=1,
        status="success",
        construction="direct_trial",
        source_trial_ids=(101,),
        ground_truth_id=0,
        ground_truth_name="left_hand",
        preprocessing_latency_ms=1.0,
        population=cli.Prediction(
            class_id=1,
            class_name="right_hand",
            confidence=0.6,
            probabilities=(0.1, 0.6, 0.2, 0.1),
            model_latency_ms=3.0,
            total_latency_ms=4.0,
            adapter_timing=None,
        ),
        personal=cli.Prediction(
            class_id=0,
            class_name="left_hand",
            confidence=0.8,
            probabilities=(0.8, 0.1, 0.05, 0.05),
            model_latency_ms=2.5,
            total_latency_ms=3.5,
            adapter_timing=None,
        ),
        models_agree=False,
        population_correct=False,
        personal_correct=True,
    )
    output = tmp_path / "comparison.csv"

    cli.save_csv([result], output)

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["ground_truth_name"] == "left_hand"
    assert rows[0]["population_class_name"] == "right_hand"
    assert rows[0]["personal_class_name"] == "left_hand"
    assert rows[0]["models_agree"] == "False"


def test_main_compares_both_models_on_the_same_preprocessed_windows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    population_path = _write_contract_package(tmp_path / "population")
    personal_path = _write_contract_package(tmp_path / "personal")
    shared_preprocessor = IdentityPreprocessor()
    population_runtime = _runtime(
        population_path,
        probabilities=(0.1, 0.7, 0.1, 0.1),
        preprocessor=shared_preprocessor,
    )
    personal_runtime = _runtime(
        personal_path,
        probabilities=(0.8, 0.1, 0.05, 0.05),
        preprocessor=IdentityPreprocessor(),
    )
    metadata = _metadata()
    data = _session_data(labels=(0, 0))

    class FakeDataset:
        def __init__(self, path: str | Path) -> None:
            self.path = Path(path)

        @property
        def metadata(self) -> HDF5Metadata:
            return metadata

        def load(self, session: str) -> dict[str, np.ndarray]:
            assert session == "1test"
            return data

    def fake_load_runtime_package(
        package_path: str | Path,
        metadata_value: HDF5Metadata,
        *,
        device: str = "cpu",
    ) -> ModelRuntimePackage:
        assert metadata_value is metadata
        assert device == "cpu"
        resolved = Path(package_path).resolve()
        if resolved == population_path.resolve():
            return population_runtime
        if resolved == personal_path.resolve():
            return personal_runtime
        raise AssertionError(f"Unexpected package path: {resolved}")

    monkeypatch.setattr(cli, "EEGHDF5", FakeDataset)
    monkeypatch.setattr(
        cli.ModelFactory,
        "load_runtime_package",
        staticmethod(fake_load_runtime_package),
    )

    jsonl_output = tmp_path / "outputs" / "windows.jsonl"
    csv_output = tmp_path / "outputs" / "windows.csv"
    summary_output = tmp_path / "outputs" / "summary.json"

    return_code = cli.main(
        [
            "--data",
            str(tmp_path / "subject_01.h5"),
            "--session",
            "1test",
            "--population-package",
            str(population_path),
            "--personal-package",
            str(personal_path),
            "--device",
            "cpu",
            "--window-mode",
            "direct_trial",
            "--max-windows",
            "2",
            "--print-every",
            "0",
            "--jsonl-output",
            str(jsonl_output),
            "--csv-output",
            str(csv_output),
            "--summary-output",
            str(summary_output),
        ]
    )

    assert return_code == 0
    assert shared_preprocessor.calls == 2
    assert len(population_runtime.model.received) == 2
    assert len(personal_runtime.model.received) == 2

    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    assert summary["windowing"]["successful_windows"] == 2
    assert summary["windowing"]["window_completion_rate"] == 1.0
    assert summary["comparison"]["agreement_rate"] == 0.0
    assert summary["comparison"]["population"]["accuracy"] == 0.0
    assert summary["comparison"]["personal"]["accuracy"] == 1.0
    assert summary["comparison"]["gain"]["accuracy"] == 1.0

    lines = jsonl_output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert csv_output.is_file()
