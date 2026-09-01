from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from bci_dayloop.data.hdf5_dataset import HDF5Metadata, write_hdf5
from bci_dayloop.data.sequential_dataset import (
    load_sequential_dataset,
    validate_package_window_contract,
)
from bci_dayloop.data.workload import (
    DIFF_CONDITION,
    EASY_CONDITION,
    CLASS_NAMES as WORKLOAD_CLASS_NAMES,
    WorkloadCondition,
    build_workload_session,
    write_workload_hdf5,
)
from bci_dayloop.runtime.types import ModelOutput


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts import evaluate_neuroonline_sequential as sequential_evaluator


def _write_bnci(path: Path) -> Path:
    data = np.arange(4 * 2 * 8, dtype=np.float32).reshape(4, 2, 8)
    return write_hdf5(
        path,
        data,
        np.asarray([0, 1, 2, 3], dtype=np.int64),
        np.ones(4, dtype=np.int64),
        ["1test"] * 4,
        np.asarray([30, 10, 40, 20], dtype=np.int64),
        HDF5Metadata(
            sample_rate=2.0,
            channel_names=["C3", "C4"],
            class_names=["left", "right", "feet", "tongue"],
            unit="uV",
            dataset_name="bnci-test",
        ),
    )


def _write_workload(path: Path) -> Path:
    easy = WorkloadCondition(
        condition=EASY_CONDITION,
        data=np.arange(2 * 2 * 4, dtype=np.float32).reshape(2, 2, 4),
        channel_names=("C3", "C4"),
        sample_rate=2.0,
        unit="uV",
        source_set=path.parent / "easy.set",
    )
    diff = WorkloadCondition(
        condition=DIFF_CONDITION,
        data=(100 + np.arange(2 * 2 * 4, dtype=np.float32)).reshape(2, 2, 4),
        channel_names=("C3", "C4"),
        sample_rate=2.0,
        unit="uV",
        source_set=path.parent / "diff.set",
    )
    session = build_workload_session(
        easy,
        diff,
        subject_id="P03",
        session_id="S1",
    )
    return write_workload_hdf5(
        path,
        [session],
        subject_id="P03",
        data_root=path.parent,
    )


def _write_seed(path: Path) -> Path:
    data = np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4)
    with h5py.File(path, "w") as handle:
        handle.attrs["dataset_name"] = "seed"
        handle.attrs["subject_id"] = "SEED-01"
        handle.attrs["class_names"] = json.dumps(["negative", "neutral", "positive"])
        handle.attrs["unit"] = "uV"
        handle.attrs["window_sec"] = 2.0
        group = handle.create_group("sessions").create_group("session_1")
        group.attrs["sample_rate"] = 2.0
        group.attrs["channel_names"] = json.dumps(["FP1", "FP2"])
        group.create_dataset("data", data=data)
        group.create_dataset("labels", data=np.asarray([2, 0, 1], dtype=np.int64))
        group.create_dataset("trial_ids", data=np.asarray([b"trial-30", b"trial-10", b"trial-20"]))
        group.create_dataset("trial_ordinals", data=np.asarray([1, 2, 3], dtype=np.int64))
    return path


def test_bnci_adapter_preserves_four_second_hdf5_order(tmp_path: Path) -> None:
    dataset = load_sequential_dataset(_write_bnci(tmp_path / "bnci.h5"), session="1test")

    assert dataset.metadata.window_sec == pytest.approx(4.0)
    assert dataset.data.shape == (4, 2, 8)
    assert dataset.trial_ids.tolist() == [30, 10, 40, 20]
    assert dataset.trial_ordinals.tolist() == [1, 2, 3, 4]
    assert dataset.session_ids.tolist() == ["1test"] * 4
    assert dataset.window_ids.tolist() == [
        "30:trial4s",
        "10:trial4s",
        "40:trial4s",
        "20:trial4s",
    ]


def test_workload_adapter_preserves_two_second_windows_and_identifiers(
    tmp_path: Path,
) -> None:
    dataset = load_sequential_dataset(_write_workload(tmp_path / "workload.h5"), session="S1")

    assert dataset.metadata.window_sec == pytest.approx(2.0)
    assert dataset.metadata.class_names == WORKLOAD_CLASS_NAMES
    assert dataset.data.shape == (4, 2, 4)
    assert dataset.labels.tolist() == [0, 1, 0, 1]
    assert dataset.trial_ordinals.tolist() == [1, 2, 3, 4]
    assert dataset.window_ids.tolist() == [
        "P03:S1:MATBeasy:000000",
        "P03:S1:MATBdiff:000000",
        "P03:S1:MATBeasy:000001",
        "P03:S1:MATBdiff:000001",
    ]
    # The sequential evaluator reports persisted source identities verbatim.
    assert dataset.subject_ids.tolist() == ["P03"] * 4
    assert dataset.session_ids.tolist() == ["S1"] * 4
    assert dataset.trial_ids.tolist() == dataset.window_ids.tolist()


def test_workload_adapter_rejects_noncausal_trial_ordinals(tmp_path: Path) -> None:
    path = _write_workload(tmp_path / "workload.h5")
    with h5py.File(path, "r+") as handle:
        handle["sessions"]["S1"]["trial_ordinals"][:] = [1, 3, 2, 4]

    with pytest.raises(ValueError, match="trial_ordinals must preserve"):
        load_sequential_dataset(path, session="S1")


def test_seed_adapter_preserves_trials_metadata_and_chronological_order(
    tmp_path: Path,
) -> None:
    path = _write_seed(tmp_path / "seed.h5")
    dataset = load_sequential_dataset(path, session="session_1")

    assert dataset.metadata.dataset_name == "seed"
    assert dataset.metadata.sample_rate == pytest.approx(2.0)
    assert dataset.metadata.channel_names == ("FP1", "FP2")
    assert dataset.metadata.class_names == ("negative", "neutral", "positive")
    assert dataset.metadata.unit == "uV"
    assert dataset.subject_ids.tolist() == ["SEED-01"] * 3
    assert dataset.session_ids.tolist() == ["session_1"] * 3
    assert dataset.trial_ids.tolist() == ["trial-30", "trial-10", "trial-20"]
    assert dataset.labels.tolist() == [2, 0, 1]
    assert dataset.trial_ordinals.tolist() == [1, 2, 3]
    assert dataset.window_ids.tolist() == [
        "SEED-01:session_1:trial-30",
        "SEED-01:session_1:trial-10",
        "SEED-01:session_1:trial-20",
    ]
    assert dataset.data.shape == (3, 2, 4)
    with h5py.File(path, "r") as handle:
        np.testing.assert_array_equal(dataset.data, handle["sessions"]["session_1"]["data"][:])


def test_dataset_package_window_mismatch_is_fail_closed(tmp_path: Path) -> None:
    dataset = load_sequential_dataset(_write_workload(tmp_path / "workload.h5"), session="S1")

    with pytest.raises(ValueError, match="window contracts differ"):
        validate_package_window_contract(dataset, package_window_sec=4.0)


class _TwoClassRuntime:
    def __init__(self) -> None:
        self.raw_windows = []

    def prepare(self, raw_window):
        self.raw_windows.append(raw_window)
        return raw_window

    def predict_prepared(self, prepared, *, return_features: bool = False) -> ModelOutput:
        del return_features
        predicted = (len(self.raw_windows) - 1) % 2
        probabilities = torch.tensor(
            [[0.8, 0.2] if predicted == 0 else [0.2, 0.8]],
            dtype=torch.float32,
        )
        return ModelOutput(
            logits=torch.log(probabilities),
            probabilities=probabilities,
            predicted_class=predicted,
            confidence=float(probabilities[0, predicted]),
            diagnostics={},
        )


def _workload_settings(path: Path, tmp_path: Path):
    args = sequential_evaluator.build_parser().parse_args(
        [
            "--data", str(path),
            "--session", "S1",
            "--model-package", str(tmp_path / "package"),
            "--online-strategy", "none",
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    return sequential_evaluator.resolve_settings(
        args,
        {"project": {"run_dir": str(tmp_path / "runs")}, "online": {"strategy": "none"}},
    )


def _loaded_workload_package(runtime: _TwoClassRuntime, *, window_sec: float):
    return SimpleNamespace(
        runtime_model=runtime,
        class_names=WORKLOAD_CLASS_NAMES,
        window_sec=window_sec,
        package_path=Path("package"),
        model_type="fake",
        model_name="fake-workload",
        step_sec=0.5,
        is_test_head=False,
        warning_message=None,
    )


def test_workload_dataset_enters_the_existing_evaluator_without_reordering(
    tmp_path: Path,
) -> None:
    path = _write_workload(tmp_path / "workload.h5")
    settings = _workload_settings(path, tmp_path)
    dataset = sequential_evaluator.load_sequential_dataset(settings)
    runtime = _TwoClassRuntime()
    loaded = SimpleNamespace(
        runtime_model=runtime,
        class_names=WORKLOAD_CLASS_NAMES,
    )

    records, summary = sequential_evaluator.evaluate_mode(
        mode="none",
        loaded=loaded,
        dataset=dataset,
        settings=settings,
    )

    assert summary["num_trials"] == 4
    assert [item["trial_ordinal"] for item in records] == [1, 2, 3, 4]
    assert [item["source_trial_id"] for item in records] == dataset.window_ids.tolist()
    assert [window.window_id for window in runtime.raw_windows] == dataset.window_ids.tolist()
    for index, window in enumerate(runtime.raw_windows):
        np.testing.assert_array_equal(window.data, dataset.data[index])


def test_workload_window_mismatch_fails_before_prepare(tmp_path: Path) -> None:
    path = _write_workload(tmp_path / "workload.h5")
    settings = _workload_settings(path, tmp_path)
    runtime = _TwoClassRuntime()
    loaded = _loaded_workload_package(runtime, window_sec=4.0)

    with pytest.raises(ValueError, match="window contracts differ"):
        sequential_evaluator.run_evaluation(
            settings,
            package_loader=lambda *_args, **_kwargs: loaded,
        )

    assert runtime.raw_windows == []
