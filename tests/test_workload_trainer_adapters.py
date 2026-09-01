from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from bci_dayloop.data.sequential_dataset import (
    load_sequential_dataset,
    resolve_population_split_plan,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _script_module(filename: str, name: str) -> object:
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


def _workload_h5(path: Path, *, window_sec: float = 2.0) -> Path:
    data = np.arange(4 * 2 * 500, dtype=np.float32).reshape(4, 2, 500)
    with h5py.File(path, "w") as handle:
        handle.attrs["dataset_name"] = "workload_pbci_hackathon"
        handle.attrs["subject_id"] = "P01"
        handle.attrs["window_sec"] = window_sec
        handle.attrs["unit"] = "V"
        handle.attrs["class_names"] = json.dumps(["low_workload", "high_workload"])
        group = handle.create_group("sessions").create_group("S1")
        group.attrs["sample_rate"] = 250.0
        group.attrs["channel_names"] = json.dumps(["C3", "C4"])
        group.create_dataset("data", data=data)
        group.create_dataset("labels", data=np.asarray([0, 1, 0, 1], dtype=np.int64))
        group.create_dataset("condition_ids", data=np.asarray([0, 1, 0, 1], dtype=np.int8))
        group.create_dataset("source_epoch_indices", data=np.asarray([0, 0, 1, 1], dtype=np.int64))
        group.create_dataset("trial_ordinals", data=np.arange(1, 5, dtype=np.int64))
        group.create_dataset("window_ids", data=np.asarray([b"P01:S1:A:0", b"P01:S1:B:0", b"P01:S1:A:1", b"P01:S1:B:1"]))
        second = handle["sessions"].create_group("S2")
        second.attrs["sample_rate"] = 250.0
        second.attrs["channel_names"] = json.dumps(["C3", "C4"])
        second.create_dataset("data", data=data + 10_000)
        second.create_dataset("labels", data=np.asarray([0, 1, 0, 1], dtype=np.int64))
        second.create_dataset("condition_ids", data=np.asarray([0, 1, 0, 1], dtype=np.int8))
        second.create_dataset("source_epoch_indices", data=np.asarray([0, 0, 1, 1], dtype=np.int64))
        second.create_dataset("trial_ordinals", data=np.arange(1, 5, dtype=np.int64))
        second.create_dataset("window_ids", data=np.asarray([b"P01:S2:A:0", b"P01:S2:B:0", b"P01:S2:A:1", b"P01:S2:B:1"]))
    return path


class _LaBraMPreprocessor:
    class config:
        target_sample_rate = 200.0
        patch_samples = 200

    def transform(self, values: np.ndarray, *_: object, **__: object) -> np.ndarray:
        return np.zeros((len(values), values.shape[1], 2, 200), dtype=np.float32)


def test_labram_workload_loader_uses_persisted_two_second_trials(tmp_path: Path) -> None:
    module = _script_module("train_labram_population_head.py", "test_labram_trainer")
    loaded = module.load_preprocessed_subject_session(
        subject_id=1, path=_workload_h5(tmp_path / "subject_01.h5"), session_name="S1",
        preprocessor=_LaBraMPreprocessor(), reference_metadata=None, expected_window_sec=2.0,
        trial_window_anchor="end", maximum_per_class=None, seed=1,
    )
    values, labels, metadata, summary = loaded
    assert values.shape == (4, 2, 2, 200)
    assert labels.tolist() == [0, 1, 0, 1]
    assert metadata.class_names == ("low_workload", "high_workload")
    assert summary["selected_raw_shape"] == [4, 2, 500]


def test_labram_workload_window_mismatch_fails_closed(tmp_path: Path) -> None:
    module = _script_module("train_labram_population_head.py", "test_labram_trainer_mismatch")
    with pytest.raises(ValueError, match="persisted window duration"):
        module.load_preprocessed_subject_session(
            subject_id=1, path=_workload_h5(tmp_path / "subject_01.h5"), session_name="S1",
            preprocessor=_LaBraMPreprocessor(), reference_metadata=None, expected_window_sec=4.0,
            trial_window_anchor="end", maximum_per_class=None, seed=1,
        )


def test_cbramod_workload_loader_uses_adapter_and_two_segments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _script_module("train_cbramod_population_head.py", "test_cbramod_trainer")
    monkeypatch.setattr(module, "prepare_cbramod_trials", lambda **kwargs: np.zeros((4, 22, 2, 200), dtype=np.float32))
    monkeypatch.setattr(module, "extract_frozen_features", lambda **kwargs: torch.zeros((4, 22, 2, 200)))

    class _Config:
        window_seconds = 2.0
    class _Preprocessor:
        config = _Config()
    class _Backbone:
        config = object()

    split, metadata = module.build_subject_feature_split(
        subject_id=1, session_name="S1", subject_path=_workload_h5(tmp_path / "subject_01.h5"),
        reference_metadata=None, canonicalizer=object(), preprocessor=_Preprocessor(),
        backbone=_Backbone(), feature_batch_size=1, direct_trial_anchor="end",
    )
    assert tuple(split.features.shape) == (4, 22, 2, 200)
    assert split.labels.tolist() == [0, 1, 0, 1]
    assert metadata.class_names == ("low_workload", "high_workload")


def test_cbramod_parser_accepts_window_sec_alias() -> None:
    module = _script_module("train_cbramod_population_head.py", "test_cbramod_parser")
    args = module.build_argument_parser().parse_args(["--target-subject", "1", "--window-sec", "2"])
    assert args.window_seconds == 2.0


def test_cbramod_spline_cli_contract() -> None:
    module = _script_module("train_cbramod_population_head.py", "test_cbramod_spline_cli")
    args = module.build_argument_parser().parse_args([
        "--target-subject", "1", "--normalization", "fixed_100uv",
        "--missing-channel-policy", "spherical_spline",
        "--min-observed-channels", "21", "--spline-alpha", "1e-5",
    ])
    assert args.normalization == "fixed_100uv"
    assert args.missing_channel_policy == "spherical_spline"
    assert args.min_observed_channels == 21 and args.spline_alpha == 1e-5


def test_cbramod_workload_preprocessing_manifest_provenance() -> None:
    module = _script_module("train_cbramod_population_head.py", "test_cbramod_manifest")
    config = module.CBraModConfig(
        checkpoint_path="unused.pt", target_sample_rate=200.0,
        window_seconds=2.0, time_segments=2, points_per_patch=200,
        input_unit="uV", normalization="fixed_100uv",
        missing_channel_policy="spherical_spline", min_observed_channels=21,
        spline_alpha=1e-5, filter_enabled=False, reference_mode="none",
    )
    manifest = module.build_preprocessing_manifest(
        config=config, source_unit="V", direct_trial_anchor="end"
    )
    assert manifest["source_unit"] == "V"
    assert manifest["input_unit"] == "uV"
    assert manifest["normalization"] == "fixed_100uv"
    assert manifest["window_seconds"] == 2.0
    assert manifest["time_segments"] == 2
    assert manifest["points_per_patch"] == 200
    assert manifest["missing_channel_policy"] == "spherical_spline"
    assert manifest["min_observed_channels"] == 21
    assert manifest["spline_alpha"] == 1e-5


def test_population_split_plan_excludes_target_and_fails_closed() -> None:
    plan = resolve_population_split_plan(range(1, 16), 1, "S1", "S2", "S2")
    assert plan.train_subjects == tuple(range(2, 16))
    assert plan.validation_subjects == tuple(range(2, 16))
    assert plan.final_test_subjects == (1,)
    assert 1 not in plan.train_subjects and 1 not in plan.validation_subjects
    with pytest.raises(ValueError):
        resolve_population_split_plan(range(2, 16), 1, "S1", "S2", "S2")


def test_workload_sessions_are_selected_strictly(tmp_path: Path) -> None:
    path = _workload_h5(tmp_path / "subject_01.h5")
    s1 = load_sequential_dataset(path, session="S1")
    s2 = load_sequential_dataset(path, session="S2")
    assert not np.array_equal(s1.data, s2.data)
    assert s1.window_ids[0] == "P01:S1:A:0"
    assert s2.window_ids[0] == "P01:S2:A:0"
    with pytest.raises(ValueError, match="Session"):
        load_sequential_dataset(path, session="missing")


def test_cbramod_feature_split_merge_preserves_string_source_ids() -> None:
    module = _script_module("train_cbramod_population_head.py", "test_cbramod_merge")
    left = module.FeatureSplit(
        features=torch.zeros((1, 22, 2, 200)), labels=torch.tensor([0]),
        source_trial_ids=np.asarray(["P02:S1:trial-1"]),
        subject_ids=np.asarray([2]), session_name="S1",
    )
    right = module.FeatureSplit(
        features=torch.zeros((1, 22, 2, 200)), labels=torch.tensor([1]),
        source_trial_ids=np.asarray(["P02:S2:trial-1"]),
        subject_ids=np.asarray([2]), session_name="S2",
    )
    merged = module.combine_feature_splits([left, right], session_name="population")
    assert merged.source_trial_ids.tolist() == ["P02:S1:trial-1", "P02:S2:trial-1"]
    assert len(set(merged.source_trial_ids.tolist())) == 2
