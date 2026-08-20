from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_population_trainer_defaults_to_loso_and_parses_within_subject_cli() -> None:
    """Exercise the real parser without relying on this environment's SciPy."""
    code = r'''
import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
from bci_dayloop.data.hdf5_dataset import HDF5Metadata, write_hdf5

scipy = types.ModuleType("scipy")
signal = types.ModuleType("scipy.signal")
for name in ("butter", "resample_poly", "sosfiltfilt"):
    setattr(signal, name, lambda *args, **kwargs: None)
scipy.signal = signal
sys.modules["scipy"] = scipy
sys.modules["scipy.signal"] = signal

root = Path.cwd()
sys.path.insert(0, str(root / "scripts"))
spec = importlib.util.spec_from_file_location(
    "population_head_cli_test",
    root / "scripts" / "train_50m_population_head.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)

parser = module.build_argument_parser()
default_args = parser.parse_args([])
assert default_args.split_mode == "loso"
assert default_args.overwrite is False
args = parser.parse_args([
    "--split-mode", "within-subject",
    "--target-subject", "1",
    "--train-session", "session_a",
    "--test-session", "session_b",
    "--validation-ratio", "0.2",
    "--class-names", "left_hand", "right_hand", "both_hand", "rest",
    "--overwrite",
])
assert args.split_mode == "within-subject"
assert args.train_session == ["session_a"]
assert args.class_names == ["left_hand", "right_hand", "both_hand", "rest"]
assert args.validation_ratio == 0.2
assert args.overwrite is True

with tempfile.TemporaryDirectory() as directory:
    run_dir = Path(directory) / "existing_run"
    run_dir.mkdir()
    try:
        module.create_run_directory(run_dir, overwrite=False)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing run directory must be rejected by default.")
    module.create_run_directory(run_dir, overwrite=True)
    assert run_dir.is_dir()

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "single_subject.h5"
    source_labels = np.repeat(np.arange(4, dtype=np.int64), 20)
    test_labels = np.repeat(np.arange(4, dtype=np.int64), 2)
    labels = np.concatenate((source_labels, test_labels))
    sessions = ["source"] * len(source_labels) + ["test"] * len(test_labels)
    write_hdf5(
        path,
        np.zeros((len(labels), 2, 500), dtype=np.float32),
        labels,
        np.ones(len(labels), dtype=np.int64),
        sessions,
        np.arange(len(labels), dtype=np.int64),
        HDF5Metadata(
            250.0,
            ["C3", "C4"],
            ["left_hand", "right_hand", "both_hand", "rest"],
            "uV",
            "synthetic",
        ),
    )
    train, validation, metadata, class_names, split, all_trials = (
        module.build_within_subject_splits(
            subject_id=1,
            path=path,
            train_sessions=["source"],
            test_session="test",
            validation_ratio=0.2,
            seed=42,
            window_seconds=2.0,
            stride_seconds=2.0,
            max_windows_per_class=None,
            window_construction="direct_trial",
            direct_trial_anchor="end",
            explicit_class_names=None,
        )
    )
    test = module.build_within_subject_test_split(
        subject_id=1,
        path=path,
        metadata=metadata,
        class_names=class_names,
        split=split,
        all_trial_metadata=all_trials,
        window_seconds=2.0,
        stride_seconds=2.0,
        max_windows_per_class=None,
        window_construction="direct_trial",
        direct_trial_anchor="end",
    )
    assert train.bundle.window_set.windows.shape == (64, 2, 500)
    assert validation.bundle.window_set.windows.shape == (16, 2, 500)
    assert test.bundle.window_set.windows.shape == (8, 2, 500)

    yaxin_path = Path(directory) / "yaxin_combined.h5"
    session_sizes = {"S1": 10, "S2": 20, "S3": 10, "S4": 20, "S5": 10, "S6": 20}
    yaxin_labels = np.concatenate(
        [np.repeat(np.arange(4, dtype=np.int64), session_sizes[name]) for name in session_sizes]
    )
    yaxin_sessions = np.concatenate(
        [np.full(session_sizes[name] * 4, name, dtype=object) for name in session_sizes]
    )
    write_hdf5(
        yaxin_path,
        np.zeros((len(yaxin_labels), 2, 500), dtype=np.float32),
        yaxin_labels,
        np.ones(len(yaxin_labels), dtype=np.int64),
        yaxin_sessions,
        np.arange(len(yaxin_labels), dtype=np.int64),
        HDF5Metadata(
            250.0,
            ["C3", "C4"],
            ["left_hand", "right_hand", "both_hand", "rest"],
            "uV",
            "synthetic-yaxin",
        ),
    )
    yaxin_train, yaxin_validation, _, _, yaxin_split, yaxin_trials = (
        module.build_within_subject_splits(
            subject_id=1,
            path=yaxin_path,
            train_sessions=["S1", "S2", "S3", "S4", "S5"],
            test_session="S6",
            validation_ratio=0.2,
            seed=42,
            window_seconds=2.0,
            stride_seconds=2.0,
            max_windows_per_class=None,
            window_construction="direct_trial",
            direct_trial_anchor="end",
            explicit_class_names=None,
        )
    )
    assert len(yaxin_split.train_indices) == 224
    assert len(yaxin_split.validation_indices) == 56
    assert len(yaxin_split.test_indices) == 80
    np.testing.assert_array_equal(
        np.bincount(yaxin_trials["labels"][yaxin_split.train_indices], minlength=4),
        np.asarray([56, 56, 56, 56]),
    )
    np.testing.assert_array_equal(
        np.bincount(yaxin_trials["labels"][yaxin_split.validation_indices], minlength=4),
        np.asarray([14, 14, 14, 14]),
    )
    np.testing.assert_array_equal(
        np.bincount(yaxin_trials["labels"][yaxin_split.test_indices], minlength=4),
        np.asarray([20, 20, 20, 20]),
    )
    assert yaxin_train.bundle.window_set.windows.shape[0] == 224
    assert yaxin_validation.bundle.window_set.windows.shape[0] == 56
    yaxin_metadata = module.build_within_subject_split_metadata(
        subject_id=1,
        split=yaxin_split,
        all_trial_metadata=yaxin_trials,
        class_names=["left_hand", "right_hand", "both_hand", "rest"],
        validation_ratio=0.2,
        seed=42,
    )
    assert yaxin_metadata["train_session"] is None
    assert yaxin_metadata["train_sessions"] == ["S1", "S2", "S3", "S4", "S5"]
    assert yaxin_metadata["test_session"] == "S6"
    assert yaxin_metadata["train_class_counts"] == {
        "left_hand": 56,
        "right_hand": 56,
        "both_hand": 56,
        "rest": 56,
    }
    cache_identity = module.build_feature_cache_split_identity(
        split_mode="within-subject",
        train_sessions=["S1", "S2", "S3", "S4", "S5"],
        test_session="S6",
        validation_session=None,
        validation_ratio=0.2,
        split_seed=42,
    )
    cache_path = Path(directory) / "features.pt"
    module.save_population_feature_cache(
        dataset=module.TensorDataset(
            module.torch.zeros((224, 1)),
            module.torch.from_numpy(yaxin_train.bundle.window_set.labels.copy()),
        ),
        bundle=yaxin_train.bundle,
        path=cache_path,
        split_name="within_subject_train",
        class_names=["left_hand", "right_hand", "both_hand", "rest"],
        subject_ids=[1],
        data_reader="eeg",
        subject_identities={"1": {"canonical_subject_id": 1, "source_subject_id": "1"}},
        backbone_sha256="backbone",
        preprocessing_hash="preprocessing",
        split_identity=cache_identity,
    )
    saved_cache = module.torch.load(cache_path, weights_only=False)
    assert saved_cache["split_identity"] == cache_identity

multi_args = parser.parse_args([
    "--split-mode", "within-subject",
    "--target-subject", "1",
    "--train-session", "S1", "S2", "S3", "S4", "S5",
    "--test-session", "S6",
])
assert multi_args.train_session == ["S1", "S2", "S3", "S4", "S5"]
assert module.build_feature_cache_split_identity(
    split_mode="within-subject",
    train_sessions=["S1"],
    test_session="S6",
    validation_session=None,
    validation_ratio=0.2,
    split_seed=42,
) != module.build_feature_cache_split_identity(
    split_mode="within-subject",
    train_sessions=["S1", "S2", "S3"],
    test_session="S6",
    validation_session=None,
    validation_ratio=0.2,
    split_seed=42,
)
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
