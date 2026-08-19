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
assert parser.parse_args([]).split_mode == "loso"
args = parser.parse_args([
    "--split-mode", "within-subject",
    "--target-subject", "1",
    "--train-session", "session_a",
    "--test-session", "session_b",
    "--validation-ratio", "0.2",
    "--class-names", "left_hand", "right_hand", "both_hand", "rest",
])
assert args.split_mode == "within-subject"
assert args.class_names == ["left_hand", "right_hand", "both_hand", "rest"]
assert args.validation_ratio == 0.2

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
            train_session="source",
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
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
