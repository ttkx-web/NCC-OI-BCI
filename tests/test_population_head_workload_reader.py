from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_workload_reader_uses_existing_loso_population_split() -> None:
    """Exercise the trainer's reader → LOSO windows boundary on real H5 data."""
    code = r'''
import importlib.util
import sys
import types
from pathlib import Path

import torch

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
    "population_head_workload_test",
    root / "scripts" / "train_50m_population_head.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)

parser = module.build_argument_parser()
assert parser.parse_args([]).data_reader == "eeg"

eeg_root = root / "data/processed/bnci2014_001"
eeg_common = dict(
    data_root=eeg_root,
    data_pattern="subject_{subject:02d}.h5",
    data_reader="eeg",
    window_seconds=4.0,
    stride_seconds=4.0,
    max_windows_per_class_per_subject=None,
    window_construction="direct_trial",
    direct_trial_anchor="end",
)
eeg_train = module.build_population_split(
    subjects=[2, 3], session_name="0train", base_seed=10,
    shuffle_trials_within_class=True, **eeg_common,
)
eeg_validation = module.build_population_split(
    subjects=[2, 3], session_name="1test", base_seed=20,
    shuffle_trials_within_class=False, reference_metadata=eeg_train.metadata,
    **eeg_common,
)
module.validate_no_source_leakage(
    eeg_train.bundle.window_set, eeg_validation.bundle.window_set,
    left_name="eeg train", right_name="eeg validation",
)
assert set(eeg_train.bundle.window_subject_ids.tolist()) == {2, 3}
assert set(eeg_validation.bundle.window_subject_ids.tolist()) == {2, 3}

root_data = root / "data/processed/workload"
for subject in (1, 2, 3):
    assert (root_data / f"subject_{subject:02d}.h5").is_file()

common = dict(
    data_root=root_data,
    data_pattern="subject_{subject:02d}.h5",
    data_reader="workload",
    window_seconds=2.0,
    stride_seconds=2.0,
    max_windows_per_class_per_subject=None,
    window_construction="direct_trial",
    direct_trial_anchor="end",
)
train = module.build_population_split(
    subjects=[2, 3], session_name="S1", base_seed=1_000,
    shuffle_trials_within_class=True, **common,
)
validation = module.build_population_split(
    subjects=[2, 3], session_name="S2", base_seed=2_000,
    shuffle_trials_within_class=False, reference_metadata=train.metadata,
    **common,
)
target = module.build_population_split(
    subjects=[1], session_name="S2", base_seed=3_000,
    shuffle_trials_within_class=False, reference_metadata=train.metadata,
    **common,
)

assert train.bundle.window_set.windows.shape == (596, 61, 500)
assert validation.bundle.window_set.windows.shape == (596, 61, 500)
assert target.bundle.window_set.windows.shape == (298, 61, 500)
for bundle, subjects in ((train.bundle, {2, 3}), (validation.bundle, {2, 3}), (target.bundle, {1})):
    assert set(bundle.window_subject_ids.tolist()) == subjects
    assert module.class_counts(bundle.window_set.labels, 2) == {
        0: len(bundle.window_set.labels) // 2,
        1: len(bundle.window_set.labels) // 2,
    }
module.validate_no_source_leakage(
    train.bundle.window_set, validation.bundle.window_set,
    left_name="train", right_name="validation",
)
module.validate_no_source_leakage(
    train.bundle.window_set, target.bundle.window_set,
    left_name="train", right_name="target",
)
module.validate_no_source_leakage(
    validation.bundle.window_set, target.bundle.window_set,
    left_name="validation", right_name="target",
)

config = module.Model50MConfig(
    checkpoint_path=root / "checkpoints/backbones/50m/model_deploy.pt",
    window_seconds=2.0,
    target_sample_rate=100.0,
    output_layer_idx=8,
    aggregation="flatten",
    num_classes=2,
)
assert config.num_tokens == 128
assert config.classifier_input_dim == 65_536
assert tuple(torch.nn.Linear(config.classifier_input_dim, 2)(torch.zeros(1, 65_536)).shape) == (1, 2)
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
