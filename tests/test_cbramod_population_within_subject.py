from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from bci_dayloop.data.sequential_dataset import SequentialDatasetMetadata
from bci_dayloop.data.workload import (
    DIFF_CONDITION,
    EASY_CONDITION,
    WorkloadCondition,
    build_workload_session,
    write_workload_hdf5,
)
from bci_dayloop.models.cbramod.config import (
    CBRAMOD_NEURACLE_LIVE19_SPLINE22_PROFILE,
    CBRAMOD_STRICT22_PROFILE,
    resolve_cbramod_deployment_profile,
)


ROOT = Path(__file__).resolve().parents[1]


def load_trainer_module():
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "cbramod_population_head_test",
        ROOT / "scripts" / "train_cbramod_population_head.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cbramod_population_cli_defaults_to_loso_and_resolves_semantics() -> None:
    module = load_trainer_module()
    parser = module.build_argument_parser()
    assert parser.parse_args(["--target-subject", "1"]).split_mode == "loso"

    args = parser.parse_args(
        [
            "--target-subject",
            "1",
            "--split-mode",
            "within-subject",
            "--train-session",
            "source",
            "--test-session",
            "held_out",
            "--validation-ratio",
            "0.2",
            "--class-names",
            "left_hand",
            "right_hand",
            "both_hand",
            "rest",
        ]
    )
    assert args.split_mode == "within-subject"
    assert args.class_names == [
        "left_hand",
        "right_hand",
        "both_hand",
        "rest",
    ]

    metadata = SequentialDatasetMetadata(
        sample_rate=250.0,
        channel_names=("C3",),
        class_names=("metadata_0", "metadata_1", "metadata_2", "metadata_3"),
        unit="uV",
        dataset_name="unit-test",
        window_sec=4.0,
    )
    assert module.resolve_class_names(
        metadata=metadata,
        explicit_class_names=args.class_names,
    ) == ("left_hand", "right_hand", "both_hand", "rest")


def test_cbramod_deployment_profile_is_the_channel_adaptation_authority() -> None:
    module = load_trainer_module()
    parser = module.build_argument_parser()
    default_args = parser.parse_args(["--target-subject", "1"])

    assert default_args.deployment_profile == CBRAMOD_STRICT22_PROFILE
    assert default_args.missing_channel_policy is None
    assert default_args.min_observed_channels is None
    assert default_args.spline_alpha is None

    args = parser.parse_args(
        [
            "--target-subject",
            "1",
            "--deployment-profile",
            CBRAMOD_NEURACLE_LIVE19_SPLINE22_PROFILE,
        ]
    )
    profile = resolve_cbramod_deployment_profile(args.deployment_profile)
    assert profile.missing_channel_policy == "spherical_spline"
    assert profile.min_observed_channels == 19
    assert profile.spline_alpha == 1e-5


def test_workload_binary_sequential_adapter_supports_cbramod_within_subject(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A binary Workload stream must not inherit a four-class MI restriction."""
    module = load_trainer_module()
    channels = tuple(f"C{index}" for index in range(22))

    def session(name: str):
        easy = WorkloadCondition(
            condition=EASY_CONDITION,
            data=np.zeros((4, 22, 400), dtype=np.float32),
            channel_names=channels,
            sample_rate=200.0,
            unit="uV",
            source_set=tmp_path / f"{name}_easy.set",
        )
        difficult = WorkloadCondition(
            condition=DIFF_CONDITION,
            data=np.ones((4, 22, 400), dtype=np.float32),
            channel_names=channels,
            sample_rate=200.0,
            unit="uV",
            source_set=tmp_path / f"{name}_diff.set",
        )
        return build_workload_session(easy, difficult, subject_id="P01", session_id=name)

    path = write_workload_hdf5(
        tmp_path / "workload.h5", [session("S1"), session("S2")],
        subject_id="P01", data_root=tmp_path,
    )

    def fake_encode(*, loaded, metadata, subject_id, session_name, **_kwargs):
        labels = np.asarray(loaded["labels"], dtype=np.int64)
        trial_ids = np.asarray(loaded["trial_ids"], dtype=str)
        return (
            module.FeatureSplit(
                features=torch.zeros((len(labels), 1, 1, 1)),
                labels=torch.from_numpy(labels),
                source_trial_ids=np.asarray(
                    [f"{subject_id}:{session_name}:{trial_id}" for trial_id in trial_ids],
                    dtype=str,
                ),
                subject_ids=np.asarray(loaded["subject_ids"], dtype=np.int64),
                session_name=session_name,
            ),
            metadata,
        )

    monkeypatch.setattr(module, "build_feature_split_from_session_data", fake_encode)
    train, validation, metadata, split, all_trials = (
        module.build_within_subject_train_validation_splits(
            subject_id=1,
            subject_path=path,
            train_session="S1",
            test_session="S2",
            validation_ratio=0.25,
            seed=7,
            class_names=("low_workload", "high_workload"),
            canonicalizer=None,
            preprocessor=None,
            backbone=None,
            feature_batch_size=1,
            direct_trial_anchor="end",
        )
    )
    assert metadata.class_names == ("low_workload", "high_workload")
    assert set(train.labels.tolist()) == {0, 1}
    assert set(validation.labels.tolist()) == {0, 1}
    assert len(split.test_indices) == 8
    assert all_trials["subject_ids"].dtype == np.dtype(np.int64)
    assert not set(train.source_trial_ids).intersection(validation.source_trial_ids)
