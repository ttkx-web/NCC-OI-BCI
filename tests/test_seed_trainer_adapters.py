from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from bci_dayloop.data.preprocessing import PreprocessingConfig
from bci_dayloop.data.sequential_dataset import load_sequential_dataset
from bci_dayloop.models.cbramod.config import (
    BCICIV2A_22_CHANNELS,
    CBraModConfig,
)
from bci_dayloop.models.labram_linear import LaBraMLinearAdapter
from bci_dayloop.packages.exporter import export_labram_runtime_package
from bci_dayloop.utils.config import load_yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SEED_CLASSES = ("negative", "neutral", "positive")


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


def _write_seed(path: Path) -> Path:
    base = np.arange(3 * 22 * 400, dtype=np.float32).reshape(3, 22, 400)
    with h5py.File(path, "w") as handle:
        handle.attrs["dataset_name"] = "seed"
        handle.attrs["subject_id"] = "SEED-01"
        handle.attrs["class_names"] = json.dumps(list(SEED_CLASSES))
        handle.attrs["unit"] = "uV"
        handle.attrs["window_sec"] = 2.0
        sessions = handle.create_group("sessions")
        for session_name, offset in (("S1", 0.0), ("S2", 1_000_000.0)):
            group = sessions.create_group(session_name)
            group.attrs["sample_rate"] = 200.0
            group.attrs["channel_names"] = json.dumps(
                list(BCICIV2A_22_CHANNELS)
            )
            group.create_dataset("data", data=base + offset)
            group.create_dataset(
                "labels", data=np.asarray([2, 0, 1], dtype=np.int64)
            )
            group.create_dataset(
                "trial_ids",
                data=np.asarray(
                    [
                        f"{session_name}-trial-30".encode(),
                        f"{session_name}-trial-10".encode(),
                        f"{session_name}-trial-20".encode(),
                    ]
                ),
            )
            group.create_dataset(
                "trial_ordinals", data=np.asarray([1, 2, 3], dtype=np.int64)
            )
    return path


class _SeedLaBraMPreprocessor:
    class config:
        target_sample_rate = 200.0
        patch_samples = 200

    def transform(
        self,
        values: np.ndarray,
        sample_rate: float,
        unit: str,
        *,
        reshape: bool,
    ) -> np.ndarray:
        assert sample_rate == 200.0
        assert unit == "uV"
        assert reshape is True
        return np.asarray(values, dtype=np.float32).reshape(
            values.shape[0], values.shape[1], 2, 200
        )


class _TinyLaBraMEncoder(nn.Module):
    embed_dim = 8


def test_seed_labram_trainer_adapter_preserves_contract_and_infers_classes(
    tmp_path: Path,
) -> None:
    path = _write_seed(tmp_path / "seed.h5")
    trainer = _script_module(
        "train_labram_population_head.py", "test_seed_labram_trainer"
    )
    values, labels, metadata, summary = trainer.load_preprocessed_subject_session(
        subject_id=1,
        path=path,
        session_name="S2",
        preprocessor=_SeedLaBraMPreprocessor(),
        reference_metadata=None,
        expected_window_sec=2.0,
        trial_window_anchor="end",
        maximum_per_class=None,
        seed=1,
    )

    assert metadata.dataset_name == "seed"
    assert metadata.class_names == SEED_CLASSES
    assert labels.tolist() == [2, 0, 1]
    assert values.shape == (3, 22, 2, 200)
    assert summary["session"] == "S2"
    assert summary["num_trials"] == 3

    adapter = LaBraMLinearAdapter(
        channel_names=list(metadata.channel_names),
        n_classes=len(metadata.class_names),
        device="cpu",
        amp=False,
        freeze_encoder=True,
        n_patches=2,
        encoder=_TinyLaBraMEncoder(),
    )
    assert adapter.n_classes == 3
    assert adapter.head(torch.ones((2, adapter.embedding_dim))).shape == (2, 3)


def test_seed_cbramod_trainer_and_package_contract_are_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_seed(tmp_path / "seed.h5")
    trainer = _script_module(
        "train_cbramod_population_head.py", "test_seed_cbramod_trainer"
    )
    captured: dict[str, object] = {}

    def fake_prepare(**kwargs: object) -> np.ndarray:
        captured.update(kwargs)
        return np.zeros((3, 22, 2, 200), dtype=np.float32)

    monkeypatch.setattr(trainer, "prepare_cbramod_trials", fake_prepare)
    monkeypatch.setattr(
        trainer,
        "extract_frozen_features",
        lambda **_kwargs: torch.zeros((3, 22, 2, 200)),
    )

    config = CBraModConfig(
        checkpoint_path="unused.pt",
        target_sample_rate=200.0,
        window_seconds=2.0,
        time_segments=2,
        points_per_patch=200,
        input_unit="uV",
        normalization="none",
        missing_channel_policy="error",
        min_observed_channels=None,
        num_classes=3,
    )

    class _Preprocessor:
        def __init__(self, config: CBraModConfig) -> None:
            self.config = config

    class _Backbone:
        pass

    backbone = _Backbone()
    backbone.config = config

    split, metadata = trainer.build_subject_feature_split(
        subject_id=1,
        session_name="S1",
        subject_path=path,
        reference_metadata=None,
        canonicalizer=object(),
        preprocessor=_Preprocessor(config),
        backbone=backbone,
        feature_batch_size=1,
        direct_trial_anchor="end",
    )

    assert metadata.class_names == SEED_CLASSES
    assert split.labels.tolist() == [2, 0, 1]
    assert tuple(split.features.shape) == (3, 22, 2, 200)
    assert split.source_trial_ids.tolist() == [
        "1:S1:S1-trial-30",
        "1:S1:S1-trial-10",
        "1:S1:S1-trial-20",
    ]
    assert np.asarray(captured["trial_ids"]).tolist() == [
        "S1-trial-30",
        "S1-trial-10",
        "S1-trial-20",
    ]

    preprocessing = trainer.build_preprocessing_manifest(
        config=config,
        source_unit=metadata.unit,
        direct_trial_anchor="end",
    )
    exporter = _script_module(
        "export_cbramod_model_package.py", "test_seed_cbramod_exporter"
    )
    runtime_kwargs = exporter.runtime_kwargs_from_training_report(
        {"head_type": "linear", "preprocessing": preprocessing}
    )
    backbone = tmp_path / "backbone.pt"
    classifier = tmp_path / "classifier.pt"
    report = tmp_path / "training_report.json"
    backbone.write_bytes(b"backbone")
    classifier.write_bytes(b"classifier")
    report.write_text("{}", encoding="utf-8")
    package_payload, preprocessing_payload, _ = exporter.build_package_payload(
        class_names=metadata.class_names,
        command_map={},
        runtime_kwargs=runtime_kwargs,
        package_id="cbramod-seed-subject-01-2s",
        package_version="v1",
        dataset_name=metadata.dataset_name,
        step_sec=0.5,
        confidence_threshold=0.0,
        backbone_path=backbone,
        classifier_path=classifier,
        metrics={},
        training_report_path=report,
    )

    assert preprocessing["source_unit"] == "uV"
    assert preprocessing["window_seconds"] == 2.0
    assert preprocessing["time_segments"] == 2
    assert tuple(preprocessing["standard_channels"]) == BCICIV2A_22_CHANNELS
    assert package_payload["model"]["dataset"] == "seed"
    assert package_payload["model"]["num_classes"] == 3
    assert tuple(package_payload["model"]["class_names"]) == SEED_CLASSES
    assert package_payload["input_contract"]["window_sec"] == 2.0
    assert package_payload["input_contract"]["sample_rate"] == 200.0
    assert package_payload["input_contract"]["num_samples"] == 400
    assert package_payload["input_contract"]["tensor_layout"] == "BCTP"
    assert tuple(package_payload["input_contract"]["channel_names"]) == (
        BCICIV2A_22_CHANNELS
    )
    assert preprocessing_payload["transform"]["time_segments"] == 2
    assert preprocessing_payload["transform"]["points_per_patch"] == 200
    assert preprocessing_payload["transform"]["normalization"] == "none"


def test_seed_subject_session_labels_and_source_order_are_preserved(
    tmp_path: Path,
) -> None:
    path = _write_seed(tmp_path / "seed.h5")
    s1 = load_sequential_dataset(path, session="S1")
    s2 = load_sequential_dataset(path, session="S2")

    assert s1.subject_ids.tolist() == ["SEED-01"] * 3
    assert s1.session_ids.tolist() == ["S1"] * 3
    assert s2.session_ids.tolist() == ["S2"] * 3
    assert s1.labels.tolist() == [2, 0, 1]
    assert s1.trial_ordinals.tolist() == [1, 2, 3]
    assert s1.trial_ids.tolist() == ["S1-trial-30", "S1-trial-10", "S1-trial-20"]
    assert s1.window_ids.tolist() == [
        "SEED-01:S1:S1-trial-30",
        "SEED-01:S1:S1-trial-10",
        "SEED-01:S1:S1-trial-20",
    ]
    assert not np.array_equal(s1.data, s2.data)
    with pytest.raises(ValueError, match="session is missing"):
        load_sequential_dataset(path, session="S3")


def test_seed_trainer_clis_accept_shared_dataset_and_split_contract() -> None:
    common = [
        "--data-root",
        "data/processed/seed",
        "--data-pattern",
        "subject_{subject:02d}.h5",
        "--subjects",
        "1",
        "2",
        "3",
        "--target-subject",
        "1",
        "--train-session",
        "S1",
        "--validation-session",
        "S2",
        "--final-test-session",
        "S3",
    ]
    labram = _script_module(
        "train_labram_population_head.py", "test_seed_labram_cli"
    )
    labram_args = labram.build_parser().parse_args(
        [*common, "--dataset-name", "seed", "--window-sec", "2"]
    )
    assert labram_args.dataset_name == "seed"
    assert labram_args.subjects == [1, 2, 3]
    assert (
        labram_args.train_session,
        labram_args.validation_session,
        labram_args.final_test_session,
    ) == ("S1", "S2", "S3")
    assert labram_args.window_sec == 2.0

    cbramod = _script_module(
        "train_cbramod_population_head.py", "test_seed_cbramod_cli"
    )
    cbramod_args = cbramod.build_argument_parser().parse_args(
        [*common, "--window-sec", "2"]
    )
    assert cbramod_args.subjects == [1, 2, 3]
    assert (
        cbramod_args.train_session,
        cbramod_args.validation_session,
        cbramod_args.final_test_session,
    ) == ("S1", "S2", "S3")
    assert cbramod_args.window_seconds == 2.0


def test_seed_labram_runtime_package_export_contract(
    tmp_path: Path,
) -> None:
    adapter = LaBraMLinearAdapter(
        channel_names=list(BCICIV2A_22_CHANNELS),
        n_classes=len(SEED_CLASSES),
        device="cpu",
        amp=False,
        freeze_encoder=True,
        n_patches=2,
        encoder=_TinyLaBraMEncoder(),
    )
    source_backbone = tmp_path / "seed_labram_backbone.pt"
    source_classifier = tmp_path / "seed_labram_head.pt"
    source_backbone.write_bytes(b"synthetic-backbone-provenance")
    source_classifier.write_bytes(b"synthetic-head-provenance")

    package_dir = export_labram_runtime_package(
        output_dir=tmp_path / "seed_labram_package",
        adapter=adapter,
        backbone_checkpoint=source_backbone,
        classifier_checkpoint=source_classifier,
        preprocessing_config=PreprocessingConfig(
            target_sample_rate=200.0,
            patch_samples=200,
        ),
        class_names=SEED_CLASSES,
        command_map={},
        dataset_name="seed",
        package_id="labram_seed_subject_01_population_2s",
        package_version="v1",
    )

    payload = load_yaml(package_dir / "package.yaml")
    contract = payload["input_contract"]
    assert payload["schema_version"] == 2
    assert payload["model"]["type"] == "labram"
    assert payload["model"]["dataset"] == "seed"
    assert payload["model"]["class_names"] == list(SEED_CLASSES)
    assert payload["model"]["num_classes"] == 3
    assert contract["channel_names"] == list(BCICIV2A_22_CHANNELS)
    assert contract["sample_rate"] == 200.0
    assert contract["window_sec"] == 2.0
    assert contract["num_samples"] == 400
    assert contract["tensor_layout"] == "BCTP"
