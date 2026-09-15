from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from bci_dayloop.data.dataset_adapter_registry import (
    DEFAULT_DATASET_ADAPTER_REGISTRY,
    LegacyHDF5Adapter,
    inspect_hdf5_dataset,
)
from bci_dayloop.data.dataset_registry import (
    AWAKENING_2S,
    DATASET_REGISTRY,
    get_dataset_spec,
)
from bci_dayloop.models.model_50m.classifier import LinearClassificationHead
from bci_dayloop.models.model_50m.config import STANDARD_64_CHANNELS
from bci_dayloop.models.model_50m import preprocessing as preprocessing_module
from bci_dayloop.models.model_50m.preprocessing import Model50MPreprocessor
from bci_dayloop.models.model_50m.tokenization import Model50MTokenizer
from bci_dayloop.training.model_50m.awakening import (
    awakening_model_config,
    channel_profile_contract,
    validate_awakening_dataset,
)
from bci_dayloop.training.model_50m.data import build_population_split
from bci_dayloop.training.model_50m.evaluation import evaluate_binary_feature_dataset
from bci_dayloop.training.model_50m.features import (
    load_population_feature_cache,
    save_population_feature_cache,
)
from bci_dayloop.training.model_50m.linear_head import WindowSet
from bci_dayloop.training.model_50m.types import WindowBundle
from bci_dayloop.utils.config import load_yaml


def _write_manifest_files(root: Path) -> None:
    for name in ("subject_mapping.json", "split_manifest.json", "conversion_report.json"):
        (root / name).write_text(json.dumps({"name": name}), encoding="utf-8")


def _write_subject(
    root: Path,
    *,
    subject: int,
    labels: list[int],
    sessions: list[str],
    parents: list[int],
) -> Path:
    n = len(labels)
    path = root / f"subject_{subject:02d}.h5"
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=np.ones((n, 62, 400), dtype=np.float32))
        handle.create_dataset("labels", data=np.asarray(labels, dtype=np.int64))
        handle.create_dataset("subject_ids", data=np.full(n, subject, dtype=np.int64))
        handle.create_dataset("session_ids", data=np.asarray(sessions, dtype=string_dtype))
        handle.create_dataset("trial_ids", data=np.arange(n, dtype=np.int64))
        handle.create_dataset("source_lance_row_ids", data=np.asarray(parents, dtype=np.int64))
        handle.create_dataset(
            "source_subwindow_indices", data=np.arange(n, dtype=np.int64) % 5
        )
        handle.create_dataset(
            "source_sample_ids",
            data=np.asarray([f"sample-{subject}-{i}" for i in range(n)], dtype=string_dtype),
        )
        handle.attrs["sample_rate"] = 200.0
        handle.attrs["window_seconds"] = 2.0
        handle.attrs["samples_per_window"] = 400
        handle.attrs["dataset_name"] = "awakening"
        handle.attrs["unit"] = "uV"
        handle.attrs["no_temporal_padding"] = True
        handle.attrs["session_semantics"] = AWAKENING_2S.session_semantics
        handle.attrs["channel_names"] = json.dumps(AWAKENING_2S.source_channels)
        handle.attrs["class_names"] = json.dumps(AWAKENING_2S.class_names)
        handle.attrs["label_mapping"] = json.dumps(
            {"0": "non_awakening", "1": "awakening"}
        )
    return path


def test_awakening_is_registered_with_flat_legacy_contract() -> None:
    assert DATASET_REGISTRY["awakening"] is AWAKENING_2S
    assert get_dataset_spec("Awakening") is AWAKENING_2S
    assert AWAKENING_2S.canonical_subjects == tuple(range(1, 22))
    assert AWAKENING_2S.train_session == "S1"
    assert AWAKENING_2S.validation_session == "S2"
    assert AWAKENING_2S.class_names == ("non_awakening", "awakening")


def test_dataset_audit_accepts_single_class_subject_and_checks_parent_split(tmp_path) -> None:
    _write_manifest_files(tmp_path)
    first = _write_subject(
        tmp_path,
        subject=1,
        labels=[0, 1, 0, 1],
        sessions=["S1", "S1", "S2", "S2"],
        parents=[10, 11, 12, 13],
    )
    _write_subject(
        tmp_path,
        subject=8,
        labels=[1, 1],
        sessions=["S1", "S2"],
        parents=[80, 81],
    )
    adapter = DEFAULT_DATASET_ADAPTER_REGISTRY.resolve(inspect_hdf5_dataset(first))
    assert isinstance(adapter, LegacyHDF5Adapter)
    audit = validate_awakening_dataset(tmp_path, subjects=[1, 8])
    assert audit.window_count == 6
    assert audit.counts_by_subject["8"]["classes"] == {
        "non_awakening": 0,
        "awakening": 2,
    }
    loaded = build_population_split(
        subjects=[1, 8],
        data_root=tmp_path,
        data_pattern="subject_{subject:02d}.h5",
        data_reader="eeg",
        session_name="S1",
        window_seconds=2.0,
        stride_seconds=2.0,
        base_seed=42,
        shuffle_trials_within_class=True,
        max_windows_per_class_per_subject=None,
        window_construction="direct_trial",
        direct_trial_anchor="start",
        require_all_classes_per_subject=False,
    )
    assert set(loaded.bundle.window_set.labels.tolist()) == {0, 1}
    assert set(loaded.bundle.window_subject_ids.tolist()) == {1, 8}

    with h5py.File(first, "r+") as handle:
        handle["source_lance_row_ids"][:] = [10, 11, 10, 13]
    with pytest.raises(ValueError, match="crosses subject/logical split or label"):
        validate_awakening_dataset(tmp_path, subjects=[1, 8])


def test_channel_mapping_and_two_second_preprocessing_contract(monkeypatch) -> None:
    profile = channel_profile_contract()
    assert profile["mapped_channel_count"] == 60
    assert profile["ignored_source_channels"] == ["TP9", "TP10"]
    assert profile["missing_model_channels"] == ["AF7", "AF8", "F9", "F10"]
    config = awakening_model_config(checkpoint="unused.pt")
    rng = np.random.default_rng(4)
    original_resample = preprocessing_module._resample_signal
    resample_calls = 0

    def counted_resample(*args, **kwargs):
        nonlocal resample_calls
        resample_calls += 1
        return original_resample(*args, **kwargs)

    monkeypatch.setattr(preprocessing_module, "_resample_signal", counted_resample)
    result = Model50MPreprocessor(config)(
        signal=rng.normal(size=(62, 400)).astype(np.float32),
        channel_names=AWAKENING_2S.source_channels,
        original_sample_rate=200.0,
        input_unit="uV",
    )
    assert result.shape == (64, 200)
    assert result.mapped_channel_count == 60
    assert result.padded_points == 0
    assert result.cropped_points == 0
    assert resample_calls == 1
    assert set(result.unknown_channel_names) == {"TP9", "TP10"}
    tokenized = Model50MTokenizer(config)(result)
    assert tokenized.num_tokens == 128
    assert config.classifier_input_dim == 65536


def test_standard_channel_json_matches_runtime_contract() -> None:
    path = Path(__file__).resolve().parents[1] / "docs" / "standard_64_channels.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert tuple(payload["STANDARD_64_CHANNELS"]) == STANDARD_64_CHANNELS
    config = load_yaml(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "datasets"
        / "awakening_2s.yaml"
    )
    assert config["dataset_name"] == AWAKENING_2S.name
    assert config["channel_profile"]["id"] == AWAKENING_2S.channel_profile
    assert config["channel_profile"]["ignored_source_channels"] == ["TP9", "TP10"]
    assert config["channel_profile"]["missing_model_channels"] == [
        "AF7",
        "AF8",
        "F9",
        "F10",
    ]


def test_strict_feature_cache_rejects_contract_mismatch(tmp_path) -> None:
    features = torch.randn(3, 4, dtype=torch.float16)
    labels = torch.tensor([0, 1, 1], dtype=torch.int64)
    dataset = TensorDataset(features, labels)
    window_set = WindowSet(
        windows=np.ones((3, 1, 2), dtype=np.float32),
        labels=labels.numpy(),
        source_trial_ids=((1,), (2,), (3,)),
        construction="direct_source_trial",
    )
    bundle = WindowBundle(
        window_set=window_set,
        window_subject_ids=np.ones(3, dtype=np.int64),
    )
    contract = {"feature_dim": 4, "dataset_name": "awakening", "seed": 42}
    path = tmp_path / "cache.pt"
    save_population_feature_cache(
        dataset=dataset,
        bundle=bundle,
        path=path,
        split_name="population_train",
        class_names=AWAKENING_2S.class_names,
        subject_ids=[1],
        data_reader="eeg",
        subject_identities={"1": {"canonical_subject_id": 1}},
        backbone_sha256="abc",
        preprocessing_hash="def",
        split_identity={"population_train": "S1"},
        cache_contract=contract,
    )
    loaded = load_population_feature_cache(
        path,
        expected_contract=contract,
        expected_split="population_train",
        expected_subject=1,
        mmap=False,
    )
    assert loaded.dataset.tensors[0].shape == (3, 4)
    with pytest.raises(ValueError, match="contract mismatch"):
        load_population_feature_cache(
            path,
            expected_contract={**contract, "seed": 43},
            expected_split="population_train",
            expected_subject=1,
            mmap=False,
        )


def test_linear_head_and_awakening_positive_class_auroc() -> None:
    head = LinearClassificationHead(input_dim=65536, num_classes=2)
    assert head(torch.zeros(2, 65536)).shape == (2, 2)

    small_head = torch.nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        small_head.weight.copy_(torch.tensor([[-1.0], [1.0]]))
    dataset = TensorDataset(
        torch.tensor([[-2.0], [2.0], [-1.0], [1.0]]),
        torch.tensor([0, 1, 0, 1]),
    )
    metrics = evaluate_binary_feature_dataset(
        head=small_head,
        dataset=dataset,
        subject_ids=torch.tensor([1, 1, 8, 8]),
        criterion=torch.nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        class_names=AWAKENING_2S.class_names,
        positive_class_index=1,
        batch_size=2,
    )
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["positive_class_index"] == 1
    assert metrics["class_counts"] == {"non_awakening": 2, "awakening": 2}
    with pytest.raises(ValueError, match="positive_class_index must be 1"):
        evaluate_binary_feature_dataset(
            head=small_head,
            dataset=dataset,
            subject_ids=torch.tensor([1, 1, 8, 8]),
            criterion=torch.nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            class_names=AWAKENING_2S.class_names,
            positive_class_index=0,
            batch_size=2,
        )


def test_awakening_cli_parsers_keep_safe_defaults() -> None:
    from scripts.cache_50m_awakening_features import build_parser as cache_parser
    from scripts.evaluate_50m_awakening_head import build_parser as eval_parser
    from scripts.train_50m_awakening_head import build_parser as train_parser

    cache = cache_parser().parse_args(["--dry-run"])
    assert cache.dry_run is True and cache.overwrite is False
    train = train_parser().parse_args(["--dry-run"])
    assert train.class_weight == "balanced" and train.dry_run is True
    evaluate = eval_parser().parse_args(
        ["--classifier-checkpoint", "head.pt", "--output", "metrics.json"]
    )
    assert evaluate.split == "target_final_test"
    assert evaluate.overwrite is False
