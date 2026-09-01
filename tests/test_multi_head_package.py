from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from bci_dayloop.data.trial_reader import open_trial_reader
from bci_dayloop.inference import MultiHeadDecodeResult, MultiHeadPredictor, SlidingWindowDecoder
from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.packages import (
    export_50m_multi_head_runtime_package,
    load_multi_head_runtime_package,
)
from bci_dayloop.runtime.types import RawEEGWindow
from bci_dayloop.utils.config import dump_yaml, load_yaml


ROOT = Path(__file__).resolve().parents[1]
BACKBONE = ROOT / "checkpoints/backbones/50m/model_deploy.pt"
WORKLOAD = ROOT / (
    "checkpoints/heads/stage1/bnci2014_001/subject_01/Workload/"
    "subject_01/population/2s_flatten/head.pt"
)
ATTENTION = ROOT / (
    "checkpoints/heads/stage1/bnci2014_001/subject_01/MEMA/"
    "subject_01/population/2s_flatten/head.pt"
)
EMOTION = ROOT / (
    "checkpoints/heads/stage1/bnci2014_001/subject_01/SEED/"
    "subject_01/population/2s_flatten/head.pt"
)
YAXIN = ROOT / "data/processed/yaxin/smr_control_yaxin_0819_combined.h5"
HAS_ARTIFACTS = all(path.is_file() for path in (BACKBONE, WORKLOAD, ATTENTION, EMOTION))


def _config() -> Model50MConfig:
    return Model50MConfig(
        checkpoint_path=BACKBONE,
        device="cpu",
        target_sample_rate=100.0,
        window_seconds=2.0,
        patch_seconds=1.0,
        patch_stride_seconds=1.0,
        model_n_time_patches=10,
        output_layer_idx=8,
        aggregation="flatten",
        num_classes=3,
        head_type="linear",
    )


@pytest.fixture(scope="module")
def package_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not HAS_ARTIFACTS:
        pytest.skip("requires local 50M checkpoint and three trained heads")
    return export_50m_multi_head_runtime_package(
        output_dir=tmp_path_factory.mktemp("multi_head_package") / "package",
        config=_config(),
        workload_head=WORKLOAD,
        attention_head=ATTENTION,
        emotion_head=EMOTION,
    )


def _assert_equal(first: object, second: object) -> None:
    for task in ("workload", "attention", "emotion"):
        before = getattr(first, task)
        after = getattr(second, task)
        assert before.label_id == after.label_id
        assert before.label == after.label
        assert before.confidence == pytest.approx(after.confidence)
        np.testing.assert_allclose(before.probabilities, after.probabilities)


def test_export_writes_one_backbone_three_heads_and_relative_manifest(package_path: Path) -> None:
    manifest = load_yaml(package_path / "package.yaml")
    assert manifest["schema_version"] == 2
    assert manifest["model"]["type"] == "model_50m_multi_head"
    assert manifest["model"]["tasks"] == ["workload", "attention", "emotion"]
    assert manifest["feature_contract"] == {
        "embedding_layer": 9,
        "output_layer_idx": 8,
        "aggregation": "flatten",
        "feature_dim": 65_536,
        "num_tokens": 128,
        "channel_mapping": "STANDARD_64_CHANNELS+zero_fill+channel_valid_mask",
    }
    assert (package_path / "backbone.pt").is_file()
    assert sorted(path.name for path in (package_path / "heads").glob("*.pt")) == [
        "attention.pt", "emotion.pt", "workload.pt"
    ]
    for task, output_dim in (("workload", 2), ("attention", 3), ("emotion", 3)):
        entry = manifest["heads"][task]
        assert entry["checkpoint"] == f"heads/{task}.pt"
        assert entry["input_dim"] == 65_536
        assert entry["output_dim"] == output_dim
        assert not Path(entry["checkpoint"]).is_absolute()


def test_export_fails_fast_for_incompatible_feature_dim(tmp_path: Path) -> None:
    if not HAS_ARTIFACTS:
        pytest.skip("requires local 50M checkpoint and three trained heads")
    payload = torch.load(WORKLOAD, map_location="cpu", weights_only=True)
    payload["metadata"] = dict(payload["metadata"])
    payload["metadata"]["feature_dim"] = 512
    incompatible = tmp_path / "bad_feature.pt"
    torch.save(payload, incompatible)
    with pytest.raises(ValueError, match="task=workload; field=feature_dim"):
        export_50m_multi_head_runtime_package(
            output_dir=tmp_path / "package",
            config=_config(),
            workload_head=incompatible,
            attention_head=ATTENTION,
            emotion_head=EMOTION,
        )


def test_export_fails_fast_for_incompatible_aggregation(tmp_path: Path) -> None:
    if not HAS_ARTIFACTS:
        pytest.skip("requires local 50M checkpoint and three trained heads")
    payload = torch.load(ATTENTION, map_location="cpu", weights_only=True)
    payload["metadata"] = dict(payload["metadata"])
    payload["metadata"]["aggregation"] = "mean"
    incompatible = tmp_path / "bad_aggregation.pt"
    torch.save(payload, incompatible)
    with pytest.raises(ValueError, match="task=attention; field=aggregation"):
        export_50m_multi_head_runtime_package(
            output_dir=tmp_path / "package",
            config=_config(),
            workload_head=WORKLOAD,
            attention_head=incompatible,
            emotion_head=EMOTION,
        )


def test_loader_fails_for_missing_or_corrupt_head_manifest(package_path: Path) -> None:
    attention = package_path / "heads" / "attention.pt"
    hidden = package_path / "heads" / "attention.hidden"
    attention.replace(hidden)
    try:
        with pytest.raises(FileNotFoundError, match="attention head"):
            load_multi_head_runtime_package(package_path)
    finally:
        hidden.replace(attention)

    manifest_path = package_path / "package.yaml"
    manifest = load_yaml(manifest_path)
    manifest["heads"]["attention"]["output_dim"] = 2
    dump_yaml(manifest, manifest_path)
    try:
        with pytest.raises(ValueError, match="attention: manifest output_dim=2, actual=3"):
            load_multi_head_runtime_package(package_path)
    finally:
        manifest["heads"]["attention"]["output_dim"] = 3
        dump_yaml(manifest, manifest_path)


@pytest.mark.skipif(not (HAS_ARTIFACTS and YAXIN.is_file()), reason="requires local package artifacts and yaxin H5")
def test_direct_package_and_decoder_predictions_are_equivalent(package_path: Path) -> None:
    reader = open_trial_reader(data_reader="eeg", path=YAXIN, canonical_subject_id=1)
    source = reader.load(session="S6")
    raw = np.asarray(source["data"][0], dtype=np.float32)
    window = RawEEGWindow(
        data=raw,
        channel_names=list(reader.metadata.channel_names),
        sample_rate=float(reader.metadata.sample_rate),
        unit=str(reader.metadata.unit),
        trial_id=str(source["trial_ids"][0]),
    )
    direct = MultiHeadPredictor.from_checkpoints(
        backbone_checkpoint=BACKBONE,
        workload_head=WORKLOAD,
        attention_head=ATTENTION,
        emotion_head=EMOTION,
        device="cpu",
    )
    packaged = load_multi_head_runtime_package(package_path, device="cpu")
    direct_result = direct.predict(window)
    package_result = packaged.predict(window)
    _assert_equal(direct_result, package_result)
    direct_decoder = SlidingWindowDecoder(
        predictor=direct,
        channel_names=reader.metadata.channel_names,
        sample_rate=float(reader.metadata.sample_rate),
        input_unit=str(reader.metadata.unit),
        window_sec=2.0,
        step_sec=2.0,
    )
    direct_decoded = direct_decoder.push(raw)
    assert isinstance(direct_decoded, MultiHeadDecodeResult)
    _assert_equal(direct_result, direct_decoded.prediction)

    package_decoder = SlidingWindowDecoder(
        predictor=packaged,
        channel_names=reader.metadata.channel_names,
        sample_rate=float(reader.metadata.sample_rate),
        input_unit=str(reader.metadata.unit),
        window_sec=2.0,
        step_sec=2.0,
    )
    package_decoded = package_decoder.push(raw)
    assert isinstance(package_decoded, MultiHeadDecodeResult)
    _assert_equal(package_result, package_decoded.prediction)
    _assert_equal(direct_decoded.prediction, package_decoded.prediction)
    diagnostics = packaged.last_diagnostics
    assert diagnostics is not None
    assert diagnostics.preprocessing_calls == 1
    assert diagnostics.backbone_forwards == 1
    assert diagnostics.head_forwards == {
        "workload": 1,
        "attention": 1,
        "emotion": 1,
    }


def test_package_is_loadable_after_move(package_path: Path) -> None:
    moved = package_path.parent / "moved_package"
    package_path.replace(moved)
    try:
        predictor = load_multi_head_runtime_package(moved, device="cpu")
        assert predictor.window_seconds == 2.0
    finally:
        moved.replace(package_path)
