from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bci_dayloop.applications.three_mental_states.contract import (
    AWAKENING_CLASS_NAMES,
    FOUR_HEAD_TASKS,
)
from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.packages import (
    export_50m_multi_head_runtime_package,
    load_inference_package,
    load_multi_head_runtime_package,
)
from bci_dayloop.utils.config import dump_yaml, load_yaml


ROOT = Path(__file__).resolve().parents[1]
BACKBONE = ROOT / "checkpoints/backbones/50m/model_deploy.pt"
HEADS = {
    "workload": ROOT / "checkpoints/heads/stage1/bnci2014_001/subject_01/Workload/subject_01/population/2s_flatten/head.pt",
    "attention": ROOT / "checkpoints/heads/stage1/bnci2014_001/subject_01/MEMA/subject_01/population/2s_flatten/head.pt",
    "emotion": ROOT / "checkpoints/heads/stage1/bnci2014_001/subject_01/SEED/subject_01/population/2s_flatten/head.pt",
    "awakening": ROOT / "checkpoints/heads/stage1/bnci2014_001/subject_01/awakening/subject_01/population/2s_flatten/head.pt",
}
HAS_ARTIFACTS = BACKBONE.is_file() and all(path.is_file() for path in HEADS.values())


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
        num_classes=2,
        head_type="linear",
    )


@pytest.fixture(scope="module")
def four_head_package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not HAS_ARTIFACTS:
        pytest.skip("requires the local backbone and four trained heads")
    return export_50m_multi_head_runtime_package(
        output_dir=tmp_path_factory.mktemp("four_head_package") / "package",
        config=_config(),
        workload_head=HEADS["workload"],
        attention_head=HEADS["attention"],
        emotion_head=HEADS["emotion"],
        awakening_head=HEADS["awakening"],
    )


def test_four_head_export_and_manifest_driven_load(four_head_package: Path) -> None:
    manifest = load_yaml(four_head_package / "package.yaml")
    assert tuple(manifest["model"]["tasks"]) == FOUR_HEAD_TASKS
    assert sorted(path.name for path in (four_head_package / "heads").glob("*.pt")) == [
        "attention.pt", "awakening.pt", "emotion.pt", "workload.pt"
    ]
    awakening = manifest["heads"]["awakening"]
    assert tuple(awakening["class_names"]) == AWAKENING_CLASS_NAMES
    assert awakening["input_dim"] == 65_536
    assert awakening["output_dim"] == 2
    assert awakening["positive_class_index"] == 1

    loaded = load_inference_package(four_head_package, device="cpu")
    assert tuple(task.task_id for task in loaded.tasks) == FOUR_HEAD_TASKS
    assert tuple(loaded.predictor.head_info) == FOUR_HEAD_TASKS


@pytest.mark.parametrize("tasks", [["workload", "attention", "emotion", "awakening", "awakening"], ["workload", ""]])
def test_loader_rejects_duplicate_or_empty_task_ids(four_head_package: Path, tasks: list[str]) -> None:
    manifest_path = four_head_package / "package.yaml"
    manifest = load_yaml(manifest_path)
    original = list(manifest["model"]["tasks"])
    manifest["model"]["tasks"] = tasks
    dump_yaml(manifest, manifest_path)
    try:
        with pytest.raises(ValueError, match="task_id"):
            load_multi_head_runtime_package(four_head_package, device="cpu")
    finally:
        manifest["model"]["tasks"] = original
        dump_yaml(manifest, manifest_path)


def test_export_rejects_wrong_awakening_class_order(tmp_path: Path) -> None:
    if not HAS_ARTIFACTS:
        pytest.skip("requires the local backbone and four trained heads")
    payload = torch.load(HEADS["awakening"], map_location="cpu", weights_only=True)
    payload["metadata"] = dict(payload["metadata"])
    payload["metadata"]["class_names"] = ["awakening", "non_awakening"]
    bad = tmp_path / "bad_awakening.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="class_names"):
        export_50m_multi_head_runtime_package(
            output_dir=tmp_path / "package",
            config=_config(),
            workload_head=HEADS["workload"],
            attention_head=HEADS["attention"],
            emotion_head=HEADS["emotion"],
            awakening_head=bad,
        )


def test_export_rejects_awakening_backbone_hash_mismatch(tmp_path: Path) -> None:
    if not HAS_ARTIFACTS:
        pytest.skip("requires the local backbone and four trained heads")
    payload = torch.load(HEADS["awakening"], map_location="cpu", weights_only=True)
    payload["metadata"] = dict(payload["metadata"])
    payload["metadata"]["backbone_checkpoint_sha256"] = "0" * 64
    bad = tmp_path / "wrong_backbone_awakening.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="do not all reference the supplied backbone"):
        export_50m_multi_head_runtime_package(
            output_dir=tmp_path / "package",
            config=_config(),
            workload_head=HEADS["workload"],
            attention_head=HEADS["attention"],
            emotion_head=HEADS["emotion"],
            awakening_head=bad,
        )


def test_loader_rejects_head_hash_mismatch(four_head_package: Path) -> None:
    head = four_head_package / "heads/awakening.pt"
    original = head.read_bytes()
    head.write_bytes(original + b"corrupt")
    try:
        with pytest.raises(ValueError, match="awakening head SHA256 mismatch"):
            load_multi_head_runtime_package(four_head_package, device="cpu")
    finally:
        head.write_bytes(original)
