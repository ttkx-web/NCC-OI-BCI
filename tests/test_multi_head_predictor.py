from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from bci_dayloop.data.trial_reader import open_trial_reader
from bci_dayloop.inference.multi_head import (
    HeadCheckpointInfo,
    MultiHeadPrediction,
    MultiHeadPredictor,
    TASK_OUTPUT_DIMS,
    _LoadedHead,
)
from bci_dayloop.applications.three_mental_states.contract import TASKS, ThreeMentalStatePrediction
from bci_dayloop.applications.three_mental_states.predictor import ThreeMentalStatePredictor
from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.models.model_50m.preprocessing import PreprocessResult
from bci_dayloop.runtime.types import RawEEGWindow


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


class _CountingPreprocessor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, **_kwargs: object) -> PreprocessResult:
        self.calls += 1
        return PreprocessResult(
            signal=np.ones((64, 200), dtype=np.float32),
            channel_valid_mask=np.ones(64, dtype=np.float32),
            original_channel_names=("C3", "C4"),
            canonical_channel_names=("C3", "C4"),
            unknown_channel_names=(),
            original_sample_rate=100.0,
            target_sample_rate=100.0,
            mapped_channel_count=2,
            missing_channel_count=62,
            duplicate_channel_count=0,
            padded_points=0,
            cropped_points=0,
            notes=(),
        )


class _FakeBatch:
    def __init__(self) -> None:
        self.token_valid_mask = torch.ones((1, 128), dtype=torch.float32)


class _FakeTokenized:
    def as_batch(self, device: torch.device) -> _FakeBatch:
        return _FakeBatch()


class _FakeTokenizer:
    def __call__(self, _preprocessed: PreprocessResult) -> _FakeTokenized:
        return _FakeTokenized()


class _CountingBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.calls = 0
        self.embedding = (
            torch.arange(128 * 512, dtype=torch.float32)
            .reshape(1, 128, 512)
            / 1000.0
        )

    def extract_embeddings(
        self, *, batch: _FakeBatch, return_layer_idx: int
    ) -> torch.Tensor:
        assert return_layer_idx == 8
        assert batch.token_valid_mask.shape == (1, 128)
        self.calls += 1
        return self.embedding.clone()


class _CountingHead(nn.Linear):
    def __init__(self, output_dim: int) -> None:
        super().__init__(65_536, output_dim)
        self.calls = 0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return super().forward(features)


def _config() -> Model50MConfig:
    return Model50MConfig(
        checkpoint_path="unused",
        window_seconds=2.0,
        target_sample_rate=100.0,
        patch_seconds=1.0,
        patch_stride_seconds=1.0,
        model_n_time_patches=10,
        output_layer_idx=8,
        aggregation="flatten",
        num_classes=3,
        head_type="linear",
    )


def _loaded_head(task: str, module: nn.Module) -> _LoadedHead:
    output_dim = TASK_OUTPUT_DIMS[task]
    names = tuple(f"{task}_{index}" for index in range(output_dim))
    return _LoadedHead(
        info=HeadCheckpointInfo(
            task=task,
            checkpoint_path=Path(f"{task}.pt"),
            class_names=names,
            input_dim=65_536,
            output_dim=output_dim,
            metadata={},
        ),
        module=module,
    )


def _fake_predictor() -> tuple[MultiHeadPredictor, _CountingPreprocessor, _CountingBackbone, dict[str, _CountingHead]]:
    torch.manual_seed(7)
    preprocessor = _CountingPreprocessor()
    backbone = _CountingBackbone()
    modules = {task: _CountingHead(output_dim) for task, output_dim in TASK_OUTPUT_DIMS.items()}
    predictor = MultiHeadPredictor(
        config=_config(),
        preprocessor=preprocessor,  # type: ignore[arg-type]
        tokenizer=_FakeTokenizer(),  # type: ignore[arg-type]
        backbone=backbone,  # type: ignore[arg-type]
        heads={task: _loaded_head(task, module) for task, module in modules.items()},
    )
    return predictor, preprocessor, backbone, modules


def _window() -> RawEEGWindow:
    return RawEEGWindow(
        data=np.ones((2, 20), dtype=np.float32),
        channel_names=["C3", "C4"],
        sample_rate=100.0,
        unit="uV",
    )


def test_predictor_uses_one_shared_feature_and_returns_stable_results() -> None:
    predictor, preprocessor, backbone, modules = _fake_predictor()
    result = predictor.predict(_window())
    diagnostics = predictor.last_diagnostics

    assert diagnostics is not None
    assert preprocessor.calls == diagnostics.preprocessing_calls == 1
    assert backbone.calls == diagnostics.backbone_forwards == 1
    assert {task: module.calls for task, module in modules.items()} == {
        "workload": 1,
        "attention": 1,
        "emotion": 1,
    }
    assert diagnostics.shared_feature_shape == (1, 65_536)
    assert diagnostics.logit_shapes == {
        "workload": (1, 2),
        "attention": (1, 3),
        "emotion": (1, 3),
    }
    for task, prediction in (
        ("workload", result.workload),
        ("attention", result.attention),
        ("emotion", result.emotion),
    ):
        assert prediction.label == f"{task}_{prediction.label_id}"
        assert len(prediction.probabilities) == TASK_OUTPUT_DIMS[task]
        assert np.isfinite(prediction.probabilities).all()
        assert np.isclose(sum(prediction.probabilities), 1.0)
        assert prediction.confidence == prediction.probabilities[prediction.label_id]
    assert not backbone.training
    assert all(not module.training for module in modules.values())
    assert all(not parameter.requires_grad for module in modules.values() for parameter in module.parameters())


def test_three_state_contract_order_dimensions_and_legacy_aliases() -> None:
    assert TASKS == ("workload", "attention", "emotion")
    assert TASK_OUTPUT_DIMS == {"workload": 2, "attention": 3, "emotion": 3}
    assert MultiHeadPredictor is ThreeMentalStatePredictor
    assert MultiHeadPrediction is ThreeMentalStatePrediction


def test_predictor_fails_fast_for_incompatible_head_feature_dim() -> None:
    predictor, preprocessor, backbone, modules = _fake_predictor()
    bad = _LoadedHead(
        info=HeadCheckpointInfo(
            task="workload",
            checkpoint_path=Path("bad.pt"),
            class_names=("a", "b"),
            input_dim=1,
            output_dim=2,
            metadata={},
        ),
        module=modules["workload"],
    )
    with pytest.raises(ValueError, match="workload: input dim expected 65536, actual 1"):
        MultiHeadPredictor(
            config=predictor.config,
            preprocessor=preprocessor,  # type: ignore[arg-type]
            tokenizer=_FakeTokenizer(),  # type: ignore[arg-type]
            backbone=backbone,  # type: ignore[arg-type]
            heads={
                "workload": bad,
                "attention": _loaded_head("attention", modules["attention"]),
                "emotion": _loaded_head("emotion", modules["emotion"]),
            },
        )


@pytest.mark.skipif(
    not WORKLOAD.is_file(),
    reason="requires the local workload Linear head checkpoint",
)
def test_checkpoint_contract_mismatch_fails_before_backbone_loading(tmp_path: Path) -> None:
    payload = torch.load(WORKLOAD, map_location="cpu", weights_only=True)
    payload["metadata"] = dict(payload["metadata"])
    payload["metadata"]["aggregation"] = "mean"
    incompatible = tmp_path / "incompatible_workload.pt"
    torch.save(payload, incompatible)

    with pytest.raises(
        ValueError,
        match="workload: incompatible shared feature contract for aggregation: "
        "expected 'flatten', actual 'mean'",
    ):
        MultiHeadPredictor.from_checkpoints(
            backbone_checkpoint=BACKBONE,
            workload_head=incompatible,
            attention_head=ATTENTION,
            emotion_head=EMOTION,
            device="cpu",
        )


@pytest.mark.skipif(
    not all(path.is_file() for path in (BACKBONE, WORKLOAD, ATTENTION, EMOTION, YAXIN)),
    reason="requires local three-head checkpoints and canonical yaxin H5",
)
def test_real_yaxin_window_runs_through_formal_predictor() -> None:
    reader = open_trial_reader(data_reader="eeg", path=YAXIN, canonical_subject_id=1)
    source = reader.load(session="S6")
    predictor = MultiHeadPredictor.from_checkpoints(
        backbone_checkpoint=BACKBONE,
        workload_head=WORKLOAD,
        attention_head=ATTENTION,
        emotion_head=EMOTION,
        device="cpu",
    )
    result = predictor.predict(
        RawEEGWindow(
            data=np.asarray(source["data"][0], dtype=np.float32),
            channel_names=list(reader.metadata.channel_names),
            sample_rate=float(reader.metadata.sample_rate),
            unit=str(reader.metadata.unit),
            trial_id=str(source["trial_ids"][0]),
        )
    )
    diagnostics = predictor.last_diagnostics
    assert diagnostics is not None
    assert diagnostics.preprocessing_calls == 1
    assert diagnostics.backbone_forwards == 1
    assert diagnostics.shared_feature_shape == (1, 65_536)
    assert diagnostics.logit_shapes == {
        "workload": (1, 2),
        "attention": (1, 3),
        "emotion": (1, 3),
    }
    assert result.workload.label in {"low_workload", "high_workload"}
    assert result.attention.label in {"relaxing", "neutral", "concentrating"}
    assert result.emotion.label in {"negative", "neutral", "positive"}
