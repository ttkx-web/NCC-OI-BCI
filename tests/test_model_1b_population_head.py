from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from bci_dayloop.models.model_1b.classifier import Model1BFlattenLinearHead, classifier_input_dim
from bci_dayloop.models.model_1b.config import Model1BConfig
from bci_dayloop.training.model_1b import population
from bci_dayloop.training.model_50m import data as model_50m_data
from bci_dayloop.training.model_50m.types import ExtendedMetrics


def _metrics() -> ExtendedMetrics:
    return ExtendedMetrics(
        loss=1.0,
        accuracy=0.5,
        balanced_accuracy=0.5,
        macro_f1=0.5,
        confusion_matrix=[[1, 0], [1, 0]],
        per_class=[],
    )


@pytest.mark.parametrize(
    ("seconds", "expected"),
    ((1.0, 131_072), (2.0, 262_144), (3.0, 393_216), (4.0, 524_288)),
)
def test_1b_flatten_head_dimension_follows_active_token_contract(seconds: float, expected: int) -> None:
    config = Model1BConfig(checkpoint_path="unused.pt", window_seconds=seconds)
    assert classifier_input_dim(config) == config.n_channels * config.num_time_patches * config.d_model
    assert classifier_input_dim(config) == expected
    head = Model1BFlattenLinearHead(input_dim=classifier_input_dim(config), num_classes=4)
    assert head.linear.weight.shape == (4, expected)


def test_only_linear_head_receives_gradients() -> None:
    backbone = nn.Linear(2, 2)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    runner = SimpleNamespace(backbone=backbone)
    population._assert_frozen_backbone(runner)  # type: ignore[arg-type]

    head = Model1BFlattenLinearHead(input_dim=8, num_classes=2)
    logits = head(torch.ones((2, 8), dtype=torch.float32))
    logits.sum().backward()
    assert all(parameter.grad is None for parameter in backbone.parameters())
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_head_artifact_round_trip_and_contract_guards(tmp_path: Path) -> None:
    config = Model1BConfig(checkpoint_path="unused.pt", window_seconds=1.0)
    head = Model1BFlattenLinearHead(input_dim=classifier_input_dim(config), num_classes=2)
    args = Namespace(
        split_mode="loso", epochs=1, head_batch_size=2, head_lr=1e-3,
        weight_decay=0.0, patience=0, metric_for_best="val_bacc", window_seed=42, seed=42,
    )
    payload = population._head_payload(
        state=head.state_dict(), config=config, class_names=["left", "right"],
        backbone_path=tmp_path / "backbone.pt", backbone_sha256="abc123",
        split={"excluded_target_subject": 1, "train_session": "0train"}, args=args,
        best_epoch=1, validation_metrics=_metrics(), final_test_metrics=_metrics(), training_seconds=0.1,
    )
    path = population.save_1b_head_checkpoint(tmp_path / "head.pt", payload, overwrite=False)
    loaded, metadata = population.load_1b_head_checkpoint(
        path, window_seconds=1.0, class_names=["left", "right"], backbone_sha256="abc123",
    )
    assert loaded.linear.weight.shape == (2, 131_072)
    assert metadata["head_state_dict"].keys() == head.state_dict().keys()
    assert metadata["contains_backbone_weights"] is False
    assert metadata["contains_optimizer_state"] is False
    with pytest.raises(ValueError, match="window_seconds"):
        population.load_1b_head_checkpoint(path, window_seconds=2.0)
    with pytest.raises(ValueError, match="class_names order"):
        population.load_1b_head_checkpoint(path, class_names=["right", "left"])
    with pytest.raises(ValueError, match="SHA-256"):
        population.load_1b_head_checkpoint(path, backbone_sha256="different")


def test_population_script_reuses_50m_split_semantics() -> None:
    assert population.build_population_split is model_50m_data.build_population_split
    assert population.build_within_subject_splits is model_50m_data.build_within_subject_splits
    assert population.build_within_subject_test_split is model_50m_data.build_within_subject_test_split
    parser = population.build_argument_parser()
    assert parser.parse_args([]).split_mode == "loso"
    within = parser.parse_args([
        "--split-mode", "within-subject", "--target-subject", "1",
        "--train-session", "0train", "--test-session", "1test", "--window-seconds", "4",
    ])
    population.validate_args(within)
    assert within.window_seconds == 4.0


def test_head_checkpoint_rejects_metadata_state_shape_disagreement(tmp_path: Path) -> None:
    config = Model1BConfig(checkpoint_path="unused.pt", window_seconds=1.0)
    head = Model1BFlattenLinearHead(input_dim=classifier_input_dim(config), num_classes=2)
    args = Namespace(split_mode="loso", epochs=1, head_batch_size=1, head_lr=1e-3, weight_decay=0.0, patience=0, metric_for_best="val_bacc", window_seed=1, seed=1)
    payload = population._head_payload(
        state=head.state_dict(), config=config, class_names=["a", "b"], backbone_path=tmp_path / "b.pt",
        backbone_sha256="sha", split={}, args=args, best_epoch=1, validation_metrics=_metrics(),
        final_test_metrics=_metrics(), training_seconds=0.0,
    )
    payload["head_state_dict"]["linear.weight"] = torch.zeros((2, 3))
    with pytest.raises(ValueError, match="weight shape"):
        population.validate_head_checkpoint_compatibility(payload)
