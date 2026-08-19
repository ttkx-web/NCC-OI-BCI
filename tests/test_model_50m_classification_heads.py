from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import pytest

from bci_dayloop.models.model_50m.classifier import (
    LinearClassificationHead,
    MLPClassificationHead,
    Model50MClassifier,
    build_classifier_metadata,
    load_classifier_checkpoint,
    read_classifier_head_config,
    save_classifier_checkpoint,
)
from bci_dayloop.models.model_50m.config import Model50MConfig


class TinyFrozenBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def freeze(self) -> "TinyFrozenBackbone":
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()
        return self


def tiny_config(**overrides: object) -> Model50MConfig:
    values: dict[str, object] = {
        "checkpoint_path": "unused.pt",
        "filter_enabled": False,
        "n_channels": 1,
        "standard_channels": ("C3",),
        "target_sample_rate": 2.0,
        "window_seconds": 2.0,
        "patch_seconds": 1.0,
        "patch_stride_seconds": 1.0,
        "model_n_time_patches": 2,
        "d_model": 8,
        "n_heads": 2,
        "depth": 1,
        "output_layer_idx": 0,
        "aggregation": "mean",
        "num_classes": 3,
    }
    values.update(overrides)
    return Model50MConfig(**values)  # type: ignore[arg-type]


def build_classifier(config: Model50MConfig) -> Model50MClassifier:
    return Model50MClassifier(config=config, backbone=TinyFrozenBackbone())  # type: ignore[arg-type]


def test_default_linear_config_matches_explicit_linear() -> None:
    default = tiny_config()
    explicit = tiny_config(head_type="linear")
    assert default.head_type == explicit.head_type == "linear"
    assert default.head_norm == explicit.head_norm == "none"
    assert default.head_dropout == explicit.head_dropout == 0.0

    torch.manual_seed(7)
    default_head = build_classifier(default).head
    torch.manual_seed(7)
    explicit_head = build_classifier(explicit).head
    features = torch.randn(3, default.classifier_input_dim)
    assert isinstance(default_head, LinearClassificationHead)
    torch.testing.assert_close(default_head(features), explicit_head(features))


def test_linear_rejects_mlp_only_normalization_and_dropout() -> None:
    with pytest.raises(ValueError, match="head_norm is only supported"):
        tiny_config(head_norm="layernorm")
    with pytest.raises(ValueError, match="head_dropout is only supported"):
        tiny_config(head_dropout=0.2)


def test_mlp_without_normalization_forwards_logits() -> None:
    config = tiny_config(head_type="mlp", head_hidden_dim=5)
    head = build_classifier(config).head
    features = torch.randn(4, config.classifier_input_dim)

    logits = head(features)

    assert isinstance(head, MLPClassificationHead)
    assert logits.shape == (4, config.num_classes)


def test_layernorm_mlp_supports_backward() -> None:
    config = tiny_config(
        head_type="mlp",
        head_hidden_dim=5,
        head_dropout=0.2,
        head_norm="layernorm",
    )
    head = build_classifier(config).head
    logits = head(torch.randn(3, config.classifier_input_dim))
    logits.sum().backward()

    assert isinstance(head.normalization, nn.LayerNorm)
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_batchnorm_mlp_supports_batched_backward() -> None:
    config = tiny_config(
        head_type="mlp",
        head_hidden_dim=5,
        head_norm="batchnorm",
    )
    head = build_classifier(config).head
    logits = head(torch.randn(2, config.classifier_input_dim))
    logits.sum().backward()

    assert isinstance(head.normalization, nn.BatchNorm1d)
    assert logits.shape == (2, config.num_classes)
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_backbone_is_frozen_and_optimizer_contains_only_head() -> None:
    classifier = build_classifier(tiny_config(head_type="mlp", head_hidden_dim=5))
    optimizer = torch.optim.AdamW(classifier.head.parameters(), lr=1e-3)
    backbone_ids = {id(parameter) for parameter in classifier.backbone.parameters()}
    head_ids = {id(parameter) for parameter in classifier.head.parameters()}
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert all(not parameter.requires_grad for parameter in classifier.backbone.parameters())
    assert all(parameter.requires_grad for parameter in classifier.head.parameters())
    assert not backbone_ids & optimizer_ids
    assert optimizer_ids == head_ids


def test_mlp_checkpoint_metadata_round_trips(tmp_path: Path) -> None:
    config = tiny_config(
        head_type="mlp",
        head_hidden_dim=5,
        head_dropout=0.25,
        head_norm="layernorm",
    )
    source = build_classifier(config)
    checkpoint = save_classifier_checkpoint(source, tmp_path / "mlp_head.pt")
    restored = build_classifier(config)

    report = load_classifier_checkpoint(restored, checkpoint)

    assert report.metadata["head_type"] == "mlp"
    assert report.metadata["head_hidden_dim"] == 5
    assert report.metadata["head_dropout"] == 0.25
    assert report.metadata["head_norm"] == "layernorm"
    assert read_classifier_head_config(checkpoint) == {
        "head_type": "mlp",
        "head_hidden_dim": 5,
        "head_dropout": 0.25,
        "head_norm": "layernorm",
    }
    for source_parameter, restored_parameter in zip(
        source.head.parameters(), restored.head.parameters(), strict=True
    ):
        torch.testing.assert_close(source_parameter, restored_parameter)


def test_checkpoint_preserves_explicit_class_semantics(tmp_path: Path) -> None:
    classifier = build_classifier(tiny_config(num_classes=3))
    class_names = ["left_hand", "both_hand", "rest"]
    checkpoint = save_classifier_checkpoint(
        classifier,
        tmp_path / "semantic_head.pt",
        extra_metadata={
            "num_classes": 3,
            "class_names": class_names,
            "label_mapping": {
                "0": "left_hand",
                "1": "both_hand",
                "2": "rest",
            },
        },
    )
    restored = build_classifier(tiny_config(num_classes=3))

    report = load_classifier_checkpoint(restored, checkpoint)

    assert report.metadata["num_classes"] == 3
    assert report.metadata["class_names"] == class_names
    assert report.metadata["label_mapping"] == {
        "0": "left_hand",
        "1": "both_hand",
        "2": "rest",
    }


def test_legacy_linear_checkpoint_defaults_missing_head_metadata(tmp_path: Path) -> None:
    config = tiny_config()
    source = build_classifier(config)
    metadata = build_classifier_metadata(config)
    for key in ("head_type", "head_hidden_dim", "head_dropout", "head_norm"):
        metadata.pop(key)
    checkpoint = tmp_path / "legacy_linear.pt"
    torch.save(
        {
            "format_version": 1,
            "head_state_dict": source.head.state_dict(),
            "metadata": metadata,
        },
        checkpoint,
    )
    restored = build_classifier(config)

    load_classifier_checkpoint(restored, checkpoint, strict_metadata=True)

    assert read_classifier_head_config(checkpoint) == {
        "head_type": "linear",
        "head_hidden_dim": 512,
        "head_dropout": 0.0,
        "head_norm": "none",
    }
    for source_parameter, restored_parameter in zip(
        source.head.parameters(), restored.head.parameters(), strict=True
    ):
        torch.testing.assert_close(source_parameter, restored_parameter)
