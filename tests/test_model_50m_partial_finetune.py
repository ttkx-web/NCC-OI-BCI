from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from bci_dayloop.models.model_50m.backbone import Model50MBackbone
from bci_dayloop.models.model_50m.classifier import (
    Model50MClassifier,
    load_classifier_checkpoint,
    save_classifier_checkpoint,
)
from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.models.model_50m.finetuning import (
    resolve_embedding_layer,
    resolve_trainable_block_indices,
    uses_frozen_feature_cache,
)
from bci_dayloop.models.model_50m.tokenization import Model50MBatchedInput


def make_config() -> Model50MConfig:
    return Model50MConfig(
        checkpoint_path="unused.pt",
        device="cpu",
        target_sample_rate=2.0,
        window_seconds=1.0,
        patch_seconds=1.0,
        patch_stride_seconds=1.0,
        n_channels=2,
        standard_channels=("C3", "C4"),
        filter_enabled=False,
        zscore_enabled=False,
        d_model=4,
        n_heads=2,
        depth=4,
        mlp_ratio=1.0,
        dropout=0.0,
        model_n_time_patches=1,
        output_layer_idx=2,
        aggregation="mean",
        num_classes=3,
    )


def make_classifier() -> Model50MClassifier:
    config = make_config()
    backbone = Model50MBackbone(
        config=config,
        load_checkpoint=False,
        freeze=True,
    )
    return Model50MClassifier(config=config, backbone=backbone)


def make_batch(batch_size: int = 2) -> Model50MBatchedInput:
    token_inputs = torch.arange(
        batch_size * 2 * 2,
        dtype=torch.float32,
    ).reshape(batch_size, 2, 2)
    return Model50MBatchedInput(
        token_inputs=token_inputs,
        token_channel_indices=torch.tensor([[0, 1]]).repeat(batch_size, 1),
        token_time_indices=torch.zeros(batch_size, 2, dtype=torch.long),
        token_valid_mask=torch.ones(batch_size, 2, dtype=torch.float32),
        channel_valid_mask=torch.ones(batch_size, 2, dtype=torch.float32),
    )


def block_parameter_ids(classifier: Model50MClassifier, index: int) -> set[int]:
    return {
        id(parameter)
        for parameter in classifier.backbone.model.encoder.encoder.layers[index].parameters()
    }


def test_auto_embedding_layer_uses_existing_output_layer_contract() -> None:
    assert resolve_embedding_layer(
        requested="auto",
        output_layer_idx=8,
        depth=12,
    ) == 9
    assert resolve_embedding_layer(
        requested="3",
        output_layer_idx=8,
        depth=12,
    ) == 3
    with pytest.raises(ValueError, match="1-based range"):
        resolve_embedding_layer(
            requested="13",
            output_layer_idx=8,
            depth=12,
        )


@pytest.mark.parametrize(
    ("unfreeze_last_n_blocks", "expected"),
    [(0, ()), (1, (2,)), (2, (1, 2))],
)
def test_trainable_blocks_end_at_embedding_layer(
    unfreeze_last_n_blocks: int,
    expected: tuple[int, ...],
) -> None:
    assert resolve_trainable_block_indices(
        embedding_layer=3,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
    ) == expected


@pytest.mark.parametrize("unfreeze_last_n_blocks", [-1, 4])
def test_invalid_unfreeze_count_is_rejected(unfreeze_last_n_blocks: int) -> None:
    with pytest.raises(ValueError):
        resolve_trainable_block_indices(
            embedding_layer=3,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )


def test_feature_cache_is_only_allowed_for_frozen_backbone() -> None:
    assert uses_frozen_feature_cache(unfreeze_last_n_blocks=0)
    assert not uses_frozen_feature_cache(unfreeze_last_n_blocks=1)


def test_frozen_baseline_keeps_all_backbone_parameters_frozen() -> None:
    classifier = make_classifier()
    classifier.backbone.set_trainable_encoder_blocks(())

    assert classifier.backbone.trainable_encoder_block_indices == ()
    assert all(
        not parameter.requires_grad
        for parameter in classifier.backbone.parameters()
    )
    assert all(parameter.requires_grad for parameter in classifier.head.parameters())
    classifier.train()
    assert not classifier.backbone.training


def test_partial_finetune_unfreezes_only_selected_blocks_and_gradients() -> None:
    classifier = make_classifier()
    classifier.backbone.set_trainable_encoder_blocks((1, 2))
    classifier.train()

    assert classifier.backbone.trainable_encoder_block_indices == (1, 2)
    assert classifier.backbone.model.encoder.encoder.layers[1].training
    assert classifier.backbone.model.encoder.encoder.layers[2].training
    assert not classifier.backbone.model.encoder.encoder.layers[0].training
    assert not classifier.backbone.model.encoder.encoder.layers[3].training

    trainable_ids = {
        id(parameter)
        for parameter in classifier.backbone.parameters()
        if parameter.requires_grad
    }
    assert trainable_ids == block_parameter_ids(classifier, 1) | block_parameter_ids(
        classifier,
        2,
    )

    logits = classifier(make_batch())
    F.cross_entropy(logits, torch.tensor([0, 1])).backward()

    assert any(parameter.grad is not None for parameter in classifier.head.parameters())
    for index in (1, 2):
        assert any(
            parameter.grad is not None
            for parameter in classifier.backbone.model.encoder.encoder.layers[index].parameters()
        )
    for index in (0, 3):
        assert all(
            parameter.grad is None
            for parameter in classifier.backbone.model.encoder.encoder.layers[index].parameters()
        )


def test_partial_finetune_checkpoint_restores_backbone_and_head(
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    source = make_classifier()
    source.backbone.set_trainable_encoder_blocks((2,))
    source.train()
    loss = F.cross_entropy(source(make_batch()), torch.tensor([0, 1]))
    loss.backward()
    optimizer = torch.optim.AdamW(
        list(source.head.parameters())
        + list(source.backbone.model.encoder.encoder.layers[2].parameters()),
        lr=1e-2,
    )
    optimizer.step()
    source.eval()
    expected_logits = source(make_batch()).detach()

    path = save_classifier_checkpoint(
        source,
        tmp_path / "partial.pt",
        extra_metadata={"unfreeze_last_n_blocks": 1},
        backbone_state_dict=source.backbone.model.state_dict(),
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["format_version"] == 3
    assert "backbone_state_dict" in payload

    torch.manual_seed(9)
    restored = make_classifier()
    restored.backbone.set_trainable_encoder_blocks((2,))
    load_classifier_checkpoint(restored, path, strict_metadata=True)
    restored.eval()
    torch.testing.assert_close(restored(make_batch()), expected_logits)
