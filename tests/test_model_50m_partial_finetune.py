from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize

from bci_dayloop.models.model_50m.backbone import Model50MBackbone
from bci_dayloop.models.model_50m.classifier import (
    Model50MClassifier,
    load_classifier_checkpoint,
    save_classifier_checkpoint,
)
from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.models.model_50m.finetuning import (
    resolve_backbone_adaptation,
    resolve_embedding_layer,
    resolve_trainable_block_indices,
    uses_frozen_feature_cache,
)
from bci_dayloop.models.model_50m.lora import (
    inject_lora_adapters,
    lora_state_dict,
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


def test_legacy_adaptation_resolution_preserves_frozen_and_partial_modes() -> None:
    assert resolve_backbone_adaptation(
        requested=None,
        unfreeze_last_n_blocks=0,
    ) == "frozen"
    assert resolve_backbone_adaptation(
        requested=None,
        unfreeze_last_n_blocks=2,
    ) == "partial"
    assert resolve_backbone_adaptation(
        requested="lora",
        unfreeze_last_n_blocks=0,
    ) == "lora"
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_backbone_adaptation(
            requested="lora",
            unfreeze_last_n_blocks=1,
        )


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


def test_lora_initial_injection_is_an_exact_identity() -> None:
    torch.manual_seed(11)
    classifier = make_classifier()
    classifier.eval()
    baseline = classifier(make_batch()).detach()

    report = inject_lora_adapters(
        classifier.backbone,
        block_indices=(1, 2),
        target_modules=("q", "v"),
        rank=2,
        alpha=4.0,
        dropout=0.0,
    )
    classifier.eval()

    assert report.block_indices == (1, 2)
    torch.testing.assert_close(classifier(make_batch()), baseline, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("last_n_blocks", "expected_blocks"),
    [(1, {8}), (2, {7, 8})],
)
def test_lora_selects_effective_blocks_before_embedding_layer(
    last_n_blocks: int,
    expected_blocks: set[int],
) -> None:
    config = make_config()
    config = Model50MConfig(
        **{
            **{
                field: getattr(config, field)
                for field in config.__dataclass_fields__
            },
            "depth": 12,
            "output_layer_idx": 8,
        }
    )
    backbone = Model50MBackbone(config=config, load_checkpoint=False, freeze=True)
    selected = resolve_trainable_block_indices(
        embedding_layer=9,
        unfreeze_last_n_blocks=last_n_blocks,
    )
    inject_lora_adapters(
        backbone,
        block_indices=selected,
        target_modules=("q", "v"),
        rank=2,
        alpha=4.0,
        dropout=0.0,
    )

    for index, layer in enumerate(backbone.model.encoder.encoder.layers):
        injected = parametrize.is_parametrized(layer.self_attn, "in_proj_weight")
        assert injected is (index in expected_blocks)


def test_lora_only_adapters_and_head_receive_gradients() -> None:
    classifier = make_classifier()
    report = inject_lora_adapters(
        classifier.backbone,
        block_indices=(1, 2),
        target_modules=("q", "v"),
        rank=2,
        alpha=4.0,
        dropout=0.0,
    )
    classifier.train()
    loss = F.cross_entropy(classifier(make_batch()), torch.tensor([0, 1]))
    loss.backward()

    assert all(
        not parameter.requires_grad
        for name, parameter in classifier.backbone.model.named_parameters()
        if ".lora_" not in name
    )
    assert all(
        parameter.requires_grad
        for adapter in report.adapter_modules
        for parameter in adapter.parameters()
    )
    assert any(
        parameter.grad is not None
        for adapter in report.adapter_modules
        for parameter in adapter.parameters()
    )
    assert any(parameter.grad is not None for parameter in classifier.head.parameters())


def test_lora_checkpoint_restores_adapters_and_head(tmp_path: Path) -> None:
    torch.manual_seed(17)
    source = make_classifier()
    base_state = {
        key: value.detach().clone()
        for key, value in source.backbone.model.state_dict().items()
    }
    report = inject_lora_adapters(
        source.backbone,
        block_indices=(2,),
        target_modules=("q", "v"),
        rank=2,
        alpha=4.0,
        dropout=0.0,
    )
    source.train()
    optimizer = torch.optim.AdamW(
        list(source.head.parameters())
        + [
            parameter
            for adapter in report.adapter_modules
            for parameter in adapter.parameters()
        ],
        lr=1e-2,
    )
    optimizer.zero_grad()
    F.cross_entropy(source(make_batch()), torch.tensor([0, 1])).backward()
    optimizer.step()
    source.eval()
    expected_logits = source(make_batch()).detach()

    path = save_classifier_checkpoint(
        source,
        tmp_path / "lora.pt",
        extra_metadata={
            "backbone_adaptation": "lora",
            "lora_block_indices": [2],
            "lora_target_modules": ["q", "v"],
            "lora_rank": 2,
            "lora_alpha": 4.0,
            "lora_dropout": 0.0,
        },
        lora_state_dict=lora_state_dict(source.backbone.model),
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["format_version"] == 4
    assert set(payload["lora_state_dict"]) == set(lora_state_dict(source.backbone.model))

    torch.manual_seed(19)
    restored = make_classifier()
    restored.backbone.model.load_state_dict(base_state, strict=True)
    load_classifier_checkpoint(restored, path, strict_metadata=True)
    restored.eval()
    torch.testing.assert_close(restored(make_batch()), expected_logits)
