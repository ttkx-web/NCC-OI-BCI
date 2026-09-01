from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn.functional as F

from bci_dayloop.models.model_50m.backbone import Model50MBackbone
from bci_dayloop.models.model_50m.classifier import (
    Model50MClassifier,
    load_classifier_checkpoint,
    save_classifier_checkpoint,
)
from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.models.model_50m.lora import lora_state_dict
from bci_dayloop.models.model_50m.tokenization import Model50MBatchedInput
from bci_dayloop.training.model_50m import adaptation, runner
from bci_dayloop.training.model_50m.engine import build_optimizer


def _classifier() -> Model50MClassifier:
    config = Model50MConfig(
        checkpoint_path="unused.pt", device="cpu", target_sample_rate=2.0,
        window_seconds=1.0, patch_seconds=1.0, patch_stride_seconds=1.0,
        n_channels=2, standard_channels=("C3", "C4"), filter_enabled=False,
        zscore_enabled=False, d_model=4, n_heads=2, depth=4, mlp_ratio=1.0,
        dropout=0.0, model_n_time_patches=1, output_layer_idx=2,
        aggregation="mean", num_classes=3,
    )
    backbone = Model50MBackbone(config=config, load_checkpoint=False, freeze=True)
    return Model50MClassifier(config=config, backbone=backbone)


def _batch() -> Model50MBatchedInput:
    return Model50MBatchedInput(
        token_inputs=torch.arange(8, dtype=torch.float32).reshape(2, 2, 2),
        token_channel_indices=torch.tensor([[0, 1], [0, 1]]),
        token_time_indices=torch.zeros(2, 2, dtype=torch.long),
        token_valid_mask=torch.ones(2, 2),
        channel_valid_mask=torch.ones(2, 2),
    )


def test_partial_adaptation_plan_scope_and_live_gradients() -> None:
    classifier = _classifier()
    plan = adaptation.resolve_adaptation_plan(
        config=classifier.config,
        requested_embedding_layer="auto",
        requested_adaptation="partial",
        unfreeze_last_n_blocks=2,
        lora_last_n_blocks=2,
    )
    assert plan.trainable_block_indices == (1, 2)
    setup = adaptation.configure_adaptation(
        backbone=classifier.backbone, classifier=classifier, plan=plan,
        lora_target_modules=("q", "v"), lora_rank=2, lora_alpha=4.0,
        lora_dropout=0.0,
    )
    assert {id(p) for p in setup.trainable_backbone_parameters} == {
        id(p) for index in (1, 2)
        for p in classifier.backbone.model.encoder.encoder.layers[index].parameters()
    }
    batch = _batch()
    logits = adaptation.forward_live_logits(
        classifier=classifier,
        token_inputs=batch.token_inputs,
        token_channel_indices=batch.token_channel_indices,
        token_time_indices=batch.token_time_indices,
        token_valid_mask=batch.token_valid_mask,
        channel_valid_mask=batch.channel_valid_mask,
    )
    logits.sum().backward()
    assert any(p.grad is not None for p in setup.trainable_backbone_parameters)
    assert all(
        p.grad is None
        for p in classifier.backbone.model.encoder.encoder.layers[0].parameters()
    )


def test_adaptation_exports_remain_runner_compatible() -> None:
    assert runner.tokenize_windows_for_finetuning is adaptation.tokenize_windows_for_finetuning
    assert runner.forward_live_logits is adaptation.forward_live_logits


def test_lora_adaptation_injects_trainable_adapters_and_round_trips(
    tmp_path,
) -> None:
    """The public adaptation path must yield a loadable, trainable LoRA model."""
    source = _classifier()
    base_backbone_state = deepcopy(source.backbone.model.state_dict())
    plan = adaptation.resolve_adaptation_plan(
        config=source.config,
        requested_embedding_layer="auto",
        requested_adaptation="lora",
        unfreeze_last_n_blocks=0,
        lora_last_n_blocks=2,
    )
    assert plan.mode == "lora"
    assert plan.lora_enabled is True
    assert plan.partial_finetuning_enabled is False
    assert plan.requires_live_backbone_forward is True
    assert plan.feature_cache_enabled is False

    setup = adaptation.configure_adaptation(
        backbone=source.backbone,
        classifier=source,
        plan=plan,
        lora_target_modules=("q", "v"),
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
    )
    assert setup.lora_report is not None
    assert len(setup.lora_report.adapter_modules) > 0
    assert setup.lora_report.block_indices == plan.trainable_block_indices
    assert setup.lora_report.target_modules == ("q", "v")
    assert setup.lora_parameters
    assert setup.lora_parameter_ids
    assert setup.trainable_lora_parameter_count > 0
    assert setup.trainable_original_backbone_parameter_count == 0
    assert setup.trainable_backbone_parameters == []
    assert all(parameter.requires_grad for parameter in setup.lora_parameters)
    assert all(not parameter.requires_grad for parameter in source.backbone.parameters()
               if id(parameter) not in setup.lora_parameter_ids)
    assert all(parameter.requires_grad for parameter in source.head.parameters())
    assert runner.configure_adaptation is adaptation.configure_adaptation

    optimizer = build_optimizer(
        classifier=source,
        backbone=source.backbone,
        head_parameters=list(source.head.parameters()),
        trainable_backbone_parameters=setup.trainable_backbone_parameters,
        lora_parameters=setup.lora_parameters,
        lora_parameter_ids=setup.lora_parameter_ids,
        head_lr=1e-3,
        backbone_lr=2e-4,
        lora_lr=3e-4,
        weight_decay=1e-2,
        partial_enabled=False,
        lora_enabled=True,
    )
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    head_parameter_ids = {id(parameter) for parameter in source.head.parameters()}
    original_backbone_ids = {
        id(parameter)
        for parameter in source.backbone.parameters()
        if id(parameter) not in setup.lora_parameter_ids
    }
    assert len(optimizer.param_groups) == 2
    assert [group["lr"] for group in optimizer.param_groups] == [1e-3, 3e-4]
    assert all(group["weight_decay"] == 1e-2 for group in optimizer.param_groups)
    assert optimizer_parameter_ids == head_parameter_ids | setup.lora_parameter_ids
    assert not optimizer_parameter_ids & original_backbone_ids

    batch = _batch()
    labels = torch.tensor([0, 1], dtype=torch.long)
    optimizer.zero_grad(set_to_none=True)
    logits = adaptation.forward_live_logits(
        classifier=source,
        token_inputs=batch.token_inputs,
        token_channel_indices=batch.token_channel_indices,
        token_time_indices=batch.token_time_indices,
        token_valid_mask=batch.token_valid_mask,
        channel_valid_mask=batch.channel_valid_mask,
    )
    F.cross_entropy(logits, labels).backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in setup.lora_parameters
    )
    assert any(parameter.grad is not None for parameter in source.head.parameters())
    assert all(parameter.grad is None for parameter in source.backbone.parameters()
               if id(parameter) not in setup.lora_parameter_ids)
    optimizer.step()

    saved_lora_state = lora_state_dict(source.backbone.model)
    assert saved_lora_state
    assert any(key.endswith(".lora_A") for key in saved_lora_state)
    assert any(key.endswith(".lora_B") for key in saved_lora_state)
    checkpoint_path = tmp_path / "lora_head.pt"
    save_classifier_checkpoint(
        source,
        checkpoint_path,
        extra_metadata={
            "lora_block_indices": list(plan.trainable_block_indices),
            "lora_target_modules": ["q", "v"],
            "lora_rank": 2,
            "lora_alpha": 4.0,
            "lora_dropout": 0.0,
        },
        lora_state_dict=saved_lora_state,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["format_version"] == 4
    assert payload["lora_state_dict"]
    assert "head_state_dict" in payload
    assert "backbone_state_dict" not in payload

    loaded = _classifier()
    loaded.backbone.model.load_state_dict(base_backbone_state, strict=True)
    load_classifier_checkpoint(loaded, checkpoint_path)
    loaded_lora_state = lora_state_dict(loaded.backbone.model)
    assert loaded_lora_state.keys() == saved_lora_state.keys()
    for key, value in saved_lora_state.items():
        torch.testing.assert_close(loaded_lora_state[key], value)
    for key, value in source.head.state_dict().items():
        torch.testing.assert_close(loaded.head.state_dict()[key], value)
    with torch.no_grad():
        source_logits = adaptation.forward_live_logits(
            classifier=source,
            token_inputs=batch.token_inputs,
            token_channel_indices=batch.token_channel_indices,
            token_time_indices=batch.token_time_indices,
            token_valid_mask=batch.token_valid_mask,
            channel_valid_mask=batch.channel_valid_mask,
        )
        loaded_logits = adaptation.forward_live_logits(
            classifier=loaded,
            token_inputs=batch.token_inputs,
            token_channel_indices=batch.token_channel_indices,
            token_time_indices=batch.token_time_indices,
            token_valid_mask=batch.token_valid_mask,
            channel_valid_mask=batch.channel_valid_mask,
        )
    torch.testing.assert_close(loaded_logits, source_logits)
