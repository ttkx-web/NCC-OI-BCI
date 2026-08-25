from __future__ import annotations

import torch

from bci_dayloop.models.model_50m.backbone import Model50MBackbone
from bci_dayloop.models.model_50m.classifier import Model50MClassifier
from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.models.model_50m.tokenization import Model50MBatchedInput
from bci_dayloop.training.model_50m import adaptation, runner


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
