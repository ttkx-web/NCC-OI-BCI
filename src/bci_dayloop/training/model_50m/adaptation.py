"""Partial/LoRA configuration and live-token forward helpers for 50M training."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import TensorDataset

from bci_dayloop.data.hdf5_dataset import HDF5Metadata
from bci_dayloop.models.model_50m.classifier import Model50MClassifier
from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.models.model_50m.finetuning import (
    resolve_backbone_adaptation,
    resolve_embedding_layer,
    resolve_trainable_block_indices,
    uses_frozen_feature_cache,
)
from bci_dayloop.models.model_50m.lora import (
    inject_lora_adapters,
    load_lora_state_dict,
    lora_state_dict,
    normalize_lora_target_modules,
)
from bci_dayloop.models.model_50m.preprocessing import Model50MPreprocessor
from bci_dayloop.models.model_50m.tokenization import (
    Model50MBatchedInput,
    Model50MTokenizer,
    stack_model50m_tokens,
)
from bci_dayloop.training.model_50m.linear_head import WindowSet


@dataclass(frozen=True, slots=True)
class AdaptationPlan:
    """Resolved adaptation scope before the backbone is constructed."""

    embedding_layer: int
    mode: str
    trainable_block_indices: tuple[int, ...]
    partial_finetuning_enabled: bool
    lora_enabled: bool
    requires_live_backbone_forward: bool
    feature_cache_enabled: bool


@dataclass(slots=True)
class AdaptationSetup:
    """Trainable parameter groups selected for one constructed classifier."""

    lora_report: Any | None
    lora_parameters: list[torch.nn.Parameter]
    lora_parameter_ids: set[int]
    trainable_backbone_parameters: list[torch.nn.Parameter]
    selected_block_parameter_ids: set[int]
    original_backbone_parameter_count: int
    trainable_lora_parameter_count: int
    trainable_head_parameter_count: int
    trainable_original_backbone_parameter_count: int
    total_model_parameter_count: int
    total_trainable_parameter_count: int


def resolve_adaptation_plan(
    *,
    config: Model50MConfig,
    requested_embedding_layer: str,
    requested_adaptation: str | None,
    unfreeze_last_n_blocks: int,
    lora_last_n_blocks: int,
) -> AdaptationPlan:
    """Resolve the existing block selection and cache policy unchanged."""
    embedding_layer = resolve_embedding_layer(
        requested=requested_embedding_layer,
        output_layer_idx=config.output_layer_idx,
        depth=config.depth,
    )
    mode = resolve_backbone_adaptation(
        requested=requested_adaptation,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
    )
    partial_enabled = mode == "partial"
    lora_enabled = mode == "lora"
    if partial_enabled:
        blocks = resolve_trainable_block_indices(
            embedding_layer=embedding_layer,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )
    elif lora_enabled:
        blocks = resolve_trainable_block_indices(
            embedding_layer=embedding_layer,
            unfreeze_last_n_blocks=lora_last_n_blocks,
        )
    else:
        blocks = ()
    return AdaptationPlan(
        embedding_layer=embedding_layer,
        mode=mode,
        trainable_block_indices=blocks,
        partial_finetuning_enabled=partial_enabled,
        lora_enabled=lora_enabled,
        requires_live_backbone_forward=partial_enabled or lora_enabled,
        feature_cache_enabled=uses_frozen_feature_cache(
            unfreeze_last_n_blocks=0 if mode == "frozen" else 1
        ),
    )


def configure_adaptation(
    *,
    backbone: Any,
    classifier: Model50MClassifier,
    plan: AdaptationPlan,
    lora_target_modules: Sequence[str],
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
) -> AdaptationSetup:
    """Apply the established partial/LoRA scope and validate its parameters."""
    lora_report = None
    # The modes are mutually exclusive: both partial and LoRA need a live
    # backbone forward, but only LoRA injects adapter parametrizations.
    if plan.lora_enabled:
        normalized_lora_targets = normalize_lora_target_modules(lora_target_modules)
        lora_report = inject_lora_adapters(
            backbone,
            block_indices=plan.trainable_block_indices,
            target_modules=normalized_lora_targets,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
    elif plan.partial_finetuning_enabled:
        backbone.set_trainable_encoder_blocks(plan.trainable_block_indices)
    else:
        backbone.set_trainable_encoder_blocks(())
    classifier.eval()

    selected_block_parameter_ids = {
        id(parameter)
        for block_index in plan.trainable_block_indices
        for parameter in backbone.model.encoder.encoder.layers[block_index].parameters()
    }
    lora_parameters = (
        [
            parameter
            for adapter in lora_report.adapter_modules
            for parameter in adapter.parameters()
            if parameter.requires_grad
        ]
        if lora_report is not None
        else []
    )
    lora_parameter_ids = {id(parameter) for parameter in lora_parameters}
    if plan.lora_enabled:
        for parameter in backbone.parameters():
            if id(parameter) not in lora_parameter_ids:
                parameter.requires_grad = False
        trainable_backbone_parameters: list[torch.nn.Parameter] = []
    else:
        trainable_backbone_parameters = [
            parameter
            for parameter in backbone.parameters()
            if parameter.requires_grad and id(parameter) not in lora_parameter_ids
        ]
    original_backbone_parameter_count = sum(
        parameter.numel()
        for parameter in backbone.parameters()
        if id(parameter) not in lora_parameter_ids
    )
    trainable_lora_parameter_count = sum(parameter.numel() for parameter in lora_parameters)
    trainable_head_parameter_count = sum(parameter.numel() for parameter in classifier.head.parameters())
    trainable_original_backbone_parameter_count = sum(
        parameter.numel() for parameter in trainable_backbone_parameters
    )
    total_model_parameter_count = sum(parameter.numel() for parameter in classifier.parameters())
    total_trainable_parameter_count = (
        trainable_original_backbone_parameter_count
        + trainable_lora_parameter_count
        + trainable_head_parameter_count
    )
    if plan.partial_finetuning_enabled and (
        {id(parameter) for parameter in trainable_backbone_parameters}
        != selected_block_parameter_ids
    ):
        raise RuntimeError(
            "Backbone trainable parameters do not exactly match the selected "
            "encoder blocks."
        )
    if plan.partial_finetuning_enabled and (
        tuple(backbone.trainable_encoder_block_indices)
        != plan.trainable_block_indices
    ):
        raise RuntimeError("Backbone did not retain the requested trainable encoder blocks.")
    if plan.lora_enabled:
        if lora_report is None or not lora_report.adapter_modules:
            raise RuntimeError("LoRA adaptation did not inject any adapter modules.")
        if tuple(lora_report.block_indices) != plan.trainable_block_indices:
            raise RuntimeError("Injected LoRA block indices differ from the adaptation plan.")
        if lora_report.target_modules != normalized_lora_targets:
            raise RuntimeError("Injected LoRA targets differ from the requested targets.")
        if not lora_parameters or not lora_parameter_ids:
            raise RuntimeError("LoRA adaptation did not expose trainable adapter parameters.")
        if trainable_lora_parameter_count <= 0:
            raise RuntimeError("LoRA adaptation has no trainable adapter parameters.")
        if trainable_backbone_parameters or trainable_original_backbone_parameter_count:
            raise RuntimeError("LoRA adaptation must freeze all original backbone parameters.")
        if any(not parameter.requires_grad for parameter in lora_parameters):
            raise RuntimeError("LoRA adapter parameters must require gradients.")
        if any(
            parameter.requires_grad
            for parameter in backbone.parameters()
            if id(parameter) not in lora_parameter_ids
        ):
            raise RuntimeError("LoRA adaptation left an original backbone parameter trainable.")
        if any(not parameter.requires_grad for parameter in classifier.head.parameters()):
            raise RuntimeError("Classification head parameters must require gradients in LoRA mode.")
        saved_lora_state = lora_state_dict(backbone.model)
        if not saved_lora_state or not any(
            key.endswith(".lora_A") for key in saved_lora_state
        ) or not any(key.endswith(".lora_B") for key in saved_lora_state):
            raise RuntimeError("LoRA adaptation did not produce a complete adapter state.")
    return AdaptationSetup(
        lora_report=lora_report,
        lora_parameters=lora_parameters,
        lora_parameter_ids=lora_parameter_ids,
        trainable_backbone_parameters=trainable_backbone_parameters,
        selected_block_parameter_ids=selected_block_parameter_ids,
        original_backbone_parameter_count=original_backbone_parameter_count,
        trainable_lora_parameter_count=trainable_lora_parameter_count,
        trainable_head_parameter_count=trainable_head_parameter_count,
        trainable_original_backbone_parameter_count=trainable_original_backbone_parameter_count,
        total_model_parameter_count=total_model_parameter_count,
        total_trainable_parameter_count=total_trainable_parameter_count,
    )


def tokenize_windows_for_finetuning(
    *,
    window_set: WindowSet,
    metadata: HDF5Metadata,
    config: Model50MConfig,
    preprocess_batch_size: int,
    split_name: str,
    log_every: int,
) -> TensorDataset:
    """Stage token inputs on CPU while keeping every backbone forward live.

    Preprocessing/tokenization are fixed input transformations. Unlike the
    frozen feature cache, this dataset stores no backbone embeddings, so each
    training batch still executes the selected backbone layer with autograd.
    """
    if preprocess_batch_size <= 0:
        raise ValueError("preprocess_batch_size must be positive.")

    preprocessor = Model50MPreprocessor(config)
    tokenizer = Model50MTokenizer(config)
    token_input_chunks: list[torch.Tensor] = []
    channel_index_chunks: list[torch.Tensor] = []
    time_index_chunks: list[torch.Tensor] = []
    token_mask_chunks: list[torch.Tensor] = []
    channel_mask_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    split_start = time.perf_counter()

    for batch_start in range(0, len(window_set.windows), preprocess_batch_size):
        batch_end = min(batch_start + preprocess_batch_size, len(window_set.windows))
        tokenized_samples = []
        for raw_window in window_set.windows[batch_start:batch_end]:
            processed = preprocessor(
                signal=raw_window,
                channel_names=metadata.channel_names,
                original_sample_rate=metadata.sample_rate,
                input_unit=metadata.unit,
            )
            tokenized_samples.append(tokenizer(processed))

        batch = stack_model50m_tokens(tokenized_samples)
        token_input_chunks.append(batch.token_inputs.contiguous())
        channel_index_chunks.append(batch.token_channel_indices.contiguous())
        time_index_chunks.append(batch.token_time_indices.contiguous())
        token_mask_chunks.append(batch.token_valid_mask.contiguous())
        channel_mask_chunks.append(batch.channel_valid_mask.contiguous())
        label_chunks.append(
            torch.from_numpy(window_set.labels[batch_start:batch_end].copy()).long()
        )

        batch_number = batch_start // preprocess_batch_size + 1
        if (
            batch_number == 1
            or batch_end == len(window_set.windows)
            or batch_number % log_every == 0
        ):
            print(
                f"[TokenInputs] split={split_name} batch={batch_number} "
                f"samples={batch_end}/{len(window_set.windows)}",
                flush=True,
            )

    dataset = TensorDataset(
        torch.cat(token_input_chunks, dim=0).contiguous(),
        torch.cat(channel_index_chunks, dim=0).contiguous(),
        torch.cat(time_index_chunks, dim=0).contiguous(),
        torch.cat(token_mask_chunks, dim=0).contiguous(),
        torch.cat(channel_mask_chunks, dim=0).contiguous(),
        torch.cat(label_chunks, dim=0).contiguous(),
    )
    if len(dataset) != len(window_set.windows):
        raise RuntimeError(
            f"{split_name}: token input count {len(dataset)} does not match "
            f"window count {len(window_set.windows)}."
        )
    print(
        f"[TokenInputs] completed split={split_name} samples={len(dataset)} "
        f"time={time.perf_counter() - split_start:.1f}s",
        flush=True,
    )
    return dataset


def forward_live_logits(
    *,
    classifier: Model50MClassifier,
    token_inputs: torch.Tensor,
    token_channel_indices: torch.Tensor,
    token_time_indices: torch.Tensor,
    token_valid_mask: torch.Tensor,
    channel_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Run the adaptation forward without changing its autograd context."""
    batch = Model50MBatchedInput(
        token_inputs=token_inputs,
        token_channel_indices=token_channel_indices,
        token_time_indices=token_time_indices,
        token_valid_mask=token_valid_mask,
        channel_valid_mask=channel_valid_mask,
    ).to(classifier.device, non_blocking=True)
    return classifier(batch)
