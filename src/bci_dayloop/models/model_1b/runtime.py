"""Runtime inference for one fixed-window 1B linear-head package."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from bci_dayloop.runtime.types import ModelOutput, RawEEGWindow

from .classifier import Model1BFlattenLinearHead, classifier_input_dim, flatten_token_embeddings
from .config import Model1BConfig
from .runner import Model1BBackboneRunner, Model1BPreparedInput


class Model1BRuntime:
    """A package-bound 1B runtime; its linear head fixes one window length."""

    def __init__(
        self,
        *,
        config: Model1BConfig,
        runner: Model1BBackboneRunner,
        head: Model1BFlattenLinearHead,
        class_names: Sequence[str],
    ) -> None:
        normalized_classes = tuple(str(name) for name in class_names)
        if len(normalized_classes) != head.num_classes or len(set(normalized_classes)) != len(normalized_classes):
            raise ValueError("1B runtime class_names must be unique and match the linear head")
        if head.input_dim != classifier_input_dim(config):
            raise ValueError("1B package head input dimension does not match its fixed window contract")
        self.config = config
        self.runner = runner
        self.head = head.to(runner.backbone.device_object).eval()
        self.class_names = normalized_classes
        if any(parameter.requires_grad for parameter in self.runner.backbone.parameters()):
            raise RuntimeError("1B runtime backbone must be frozen")

    @property
    def input_contract(self):
        return self.runner.input_transform.input_contract

    def prepare(self, raw_window: RawEEGWindow) -> Model1BPreparedInput:
        """Prepare only a raw window matching this Package's fixed duration."""
        return self.runner.prepare(raw_window)

    def predict_prepared(
        self,
        prepared: Model1BPreparedInput,
        return_features: bool = False,
    ) -> ModelOutput:
        prepared.validate(self.config)
        if prepared.num_time_patches != self.config.num_time_patches:
            raise ValueError("prepared 1B tokens do not match this Package window_seconds")
        embedding = self.runner.extract_embeddings(prepared)
        features = flatten_token_embeddings(
            embedding,
            prepared.token_valid_mask.to(embedding.device),
        )
        expected_features = (prepared.batch_size, classifier_input_dim(self.config))
        if tuple(features.shape) != expected_features:
            raise RuntimeError(
                "1B flattened features do not match this Package head: "
                f"expected {expected_features}, got {tuple(features.shape)}"
            )
        with torch.inference_mode():
            logits = self.head(features)
            probabilities = torch.softmax(logits, dim=-1)
        if logits.shape != (prepared.batch_size, len(self.class_names)):
            raise RuntimeError("1B runtime head produced an invalid logits shape")
        if not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all():
            raise RuntimeError("1B runtime produced NaN or Inf logits/probabilities")
        predicted = int(torch.argmax(probabilities[0]).item())
        confidence = float(probabilities[0, predicted].item())
        return ModelOutput(
            logits=logits,
            probabilities=probabilities,
            predicted_class=predicted,
            confidence=confidence,
            features=embedding if return_features else None,
            diagnostics={
                "model_type": "model_1b",
                "window_seconds": self.config.window_seconds,
                "num_time_patches": self.config.num_time_patches,
                "token_count": self.config.num_tokens,
                "classifier_input_dim": classifier_input_dim(self.config),
                "class_names": list(self.class_names),
            },
        )

    def predict(self, raw_window: RawEEGWindow, return_features: bool = False) -> ModelOutput:
        return self.predict_prepared(self.prepare(raw_window), return_features=return_features)


def build_1b_runtime(
    *,
    backbone_checkpoint: Path | str,
    head: Model1BFlattenLinearHead,
    class_names: Sequence[str],
    device: str,
    window_seconds: float,
    target_sample_rate: float,
    patch_seconds: float,
    patch_stride_seconds: float,
    filter_enabled: bool,
    filter_low_hz: float,
    filter_high_hz: float,
    filter_order: int,
    reference_mode: str,
    zscore_enabled: bool,
    zscore_eps: float,
    missing_channel_fill_value: float,
    strict_window_duration: bool,
    window_tolerance_seconds: float,
) -> Model1BRuntime:
    """Build a frozen 1B runtime from an already-validated package head."""
    config = Model1BConfig(
        checkpoint_path=backbone_checkpoint,
        device=device,
        target_sample_rate=target_sample_rate,
        window_seconds=window_seconds,
        patch_seconds=patch_seconds,
        patch_stride_seconds=patch_stride_seconds,
        filter_enabled=filter_enabled,
        filter_low_hz=filter_low_hz,
        filter_high_hz=filter_high_hz,
        filter_order=filter_order,
        reference_mode=reference_mode,
        zscore_enabled=zscore_enabled,
        zscore_eps=zscore_eps,
        missing_channel_fill_value=missing_channel_fill_value,
        strict_window_duration=strict_window_duration,
        window_tolerance_seconds=window_tolerance_seconds,
    )
    return Model1BRuntime(
        config=config,
        runner=Model1BBackboneRunner(config),
        head=head,
        class_names=class_names,
    )
