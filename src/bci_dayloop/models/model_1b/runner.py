"""Reusable RawEEGWindow-to-embedding execution chain for the 1B backbone."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from bci_dayloop.preprocessing.canonical import SignalCanonicalizer
from bci_dayloop.runtime.types import PreparedModelInput, RawEEGWindow

from .backbone import Model1BBackbone
from .config import Model1BConfig
from .preprocessing import Model1BInputTransform
from .tokenization import Model1BBatchedInput, Model1BTokenizer


@dataclass(frozen=True, slots=True)
class Model1BPreparedInput:
    """Named, validated 1B encoder inputs produced from one raw EEG window.

    All token tensors are batch-major.  The runner deliberately retains the
    preprocessing result for traceability, but it exposes neither logits nor
    class-related fields.
    """

    token_inputs: torch.Tensor
    token_channel_indices: torch.Tensor
    token_time_indices: torch.Tensor
    token_valid_mask: torch.Tensor
    channel_valid_mask: torch.Tensor
    num_time_patches: int
    prepared_model_input: PreparedModelInput

    @property
    def batch_size(self) -> int:
        return int(self.token_inputs.shape[0])

    @property
    def num_tokens(self) -> int:
        return int(self.token_inputs.shape[1])

    @property
    def device(self) -> torch.device:
        return self.token_inputs.device

    @property
    def preprocessing_trace(self) -> tuple[str, ...]:
        """Expose the common Runtime prepared-input trace without mutation."""
        return tuple(
            str(step)
            for step in self.prepared_model_input.preprocessing_trace
        )

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Expose common Runtime observability fields for Replay and benchmark."""
        return {
            **self.prepared_model_input.diagnostics,
            "model_type": "model_1b",
            "num_time_patches": self.num_time_patches,
            "token_count": self.num_tokens,
        }

    def as_batched_input(self) -> Model1BBatchedInput:
        return Model1BBatchedInput(
            token_inputs=self.token_inputs,
            token_channel_indices=self.token_channel_indices,
            token_time_indices=self.token_time_indices,
            token_valid_mask=self.token_valid_mask,
            channel_valid_mask=self.channel_valid_mask,
        )

    def validate(self, config: Model1BConfig) -> None:
        expected_tokens = config.n_channels * config.num_time_patches
        if self.num_time_patches != config.num_time_patches:
            raise ValueError(
                "prepared 1B num_time_patches does not match its configuration"
            )
        if tuple(self.token_inputs.shape) != (
            self.batch_size,
            expected_tokens,
            config.patch_num_points,
        ):
            raise ValueError(
                "prepared 1B token_inputs must have shape "
                f"[B, {expected_tokens}, {config.patch_num_points}]"
            )
        expected_token_shape = (self.batch_size, expected_tokens)
        for name, tensor in (
            ("token_channel_indices", self.token_channel_indices),
            ("token_time_indices", self.token_time_indices),
            ("token_valid_mask", self.token_valid_mask),
        ):
            if tuple(tensor.shape) != expected_token_shape:
                raise ValueError(f"prepared 1B {name} has an invalid shape")
        if tuple(self.channel_valid_mask.shape) != (
            self.batch_size,
            config.n_channels,
        ):
            raise ValueError("prepared 1B channel_valid_mask has an invalid shape")
        tensors = (
            self.token_inputs,
            self.token_channel_indices,
            self.token_time_indices,
            self.token_valid_mask,
            self.channel_valid_mask,
        )
        if any(tensor.device != self.device for tensor in tensors):
            raise ValueError("prepared 1B tensors must all be on one device")
        if self.token_inputs.dtype != torch.float32:
            raise TypeError("prepared 1B token_inputs must be torch.float32")
        if self.token_channel_indices.dtype != torch.int64:
            raise TypeError("prepared 1B token_channel_indices must be torch.int64")
        if self.token_time_indices.dtype != torch.int64:
            raise TypeError("prepared 1B token_time_indices must be torch.int64")
        if self.token_valid_mask.dtype != torch.float32 or self.channel_valid_mask.dtype != torch.float32:
            raise TypeError("prepared 1B validity masks must be torch.float32")
        if not torch.isfinite(self.token_inputs).all():
            raise ValueError("prepared 1B token_inputs contains NaN or Inf")
        if not torch.isfinite(self.token_valid_mask).all() or not torch.isfinite(self.channel_valid_mask).all():
            raise ValueError("prepared 1B validity masks contain NaN or Inf")
        if self.token_channel_indices.numel() and (
            self.token_channel_indices.min() < 0
            or self.token_channel_indices.max() >= config.n_channels
        ):
            raise ValueError("prepared 1B token_channel_indices must be in [0, 63]")
        if self.token_time_indices.numel() and (
            self.token_time_indices.min() < 0
            or self.token_time_indices.max() >= config.model_n_time_patches
        ):
            raise ValueError("prepared 1B token_time_indices must be in [0, 9]")


class Model1BBackboneRunner:
    """Execute ``RawEEGWindow -> prepared tokens -> final encoder embedding``.

    The transform and tokenizer intentionally reuse NCC-OI-BCI's verified 50M
    implementations.  The public prepared-input and execution API, model
    configuration, strict checkpoint loader, and encoder reside in this 1B
    module.
    """

    def __init__(
        self,
        config: Model1BConfig,
        *,
        backbone: Model1BBackbone | Any | None = None,
    ) -> None:
        self.config = config
        self.canonicalizer = SignalCanonicalizer(target_unit="uV")
        self.input_transform = Model1BInputTransform(config)
        self.tokenizer = Model1BTokenizer(config)
        self.backbone = backbone if backbone is not None else Model1BBackbone(config)

    def prepare(self, raw_window: RawEEGWindow) -> Model1BPreparedInput:
        """Canonicalize, physically preprocess, and tokenize one raw window."""
        canonical = self.canonicalizer.transform(raw_window)
        actual_seconds = canonical.data.shape[1] / canonical.sample_rate
        if not math.isclose(
            actual_seconds,
            self.config.window_seconds,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "1B raw window must exactly match configured whole-second "
                f"duration {self.config.window_seconds:g}s; got {actual_seconds:.9g}s"
            )
        transformed = self.input_transform.transform(canonical)
        signal = transformed.get_tensor("signal")
        channel_valid_mask = transformed.get_tensor("channel_valid_mask")
        if signal.device.type != "cpu" or channel_valid_mask.device.type != "cpu":
            raise RuntimeError("1B preprocessing must produce CPU tensors before tokenization")
        tokenized = self.tokenizer.tokenize(
            signal=signal.squeeze(0).numpy(),
            channel_valid_mask=channel_valid_mask.squeeze(0).numpy(),
        ).as_batch()
        prepared = Model1BPreparedInput(
            token_inputs=tokenized.token_inputs,
            token_channel_indices=tokenized.token_channel_indices,
            token_time_indices=tokenized.token_time_indices,
            token_valid_mask=tokenized.token_valid_mask,
            channel_valid_mask=tokenized.channel_valid_mask,
            num_time_patches=self.config.num_time_patches,
            prepared_model_input=transformed,
        )
        prepared.validate(self.config)
        return prepared

    def extract_embeddings(self, prepared: Model1BPreparedInput) -> torch.Tensor:
        """Run the final (index 19) encoder layer in eval/inference mode."""
        prepared.validate(self.config)
        self.backbone.eval()
        with torch.inference_mode():
            embedding = self.backbone.extract_embeddings(prepared.as_batched_input())
        expected = (prepared.batch_size, prepared.num_tokens, self.config.d_model)
        backbone_device = self.backbone.device_object
        device_matches = (
            embedding.device.type == backbone_device.type
            and (backbone_device.index is None or embedding.device.index == backbone_device.index)
        )
        if (
            tuple(embedding.shape) != expected
            or embedding.dtype != torch.float32
            or not device_matches
            or not torch.isfinite(embedding).all()
        ):
            raise RuntimeError(
                "1B encoder embedding is invalid; expected "
                f"{expected} on {backbone_device}, got "
                f"{tuple(embedding.shape)} on {embedding.device}"
            )
        return embedding
