"""Checkpoint-strict 1B EEG encoder.

The deployment graph deliberately ends at the final Transformer token
embedding.  ``head.*`` in the pretraining checkpoint is TimeFreqTokenHead and
is explicitly ignored; no classification module is present here.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from .config import Model1BConfig
from .tokenization import Model1BBatchedInput


class PatchTokenizer(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, int(d_model)),
            nn.LayerNorm(int(d_model)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"1B token_inputs must be [B,S,{self.input_dim}], got {tuple(x.shape)}"
            )
        return self.proj(x)


class EEGTransformerEncoder(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, depth: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model), nhead=int(n_heads),
            dim_feedforward=int(d_model * mlp_ratio), dropout=float(dropout),
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(depth))

    def forward_until_layer(self, x: torch.Tensor, return_layer_idx: int) -> torch.Tensor:
        for index, layer in enumerate(self.encoder.layers):
            x = layer(x)
            if index == return_layer_idx:
                return x
        raise ValueError(f"1B return_layer_idx={return_layer_idx} is out of range")


class EEGBackbone1B(nn.Module):
    """The four formal checkpoint prefixes, excluding the pretraining head."""

    def __init__(self, config: Model1BConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = PatchTokenizer(config.patch_num_points, config.d_model, config.dropout)
        self.channel_embed = nn.Embedding(config.n_channels, config.d_model)
        self.time_embed = nn.Embedding(config.model_n_time_patches, config.d_model)
        self.encoder = EEGTransformerEncoder(
            d_model=config.d_model, n_heads=config.n_heads, depth=config.depth,
            mlp_ratio=config.mlp_ratio, dropout=config.dropout,
        )

    def forward(
        self,
        token_inputs: torch.Tensor,
        token_channel_indices: torch.Tensor,
        token_time_indices: torch.Tensor,
        *,
        return_layer_idx: int = 19,
    ) -> torch.Tensor:
        if token_inputs.dtype != torch.float32:
            raise TypeError("1B token_inputs must be torch.float32")
        if token_channel_indices.dtype != torch.int64 or token_time_indices.dtype != torch.int64:
            raise TypeError("1B token indices must be torch.int64")
        if token_inputs.ndim != 3 or token_channel_indices.shape != token_inputs.shape[:2] or token_time_indices.shape != token_inputs.shape[:2]:
            raise ValueError("1B token tensors must have shapes [B,S,100], [B,S], [B,S]")
        if token_inputs.shape[1] > self.config.n_channels * self.config.model_n_time_patches:
            raise ValueError("1B input token count exceeds ten checkpoint time positions")
        if token_channel_indices.numel() and (token_channel_indices.min() < 0 or token_channel_indices.max() >= self.config.n_channels):
            raise ValueError("1B token_channel_indices are outside [0, 63]")
        if token_time_indices.numel() and (token_time_indices.min() < 0 or token_time_indices.max() >= self.config.model_n_time_patches):
            raise ValueError("1B token_time_indices are outside [0, 9]")
        x = self.tokenizer(token_inputs)
        x = x + self.channel_embed(token_channel_indices) + self.time_embed(token_time_indices)
        return self.encoder.forward_until_layer(x, int(return_layer_idx))


@dataclass(frozen=True, slots=True)
class BackboneLoadReport:
    checkpoint_path: str
    checkpoint_source_key: str
    loaded_tensor_count: int
    loaded_parameter_count: int
    ignored_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    load_seconds: float
    device: str

    @property
    def ignored_pretraining_head_keys(self) -> tuple[str, ...]:
        """Compatibility spelling that makes the ignored head explicit."""
        return self.ignored_keys


class _ConfigPlaceholder:
    """Allows loading upstream's saved ``configs.Config`` without importing it."""

    def __setstate__(self, state: object) -> None:
        self.__dict__.update(state if isinstance(state, dict) else {"state": state})


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        if module == "configs" and name in {"Config", "DataConfig", "MaskConfig", "ModelConfig", "TrainConfig"}:
            return _ConfigPlaceholder
        return super().find_class(module, name)


class _CheckpointPickleModule:
    Unpickler = _CheckpointUnpickler
    load = staticmethod(pickle.load)
    dump = staticmethod(pickle.dump)
    HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL


def _safe_torch_load(path: Path, device: torch.device) -> Any:
    # Formal 1B checkpoints serialize configs.Config.  The compatibility
    # unpickler preserves only that metadata object while torch restores tensor
    # storages normally.  ``weights_only`` cannot decode that custom class.
    return torch.load(path, map_location=device, pickle_module=_CheckpointPickleModule)


def _extract_state_dict(checkpoint: Any) -> tuple[Mapping[str, torch.Tensor], str]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"1B checkpoint must be a mapping, got {type(checkpoint)!r}")
    value = checkpoint.get("model_state_dict")
    if not isinstance(value, Mapping) or not value:
        raise KeyError("1B checkpoint must contain non-empty model_state_dict")
    if not all(isinstance(key, str) and torch.is_tensor(tensor) for key, tensor in value.items()):
        raise TypeError("1B model_state_dict must contain only string tensor keys")
    return value, "model_state_dict"


def load_backbone_checkpoint(model: EEGBackbone1B, checkpoint_path: Path | str, *, device: torch.device) -> BackboneLoadReport:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"1B checkpoint was not found: {path}")
    started = time.perf_counter()
    # Keep checkpoint tensors on CPU while loading.  Mapping this 3.8 GB file
    # directly to CUDA would temporarily require a second full GPU copy in
    # addition to the already-constructed 1B model.
    raw_state_dict, source_key = _extract_state_dict(
        _safe_torch_load(path, torch.device("cpu"))
    )
    target = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    head_keys: list[str] = []
    unexpected: list[str] = []
    for key, value in raw_state_dict.items():
        if key.startswith("head."):
            head_keys.append(key)
        elif key in target:
            compatible[key] = value
        else:
            unexpected.append(key)
    missing = sorted(set(target) - set(compatible))
    if unexpected or missing:
        raise RuntimeError(
            "1B checkpoint must load tokenizer/channel_embed/time_embed/encoder exactly; "
            f"unexpected={unexpected[:20]}, missing={missing[:20]}"
        )
    if not head_keys:
        raise RuntimeError("1B checkpoint has no head.* keys to identify as ignored TimeFreqTokenHead")
    try:
        model.load_state_dict(compatible, strict=True)
    except RuntimeError as error:
        raise RuntimeError(f"1B checkpoint shapes do not match formal architecture: {path}") from error
    return BackboneLoadReport(
        checkpoint_path=str(path), checkpoint_source_key=source_key,
        loaded_tensor_count=len(compatible),
        loaded_parameter_count=sum(tensor.numel() for tensor in compatible.values()),
        ignored_keys=tuple(sorted(head_keys)), missing_keys=tuple(missing),
        load_seconds=time.perf_counter() - started, device=str(device),
    )


def resolve_device(requested: str | torch.device) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def _matches_requested_device(actual: torch.device, requested: torch.device) -> bool:
    """Treat ``cuda`` as the current CUDA device, as Tensor.to(cuda) does."""
    if actual.type != requested.type:
        return False
    return requested.index is None or actual.index == requested.index


class Model1BBackbone(nn.Module):
    """Checkpoint-backed wrapper that exposes final token embeddings, never logits."""

    def __init__(self, config: Model1BConfig, *, load_checkpoint: bool = True) -> None:
        super().__init__()
        self.config = config
        self.device_object = resolve_device(config.device)
        self.model = EEGBackbone1B(config).to(self.device_object)
        self.load_report: BackboneLoadReport | None = None
        if load_checkpoint:
            self.load_report = load_backbone_checkpoint(self.model, config.checkpoint_path, device=self.device_object)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.eval()

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def _validate_batch(self, batch: Model1BBatchedInput) -> None:
        batch.validate()
        if batch.num_tokens != self.config.num_tokens or batch.patch_len != self.config.patch_num_points:
            raise ValueError(
                f"1B batch must be [B,{self.config.num_tokens},{self.config.patch_num_points}] for {self.config.window_seconds}s"
            )
        tensors = (
            batch.token_inputs,
            batch.token_channel_indices,
            batch.token_time_indices,
            batch.token_valid_mask,
            batch.channel_valid_mask,
        )
        if any(tensor.device != batch.token_inputs.device for tensor in tensors):
            raise ValueError("all 1B prepared token tensors must be on the same device")
        if batch.token_inputs.dtype != torch.float32:
            raise TypeError("1B token_inputs must be torch.float32")
        if batch.token_channel_indices.dtype != torch.int64:
            raise TypeError("1B token_channel_indices must be torch.int64")
        if batch.token_time_indices.dtype != torch.int64:
            raise TypeError("1B token_time_indices must be torch.int64")
        if batch.token_valid_mask.dtype != torch.float32 or batch.channel_valid_mask.dtype != torch.float32:
            raise TypeError("1B validity masks must be torch.float32")
        if not torch.isfinite(batch.token_inputs).all():
            raise ValueError("1B token_inputs contains NaN or Inf")
        if batch.channel_valid_mask.shape != (batch.batch_size, self.config.n_channels):
            raise ValueError(
                "1B channel_valid_mask must have shape "
                f"[{batch.batch_size}, {self.config.n_channels}]"
            )

    def extract_embeddings(self, batch: Model1BBatchedInput) -> torch.Tensor:
        """Return final-layer token embeddings with shape ``[B, S, 2048]``."""
        self._validate_batch(batch)
        batch = batch.to(self.device_object)
        self.model.eval()
        with torch.inference_mode():
            embeddings = self.model(
                batch.token_inputs, batch.token_channel_indices, batch.token_time_indices,
                return_layer_idx=self.config.output_layer_idx,
            )
        expected = (batch.batch_size, self.config.num_tokens, self.config.d_model)
        if (
            tuple(embeddings.shape) != expected
            or embeddings.dtype != torch.float32
            or not _matches_requested_device(embeddings.device, self.device_object)
            or not torch.isfinite(embeddings).all()
        ):
            raise RuntimeError(
                "1B final encoder output is invalid; "
                f"expected shape={expected}, dtype=torch.float32, device={self.device_object}; "
                f"got shape={tuple(embeddings.shape)}, dtype={embeddings.dtype}, "
                f"device={embeddings.device}, finite={bool(torch.isfinite(embeddings).all())}"
            )
        return embeddings

    def forward(self, batch: Model1BBatchedInput) -> torch.Tensor:
        return self.extract_embeddings(batch)
