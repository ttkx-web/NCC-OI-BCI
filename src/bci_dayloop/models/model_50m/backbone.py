from __future__ import annotations

import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

try:
    from .config import Model50MConfig
    from .tokenization import Model50MBatchedInput
except ImportError:
    # 允许直接运行当前文件调试。
    from config import Model50MConfig
    from tokenization import Model50MBatchedInput


# ======================================================================
# 原始 50M Backbone 结构
# ======================================================================


class PatchTokenizer(nn.Module):
    """
    将每个 channel-time Patch 投影为 Transformer Token。

    输入：
        x: [B, S, L]

    输出：
        token_embeddings: [B, S, d_model]
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = int(input_dim)
        self.d_model = int(d_model)

        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "PatchTokenizer input must have shape [B, S, L], "
                f"got {tuple(x.shape)}."
            )

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                "Patch length mismatch: "
                f"expected {self.input_dim}, got {x.shape[-1]}."
            )

        return self.proj(x)


class EEGTransformerEncoder(nn.Module):
    """
    与原 50M 仓库一致的 Transformer Encoder。

    支持：
    - 完整前向；
    - 返回指定 Block 后的 Token Embedding。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        depth: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.depth = int(depth)

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by "
                f"n_heads={self.n_heads}."
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=int(self.d_model * float(mlp_ratio)),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.depth,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward_until_layer(
        self,
        x: torch.Tensor,
        return_layer_idx: int,
    ) -> torch.Tensor:
        """
        返回第 return_layer_idx 个 Block 后的输出。

        Args:
            x:
                [B, S, d_model]

            return_layer_idx:
                0-based Transformer Block 编号。

        Returns:
            [B, S, d_model]
        """
        return_layer_idx = int(return_layer_idx)

        if not 0 <= return_layer_idx < self.depth:
            raise ValueError(
                f"Invalid return_layer_idx={return_layer_idx}. "
                f"Expected range [0, {self.depth - 1}]."
            )

        for layer_idx, layer in enumerate(self.encoder.layers):
            x = layer(x)

            if layer_idx == return_layer_idx:
                return x

        # 正常情况下不会到这里。
        raise RuntimeError(
            f"Transformer layer {return_layer_idx} was not reached."
        )


class EEGBackbone50M(nn.Module):
    """
    50M 模型的部署 Backbone。

    保留与原始 checkpoint 一致的模块名称：

        tokenizer
        channel_embed
        time_embed
        encoder

    因此原预训练 checkpoint 中这些权重可以直接加载。
    """

    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int,
        n_heads: int,
        depth: int,
        mlp_ratio: float,
        dropout: float,
        n_channels: int,
        n_time_patches: int,
    ) -> None:
        super().__init__()

        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.depth = int(depth)
        self.n_channels = int(n_channels)
        self.n_time_patches = int(n_time_patches)

        self.tokenizer = PatchTokenizer(
            input_dim=self.input_dim,
            d_model=self.d_model,
            dropout=dropout,
        )

        self.channel_embed = nn.Embedding(
            num_embeddings=self.n_channels,
            embedding_dim=self.d_model,
        )

        self.time_embed = nn.Embedding(
            num_embeddings=self.n_time_patches,
            embedding_dim=self.d_model,
        )

        self.encoder = EEGTransformerEncoder(
            d_model=self.d_model,
            n_heads=n_heads,
            depth=self.depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def embed_tokens(
        self,
        token_inputs: torch.Tensor,
        token_channel_indices: torch.Tensor,
        token_time_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Patch Token + 通道位置编码 + 时间位置编码。

        Args:
            token_inputs:
                [B, S, L]，float32。

            token_channel_indices:
                [B, S]，int64。

            token_time_indices:
                [B, S]，int64。

        Returns:
            [B, S, d_model]
        """
        token_embeddings = self.tokenizer(token_inputs)

        channel_position = self.channel_embed(
            token_channel_indices
        )

        time_position = self.time_embed(
            token_time_indices
        )

        return (
            token_embeddings
            + channel_position
            + time_position
        )

    def extract_token_embeddings(
        self,
        token_inputs: torch.Tensor,
        token_channel_indices: torch.Tensor,
        token_time_indices: torch.Tensor,
        return_layer_idx: int | None = None,
    ) -> torch.Tensor:
        """
        提取指定 Transformer Block 后的 Token Embedding。

        Returns:
            [B, S, d_model]
        """
        if return_layer_idx is None:
            return_layer_idx = self.depth - 1

        x = self.embed_tokens(
            token_inputs=token_inputs,
            token_channel_indices=token_channel_indices,
            token_time_indices=token_time_indices,
        )

        return self.encoder.forward_until_layer(
            x=x,
            return_layer_idx=int(return_layer_idx),
        )

    def forward(
        self,
        token_inputs: torch.Tensor,
        token_channel_indices: torch.Tensor,
        token_time_indices: torch.Tensor,
        return_layer_idx: int | None = None,
    ) -> torch.Tensor:
        """
        部署时 forward 直接返回 Token Embedding，
        不执行预训练重建 Head。
        """
        return self.extract_token_embeddings(
            token_inputs=token_inputs,
            token_channel_indices=token_channel_indices,
            token_time_indices=token_time_indices,
            return_layer_idx=return_layer_idx,
        )


# ======================================================================
# Checkpoint 加载
# ======================================================================


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
    def is_complete(self) -> bool:
        return len(self.missing_keys) == 0


def _safe_torch_load(
    checkpoint_path: Path,
    map_location: torch.device,
) -> Any:
    """
    加载只包含 tensor 和普通 Python 数据结构的部署 checkpoint。
    """
    try:
        return torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        # 兼容不支持 weights_only 参数的旧版 PyTorch。
        return torch.load(
            checkpoint_path,
            map_location=map_location,
        )


def _looks_like_state_dict(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False

    if not value:
        return False

    has_tensor = False

    for key, item in value.items():
        if not isinstance(key, str):
            return False

        if torch.is_tensor(item):
            has_tensor = True

    return has_tensor


def _extract_state_dict(
    checkpoint: Any,
) -> tuple[Mapping[str, torch.Tensor], str]:
    """
    支持 50M 仓库当前使用的几种 checkpoint 格式。

    优先级：
        backbone_state_dict
        model_state_dict
        state_dict
        checkpoint 本身
    """
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Checkpoint must be a mapping or state_dict, "
            f"got {type(checkpoint)!r}."
        )

    for key in (
        "backbone_state_dict",
        "model_state_dict",
        "state_dict",
    ):
        candidate = checkpoint.get(key)

        if _looks_like_state_dict(candidate):
            return candidate, key

    if _looks_like_state_dict(checkpoint):
        return checkpoint, "<checkpoint_root>"

    raise KeyError(
        "Cannot find model weights in checkpoint. "
        "Expected backbone_state_dict, model_state_dict, "
        "state_dict, or a raw state_dict."
    )


def _strip_runtime_prefixes(key: str) -> str:
    """
    去除 DDP、torch.compile 等产生的外层前缀。
    """
    prefixes = (
        "module.",
        "_orig_mod.",
    )

    changed = True

    while changed:
        changed = False

        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True

    return key


def _normalize_backbone_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    将不同 checkpoint 格式统一成 Backbone 自身的 key。

    支持：
        tokenizer.*
        module.tokenizer.*
        backbone.tokenizer.*
        module.backbone.tokenizer.*
    """
    normalized: dict[str, torch.Tensor] = {}

    for original_key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue

        key = _strip_runtime_prefixes(original_key)

        # 下游完整模型通常保存为 backbone.xxx。
        if key.startswith("backbone."):
            key = key[len("backbone."):]

        normalized[key] = value

    return normalized


def load_backbone_checkpoint(
    model: EEGBackbone50M,
    checkpoint_path: Path | str,
    *,
    device: torch.device,
    allow_partial: bool = False,
) -> BackboneLoadReport:
    """
    将预训练 checkpoint 加载到部署 Backbone。

    预训练 checkpoint 中的 head.* 会被忽略；
    tokenizer/channel_embed/time_embed/encoder 必须完整加载。
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"50M checkpoint was not found: {checkpoint_path}"
        )

    if not checkpoint_path.is_file():
        raise ValueError(
            f"Checkpoint path is not a file: {checkpoint_path}"
        )

    start_time = time.perf_counter()

    checkpoint = _safe_torch_load(
        checkpoint_path=checkpoint_path,
        map_location=device,
    )

    raw_state_dict, source_key = _extract_state_dict(
        checkpoint
    )

    normalized_state_dict = _normalize_backbone_state_dict(
        raw_state_dict
    )

    target_state_dict = model.state_dict()
    target_keys = set(target_state_dict.keys())

    compatible_state_dict: dict[str, torch.Tensor] = {}
    ignored_keys: list[str] = []

    for key, value in normalized_state_dict.items():
        if key in target_keys:
            compatible_state_dict[key] = value
        else:
            ignored_keys.append(key)

    missing_keys = sorted(
        target_keys - set(compatible_state_dict.keys())
    )

    if not compatible_state_dict:
        raise RuntimeError(
            "No Backbone parameters matched the checkpoint. "
            "Please check whether the checkpoint belongs to the "
            "current 50M architecture."
        )

    if missing_keys and not allow_partial:
        preview = "\n".join(
            f"  - {key}" for key in missing_keys[:20]
        )

        raise RuntimeError(
            "Checkpoint is missing parameters required by the "
            "50M Backbone.\n"
            f"Missing key count: {len(missing_keys)}\n"
            f"{preview}\n"
            "If this is intentional, call with allow_partial=True."
        )

    try:
        model.load_state_dict(
            compatible_state_dict,
            strict=not allow_partial,
        )
    except RuntimeError as error:
        raise RuntimeError(
            "Failed to load the 50M checkpoint. "
            "The checkpoint tensor shapes do not match the current "
            "Model50MConfig. Check d_model, depth, n_heads, "
            "n_channels, patch length and time-patch count.\n"
            f"Checkpoint: {checkpoint_path}"
        ) from error

    loaded_parameter_count = sum(
        tensor.numel()
        for tensor in compatible_state_dict.values()
    )

    elapsed = time.perf_counter() - start_time

    return BackboneLoadReport(
        checkpoint_path=str(checkpoint_path),
        checkpoint_source_key=source_key,
        loaded_tensor_count=len(compatible_state_dict),
        loaded_parameter_count=int(loaded_parameter_count),
        ignored_keys=tuple(sorted(ignored_keys)),
        missing_keys=tuple(missing_keys),
        load_seconds=float(elapsed),
        device=str(device),
    )


# ======================================================================
# 部署侧总包装
# ======================================================================


def resolve_device(
    requested_device: str | torch.device,
) -> torch.device:
    requested = str(requested_device).lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")

        return torch.device("cpu")

    device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() "
            "is False. Use device='cpu', 'mps' or 'auto'."
        )

    if device.type == "mps":
        mps_available = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )

        if not mps_available:
            raise RuntimeError(
                "MPS was requested, but it is not available."
            )

    return device


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elif device.type == "mps":
        if hasattr(torch, "mps"):
            torch.mps.synchronize()


class Model50MBackbone(nn.Module):
    """
    Pipeline 使用的 50M Backbone 封装。

    输入：
        Model50MBatchedInput

    输出：
        token_embeddings [B, S, d_model]

    默认 10 秒配置：
        输入：[B, 640, 100]
        输出：[B, 640, 512]
    """

    def __init__(
        self,
        config: Model50MConfig,
        *,
        load_checkpoint: bool = True,
        freeze: bool = True,
        allow_partial_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self.config = config
        self.device_object = resolve_device(config.device)

        # 当前 10 秒配置中，该值为 10。
        #
        # 后续改为 4 秒时，建议在 config 中额外加入：
        #     model_n_time_patches = 10
        #
        # 使模型仍加载原 checkpoint 的 10 个 time embedding，
        # 但实际输入只使用 0、1、2、3 四个时间位置。
        self.model_n_time_patches = int(
            getattr(
                config,
                "model_n_time_patches",
                config.num_time_patches,
            )
        )

        self.model = EEGBackbone50M(
            input_dim=config.patch_num_points,
            d_model=config.d_model,
            n_heads=config.n_heads,
            depth=config.depth,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            n_channels=config.n_channels,
            n_time_patches=self.model_n_time_patches,
        )

        self.model.to(self.device_object)

        self.load_report: BackboneLoadReport | None = None
        self._frozen = False
        self._trainable_encoder_block_indices: tuple[int, ...] = ()

        if load_checkpoint:
            self.load_report = load_backbone_checkpoint(
                model=self.model,
                checkpoint_path=config.checkpoint_path,
                device=self.device_object,
                allow_partial=allow_partial_checkpoint,
            )

        if freeze:
            self.freeze()
        else:
            self.unfreeze()

    @property
    def device(self) -> torch.device:
        return self.device_object

    @property
    def output_dim(self) -> int:
        return int(self.config.d_model)

    @property
    def num_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.model.parameters()
        )

    @property
    def trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )

    def freeze(self) -> "Model50MBackbone":
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self._frozen = True
        self._trainable_encoder_block_indices = ()
        # 同时更新 wrapper 和内部 checkpoint 模型的 module mode，确保
        # 在线冻结时不会出现 wrapper.training 与 model.training 不一致。
        self.train(False)
        return self

    def unfreeze(self) -> "Model50MBackbone":
        for parameter in self.model.parameters():
            parameter.requires_grad = True

        self._frozen = False
        self._trainable_encoder_block_indices = tuple(range(self.config.depth))
        self.train(True)
        return self

    @property
    def trainable_encoder_block_indices(self) -> tuple[int, ...]:
        """0-based encoder blocks whose parameters are trainable."""
        return self._trainable_encoder_block_indices

    def set_trainable_encoder_blocks(
        self,
        block_indices: Sequence[int],
    ) -> "Model50MBackbone":
        """Freeze the backbone except for the specified encoder blocks.

        This is intentionally narrower than :meth:`unfreeze`: patch/token
        embeddings and all blocks outside ``block_indices`` remain frozen.
        ``forward`` also stops using ``torch.no_grad`` when at least one block
        is trainable, preserving the autograd graph needed by those blocks.
        """
        normalized_indices = tuple(sorted({int(index) for index in block_indices}))
        invalid_indices = [
            index
            for index in normalized_indices
            if not 0 <= index < self.config.depth
        ]
        if invalid_indices:
            raise ValueError(
                "Encoder block indices must be in "
                f"[0, {self.config.depth - 1}], got {invalid_indices}."
            )

        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for index in normalized_indices:
            for parameter in self.model.encoder.encoder.layers[index].parameters():
                parameter.requires_grad = True

        self._trainable_encoder_block_indices = normalized_indices
        self._frozen = not normalized_indices
        self.train(self.training)
        return self

    def train(self, mode: bool = True) -> "Model50MBackbone":
        super().train(mode)

        # Frozen linear probe keeps the entire backbone deterministic. During
        # partial fine-tuning only the explicitly selected encoder blocks
        # follow train/eval mode; all other backbone modules remain in eval.
        if self._frozen:
            super().train(False)
            self.model.eval()
        elif self._trainable_encoder_block_indices:
            self.model.eval()
            for index in self._trainable_encoder_block_indices:
                self.model.encoder.encoder.layers[index].train(mode)

        return self

    def _validate_batch(
        self,
        batch: Model50MBatchedInput,
    ) -> None:
        batch.validate()

        if batch.token_inputs.dtype != torch.float32:
            raise TypeError(
                "token_inputs must be torch.float32, "
                f"got {batch.token_inputs.dtype}."
            )

        if batch.token_channel_indices.dtype != torch.int64:
            raise TypeError(
                "token_channel_indices must be torch.int64, "
                f"got {batch.token_channel_indices.dtype}."
            )

        if batch.token_time_indices.dtype != torch.int64:
            raise TypeError(
                "token_time_indices must be torch.int64, "
                f"got {batch.token_time_indices.dtype}."
            )

        if batch.token_valid_mask.dtype != torch.float32:
            raise TypeError(
                "token_valid_mask must be torch.float32, "
                f"got {batch.token_valid_mask.dtype}."
            )

        expected_tokens = self.config.num_tokens

        if batch.num_tokens != expected_tokens:
            raise ValueError(
                "Unexpected number of EEG Tokens: "
                f"expected {expected_tokens}, "
                f"got {batch.num_tokens}."
            )

        if batch.patch_len != self.config.patch_num_points:
            raise ValueError(
                "Unexpected Patch length: "
                f"expected {self.config.patch_num_points}, "
                f"got {batch.patch_len}."
            )

        if batch.token_channel_indices.numel() > 0:
            max_channel = int(
                batch.token_channel_indices.max().item()
            )
            min_channel = int(
                batch.token_channel_indices.min().item()
            )

            if min_channel < 0:
                raise ValueError(
                    "token_channel_indices contains a negative value."
                )

            if max_channel >= self.config.n_channels:
                raise ValueError(
                    "Channel index exceeds channel embedding range: "
                    f"max={max_channel}, "
                    f"n_channels={self.config.n_channels}."
                )

        if batch.token_time_indices.numel() > 0:
            max_time = int(
                batch.token_time_indices.max().item()
            )
            min_time = int(
                batch.token_time_indices.min().item()
            )

            if min_time < 0:
                raise ValueError(
                    "token_time_indices contains a negative value."
                )

            if max_time >= self.model_n_time_patches:
                raise ValueError(
                    "Time index exceeds time embedding range: "
                    f"max={max_time}, "
                    f"model_n_time_patches="
                    f"{self.model_n_time_patches}."
                )

    def forward(
        self,
        batch: Model50MBatchedInput,
        *,
        return_layer_idx: int | None = None,
    ) -> torch.Tensor:
        """
        提取 Token Embedding。

        token_valid_mask 不在 Backbone 内参与计算，
        后续 Flatten 或 Mean Pooling 时再使用。
        """
        self._validate_batch(batch)

        batch = batch.to(self.device_object)

        if return_layer_idx is None:
            return_layer_idx = self.config.output_layer_idx

        context = (
            torch.no_grad()
            if self._frozen
            else nullcontext()
        )

        with context:
            token_embeddings = (
                self.model.extract_token_embeddings(
                    token_inputs=batch.token_inputs,
                    token_channel_indices=(
                        batch.token_channel_indices
                    ),
                    token_time_indices=batch.token_time_indices,
                    return_layer_idx=return_layer_idx,
                )
            )

        expected_shape = (
            batch.batch_size,
            batch.num_tokens,
            self.config.d_model,
        )

        if tuple(token_embeddings.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected Backbone output shape: "
                f"expected {expected_shape}, "
                f"got {tuple(token_embeddings.shape)}."
            )

        if not torch.isfinite(token_embeddings).all():
            raise RuntimeError(
                "50M Backbone output contains NaN or Inf."
            )

        return token_embeddings

    def extract_embeddings(
        self,
        batch: Model50MBatchedInput,
        *,
        return_layer_idx: int | None = None,
    ) -> torch.Tensor:
        """forward() 的语义化别名。"""
        return self.forward(
            batch=batch,
            return_layer_idx=return_layer_idx,
        )

    def health_check(self) -> dict[str, Any]:
        """
        使用全零输入完成一次 Backbone Smoke Test。

        这里只检查模型结构和前向，不检查真实 EEG 效果。
        """
        batch_size = 1
        num_tokens = self.config.num_tokens
        patch_len = self.config.patch_num_points
        num_time_patches = self.config.num_time_patches

        token_inputs = torch.zeros(
            batch_size,
            num_tokens,
            patch_len,
            dtype=torch.float32,
        )

        token_channel_indices = (
            torch.arange(
                self.config.n_channels,
                dtype=torch.long,
            )
            .repeat_interleave(num_time_patches)
            .unsqueeze(0)
        )

        token_time_indices = (
            torch.arange(
                num_time_patches,
                dtype=torch.long,
            )
            .repeat(self.config.n_channels)
            .unsqueeze(0)
        )

        token_valid_mask = torch.ones(
            batch_size,
            num_tokens,
            dtype=torch.float32,
        )

        channel_valid_mask = torch.ones(
            batch_size,
            self.config.n_channels,
            dtype=torch.float32,
        )

        smoke_batch = Model50MBatchedInput(
            token_inputs=token_inputs,
            token_channel_indices=token_channel_indices,
            token_time_indices=token_time_indices,
            token_valid_mask=token_valid_mask,
            channel_valid_mask=channel_valid_mask,
        )

        _synchronize_device(self.device_object)
        start_time = time.perf_counter()

        output = self.forward(smoke_batch)

        _synchronize_device(self.device_object)
        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        return {
            "status": "ok",
            "device": str(self.device_object),
            "input_shape": tuple(token_inputs.shape),
            "output_shape": tuple(output.shape),
            "output_layer_idx": self.config.output_layer_idx,
            "num_parameters": self.num_parameters,
            "trainable_parameters": self.trainable_parameters,
            "forward_ms": float(elapsed_ms),
            "checkpoint_loaded": self.load_report is not None,
            "checkpoint_path": (
                self.load_report.checkpoint_path
                if self.load_report is not None
                else None
            ),
        }


def build_model50m_backbone(
    config: Model50MConfig,
    *,
    freeze: bool = True,
    allow_partial_checkpoint: bool = False,
) -> Model50MBackbone:
    """构建并加载 50M Backbone。"""
    return Model50MBackbone(
        config=config,
        load_checkpoint=True,
        freeze=freeze,
        allow_partial_checkpoint=allow_partial_checkpoint,
    )
