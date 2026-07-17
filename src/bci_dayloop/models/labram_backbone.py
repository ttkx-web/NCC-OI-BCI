"""PyTorch-only LaBraM Base backbone compatible with the official checkpoint.

Architecture and state-dict naming follow the official MIT-licensed LaBraM
`modeling_finetune.py` implementation. The local implementation removes the
runtime dependency on legacy timm registration utilities while preserving the
`labram_base_patch200_200` constructor and `[B,C,A,200]` input contract.
"""

from __future__ import annotations

import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F


def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: int | None = None,
        qk_norm: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = attn_head_dim or dim // num_heads
        all_head_dim = head_dim * num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        self.q_norm = qk_norm(head_dim) if qk_norm is not None else None
        self.k_norm = qk_norm(head_dim) if qk_norm is not None else None
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rel_pos_bias: torch.Tensor | None = None) -> torch.Tensor:
        batch, tokens, _ = x.shape
        qkv_bias = None
        if self.q_bias is not None and self.v_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(batch, tokens, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)
        attention = (q * self.scale) @ k.transpose(-2, -1)
        if rel_pos_bias is not None:
            attention = attention + rel_pos_bias
        attention = self.attn_drop(attention.softmax(dim=-1))
        x = (attention @ v).transpose(1, 2).reshape(batch, tokens, -1)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        init_values: float | None = None,
        attn_head_dim: int | None = None,
        qk_norm: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, attn_head_dim, qk_norm)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)
        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(self, x: torch.Tensor, rel_pos_bias: torch.Tensor | None = None) -> torch.Tensor:
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class TemporalConv(nn.Module):
    def __init__(self, in_chans: int = 1, out_chans: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, patches, samples = x.shape
        x = x.reshape(batch, channels * patches, samples).unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        return x.permute(0, 2, 3, 1).reshape(batch, channels * patches, -1)


class NeuralTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int = 200,
        out_chans: int = 8,
        num_classes: int = 0,
        embed_dim: int = 200,
        depth: int = 12,
        num_heads: int = 10,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        qk_norm: type[nn.Module] | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        init_values: float | None = None,
        use_abs_pos_emb: bool = True,
        num_patches_per_channel_input: int = 4,
        use_mean_pooling: bool = True,
        init_scale: float = 0.001,
        **_: object,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.time_window = num_patches_per_channel_input
        self.patch_embed = TemporalConv(out_chans=out_chans)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 129, embed_dim)) if use_abs_pos_emb else None
        self.time_embed = nn.Parameter(torch.zeros(1, num_patches_per_channel_input, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    qk_scale,
                    drop_rate,
                    attn_drop_rate,
                    rates[index],
                    norm_layer,
                    init_values,
                    None,
                    qk_norm,
                )
                for index in range(depth)
            ]
        )
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward_features(
        self,
        x: torch.Tensor,
        input_chans: torch.Tensor | None = None,
        return_patch_tokens: bool = False,
        return_all_tokens: bool = False,
    ) -> torch.Tensor:
        batch, channels, patches, samples = x.shape
        if samples != self.patch_size:
            raise ValueError(f"LaBraM requires patch size {self.patch_size}, got {samples}")
        if patches > self.time_embed.shape[1]:
            raise ValueError(f"Input has {patches} patches, model supports {self.time_embed.shape[1]}")
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(batch, -1, -1), x), dim=1)
        if self.pos_embed is not None:
            used = self.pos_embed if input_chans is None else self.pos_embed[:, input_chans]
            position = used[:, 1:].unsqueeze(2).expand(batch, -1, patches, -1).flatten(1, 2)
            x = x + torch.cat((used[:, :1].expand(batch, -1, -1), position), dim=1)
        temporal = self.time_embed[:, :patches].unsqueeze(1).expand(batch, channels, -1, -1).flatten(1, 2)
        x[:, 1:] += temporal
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(x)
            tokens = self.fc_norm(x[:, 1:])
            return tokens if return_patch_tokens else tokens.mean(dim=1)
        if return_all_tokens:
            return x
        return x[:, 1:] if return_patch_tokens else x[:, 0]

    def forward(self, x: torch.Tensor, input_chans: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.forward_features(x, input_chans=input_chans))


def labram_base_patch200_200(**kwargs: object) -> NeuralTransformer:
    return NeuralTransformer(
        patch_size=200,
        embed_dim=200,
        depth=12,
        num_heads=10,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


STANDARD_1020 = [
    "FP1", "FPZ", "FP2", "AF9", "AF7", "AF5", "AF3", "AF1", "AFZ", "AF2", "AF4", "AF6", "AF8", "AF10",
    "F9", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "F10", "FT9", "FT7", "FC5", "FC3",
    "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "FT10", "T9", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
    "C6", "T8", "T10", "TP9", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "P10", "PO9", "PO7", "PO5", "PO3",
    "PO1", "POZ", "PO2", "PO4", "PO6", "PO8", "PO10", "O1", "OZ", "O2", "O9", "CB1", "CB2", "IZ", "O10",
    "T3", "T5", "T4", "T6", "M1", "M2", "A1", "A2", "CFC1", "CFC2", "CFC3", "CFC4", "CFC5", "CFC6",
    "CFC7", "CFC8", "CCP1", "CCP2", "CCP3", "CCP4", "CCP5", "CCP6", "CCP7", "CCP8", "T1", "T2",
    "FTT9H", "TTP7H", "TPP9H", "FTT10H", "TPP8H", "TPP10H",
]


def get_input_chans(channel_names: list[str]) -> torch.Tensor:
    indices = [0]
    unknown: list[str] = []
    for channel in channel_names:
        normalized = channel.strip().upper()
        if normalized not in STANDARD_1020:
            unknown.append(channel)
        else:
            indices.append(STANDARD_1020.index(normalized) + 1)
    if unknown:
        raise ValueError(f"Channels not present in LaBraM's standard order: {unknown}")
    return torch.tensor(indices, dtype=torch.long)
