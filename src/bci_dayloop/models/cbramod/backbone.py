from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .config import CBraModConfig
from .vendor.cbramod import CBraMod


# 这些是官方发布 checkpoint 对应的网络结构。
# 不能为了适应下游任务随意修改，否则预训练权重无法严格加载。
_OFFICIAL_IN_DIM = 200
_OFFICIAL_OUT_DIM = 200
_OFFICIAL_D_MODEL = 200
_OFFICIAL_DIM_FEEDFORWARD = 800
_OFFICIAL_SEQ_LEN = 30
_OFFICIAL_N_LAYER = 12
_OFFICIAL_NHEAD = 8


class CBraModBackbone(nn.Module):
    """
    CBRaMod 预训练骨干的 Runtime 包装。

    输入：
        [B, C, S, P]
        默认严格为 [B, 22, 4, 200]。

    输出：
        [B, C, S, D]
        默认严格为 [B, 22, 4, 200]。

    本类只负责：
    - 创建官方 CBRaMod 网络；
    - 加载官方预训练权重；
    - 替换预训练输出投影；
    - 验证张量 shape；
    - 冻结或解冻 backbone。

    不负责：
    - 通道重排；
    - 滤波；
    - 重采样；
    - 归一化；
    - 原始 [C, T] EEG 切成 [C, S, P]。
    """

    def __init__(
        self,
        config: CBraModConfig,
    ) -> None:
        super().__init__()

        self.config = config
        self._validate_official_architecture_contract()

        self.model = CBraMod(
            in_dim=_OFFICIAL_IN_DIM,
            out_dim=_OFFICIAL_OUT_DIM,
            d_model=_OFFICIAL_D_MODEL,
            dim_feedforward=_OFFICIAL_DIM_FEEDFORWARD,
            seq_len=_OFFICIAL_SEQ_LEN,
            n_layer=_OFFICIAL_N_LAYER,
            nhead=_OFFICIAL_NHEAD,
        )

        self._load_pretrained_checkpoint(
            self.config.checkpoint_path,
        )

        # 官方下游封装也在加载预训练权重后执行这一步。
        # 这样 forward 返回的是 encoder feature，而不是预训练任务投影结果。
        self.model.proj_out = nn.Identity()

        self.to(self._resolve_device(self.config.device))
        self.freeze()

    @staticmethod
    def _resolve_device(
        device: str | torch.device,
    ) -> torch.device:
        resolved = torch.device(device)

        if resolved.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CBraMod was configured for CUDA, but CUDA "
                    "is unavailable."
                )

        if resolved.type == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    "CBraMod was configured for MPS, but MPS "
                    "is unavailable."
                )

        return resolved

    def _validate_official_architecture_contract(self) -> None:
        """
        CBRaMod 的频域分支将 patch_size=200 写死为 rFFT 后的 101 维，
        因此当前官方预训练权重不能接受其他 points_per_patch。
        """

        if self.config.n_channels != 22:
            raise ValueError(
                "The current CBraMod baseline is defined for "
                f"22 channels, got n_channels={self.config.n_channels}."
            )

        if self.config.time_segments != 4:
            raise ValueError(
                "The current CBraMod baseline is defined for "
                "4 time segments, got "
                f"time_segments={self.config.time_segments}."
            )

        if self.config.points_per_patch != _OFFICIAL_IN_DIM:
            raise ValueError(
                "The released CBraMod backbone requires "
                f"points_per_patch={_OFFICIAL_IN_DIM}, got "
                f"{self.config.points_per_patch}."
            )

        if self.config.backbone_output_dim != _OFFICIAL_D_MODEL:
            raise ValueError(
                "The released CBraMod backbone outputs "
                f"{_OFFICIAL_D_MODEL}-dimensional features, got "
                f"backbone_output_dim="
                f"{self.config.backbone_output_dim}."
            )

    @staticmethod
    def _extract_state_dict(
        checkpoint: object,
    ) -> dict[str, torch.Tensor]:
        """
        支持官方纯 state_dict，也兼容常见训练脚本保存的嵌套格式。
        """

        if not isinstance(checkpoint, Mapping):
            raise TypeError(
                "CBraMod checkpoint must be a state_dict or a "
                "mapping containing a state_dict, got "
                f"{type(checkpoint).__name__}."
            )

        candidate: object = checkpoint

        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "backbone",
        ):
            nested = checkpoint.get(key)

            if isinstance(nested, Mapping):
                candidate = nested
                break

        if not isinstance(candidate, Mapping):
            raise TypeError(
                "Could not find a valid state_dict in the "
                "CBraMod checkpoint."
            )

        state_dict: dict[str, torch.Tensor] = {}

        for key, value in candidate.items():
            if not isinstance(key, str):
                raise TypeError(
                    "CBraMod state_dict contains a non-string key: "
                    f"{key!r}."
                )

            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    "CBraMod state_dict contains a non-tensor value "
                    f"for key {key!r}: {type(value).__name__}."
                )

            state_dict[key] = value

        if not state_dict:
            raise ValueError(
                "CBraMod state_dict is empty."
            )

        return CBraModBackbone._strip_common_prefixes(
            state_dict,
        )

    @staticmethod
    def _strip_common_prefixes(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        兼容 DDP 的 module. 前缀，以及包装器常见的 backbone. 前缀。
        只有所有 key 都带同一前缀时才移除，避免错误修改 key。
        """

        normalized = state_dict

        for prefix in ("module.", "backbone."):
            keys = tuple(normalized.keys())

            if keys and all(key.startswith(prefix) for key in keys):
                normalized = {
                    key[len(prefix):]: value
                    for key, value in normalized.items()
                }

        return normalized

    def _load_pretrained_checkpoint(
        self,
        checkpoint_path: Path | str,
    ) -> None:
        path = Path(checkpoint_path)

        if not path.is_file():
            raise FileNotFoundError(
                "CBraMod pretrained checkpoint was not found: "
                f"{path}"
            )

        payload: Any = torch.load(
            path,
            map_location="cpu",
        )

        state_dict = self._extract_state_dict(payload)

        # strict=True 保证使用的确实是官方对应结构，
        # 不允许缺层或静默跳过不匹配参数。
        self.model.load_state_dict(
            state_dict,
            strict=True,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def output_dim(self) -> int:
        return self.config.backbone_output_dim

    def freeze(self) -> None:
        """冻结全部 CBRaMod 参数，并保持 eval 模式。"""

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.model.eval()

    def unfreeze(self) -> None:
        """
        仅为未来 cbramod-full-finetune 预留。

        当前 cbramod-frozen-head 主实验不应调用此方法。
        """

        for parameter in self.model.parameters():
            parameter.requires_grad = True

        self.model.train()

    def _validate_input(
        self,
        signal: torch.Tensor,
    ) -> None:
        if not isinstance(signal, torch.Tensor):
            raise TypeError(
                "CBraMod signal must be torch.Tensor, got "
                f"{type(signal).__name__}."
            )

        if signal.ndim != 4:
            raise ValueError(
                "CBraMod input must have shape [B, C, S, P], got "
                f"{tuple(signal.shape)}."
            )

        expected = self.config.expected_unbatched_shape

        if tuple(signal.shape[1:]) != expected:
            raise ValueError(
                "Unexpected CBraMod input shape. Expected "
                f"[B, {expected[0]}, {expected[1]}, {expected[2]}], "
                f"got {tuple(signal.shape)}."
            )

        if signal.shape[0] <= 0:
            raise ValueError(
                "CBraMod input batch size must be positive."
            )

        if not signal.is_floating_point():
            raise TypeError(
                "CBraMod input must use a floating-point dtype, "
                f"got {signal.dtype}."
            )

        if not torch.isfinite(signal).all():
            raise ValueError(
                "CBraMod input contains NaN or Inf."
            )

    def encode(
        self,
        signal: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 CBRaMod encoder 特征。

        Args:
            signal:
                [B, 22, 4, 200]。

        Returns:
            [B, 22, 4, 200]。
        """

        self._validate_input(signal)

        signal = signal.to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=True,
        )

        features = self.model(signal)

        expected_shape = (
            signal.shape[0],
            self.config.n_channels,
            self.config.time_segments,
            self.config.backbone_output_dim,
        )

        if tuple(features.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected CBraMod feature shape. Expected "
                f"{expected_shape}, got {tuple(features.shape)}."
            )

        if not torch.isfinite(features).all():
            raise RuntimeError(
                "CBraMod backbone produced NaN or Inf features."
            )

        return features

    def forward(
        self,
        signal: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode(signal)