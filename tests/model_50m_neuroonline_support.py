from __future__ import annotations

from typing import Any

import numpy as np
import torch

from bci_dayloop.models.model_50m.backend import Model50MBackend
from bci_dayloop.models.model_50m.backbone import Model50MBackbone
from bci_dayloop.models.model_50m.classifier import Model50MClassifier
from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.models.model_50m.tokenization import (
    Model50MTokenizer,
    stack_model50m_tokens,
)
from bci_dayloop.runtime.model import RuntimeModel
from bci_dayloop.runtime.types import (
    CanonicalEEGWindow,
    PreparedModelInput,
)


class Tiny50MAdapter:
    """阶段 B 使用正式 50M 模块的 checkpoint-free adapter。"""

    def __init__(self, config: Model50MConfig) -> None:
        self.config = config
        self.tokenizer = Model50MTokenizer(config)
        self.backbone = Model50MBackbone(
            config=config,
            load_checkpoint=False,
            freeze=True,
        )
        self.classifier = Model50MClassifier(
            config=config,
            backbone=self.backbone,
        )

    @property
    def device(self) -> torch.device:
        return self.backbone.device

    @property
    def num_classes(self) -> int:
        return self.config.num_classes

    def _build_model_batch(
        self,
        *,
        X: np.ndarray,
        channel_valid_masks: np.ndarray | None,
    ) -> tuple[Any, float, float]:
        if channel_valid_masks is None:
            raise ValueError("Tiny50MAdapter requires channel_valid_masks.")

        signals = np.asarray(X, dtype=np.float32)
        masks = np.asarray(channel_valid_masks, dtype=np.float32)

        if signals.ndim != 3:
            raise ValueError(
                "signals must have shape [B,C,T], got "
                f"{signals.shape}."
            )

        expected_mask_shape = (
            signals.shape[0],
            self.config.n_channels,
        )
        if tuple(masks.shape) != expected_mask_shape:
            raise ValueError(
                "channel_valid_masks shape mismatch: expected "
                f"{expected_mask_shape}, got {masks.shape}."
            )

        samples = [
            self.tokenizer.tokenize(
                signal=signals[index],
                channel_valid_mask=masks[index],
            )
            for index in range(signals.shape[0])
        ]

        return (
            stack_model50m_tokens(samples, device=self.device),
            0.0,
            0.0,
        )


def build_backend(
    *,
    aggregation: str = "flatten",
) -> Model50MBackend:
    torch.manual_seed(42)

    config = Model50MConfig(
        checkpoint_path="unused.pt",
        device="cpu",
        target_sample_rate=2.0,
        window_seconds=2.0,
        patch_seconds=1.0,
        patch_stride_seconds=1.0,
        n_channels=2,
        standard_channels=("C3", "C4"),
        filter_enabled=False,
        zscore_enabled=False,
        d_model=4,
        n_heads=2,
        depth=1,
        mlp_ratio=1.0,
        dropout=0.0,
        model_n_time_patches=2,
        output_layer_idx=0,
        aggregation=aggregation,
        num_classes=3,
    )

    return Model50MBackend(Tiny50MAdapter(config))


def build_runtime_model(
    backend: Model50MBackend,
) -> RuntimeModel:
    return RuntimeModel(
        canonicalizer=None,  # type: ignore[arg-type]
        input_transform=None,  # type: ignore[arg-type]
        backend=backend,
    )


def make_model_input(
    *,
    values: tuple[float, float],
    mask: tuple[float, float],
) -> dict[str, torch.Tensor]:
    """构造一个 batch=1 的有效 50M 输入；无效通道严格填零。"""

    channel_mask = torch.tensor([mask], dtype=torch.float32)
    signal = torch.tensor(
        [[[values[0], values[0] + 1.0, values[0] + 2.0, values[0] + 3.0],
          [values[1], values[1] + 1.0, values[1] + 2.0, values[1] + 3.0]]],
        dtype=torch.float32,
    )
    signal = signal * channel_mask.unsqueeze(-1)
    return {
        "signal": signal,
        "channel_valid_mask": channel_mask,
    }


def make_batched_model_input(
    *model_inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key: torch.cat(
            [model_input[key] for model_input in model_inputs],
            dim=0,
        )
        for key in ("signal", "channel_valid_mask")
    }


def make_prepared_input(
    model_input: dict[str, torch.Tensor],
    *,
    trial_id: str,
) -> PreparedModelInput:
    return PreparedModelInput(
        model_input=model_input,
        canonical_window=CanonicalEEGWindow(
            data=np.zeros((2, 4), dtype=np.float32),
            channel_names=["C3", "C4"],
            sample_rate=2.0,
            unit="uV",
            trial_id=trial_id,
        ),
        preprocessing_trace=["tiny_50m_prepared"],
    )


def independent_static_reference(
    backend: Model50MBackend,
    model_input: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """阶段 A 拆分前的 50M 数学公式，绝不调用在线接口。"""

    batch = backend._build_batch(model_input)
    backend.adapter.backbone.freeze()
    backend.adapter.classifier.eval()

    with torch.no_grad():
        tokens = backend.adapter.backbone.extract_embeddings(
            batch=batch,
            return_layer_idx=backend.adapter.config.output_layer_idx,
        )
        features = backend.adapter.classifier.aggregator(
            token_embeddings=tokens,
            token_valid_mask=batch.token_valid_mask,
        )
        logits = backend.adapter.classifier.head(features)

    return logits, tokens, batch.token_valid_mask


def clone_state_dict(
    module: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def changed_parameter_names(
    module: torch.nn.Module,
    before: dict[str, torch.Tensor],
) -> set[str]:
    return {
        name
        for name, parameter in module.named_parameters()
        if not torch.equal(before[name], parameter.detach().cpu())
    }
