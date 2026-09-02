from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from bci_dayloop.models.model_1b.backbone import Model1BBackbone, load_backbone_checkpoint
from bci_dayloop.models.model_1b.config import Model1BConfig
from bci_dayloop.models.model_1b.tokenization import Model1BTokenizer


class _SmallBackbone:
    """Exercise strict loader policy without allocating the 1B architecture."""

    def __init__(self) -> None:
        self.loaded: dict[str, torch.Tensor] | None = None
        self._state = {
            "tokenizer.proj.0.weight": torch.zeros((2, 2)),
            "channel_embed.weight": torch.zeros((2, 2)),
            "time_embed.weight": torch.zeros((2, 2)),
            "encoder.encoder.layers.0.linear1.weight": torch.zeros((2, 2)),
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._state

    def load_state_dict(self, state: dict[str, torch.Tensor], *, strict: bool) -> None:
        assert strict is True
        self.loaded = state


def test_strict_loader_ignores_only_timefreq_pretraining_head(tmp_path: Path) -> None:
    checkpoint = tmp_path / "one_b.pt"
    state = _SmallBackbone().state_dict() | {
        "head.head.0.weight": torch.ones((2, 2)),
        "head.head.2.bias": torch.ones(2),
    }
    torch.save({"model_state_dict": state}, checkpoint)
    model = _SmallBackbone()

    report = load_backbone_checkpoint(model, checkpoint, device=torch.device("cpu"))  # type: ignore[arg-type]

    assert report.checkpoint_source_key == "model_state_dict"
    assert report.loaded_tensor_count == 4
    assert report.ignored_pretraining_head_keys == ("head.head.0.weight", "head.head.2.bias")
    assert model.loaded is not None
    assert tuple(model.loaded) == tuple(key for key in state if not key.startswith("head."))


def test_loader_rejects_non_head_unexpected_key(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bad.pt"
    state = _SmallBackbone().state_dict() | {"classifier.weight": torch.ones((2, 2))}
    torch.save({"model_state_dict": state}, checkpoint)
    with pytest.raises(RuntimeError, match="unexpected"):
        load_backbone_checkpoint(_SmallBackbone(), checkpoint, device=torch.device("cpu"))  # type: ignore[arg-type]


@pytest.mark.parametrize("window_seconds", (1.0, 2.0, 3.0, 4.0))
def test_variable_window_tokenization_uses_checkpoint_time_positions(window_seconds: float) -> None:
    config = Model1BConfig(checkpoint_path="unused.pt", window_seconds=window_seconds, filter_enabled=False)
    tokenized = Model1BTokenizer(config).tokenize(
        np.zeros((64, config.target_num_points), dtype=np.float32),
        np.ones(64, dtype=np.float32),
    ).as_batch()
    assert tokenized.token_inputs.shape == (1, 64 * int(window_seconds), 100)
    assert tokenized.token_channel_indices.dtype == torch.int64
    assert tokenized.token_time_indices.dtype == torch.int64
    assert int(tokenized.token_time_indices.max()) == int(window_seconds) - 1


@pytest.mark.parametrize("window_seconds", (0.5, 1.5, 11.0))
def test_invalid_1b_windows_fail_fast(window_seconds: float) -> None:
    with pytest.raises(ValueError):
        Model1BConfig(checkpoint_path="unused.pt", window_seconds=window_seconds)


_RUN_FORMAL = os.environ.get("RUN_1B_CHECKPOINT_TESTS") == "1"
_FORMAL_PATH = Path("checkpoints/backbones/1b/pretrain_checkpoint_4.pt")


@pytest.mark.skipif(not _RUN_FORMAL, reason="set RUN_1B_CHECKPOINT_TESTS=1 on the measurement server")
def test_formal_checkpoint_loads_and_ignores_only_timefreq_head() -> None:
    config = Model1BConfig(checkpoint_path=_FORMAL_PATH, device="cpu", window_seconds=4.0)
    backbone = Model1BBackbone(config)
    assert backbone.load_report is not None
    assert backbone.load_report.loaded_tensor_count == 246
    assert backbone.load_report.ignored_pretraining_head_keys == (
        "head.head.0.bias", "head.head.0.weight", "head.head.2.bias", "head.head.2.weight",
    )


@pytest.mark.skipif(not _RUN_FORMAL, reason="set RUN_1B_CHECKPOINT_TESTS=1 on the measurement server")
def test_formal_1b_smoke_forward_is_finite() -> None:
    config = Model1BConfig(checkpoint_path=_FORMAL_PATH, device="cpu", window_seconds=1.0)
    backbone = Model1BBackbone(config)
    tokenized = Model1BTokenizer(config).tokenize(
        np.zeros((64, 100), dtype=np.float32), np.ones(64, dtype=np.float32)
    ).as_batch()
    embeddings = backbone(tokenized)
    assert embeddings.shape == (1, 64, 2048)
    assert torch.isfinite(embeddings).all()
