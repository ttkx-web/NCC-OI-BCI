from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from bci_dayloop.models.model_1b.backbone import (
    Model1BBackbone,
    _matches_requested_device,
    load_backbone_checkpoint,
)
from bci_dayloop.models.model_1b.config import Model1BConfig
from bci_dayloop.models.model_1b.runner import Model1BBackboneRunner
from bci_dayloop.models.model_50m.config import STANDARD_64_CHANNELS
from bci_dayloop.runtime.types import RawEEGWindow


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
        for key, tensor in state.items():
            if tuple(tensor.shape) != tuple(self._state[key].shape):
                raise RuntimeError("size mismatch")
        self.loaded = state


class _FakeBackbone:
    """CPU-only encoder stand-in for execution-chain tests."""

    device_object = torch.device("cpu")

    def eval(self) -> "_FakeBackbone":
        return self

    def extract_embeddings(self, batch: object) -> torch.Tensor:
        token_inputs = getattr(batch, "token_inputs")
        return torch.zeros(
            (token_inputs.shape[0], token_inputs.shape[1], 2048),
            dtype=torch.float32,
            device=self.device_object,
        )


def test_unsuffixed_cuda_device_accepts_cuda_zero_tensor_device() -> None:
    assert _matches_requested_device(torch.device("cuda:0"), torch.device("cuda"))
    assert not _matches_requested_device(torch.device("cuda:1"), torch.device("cuda:0"))


def _raw_window(seconds: float, *, sample_rate: float = 100.0) -> RawEEGWindow:
    points = int(round(seconds * sample_rate))
    return RawEEGWindow(
        data=np.zeros((64, points), dtype=np.float32),
        channel_names=list(STANDARD_64_CHANNELS),
        sample_rate=sample_rate,
        unit="uV",
        layout="CT",
    )


@pytest.mark.parametrize("window_seconds", (1.0, 4.0, 10.0))
def test_runner_prepares_and_extracts_variable_windows(window_seconds: float) -> None:
    config = Model1BConfig(
        checkpoint_path="unused.pt",
        window_seconds=window_seconds,
    )
    runner = Model1BBackboneRunner(config, backbone=_FakeBackbone())

    prepared = runner.prepare(_raw_window(window_seconds, sample_rate=250.0))
    embedding = runner.extract_embeddings(prepared)

    expected_tokens = 64 * int(window_seconds)
    assert prepared.token_inputs.shape == (1, expected_tokens, 100)
    assert prepared.token_channel_indices.shape == (1, expected_tokens)
    assert prepared.token_time_indices.shape == (1, expected_tokens)
    assert prepared.channel_valid_mask.shape == (1, 64)
    assert prepared.token_inputs.dtype == torch.float32
    assert prepared.token_channel_indices.dtype == torch.int64
    assert prepared.token_time_indices.dtype == torch.int64
    assert embedding.shape == (1, expected_tokens, 2048)
    assert embedding.dtype == torch.float32
    assert torch.isfinite(embedding).all()


def test_four_second_runner_contract_is_exact() -> None:
    config = Model1BConfig(checkpoint_path="unused.pt", window_seconds=4.0)
    runner = Model1BBackboneRunner(config, backbone=_FakeBackbone())
    prepared = runner.prepare(_raw_window(4.0, sample_rate=250.0))
    embedding = runner.extract_embeddings(prepared)

    assert prepared.token_inputs.shape == (1, 256, 100)
    assert embedding.shape == (1, 256, 2048)
    assert int(prepared.token_time_indices.min()) == 0
    assert int(prepared.token_time_indices.max()) == 3


@pytest.mark.parametrize("window_seconds", (0.5, 1.5, 11.0))
def test_invalid_1b_windows_fail_fast(window_seconds: float) -> None:
    with pytest.raises(ValueError):
        Model1BConfig(checkpoint_path="unused.pt", window_seconds=window_seconds)


def test_runner_rejects_non_integral_raw_window_duration() -> None:
    config = Model1BConfig(checkpoint_path="unused.pt", window_seconds=4.0)
    runner = Model1BBackboneRunner(config, backbone=_FakeBackbone())
    with pytest.raises(ValueError, match="whole-second"):
        runner.prepare(_raw_window(4.5))


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
    assert report.loaded_parameter_count == 16
    assert report.ignored_keys == ("head.head.0.weight", "head.head.2.bias")
    assert report.missing_keys == ()
    assert model.loaded is not None


@pytest.mark.parametrize("bad_key", ("classifier.weight", "module.tokenizer.proj.0.weight"))
def test_loader_rejects_non_head_and_wrong_prefix_keys(tmp_path: Path, bad_key: str) -> None:
    checkpoint = tmp_path / "bad.pt"
    state = _SmallBackbone().state_dict() | {bad_key: torch.ones((2, 2))}
    torch.save({"model_state_dict": state}, checkpoint)
    with pytest.raises(RuntimeError, match="unexpected"):
        load_backbone_checkpoint(_SmallBackbone(), checkpoint, device=torch.device("cpu"))  # type: ignore[arg-type]


def test_loader_rejects_shape_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "shape_mismatch.pt"
    state = _SmallBackbone().state_dict() | {
        "tokenizer.proj.0.weight": torch.ones((3, 2)),
        "head.head.0.weight": torch.ones((2, 2)),
    }
    torch.save({"model_state_dict": state}, checkpoint)
    with pytest.raises(RuntimeError, match="shapes"):
        load_backbone_checkpoint(_SmallBackbone(), checkpoint, device=torch.device("cpu"))  # type: ignore[arg-type]


_RUN_FORMAL = os.environ.get("RUN_1B_CHECKPOINT_TESTS") == "1"
_FORMAL_PATH = Path("checkpoints/backbones/1b/pretrain_checkpoint_4.pt")


@pytest.mark.skipif(
    not _RUN_FORMAL or not torch.cuda.is_available(),
    reason="set RUN_1B_CHECKPOINT_TESTS=1 on a CUDA measurement server",
)
def test_formal_checkpoint_gpu_load_and_smoke_forward() -> None:
    config = Model1BConfig(checkpoint_path=_FORMAL_PATH, device="cuda", window_seconds=1.0)
    runner = Model1BBackboneRunner(config)
    prepared = runner.prepare(_raw_window(1.0, sample_rate=250.0))
    embedding = runner.extract_embeddings(prepared)

    assert runner.backbone.load_report is not None
    assert runner.backbone.load_report.loaded_tensor_count == 246
    assert runner.backbone.load_report.ignored_keys == (
        "head.head.0.bias", "head.head.0.weight", "head.head.2.bias", "head.head.2.weight",
    )
    assert embedding.shape == (1, 64, 2048)
    assert embedding.device.type == "cuda"
    assert torch.isfinite(embedding).all()
