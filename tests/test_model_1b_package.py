from __future__ import annotations

import importlib.util
import os
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_1b_model_package as exporter  # noqa: E402

from bci_dayloop.models.model_1b.classifier import Model1BFlattenLinearHead, classifier_input_dim
from bci_dayloop.models.model_1b.config import Model1BConfig
from bci_dayloop.models.model_1b.runner import Model1BPreparedInput
from bci_dayloop.models.model_1b.runtime import Model1BRuntime
from bci_dayloop.models.model_50m.config import STANDARD_64_CHANNELS
from bci_dayloop.packages import loader
from bci_dayloop.runtime.types import InputContract, RawEEGWindow
from bci_dayloop.training.model_1b import population
from bci_dayloop.training.model_50m.types import ExtendedMetrics
from bci_dayloop.utils.config import load_yaml


def _metrics() -> ExtendedMetrics:
    return ExtendedMetrics(1.0, 0.5, 0.5, 0.5, [[1, 0], [1, 0]], [])


def _head_checkpoint(tmp_path: Path, seconds: float, *, backbone_sha: str) -> Path:
    config = Model1BConfig(checkpoint_path=tmp_path / "backbone.pt", window_seconds=seconds)
    head = Model1BFlattenLinearHead(input_dim=classifier_input_dim(config), num_classes=2)
    args = Namespace(
        split_mode="loso", epochs=1, head_batch_size=1, head_lr=1e-3,
        weight_decay=0.0, patience=0, metric_for_best="val_bacc", window_seed=1, seed=1,
    )
    payload = population._head_payload(
        state=head.state_dict(), config=config, class_names=["left_hand", "right_hand"],
        backbone_path=tmp_path / "backbone.pt", backbone_sha256=backbone_sha,
        split={"train_session": "0train", "validation_session": "1test", "final_test_session": "1test"},
        args=args, best_epoch=1, validation_metrics=_metrics(), final_test_metrics=_metrics(), training_seconds=0.0,
    )
    return population.save_1b_head_checkpoint(tmp_path / f"head_{int(seconds)}.pt", payload, overwrite=False)


@pytest.mark.parametrize("seconds", (1.0, 2.0, 4.0, 10.0))
def test_exported_1b_package_uses_head_window_metadata(tmp_path: Path, seconds: float) -> None:
    backbone = tmp_path / "backbone.pt"
    backbone.write_bytes(b"test backbone bytes")
    head_path = _head_checkpoint(tmp_path, seconds, backbone_sha=exporter.sha256_file(backbone))
    package = exporter.export_1b_runtime_package(
        backbone_checkpoint=backbone, head_checkpoint=head_path,
        output_dir=tmp_path / f"package_{int(seconds)}", device="cpu",
    )
    payload = load_yaml(package / "package.yaml")
    model = payload["model"]
    assert model["window_seconds"] == seconds
    assert model["num_time_patches"] == int(seconds)
    assert model["token_count"] == 64 * int(seconds)
    assert model["classifier_input_dim"] == 64 * int(seconds) * 2048
    assert payload["files"]["sha256"]["backbone"] == exporter.sha256_file(package / "backbone.pt")


def test_exporter_rejects_backbone_hash_and_label_contract_mismatch(tmp_path: Path) -> None:
    backbone = tmp_path / "backbone.pt"
    backbone.write_bytes(b"actual backbone")
    head_path = _head_checkpoint(tmp_path, 4.0, backbone_sha="wrong")
    with pytest.raises(ValueError, match="SHA-256"):
        exporter.export_1b_runtime_package(
            backbone_checkpoint=backbone, head_checkpoint=head_path, output_dir=tmp_path / "bad_hash"
        )

    good_head = _head_checkpoint(tmp_path, 2.0, backbone_sha=exporter.sha256_file(backbone))
    payload = torch.load(good_head, map_location="cpu")
    payload["label_mapping"] = {"0": "right_hand", "1": "left_hand"}
    bad_labels = tmp_path / "bad_labels.pt"
    torch.save(payload, bad_labels)
    with pytest.raises(ValueError, match="label_mapping"):
        exporter.export_1b_runtime_package(
            backbone_checkpoint=backbone, head_checkpoint=bad_labels, output_dir=tmp_path / "bad_labels"
        )


def test_exporter_rejects_invalid_head_window_and_state_shape(tmp_path: Path) -> None:
    backbone = tmp_path / "backbone.pt"
    backbone.write_bytes(b"actual backbone")
    head_path = _head_checkpoint(tmp_path, 2.0, backbone_sha=exporter.sha256_file(backbone))
    payload = torch.load(head_path, map_location="cpu")
    payload["window_seconds"] = 2.5
    bad_window = tmp_path / "bad_window.pt"
    torch.save(payload, bad_window)
    with pytest.raises(ValueError, match="integer number"):
        exporter.export_1b_runtime_package(
            backbone_checkpoint=backbone, head_checkpoint=bad_window, output_dir=tmp_path / "bad_window"
        )
    payload = torch.load(head_path, map_location="cpu")
    payload["head_state_dict"]["linear.weight"] = torch.zeros((2, 7))
    bad_shape = tmp_path / "bad_shape.pt"
    torch.save(payload, bad_shape)
    with pytest.raises(ValueError, match="weight shape"):
        exporter.export_1b_runtime_package(
            backbone_checkpoint=backbone, head_checkpoint=bad_shape, output_dir=tmp_path / "bad_shape"
        )


def test_loader_constructs_runtime_from_generic_window_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backbone = tmp_path / "backbone.pt"
    backbone.write_bytes(b"test backbone bytes")
    head_path = _head_checkpoint(tmp_path, 2.0, backbone_sha=exporter.sha256_file(backbone))
    package = exporter.export_1b_runtime_package(
        backbone_checkpoint=backbone, head_checkpoint=head_path, output_dir=tmp_path / "package", device="cpu"
    )
    captured: dict[str, object] = {}

    class _FakeRuntime:
        def __init__(self) -> None:
            self.input_contract = InputContract(
                channel_names=STANDARD_64_CHANNELS, sample_rate=100.0, window_sec=2.0,
                num_samples=200, input_unit="uV", tensor_layout="BCT",
                model_input_keys=("signal", "channel_valid_mask"),
            )

    def fake_build(**kwargs: object) -> _FakeRuntime:
        captured.update(kwargs)
        return _FakeRuntime()

    monkeypatch.setattr(loader, "build_1b_runtime", fake_build)
    loaded = loader.load_runtime_package(package, device="cpu", verify_hashes=True)
    assert loaded.model_type == "model_1b"
    assert loaded.window_sec == 2.0
    assert captured["window_seconds"] == 2.0
    assert captured["device"] == "cpu"


class _FakeBackbone(nn.Module):
    device_object = torch.device("cpu")

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)

    def extract_embeddings(self, batch: object) -> torch.Tensor:
        token_inputs = getattr(batch, "token_inputs")
        return torch.ones((token_inputs.shape[0], token_inputs.shape[1], 2048), dtype=torch.float32)


class _FakeRunner:
    def __init__(self, config: Model1BConfig) -> None:
        self.config = config
        self.backbone = _FakeBackbone()
        self.input_transform = SimpleNamespace(input_contract=InputContract(
            channel_names=STANDARD_64_CHANNELS, sample_rate=100.0, window_sec=config.window_seconds,
            num_samples=config.target_num_points, input_unit="uV", tensor_layout="BCT",
            model_input_keys=("signal", "channel_valid_mask"),
        ))

    def prepare(self, raw: RawEEGWindow) -> Model1BPreparedInput:
        if raw.data.shape[1] / raw.sample_rate != self.config.window_seconds:
            raise ValueError("1B raw window must exactly match configured whole-second duration")
        tokens = self.config.num_tokens
        channels = torch.repeat_interleave(torch.arange(64, dtype=torch.int64), self.config.num_time_patches)[None]
        times = torch.tile(torch.arange(self.config.num_time_patches, dtype=torch.int64), (64,))[None]
        return Model1BPreparedInput(
            token_inputs=torch.zeros((1, tokens, 100), dtype=torch.float32),
            token_channel_indices=channels, token_time_indices=times,
            token_valid_mask=torch.ones((1, tokens), dtype=torch.float32),
            channel_valid_mask=torch.ones((1, 64), dtype=torch.float32),
            num_time_patches=self.config.num_time_patches, prepared_model_input=None,  # type: ignore[arg-type]
        )

    def extract_embeddings(self, prepared: Model1BPreparedInput) -> torch.Tensor:
        return self.backbone.extract_embeddings(prepared.as_batched_input())


def test_runtime_rejects_a_window_other_than_its_package_contract() -> None:
    config = Model1BConfig(checkpoint_path="unused.pt", window_seconds=1.0)
    head = Model1BFlattenLinearHead(input_dim=classifier_input_dim(config), num_classes=2)
    runtime = Model1BRuntime(config=config, runner=_FakeRunner(config), head=head, class_names=("left", "right"))
    raw = RawEEGWindow(
        data=np.zeros((64, 100), dtype=np.float32), channel_names=list(STANDARD_64_CHANNELS),
        sample_rate=100.0, unit="uV", layout="CT",
    )
    output = runtime.predict(raw)
    assert output.logits.shape == (1, 2)
    assert output.probabilities.shape == (1, 2)
    assert output.predicted_class in (0, 1)
    wrong = RawEEGWindow(
        data=np.zeros((64, 200), dtype=np.float32), channel_names=list(STANDARD_64_CHANNELS),
        sample_rate=100.0, unit="uV", layout="CT",
    )
    with pytest.raises(ValueError, match="whole-second"):
        runtime.prepare(wrong)


_RUN_FORMAL_PACKAGE = os.environ.get("RUN_1B_PACKAGE_SMOKE") == "1"
_FORMAL_BACKBONE = Path(os.environ.get("ONE_B_BACKBONE_CHECKPOINT", "checkpoints/backbones/1b/pretrain_checkpoint_4.pt"))
_FORMAL_HEAD = os.environ.get("ONE_B_HEAD_CHECKPOINT")


@pytest.mark.skipif(
    not _RUN_FORMAL_PACKAGE or not _FORMAL_HEAD or not torch.cuda.is_available(),
    reason="set RUN_1B_PACKAGE_SMOKE=1 and ONE_B_HEAD_CHECKPOINT on a CUDA server",
)
def test_formal_4s_package_matches_direct_runtime_logits(tmp_path: Path) -> None:
    package = exporter.export_1b_runtime_package(
        backbone_checkpoint=_FORMAL_BACKBONE, head_checkpoint=Path(_FORMAL_HEAD),
        output_dir=tmp_path / "formal_package", device="cuda",
    )
    loaded = loader.load_runtime_package(package, device="cuda", verify_hashes=True)
    runtime = loaded.runtime_model
    assert loaded.window_sec == 4.0, "ONE_B_HEAD_CHECKPOINT must be the formal 4-second head"
    raw = RawEEGWindow(
        data=np.zeros((64, 1000), dtype=np.float32), channel_names=list(STANDARD_64_CHANNELS),
        sample_rate=250.0, unit="uV", layout="CT",
    )
    prepared = runtime.prepare(raw)
    package_output = runtime.predict_prepared(prepared)
    embedding = runtime.runner.extract_embeddings(prepared)
    direct_logits = runtime.head(
        (embedding * prepared.token_valid_mask.to(embedding.device).unsqueeze(-1)).flatten(start_dim=1)
    )
    direct_probabilities = torch.softmax(direct_logits, dim=-1)
    assert torch.allclose(package_output.logits, direct_logits, atol=1e-6, rtol=1e-5)
    assert torch.allclose(package_output.probabilities, direct_probabilities, atol=1e-6, rtol=1e-5)
    assert package_output.predicted_class == int(direct_probabilities.argmax(dim=-1)[0])
