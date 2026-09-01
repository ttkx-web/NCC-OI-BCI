from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest
import torch

from bci_dayloop.data.hdf5_dataset import HDF5Metadata, write_hdf5
from bci_dayloop.inference.neuroonline_strategy import NeuroOnlineStrategy
from bci_dayloop.models.model_50m.backend import Model50MBackend
from bci_dayloop.models.model_50m.backbone import Model50MBackbone
from bci_dayloop.models.model_50m.classifier import Model50MClassifier
from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.models.model_50m.tokenization import (
    Model50MTokenizer,
    stack_model50m_tokens,
)
from bci_dayloop.preprocessing.base import ModelInputTransform
from bci_dayloop.preprocessing.canonical import SignalCanonicalizer
from bci_dayloop.runtime.model import RuntimeModel
from bci_dayloop.runtime.types import (
    CanonicalEEGWindow,
    InputContract,
    PreparedModelInput,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts import evaluate_neuroonline_sequential as seq


CLASS_NAMES = ("left_hand", "right_hand", "feet", "tongue")


class Tiny50MSequentialAdapter:
    """使用生产 50M tokenizer/backbone/aggregator/head 的小型 adapter。"""

    def __init__(self) -> None:
        self.config = Model50MConfig(
            checkpoint_path="unused.pt",
            device="cpu",
            target_sample_rate=2.0,
            window_seconds=4.0,
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
            model_n_time_patches=4,
            output_layer_idx=0,
            aggregation="flatten",
            num_classes=len(CLASS_NAMES),
        )
        self.tokenizer = Model50MTokenizer(self.config)
        self.backbone = Model50MBackbone(
            config=self.config,
            load_checkpoint=False,
            freeze=True,
        )
        self.classifier = Model50MClassifier(
            config=self.config,
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
            raise ValueError("50M sequential test requires channel_valid_masks.")

        signals = np.asarray(X, dtype=np.float32)
        masks = np.asarray(channel_valid_masks, dtype=np.float32)
        expected_signal_shape = (
            signals.shape[0],
            self.config.n_channels,
            self.config.target_num_points,
        )
        expected_mask_shape = (signals.shape[0], self.config.n_channels)
        if tuple(signals.shape) != expected_signal_shape:
            raise ValueError(
                "signals shape mismatch: expected "
                f"{expected_signal_shape}, got {signals.shape}."
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
        return stack_model50m_tokens(samples, device=self.device), 0.0, 0.0


class Tiny50MSequentialTransform(ModelInputTransform):
    def __init__(self) -> None:
        self._contract = InputContract(
            channel_names=("C3", "C4"),
            sample_rate=2.0,
            window_sec=4.0,
            num_samples=8,
            input_unit="uV",
            tensor_layout="BCT",
            strict_window_duration=True,
            model_input_keys=("signal", "channel_valid_mask"),
        )

    @property
    def input_contract(self) -> InputContract:
        return self._contract

    def transform(self, window: CanonicalEEGWindow) -> PreparedModelInput:
        signal = torch.from_numpy(
            np.ascontiguousarray(window.data, dtype=np.float32)
        ).unsqueeze(0)
        channel_valid_mask = (signal.abs().sum(dim=-1) > 0).to(torch.float32)
        return PreparedModelInput(
            model_input={
                "signal": signal,
                "channel_valid_mask": channel_valid_mask,
            },
            canonical_window=window,
            preprocessing_trace=["tiny_50m_sequential_transform"],
        )


@dataclass
class TinyLoaded50MPackage:
    runtime_model: RuntimeModel
    package_path: Path
    model_type: str = "model_50m"
    model_name: str = "50m-linear"
    class_names: tuple[str, ...] = CLASS_NAMES
    command_map: dict[str, str] | None = None
    step_sec: float = 0.5
    confidence_threshold: float = 0.55
    is_test_head: bool = False
    warning_message: str | None = None
    metrics: dict[str, Any] | None = None
    package_metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.command_map = self.command_map or {}
        self.metrics = self.metrics or {}
        self.package_metadata = self.package_metadata or {}

    @property
    def window_sec(self) -> float:
        return float(self.runtime_model.input_contract.window_sec)

    @property
    def target_sample_rate(self) -> float:
        return float(self.runtime_model.input_contract.sample_rate)


def make_runtime() -> RuntimeModel:
    torch.manual_seed(42)
    backend = Model50MBackend(Tiny50MSequentialAdapter())
    return RuntimeModel(
        canonicalizer=SignalCanonicalizer(target_unit="uV"),
        input_transform=Tiny50MSequentialTransform(),
        backend=backend,
    )


def write_sequential_dataset(path: Path) -> Path:
    # 每个 trial 都为 4 秒；无效通道严格为零，交替 mask 验证通道语义。
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(
        [
            [[1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0, 0, 0], [2, 3, 4, 5, 6, 7, 8, 9]],
            [[3, 4, 5, 6, 7, 8, 9, 10], [0, 0, 0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0, 0, 0], [4, 5, 6, 7, 8, 9, 10, 11]],
        ],
        dtype=np.float32,
    )
    return write_hdf5(
        path,
        data,
        np.asarray([0, 1, 2, 3], dtype=np.int64),
        np.ones(4, dtype=np.int64),
        ["1test"] * 4,
        np.asarray([101, 55, 88, 7], dtype=np.int64),
        HDF5Metadata(
            sample_rate=2.0,
            channel_names=["C3", "C4"],
            class_names=list(CLASS_NAMES),
            unit="uV",
            dataset_name="tiny_50m_sequential",
        ),
    )


def make_settings(tmp_path: Path, data_path: Path) -> seq.SequentialSettings:
    args = seq.build_parser().parse_args(
        [
            "--config",
            "configs/stage1/replay_population_4s.yaml",
            "--data",
            str(data_path),
            "--model-package",
            str(tmp_path / "tiny-50m-package"),
            "--output-dir",
            str(tmp_path / "evaluation"),
            "--device",
            "cpu",
            "--online-strategy",
            "both",
        ]
    )
    return seq.resolve_settings(
        args,
        {
            "project": {"run_dir": str(tmp_path / "runs")},
            "online": {
                "strategy": "none",
                "neuroonline": {
                    "num_subject_codes": 2,
                    "num_attention_heads": 2,
                    "dropout": 0.0,
                    "learning_rate": 1e-2,
                    "weight_decay": 0.0,
                    "warmup_feedback": 2,
                    "update_interval": 2,
                    "recent_buffer_size": 4,
                    "batch_size": 2,
                    "epochs_per_update": 1,
                    "max_pending_observations": 8,
                    "seed": 42,
                },
            },
        },
    )


def clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
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
        if not torch.equal(parameter.detach().cpu(), before[name])
    }


def test_50m_uses_shared_sequential_evaluator_and_causal_update_protocol(
    tmp_path: Path,
) -> None:
    data_path = write_sequential_dataset(tmp_path / "trials.h5")
    settings = make_settings(tmp_path, data_path)
    loaded_runtimes: list[RuntimeModel] = []
    strategies: list[NeuroOnlineStrategy] = []
    online_before: dict[str, dict[str, torch.Tensor]] = {}

    def loader(*_args: object, **_kwargs: object) -> TinyLoaded50MPackage:
        runtime = make_runtime()
        loaded_runtimes.append(runtime)
        return TinyLoaded50MPackage(runtime, settings.model_package)

    def strategy_factory(config: object) -> NeuroOnlineStrategy:
        assert not isinstance(config, type)
        strategy = NeuroOnlineStrategy(config)  # type: ignore[arg-type]
        strategies.append(strategy)
        return strategy

    original_initialize = NeuroOnlineStrategy.initialize

    def recording_initialize(self: NeuroOnlineStrategy, **kwargs: object) -> None:
        original_initialize(self, **kwargs)  # type: ignore[arg-type]
        backend = self.forward_model.backend
        online_before["backbone"] = clone_state(backend.adapter.backbone)  # type: ignore[attr-defined]
        online_before["head"] = clone_state(backend.adapter.classifier.head)  # type: ignore[attr-defined]
        online_before["generator"] = clone_state(self.generator)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(NeuroOnlineStrategy, "initialize", recording_initialize)
    try:
        summary = seq.run_evaluation(
            settings,
            package_loader=loader,
            strategy_factory=strategy_factory,
        )
    finally:
        monkeypatch.undo()

    assert len(loaded_runtimes) == 2
    assert loaded_runtimes[0] is not loaded_runtimes[1]
    assert len(strategies) == 1
    strategy = strategies[0]
    online_backend = loaded_runtimes[1].backend

    assert summary["runtime_package"]["model_type"] == "model_50m"
    assert summary["runtime_package"]["package_step_sec"] == pytest.approx(0.5)
    assert summary["protocol"]["step_sec"] == pytest.approx(4.0)
    assert summary["protocol"]["package_step_sec_ignored_for_trial_sequence"] is True
    assert summary["data"]["trial_ids_preserved_in_hdf5_order"] == ["101", "55", "88", "7"]
    assert summary["static"]["metrics"].keys() == summary["neuroonline"]["metrics"].keys()
    assert summary["static"]["updates"].keys() == summary["neuroonline"]["updates"].keys()
    assert summary["identity_initialization_check"]["equivalent"] is True

    records = [
        json.loads(line)
        for line in (settings.output_dir / "trial_predictions.jsonl").read_text().splitlines()
        if '"mode":"neuroonline"' in line
    ]
    assert [record["update_step_used"] for record in records] == [0, 0, 1, 1]
    assert [record["update_applied"] for record in records] == [False, True, False, True]
    assert records[1]["model_revision_used"] == "neuroonline-0"
    assert records[1]["model_revision_after_prediction"] == "neuroonline-1"
    assert records[2]["model_revision_used"] == "neuroonline-1"
    assert summary["neuroonline"]["metrics"]["after_first_update"]["num_samples"] == 2

    assert not changed_parameter_names(online_backend.adapter.backbone, online_before["backbone"])
    assert changed_parameter_names(strategy.generator, online_before["generator"])
    assert changed_parameter_names(online_backend.adapter.classifier.head, online_before["head"])
    assert not any(parameter.requires_grad for parameter in online_backend.adapter.backbone.parameters())
    assert online_backend.adapter.backbone.training is False
    assert strategy.update_step == 2


def test_shared_evaluator_output_structure_is_model_type_independent(
    tmp_path: Path,
) -> None:
    data_path = write_sequential_dataset(tmp_path / "trials.h5")
    settings = dataclass_replace(
        make_settings(tmp_path, data_path),
        online_strategy="none",
    )
    output_shapes: dict[
        str,
        tuple[frozenset[str], frozenset[str], frozenset[str]],
    ] = {}

    for model_type in ("labram", "cbramod", "model_50m"):
        def loader(*_args: object, **_kwargs: object) -> TinyLoaded50MPackage:
            package = TinyLoaded50MPackage(make_runtime(), settings.model_package)
            package.model_type = model_type
            package.model_name = f"{model_type}-mock"
            return package

        mode_output = settings.output_dir / model_type
        summary = seq.run_evaluation(
            dataclass_replace(settings, output_dir=mode_output),
            package_loader=loader,
        )
        record = json.loads((mode_output / "trial_predictions.jsonl").read_text().splitlines()[0])
        output_shapes[model_type] = (
            frozenset(summary["static"].keys()),
            frozenset(summary["static"]["metrics"].keys()),
            frozenset(record.keys()),
        )

    assert len(set(output_shapes.values())) == 1


def test_50m_static_mode_creates_no_online_strategy_or_parameter_update(
    tmp_path: Path,
) -> None:
    data_path = write_sequential_dataset(tmp_path / "trials.h5")
    settings = dataclass_replace(
        make_settings(tmp_path, data_path),
        online_strategy="none",
    )
    runtime = make_runtime()
    backend = runtime.backend
    backbone_before = clone_state(backend.adapter.backbone)  # type: ignore[attr-defined]
    head_before = clone_state(backend.adapter.classifier.head)  # type: ignore[attr-defined]

    def loader(*_args: object, **_kwargs: object) -> TinyLoaded50MPackage:
        return TinyLoaded50MPackage(runtime, settings.model_package)

    def strategy_factory(_config: object) -> NeuroOnlineStrategy:
        raise AssertionError("Static mode must not create NeuroOnlineStrategy.")

    summary = seq.run_evaluation(
        settings,
        package_loader=loader,
        strategy_factory=strategy_factory,
    )

    assert summary["static"] is not None
    assert summary["neuroonline"] is None
    assert not changed_parameter_names(backend.adapter.backbone, backbone_before)  # type: ignore[attr-defined]
    assert not changed_parameter_names(backend.adapter.classifier.head, head_before)  # type: ignore[attr-defined]
