from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from bci_dayloop.models.model_50m.config import (
    Model50MConfig,
    STANDARD_64_CHANNELS,
)
from bci_dayloop.models.model_50m import runtime as runtime_module


CLASS_NAMES = ("left_hand", "right_hand", "feet", "tongue")


def _checkpoint_files(tmp_path: Path) -> tuple[Path, Path]:
    backbone = tmp_path / "model_deploy.pt"
    classifier = tmp_path / "head.pt"
    backbone.write_bytes(b"backbone")
    torch.save(
        {
            "format_version": 1,
            "head_state_dict": {},
            "metadata": {},
        },
        classifier,
    )
    return backbone, classifier


def _four_second_config(tmp_path: Path, *, aggregation: str = "flatten") -> Model50MConfig:
    backbone, classifier = _checkpoint_files(tmp_path)
    return Model50MConfig(
        checkpoint_path=backbone,
        classifier_path=classifier,
        target_sample_rate=100.0,
        window_seconds=4.0,
        patch_seconds=1.0,
        patch_stride_seconds=1.0,
        model_n_time_patches=10,
        aggregation=aggregation,
        num_classes=4,
    )


def test_4s_flatten_geometry_keeps_pretrained_10_time_positions(
    tmp_path: Path,
) -> None:
    config = _four_second_config(tmp_path)

    assert config.window_seconds == 4.0
    assert config.target_num_points == 400
    assert config.patch_num_points == 100
    assert config.patch_stride_points == 100
    assert config.num_time_patches == 4
    assert config.model_n_time_patches == 10
    assert config.num_tokens == 64 * 4 == 256
    assert config.classifier_input_dim == 256 * 512 == 131_072


def test_4s_mean_aggregation_uses_d_model_feature_dimension(
    tmp_path: Path,
) -> None:
    config = _four_second_config(tmp_path, aggregation="mean")

    assert config.num_tokens == 256
    assert config.classifier_input_dim == config.d_model == 512


def test_model_time_positions_cannot_be_smaller_than_input_patches(
    tmp_path: Path,
) -> None:
    backbone, classifier = _checkpoint_files(tmp_path)

    with pytest.raises(
        ValueError,
        match="cannot be smaller than the number of input time patches",
    ):
        Model50MConfig(
            checkpoint_path=backbone,
            classifier_path=classifier,
            target_sample_rate=100.0,
            window_seconds=4.0,
            patch_seconds=1.0,
            patch_stride_seconds=1.0,
            model_n_time_patches=3,
        )


def test_build_runtime_constructs_4s_config_without_loading_real_weights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backbone, classifier = _checkpoint_files(tmp_path)
    captured: dict[str, Any] = {}

    class FakeAdapter:
        def __init__(
            self,
            *,
            config: Model50MConfig,
            class_names: tuple[str, ...],
            strict_head_metadata: bool,
        ) -> None:
            captured["adapter_config"] = config
            captured["adapter_class_names"] = class_names
            captured["strict_head_metadata"] = strict_head_metadata

    class FakePreprocessor:
        def __init__(
            self,
            *,
            config: Model50MConfig,
            channel_names: tuple[str, ...],
            sample_rate: float,
            input_unit: str,
        ) -> None:
            captured["preprocessor_config"] = config
            captured["channel_names"] = channel_names
            captured["sample_rate"] = sample_rate
            captured["input_unit"] = input_unit

    monkeypatch.setattr(runtime_module, "Model50MAdapter", FakeAdapter)
    monkeypatch.setattr(
        runtime_module,
        "Model50MPipelinePreprocessor",
        FakePreprocessor,
    )

    runtime = runtime_module.build_50m_runtime(
        checkpoint_path=backbone,
        classifier_path=classifier,
        channel_names=("C3", "Cz", "C4"),
        sample_rate=250.0,
        input_unit="uV",
        class_names=CLASS_NAMES,
        device="cpu",
        target_sample_rate=100.0,
        window_seconds=4.0,
        patch_seconds=1.0,
        patch_stride_seconds=1.0,
        output_layer_idx=8,
        aggregation="flatten",
        strict_window_duration=True,
        strict_head_metadata=True,
        model_n_time_patches=10,
    )

    assert runtime.config.target_num_points == 400
    assert runtime.config.num_time_patches == 4
    assert runtime.config.model_n_time_patches == 10
    assert runtime.config.num_tokens == 256
    assert runtime.config.classifier_input_dim == 131_072
    assert runtime.class_names == CLASS_NAMES
    assert captured["adapter_config"] is runtime.config
    assert captured["preprocessor_config"] is runtime.config
    assert captured["channel_names"] == ("C3", "Cz", "C4")
    assert captured["sample_rate"] == 250.0
    assert captured["input_unit"] == "uV"


def test_build_runtime_requires_existing_checkpoint_files(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing.pt"
    torch.save(
        {"format_version": 1, "head_state_dict": {}, "metadata": {}},
        existing,
    )

    with pytest.raises(
        FileNotFoundError,
        match="50M backbone checkpoint was not found",
    ):
        runtime_module.build_50m_runtime(
            checkpoint_path=tmp_path / "missing-backbone.pt",
            classifier_path=existing,
            channel_names=("C3",),
            sample_rate=250.0,
            input_unit="uV",
            class_names=CLASS_NAMES,
            window_seconds=4.0,
            model_n_time_patches=10,
        )

    with pytest.raises(
        FileNotFoundError,
        match="50M classifier checkpoint was not found",
    ):
        runtime_module.build_50m_runtime(
            checkpoint_path=existing,
            classifier_path=tmp_path / "missing-head.pt",
            channel_names=("C3",),
            sample_rate=250.0,
            input_unit="uV",
            class_names=CLASS_NAMES,
            window_seconds=4.0,
            model_n_time_patches=10,
        )


def test_build_runtime_infers_mlp_head_from_checkpoint_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backbone, classifier = _checkpoint_files(tmp_path)
    torch.save(
        {
            "format_version": 2,
            "head_state_dict": {},
            "metadata": {
                "head_type": "mlp",
                "head_hidden_dim": 32,
                "head_dropout": 0.2,
                "head_norm": "layernorm",
            },
        },
        classifier,
    )
    captured: dict[str, Any] = {}

    class FakeAdapter:
        def __init__(self, *, config: Model50MConfig, **kwargs: Any) -> None:
            captured["config"] = config

    class FakePreprocessor:
        def __init__(self, **kwargs: Any) -> None:
            captured["preprocessor"] = kwargs

    monkeypatch.setattr(runtime_module, "Model50MAdapter", FakeAdapter)
    monkeypatch.setattr(runtime_module, "Model50MPipelinePreprocessor", FakePreprocessor)

    runtime = runtime_module.build_50m_runtime(
        checkpoint_path=backbone,
        classifier_path=classifier,
        channel_names=("C3",),
        sample_rate=250.0,
        input_unit="uV",
        class_names=CLASS_NAMES,
        target_sample_rate=100.0,
        window_seconds=4.0,
        patch_seconds=1.0,
        patch_stride_seconds=1.0,
        output_layer_idx=8,
        aggregation="flatten",
    )

    assert runtime.config.head_type == "mlp"
    assert runtime.config.head_hidden_dim == 32
    assert runtime.config.head_dropout == 0.2
    assert runtime.config.head_norm == "layernorm"

def test_build_runtime_from_metadata_forwards_4s_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backbone, classifier = _checkpoint_files(tmp_path)
    metadata = SimpleNamespace(
        channel_names=["C3", "Cz", "C4"],
        sample_rate=250.0,
        unit="uV",
        class_names=list(CLASS_NAMES),
    )
    captured: dict[str, Any] = {}
    sentinel = object()

    def fake_build_50m_runtime(**kwargs: Any) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        runtime_module,
        "build_50m_runtime",
        fake_build_50m_runtime,
    )

    result = runtime_module.build_50m_runtime_from_metadata(
        checkpoint_path=backbone,
        classifier_path=classifier,
        metadata=metadata,
        device="cpu",
        target_sample_rate=100.0,
        window_seconds=4.0,
        model_n_time_patches=10,
        patch_seconds=1.0,
        patch_stride_seconds=1.0,
        output_layer_idx=8,
        aggregation="flatten",
    )

    assert result is sentinel
    assert captured["channel_names"] == metadata.channel_names
    assert captured["sample_rate"] == 250.0
    assert captured["input_unit"] == "uV"
    assert captured["class_names"] == metadata.class_names
    assert captured["window_seconds"] == 4.0
    assert captured["model_n_time_patches"] == 10
    assert captured["strict_window_duration"] is True


def test_build_runtime_from_metadata_rejects_missing_fields(
    tmp_path: Path,
) -> None:
    backbone, classifier = _checkpoint_files(tmp_path)
    metadata = SimpleNamespace(
        channel_names=["C3"],
        sample_rate=250.0,
    )

    with pytest.raises(AttributeError, match="missing required attributes"):
        runtime_module.build_50m_runtime_from_metadata(
            checkpoint_path=backbone,
            classifier_path=classifier,
            metadata=metadata,
            target_sample_rate=100.0,
            window_seconds=4.0,
            model_n_time_patches=10,
            patch_seconds=1.0,
            patch_stride_seconds=1.0,
            output_layer_idx=8,
            aggregation="flatten",
        )


def test_predict_raw_window_returns_prediction_and_diagnostics(
    tmp_path: Path,
) -> None:
    config = _four_second_config(tmp_path)

    class Timing:
        def to_dict(self) -> dict[str, float]:
            return {
                "backbone_ms": 12.5,
                "classifier_ms": 0.4,
            }

    class FakeAdapter:
        last_timing = Timing()

        def __init__(self) -> None:
            self.last_input: Any = None

        def predict_proba(self, model_input: Any) -> np.ndarray:
            self.last_input = model_input
            return np.asarray(
                [[0.05, 0.10, 0.80, 0.05]],
                dtype=np.float32,
            )

    class FakePreprocessor:
        sample_rate = 250.0
        input_unit = "uV"
        channel_names = ("C3", "Cz", "C4")
        last_diagnostics = SimpleNamespace(
            mapped_channel_count=3,
            missing_channel_count=61,
            unknown_channel_names=("UNKNOWN",),
            notes=("mapped to standard montage",),
        )

        def transform(
            self,
            raw_window: np.ndarray,
            sample_rate: float,
            input_unit: str,
        ) -> dict[str, np.ndarray]:
            assert raw_window.shape == (3, 1000)
            assert sample_rate == 250.0
            assert input_unit == "uV"
            return {
                "signal": np.zeros((64, 4, 100), dtype=np.float32),
                "channel_mask": np.ones(64, dtype=np.float32),
            }

    adapter = FakeAdapter()
    runtime = runtime_module.Model50MRuntime(
        config=config,
        adapter=adapter,  # type: ignore[arg-type]
        preprocessor=FakePreprocessor(),  # type: ignore[arg-type]
        class_names=CLASS_NAMES,
    )

    prediction = runtime.predict_raw_window(
        np.zeros((3, 1000), dtype=np.float32)
    )

    assert prediction.prediction == 2
    assert prediction.confidence == pytest.approx(0.80)
    assert prediction.probabilities == pytest.approx((0.05, 0.10, 0.80, 0.05))
    assert prediction.adapter_timing == {
        "backbone_ms": 12.5,
        "classifier_ms": 0.4,
    }
    assert prediction.mapped_channel_count == 3
    assert prediction.missing_channel_count == 61
    assert prediction.unknown_channel_names == ("UNKNOWN",)
    assert prediction.notes == ("mapped to standard montage",)
    assert adapter.last_input["signal"].shape == (1, 64, 4, 100)


def test_health_check_reports_4s_runtime_contract(tmp_path: Path) -> None:
    config = _four_second_config(tmp_path)

    class FakeAdapter:
        def health_check(self) -> dict[str, str]:
            return {"status": "ok", "model": "fake-50m"}

    class FakePreprocessor:
        sample_rate = 250.0
        input_unit = "uV"
        channel_names = ("C3", "Cz", "C4")

    runtime = runtime_module.Model50MRuntime(
        config=config,
        adapter=FakeAdapter(),  # type: ignore[arg-type]
        preprocessor=FakePreprocessor(),  # type: ignore[arg-type]
        class_names=CLASS_NAMES,
    )

    report = runtime.health_check()

    assert report["status"] == "ok"
    assert report["runtime"]["window_seconds"] == 4.0
    assert report["runtime"]["target_shape"] == [64, 400]
    assert report["runtime"]["num_tokens"] == 256
    assert report["runtime"]["classifier_input_dim"] == 131_072
