from __future__ import annotations

"""Shared-feature offline inference for the three frozen 50M heads."""

import hashlib
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from bci_dayloop.models.model_50m.backbone import Model50MBackbone
from bci_dayloop.models.model_50m.classifier import FeatureAggregator, build_classification_head
from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.models.model_50m.preprocessing import Model50MPreprocessor, PreprocessResult
from bci_dayloop.models.model_50m.tokenization import Model50MTokenizer
from bci_dayloop.runtime.types import RawEEGWindow


TASK_OUTPUT_DIMS: dict[str, int] = {
    "workload": 2,
    "attention": 3,
    "emotion": 3,
}

_REQUIRED_SHARED_CONTRACT: dict[str, Any] = {
    "feature_dim": 65_536,
    "window_seconds": 2.0,
    "target_sample_rate": 100.0,
    "embedding_layer_resolved": 9,
    "embedding_layer_internal_index": 8,
    "output_layer_idx": 8,
    "aggregation": "flatten",
    "d_model": 512,
    "num_tokens": 128,
    "n_channels": 64,
    "model_n_time_patches": 10,
    "patch_seconds": 1.0,
    "patch_stride_seconds": 1.0,
    "backbone_adaptation": "frozen",
    "freeze_backbone": True,
    "missing_channel_strategy": "zero_with_valid_mask",
}


@dataclass(frozen=True, slots=True)
class HeadPrediction:
    """Semantic prediction from one task-specific classification head."""

    label_id: int
    label: str
    confidence: float
    probabilities: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class MultiHeadPrediction:
    """Stable public prediction contract for the three mental-state heads."""

    workload: HeadPrediction
    attention: HeadPrediction
    emotion: HeadPrediction


@dataclass(frozen=True, slots=True)
class MultiHeadInferenceDiagnostics:
    """Diagnostics from the most recent single-window inference call."""

    preprocessing_calls: int
    backbone_forwards: int
    head_forwards: dict[str, int]
    preprocessed_shape: tuple[int, ...]
    selected_embedding_shape: tuple[int, ...]
    shared_feature_shape: tuple[int, ...]
    logit_shapes: dict[str, tuple[int, ...]
    ]
    mapped_channel_count: int
    missing_channel_count: int
    missing_standard_channel_names: tuple[str, ...]
    unknown_channel_names: tuple[str, ...]
    duplicate_channel_count: int
    preprocessing_latency_ms: float
    backbone_latency_ms: float
    heads_latency_ms: float


@dataclass(frozen=True, slots=True)
class HeadCheckpointInfo:
    """Self-described, validated metadata for one loaded Linear head."""

    task: str
    checkpoint_path: Path
    class_names: tuple[str, ...]
    input_dim: int
    output_dim: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _LoadedHead:
    info: HeadCheckpointInfo
    module: nn.Module


def _resolve_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _safe_torch_load(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path}: checkpoint must be a mapping.")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_class_names(
    *,
    task: str,
    metadata: Mapping[str, Any],
    fallback: Sequence[str] | None,
) -> tuple[str, ...]:
    stored = metadata.get("class_names")
    if isinstance(stored, (list, tuple)) and all(str(item) for item in stored):
        return tuple(str(item) for item in stored)
    if fallback is None:
        raise ValueError(
            f"{task}: checkpoint metadata has no class_names. Provide an "
            "explicit matching class-name fallback."
        )
    names = tuple(str(item) for item in fallback)
    if not names or any(not item for item in names):
        raise ValueError(f"{task}: class-name fallback cannot be empty.")
    warnings.warn(
        f"{task}: checkpoint metadata has no class_names; using explicit "
        "fallback names.",
        RuntimeWarning,
        stacklevel=3,
    )
    return names


def _validate_head_metadata(task: str, metadata: Mapping[str, Any]) -> None:
    if metadata.get("head_type", "linear") != "linear":
        raise ValueError(
            f"{task}: expected head_type='linear', got "
            f"{metadata.get('head_type')!r}."
        )
    for key, expected in _REQUIRED_SHARED_CONTRACT.items():
        actual = metadata.get(key)
        if actual != expected:
            raise ValueError(
                f"{task}: incompatible shared feature contract for {key}: "
                f"expected {expected!r}, actual {actual!r}."
            )


def _validate_runtime_config(config: Model50MConfig) -> None:
    actual_contract = {
        "feature_dim": config.classifier_input_dim,
        "window_seconds": config.window_seconds,
        "target_sample_rate": config.target_sample_rate,
        "output_layer_idx": config.output_layer_idx,
        "aggregation": config.aggregation,
        "d_model": config.d_model,
        "num_tokens": config.num_tokens,
        "n_channels": config.n_channels,
        "model_n_time_patches": config.model_n_time_patches,
        "patch_seconds": config.patch_seconds,
        "patch_stride_seconds": config.patch_stride_seconds,
    }
    for field, expected in _REQUIRED_SHARED_CONTRACT.items():
        if field not in actual_contract:
            continue
        actual = actual_contract[field]
        if actual != expected:
            raise ValueError(
                f"50M runtime config field {field}: expected {expected!r}, "
                f"actual {actual!r}."
            )


def _load_head(
    *,
    task: str,
    checkpoint_path: Path | str,
    config: Model50MConfig,
    class_name_fallback: Sequence[str] | None,
) -> _LoadedHead:
    path = _resolve_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"{task}: head checkpoint not found: {path}")
    payload = _safe_torch_load(path)
    raw_metadata = payload.get("metadata")
    raw_state_dict = payload.get("head_state_dict")
    if not isinstance(raw_metadata, Mapping):
        raise TypeError(f"{task}: checkpoint metadata must be a mapping.")
    if not isinstance(raw_state_dict, Mapping):
        raise TypeError(f"{task}: checkpoint head_state_dict must be a mapping.")
    metadata = dict(raw_metadata)
    _validate_head_metadata(task, metadata)

    weight = raw_state_dict.get("linear.weight")
    bias = raw_state_dict.get("linear.bias")
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise KeyError(
            f"{task}: expected linear.weight and linear.bias in head_state_dict."
        )
    if weight.ndim != 2 or bias.ndim != 1:
        raise ValueError(f"{task}: invalid Linear state-dict tensor shapes.")
    output_dim, input_dim = (int(value) for value in weight.shape)
    if tuple(bias.shape) != (output_dim,):
        raise ValueError(
            f"{task}: linear.bias shape {tuple(bias.shape)} does not match "
            f"linear.weight shape {tuple(weight.shape)}."
        )
    expected_output_dim = TASK_OUTPUT_DIMS[task]
    if input_dim != config.classifier_input_dim:
        raise ValueError(
            f"{task}: Linear input dim expected {config.classifier_input_dim}, "
            f"actual {input_dim}."
        )
    if output_dim != expected_output_dim:
        raise ValueError(
            f"{task}: Linear output dim expected {expected_output_dim}, "
            f"actual {output_dim}."
        )
    if int(metadata.get("num_classes", -1)) != output_dim:
        raise ValueError(
            f"{task}: metadata num_classes={metadata.get('num_classes')!r} "
            f"does not match Linear output dim {output_dim}."
        )
    class_names = _resolve_class_names(
        task=task, metadata=metadata, fallback=class_name_fallback
    )
    if len(class_names) != output_dim:
        raise ValueError(
            f"{task}: {len(class_names)} class_names for Linear({input_dim}, "
            f"{output_dim})."
        )

    head_config = Model50MConfig(
        checkpoint_path=config.checkpoint_path,
        device=str(config.device),
        target_sample_rate=config.target_sample_rate,
        window_seconds=config.window_seconds,
        patch_seconds=config.patch_seconds,
        patch_stride_seconds=config.patch_stride_seconds,
        model_n_time_patches=config.model_n_time_patches,
        output_layer_idx=config.output_layer_idx,
        aggregation=config.aggregation,
        num_classes=output_dim,
        head_type="linear",
    )
    module = build_classification_head(head_config).to(config.device)
    module.load_state_dict(raw_state_dict, strict=True)
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    if (
        int(getattr(module, "input_dim", -1)) != input_dim
        or int(getattr(module, "num_classes", -1)) != output_dim
    ):
        raise RuntimeError(f"{task}: constructed head dimensions do not match checkpoint.")
    return _LoadedHead(
        info=HeadCheckpointInfo(
            task=task,
            checkpoint_path=path,
            class_names=class_names,
            input_dim=input_dim,
            output_dim=output_dim,
            metadata=metadata,
        ),
        module=module,
    )


class MultiHeadPredictor:
    """Run one frozen 50M feature extraction and three task heads once each."""

    def __init__(
        self,
        *,
        config: Model50MConfig,
        preprocessor: Model50MPreprocessor,
        tokenizer: Model50MTokenizer,
        backbone: Model50MBackbone,
        heads: Mapping[str, _LoadedHead],
    ) -> None:
        self.config = config
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.backbone = backbone
        self.aggregator = FeatureAggregator(config.aggregation).to(backbone.device).eval()
        self._heads = dict(heads)
        self._last_diagnostics: MultiHeadInferenceDiagnostics | None = None
        self._validate_initialization()

    @classmethod
    def from_checkpoints(
        cls,
        *,
        backbone_checkpoint: Path | str,
        workload_head: Path | str,
        attention_head: Path | str,
        emotion_head: Path | str,
        device: str = "cpu",
        workload_class_names: Sequence[str] | None = None,
        attention_class_names: Sequence[str] | None = None,
        emotion_class_names: Sequence[str] | None = None,
    ) -> "MultiHeadPredictor":
        """Build a predictor after fail-fast validation of all checkpoints."""
        backbone_path = _resolve_path(backbone_checkpoint)
        if not backbone_path.is_file():
            raise FileNotFoundError(f"Backbone checkpoint not found: {backbone_path}")
        config = Model50MConfig(
            checkpoint_path=backbone_path,
            device=device,
            target_sample_rate=100.0,
            window_seconds=2.0,
            patch_seconds=1.0,
            patch_stride_seconds=1.0,
            model_n_time_patches=10,
            output_layer_idx=8,
            aggregation="flatten",
            num_classes=3,
            head_type="linear",
        )
        if config.classifier_input_dim != _REQUIRED_SHARED_CONTRACT["feature_dim"]:
            raise RuntimeError(
                "Current 50M runtime feature dimension does not match the "
                f"three-head contract: {config.classifier_input_dim}."
            )
        _validate_runtime_config(config)
        return cls.from_config_and_checkpoints(
            config=config,
            workload_head=workload_head,
            attention_head=attention_head,
            emotion_head=emotion_head,
            workload_class_names=workload_class_names,
            attention_class_names=attention_class_names,
            emotion_class_names=emotion_class_names,
        )

    @classmethod
    def from_config_and_checkpoints(
        cls,
        *,
        config: Model50MConfig,
        workload_head: Path | str,
        attention_head: Path | str,
        emotion_head: Path | str,
        workload_class_names: Sequence[str] | None = None,
        attention_class_names: Sequence[str] | None = None,
        emotion_class_names: Sequence[str] | None = None,
    ) -> "MultiHeadPredictor":
        """Build from an already-resolved shared runtime configuration."""
        backbone_path = _resolve_path(config.checkpoint_path)
        if not backbone_path.is_file():
            raise FileNotFoundError(f"Backbone checkpoint not found: {backbone_path}")
        if config.classifier_input_dim != _REQUIRED_SHARED_CONTRACT["feature_dim"]:
            raise RuntimeError(
                "Current 50M runtime feature dimension does not match the "
                f"three-head contract: {config.classifier_input_dim}."
            )
        _validate_runtime_config(config)
        heads = {
            "workload": _load_head(
                task="workload", checkpoint_path=workload_head, config=config,
                class_name_fallback=workload_class_names,
            ),
            "attention": _load_head(
                task="attention", checkpoint_path=attention_head, config=config,
                class_name_fallback=attention_class_names,
            ),
            "emotion": _load_head(
                task="emotion", checkpoint_path=emotion_head, config=config,
                class_name_fallback=emotion_class_names,
            ),
        }
        expected_backbone_hash = _sha256_file(backbone_path)
        referenced_hashes = {
            str(loaded.info.metadata.get("backbone_sha256"))
            for loaded in heads.values()
        }
        if referenced_hashes != {expected_backbone_hash}:
            raise ValueError(
                "Head checkpoints do not all reference the supplied backbone: "
                f"expected {expected_backbone_hash}, actual {sorted(referenced_hashes)}."
            )
        templates = {
            tuple(loaded.info.metadata.get("channel_template", ()))
            for loaded in heads.values()
        }
        if templates != {config.standard_channels}:
            raise ValueError(
                "Head checkpoints disagree with the current STANDARD_64_CHANNELS "
                "template."
            )
        return cls(
            config=config,
            preprocessor=Model50MPreprocessor(config),
            tokenizer=Model50MTokenizer(config),
            backbone=Model50MBackbone(config=config, load_checkpoint=True, freeze=True),
            heads=heads,
        )

    @property
    def device(self) -> torch.device:
        return self.backbone.device

    @property
    def window_seconds(self) -> float:
        """Required raw-window duration for decoder contract validation."""
        return float(self.config.window_seconds)

    @property
    def head_info(self) -> Mapping[str, HeadCheckpointInfo]:
        return {task: loaded.info for task, loaded in self._heads.items()}

    @property
    def last_diagnostics(self) -> MultiHeadInferenceDiagnostics | None:
        return self._last_diagnostics

    def _validate_initialization(self) -> None:
        if set(self._heads) != set(TASK_OUTPUT_DIMS):
            raise ValueError(
                "MultiHeadPredictor requires exactly workload, attention, and "
                f"emotion heads; got {sorted(self._heads)}."
            )
        if self.config.classifier_input_dim != _REQUIRED_SHARED_CONTRACT["feature_dim"]:
            raise ValueError(
                "MultiHeadPredictor requires 65536-d flatten features, got "
                f"{self.config.classifier_input_dim}."
            )
        for task, loaded in self._heads.items():
            if loaded.info.input_dim != self.config.classifier_input_dim:
                raise ValueError(
                    f"{task}: input dim expected {self.config.classifier_input_dim}, "
                    f"actual {loaded.info.input_dim}."
                )
            expected_output = TASK_OUTPUT_DIMS[task]
            if loaded.info.output_dim != expected_output:
                raise ValueError(
                    f"{task}: output dim expected {expected_output}, "
                    f"actual {loaded.info.output_dim}."
                )
            loaded.module.to(self.device).eval()
            for parameter in loaded.module.parameters():
                parameter.requires_grad_(False)
            if loaded.module.training:
                raise RuntimeError(f"{task}: head must be in eval mode.")
        self.backbone.eval()
        if self.backbone.training:
            raise RuntimeError("Frozen 50M backbone must be in eval mode.")

    @staticmethod
    def _signal_from_window(window: RawEEGWindow) -> np.ndarray:
        signal = np.asarray(window.data)
        if window.layout == "TC":
            signal = signal.T
        elif window.layout != "CT":
            raise ValueError(f"Unsupported RawEEGWindow layout: {window.layout!r}.")
        if signal.ndim != 2:
            raise ValueError(
                "RawEEGWindow data must be [C, T] or [T, C], got "
                f"{signal.shape}."
            )
        if signal.shape[0] != len(window.channel_names):
            raise ValueError(
                "RawEEGWindow channel count does not match channel_names: "
                f"{signal.shape[0]} != {len(window.channel_names)}."
            )
        if not np.isfinite(signal).all():
            raise ValueError("RawEEGWindow data contains NaN or Inf.")
        return signal.astype(np.float32, copy=False)

    @staticmethod
    def _prediction_from_logits(
        *, task: str, logits: torch.Tensor, class_names: tuple[str, ...]
    ) -> HeadPrediction:
        if tuple(logits.shape) != (1, len(class_names)):
            raise RuntimeError(
                f"{task}: expected logits shape {(1, len(class_names))}, got "
                f"{tuple(logits.shape)}."
            )
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"{task}: logits contain NaN or Inf.")
        probabilities = torch.softmax(logits, dim=-1)
        if not torch.isfinite(probabilities).all():
            raise RuntimeError(f"{task}: probabilities contain NaN or Inf.")
        if not torch.allclose(
            probabilities.sum(dim=-1),
            torch.ones_like(probabilities[:, 0]), atol=1e-5, rtol=1e-5,
        ):
            raise RuntimeError(f"{task}: probabilities do not sum to one.")
        values = tuple(float(value) for value in probabilities[0].detach().cpu().tolist())
        label_id = int(probabilities.argmax(dim=-1).item())
        return HeadPrediction(
            label_id=label_id,
            label=class_names[label_id],
            confidence=values[label_id],
            probabilities=values,
        )

    @torch.inference_mode()
    def predict(self, window: RawEEGWindow) -> MultiHeadPrediction:
        """Predict the three task labels for exactly one runtime EEG window."""
        if self.backbone.training or any(
            loaded.module.training for loaded in self._heads.values()
        ):
            raise RuntimeError(
                "MultiHeadPredictor modules must remain in eval mode during "
                "inference."
            )
        signal = self._signal_from_window(window)
        preprocessing_started = time.perf_counter()
        preprocessed: PreprocessResult = self.preprocessor(
            signal=signal,
            channel_names=window.channel_names,
            original_sample_rate=window.sample_rate,
            input_unit=window.unit,
        )
        preprocessing_latency_ms = (
            time.perf_counter() - preprocessing_started
        ) * 1000.0
        if not np.isfinite(preprocessed.signal).all():
            raise RuntimeError("Preprocessed 50M input contains NaN or Inf.")
        batch = self.tokenizer(preprocessed).as_batch(device=self.device)
        backbone_started = time.perf_counter()
        selected_embedding = self.backbone.extract_embeddings(
            batch=batch, return_layer_idx=self.config.output_layer_idx
        )
        shared_feature = self.aggregator(
            token_embeddings=selected_embedding, token_valid_mask=batch.token_valid_mask
        )
        backbone_latency_ms = (
            time.perf_counter() - backbone_started
        ) * 1000.0
        if tuple(shared_feature.shape) != (1, self.config.classifier_input_dim):
            raise RuntimeError(
                "Shared feature must have shape [1, 65536], got "
                f"{tuple(shared_feature.shape)}."
            )
        if not torch.isfinite(shared_feature).all():
            raise RuntimeError("Shared feature contains NaN or Inf.")
        heads_started = time.perf_counter()
        logits = {task: loaded.module(shared_feature) for task, loaded in self._heads.items()}
        heads_latency_ms = (time.perf_counter() - heads_started) * 1000.0
        predictions = {
            task: self._prediction_from_logits(
                task=task, logits=value, class_names=self._heads[task].info.class_names
            )
            for task, value in logits.items()
        }
        missing_names = tuple(
            name
            for name, valid in zip(
                self.config.standard_channels, preprocessed.channel_valid_mask, strict=True
            )
            if valid < 0.5
        )
        self._last_diagnostics = MultiHeadInferenceDiagnostics(
            preprocessing_calls=1,
            backbone_forwards=1,
            head_forwards={task: 1 for task in self._heads},
            preprocessed_shape=tuple(preprocessed.signal.shape),
            selected_embedding_shape=tuple(selected_embedding.shape),
            shared_feature_shape=tuple(shared_feature.shape),
            logit_shapes={task: tuple(value.shape) for task, value in logits.items()},
            mapped_channel_count=preprocessed.mapped_channel_count,
            missing_channel_count=preprocessed.missing_channel_count,
            missing_standard_channel_names=missing_names,
            unknown_channel_names=preprocessed.unknown_channel_names,
            duplicate_channel_count=preprocessed.duplicate_channel_count,
            preprocessing_latency_ms=preprocessing_latency_ms,
            backbone_latency_ms=backbone_latency_ms,
            heads_latency_ms=heads_latency_ms,
        )
        return MultiHeadPrediction(
            workload=predictions["workload"],
            attention=predictions["attention"],
            emotion=predictions["emotion"],
        )
