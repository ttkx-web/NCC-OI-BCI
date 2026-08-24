"""One-preprocess, one-Backbone, three-head inference implementation."""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from bci_dayloop.applications.three_mental_states.contract import (
    HeadCheckpointInfo, HeadPrediction, SHARED_FEATURE_CONTRACT, TASK_OUTPUT_DIMS,
    TASKS, ThreeMentalStateDiagnostics, ThreeMentalStatePrediction,
)
from bci_dayloop.models.model_50m.backbone import Model50MBackbone
from bci_dayloop.models.model_50m.classifier import FeatureAggregator, build_classification_head
from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.models.model_50m.preprocessing import Model50MPreprocessor, PreprocessResult
from bci_dayloop.models.model_50m.tokenization import Model50MTokenizer
from bci_dayloop.packages.common import safe_torch_load, sha256_file
from bci_dayloop.runtime.types import RawEEGWindow

@dataclass(frozen=True, slots=True)
class _LoadedHead:
    info: HeadCheckpointInfo
    module: nn.Module

def _resolve_names(task: str, metadata: Mapping[str, Any], fallback: Sequence[str] | None) -> tuple[str, ...]:
    stored = metadata.get("class_names")
    if isinstance(stored, (list, tuple)) and all(str(value) for value in stored):
        return tuple(str(value) for value in stored)
    if fallback is None:
        raise ValueError(f"{task}: checkpoint metadata has no class_names. Provide an explicit matching class-name fallback.")
    names = tuple(str(value) for value in fallback)
    if not names or any(not value for value in names):
        raise ValueError(f"{task}: class-name fallback cannot be empty.")
    warnings.warn(f"{task}: checkpoint metadata has no class_names; using explicit fallback names.", RuntimeWarning, stacklevel=3)
    return names

def _validate_runtime_config(config: Model50MConfig) -> None:
    actual = {"feature_dim": config.classifier_input_dim, "window_seconds": config.window_seconds,
              "target_sample_rate": config.target_sample_rate, "output_layer_idx": config.output_layer_idx,
              "aggregation": config.aggregation, "d_model": config.d_model, "num_tokens": config.num_tokens,
              "n_channels": config.n_channels, "model_n_time_patches": config.model_n_time_patches,
              "patch_seconds": config.patch_seconds, "patch_stride_seconds": config.patch_stride_seconds}
    for field, expected in SHARED_FEATURE_CONTRACT.items():
        if field in actual and actual[field] != expected:
            raise ValueError(f"50M runtime config field {field}: expected {expected!r}, actual {actual[field]!r}.")

def _load_head(*, task: str, checkpoint_path: Path | str, config: Model50MConfig, class_name_fallback: Sequence[str] | None) -> _LoadedHead:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{task}: head checkpoint not found: {path}")
    payload = safe_torch_load(path)
    metadata, state = payload.get("metadata"), payload.get("head_state_dict")
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{task}: checkpoint metadata must be a mapping.")
    if not isinstance(state, Mapping):
        raise TypeError(f"{task}: checkpoint head_state_dict must be a mapping.")
    metadata = dict(metadata)
    if metadata.get("head_type", "linear") != "linear":
        raise ValueError(f"{task}: expected head_type='linear', got {metadata.get('head_type')!r}.")
    for key, expected in SHARED_FEATURE_CONTRACT.items():
        if metadata.get(key) != expected:
            raise ValueError(f"{task}: incompatible shared feature contract for {key}: expected {expected!r}, actual {metadata.get(key)!r}.")
    weight, bias = state.get("linear.weight"), state.get("linear.bias")
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise KeyError(f"{task}: expected linear.weight and linear.bias in head_state_dict.")
    if weight.ndim != 2 or bias.ndim != 1 or tuple(bias.shape) != (weight.shape[0],):
        raise ValueError(f"{task}: invalid Linear state-dict tensor shapes.")
    output_dim, input_dim = map(int, weight.shape)
    if input_dim != config.classifier_input_dim:
        raise ValueError(f"{task}: Linear input dim expected {config.classifier_input_dim}, actual {input_dim}.")
    if output_dim != TASK_OUTPUT_DIMS[task]:
        raise ValueError(f"{task}: Linear output dim expected {TASK_OUTPUT_DIMS[task]}, actual {output_dim}.")
    if int(metadata.get("num_classes", -1)) != output_dim:
        raise ValueError(f"{task}: metadata num_classes={metadata.get('num_classes')!r} does not match Linear output dim {output_dim}.")
    names = _resolve_names(task, metadata, class_name_fallback)
    if len(names) != output_dim:
        raise ValueError(f"{task}: {len(names)} class_names for Linear({input_dim}, {output_dim}).")
    head_config = Model50MConfig(checkpoint_path=config.checkpoint_path, device=str(config.device), target_sample_rate=config.target_sample_rate,
        window_seconds=config.window_seconds, patch_seconds=config.patch_seconds, patch_stride_seconds=config.patch_stride_seconds,
        model_n_time_patches=config.model_n_time_patches, output_layer_idx=config.output_layer_idx, aggregation=config.aggregation,
        num_classes=output_dim, head_type="linear")
    module = build_classification_head(head_config).to(config.device)
    module.load_state_dict(state, strict=True)
    module.eval()
    for parameter in module.parameters(): parameter.requires_grad_(False)
    return _LoadedHead(HeadCheckpointInfo(task, path, names, input_dim, output_dim, metadata), module)

class ThreeMentalStatePredictor:
    """Run exactly one frozen 50M feature extraction and the three fixed heads."""
    def __init__(self, *, config: Model50MConfig, preprocessor: Model50MPreprocessor, tokenizer: Model50MTokenizer, backbone: Model50MBackbone, heads: Mapping[str, _LoadedHead]) -> None:
        self.config, self.preprocessor, self.tokenizer, self.backbone = config, preprocessor, tokenizer, backbone
        self.aggregator = FeatureAggregator(config.aggregation).to(backbone.device).eval()
        self._heads = dict(heads)
        self._last_diagnostics: ThreeMentalStateDiagnostics | None = None
        self._validate_initialization()
    @classmethod
    def from_checkpoints(cls, *, backbone_checkpoint: Path | str, workload_head: Path | str, attention_head: Path | str, emotion_head: Path | str, device: str = "cpu", workload_class_names: Sequence[str] | None = None, attention_class_names: Sequence[str] | None = None, emotion_class_names: Sequence[str] | None = None) -> "ThreeMentalStatePredictor":
        backbone = Path(backbone_checkpoint).expanduser().resolve()
        if not backbone.is_file(): raise FileNotFoundError(f"Backbone checkpoint not found: {backbone}")
        config = Model50MConfig(checkpoint_path=backbone, device=device, target_sample_rate=100.0, window_seconds=2.0, patch_seconds=1.0, patch_stride_seconds=1.0, model_n_time_patches=10, output_layer_idx=8, aggregation="flatten", num_classes=3, head_type="linear")
        return cls.from_config_and_checkpoints(config=config, workload_head=workload_head, attention_head=attention_head, emotion_head=emotion_head, workload_class_names=workload_class_names, attention_class_names=attention_class_names, emotion_class_names=emotion_class_names)
    @classmethod
    def from_config_and_checkpoints(cls, *, config: Model50MConfig, workload_head: Path | str, attention_head: Path | str, emotion_head: Path | str, workload_class_names: Sequence[str] | None = None, attention_class_names: Sequence[str] | None = None, emotion_class_names: Sequence[str] | None = None) -> "ThreeMentalStatePredictor":
        backbone = Path(config.checkpoint_path).expanduser().resolve()
        if not backbone.is_file(): raise FileNotFoundError(f"Backbone checkpoint not found: {backbone}")
        if config.classifier_input_dim != SHARED_FEATURE_CONTRACT["feature_dim"]: raise RuntimeError(f"Current 50M runtime feature dimension does not match the three-head contract: {config.classifier_input_dim}.")
        _validate_runtime_config(config)
        heads = {"workload": _load_head(task="workload", checkpoint_path=workload_head, config=config, class_name_fallback=workload_class_names), "attention": _load_head(task="attention", checkpoint_path=attention_head, config=config, class_name_fallback=attention_class_names), "emotion": _load_head(task="emotion", checkpoint_path=emotion_head, config=config, class_name_fallback=emotion_class_names)}
        expected_hash = sha256_file(backbone)
        if {str(head.info.metadata.get("backbone_sha256")) for head in heads.values()} != {expected_hash}: raise ValueError("Head checkpoints do not all reference the supplied backbone: expected " + expected_hash + ".")
        if {tuple(head.info.metadata.get("channel_template", ())) for head in heads.values()} != {config.standard_channels}: raise ValueError("Head checkpoints disagree with the current STANDARD_64_CHANNELS template.")
        return cls(config=config, preprocessor=Model50MPreprocessor(config), tokenizer=Model50MTokenizer(config), backbone=Model50MBackbone(config=config, load_checkpoint=True, freeze=True), heads=heads)
    @property
    def device(self) -> torch.device: return self.backbone.device
    @property
    def window_seconds(self) -> float: return float(self.config.window_seconds)
    @property
    def head_info(self) -> Mapping[str, HeadCheckpointInfo]: return {task: head.info for task, head in self._heads.items()}
    @property
    def last_diagnostics(self) -> ThreeMentalStateDiagnostics | None: return self._last_diagnostics
    def _validate_initialization(self) -> None:
        if tuple(self._heads) != TASKS: raise ValueError(f"ThreeMentalStatePredictor requires exactly {TASKS}; got {tuple(self._heads)}.")
        if self.config.classifier_input_dim != SHARED_FEATURE_CONTRACT["feature_dim"]: raise ValueError(f"ThreeMentalStatePredictor requires 65536-d flatten features, got {self.config.classifier_input_dim}.")
        for task, loaded in self._heads.items():
            if loaded.info.input_dim != self.config.classifier_input_dim: raise ValueError(f"{task}: input dim expected {self.config.classifier_input_dim}, actual {loaded.info.input_dim}.")
            if loaded.info.output_dim != TASK_OUTPUT_DIMS[task]: raise ValueError(f"{task}: output dim expected {TASK_OUTPUT_DIMS[task]}, actual {loaded.info.output_dim}.")
            loaded.module.to(self.device).eval()
            for parameter in loaded.module.parameters(): parameter.requires_grad_(False)
        self.backbone.eval()
    @staticmethod
    def _signal(window: RawEEGWindow) -> np.ndarray:
        signal = np.asarray(window.data).T if window.layout == "TC" else np.asarray(window.data)
        if window.layout not in ("CT", "TC"): raise ValueError(f"Unsupported RawEEGWindow layout: {window.layout!r}.")
        if signal.ndim != 2 or signal.shape[0] != len(window.channel_names): raise ValueError("RawEEGWindow data/channel_names must have matching [C, T] layout.")
        if not np.isfinite(signal).all(): raise ValueError("RawEEGWindow data contains NaN or Inf.")
        return signal.astype(np.float32, copy=False)
    @staticmethod
    def _prediction(task: str, logits: torch.Tensor, names: tuple[str, ...]) -> HeadPrediction:
        if tuple(logits.shape) != (1, len(names)) or not torch.isfinite(logits).all(): raise RuntimeError(f"{task}: invalid logits.")
        probabilities = torch.softmax(logits, -1)
        if not torch.isfinite(probabilities).all(): raise RuntimeError(f"{task}: probabilities contain NaN or Inf.")
        values = tuple(float(value) for value in probabilities[0].detach().cpu().tolist()); index = int(probabilities.argmax(-1).item())
        return HeadPrediction(index, names[index], values[index], values)
    @torch.inference_mode()
    def predict(self, window: RawEEGWindow) -> ThreeMentalStatePrediction:
        if self.backbone.training or any(head.module.training for head in self._heads.values()): raise RuntimeError("ThreeMentalStatePredictor modules must remain in eval mode during inference.")
        started = time.perf_counter(); preprocessed: PreprocessResult = self.preprocessor(signal=self._signal(window), channel_names=window.channel_names, original_sample_rate=window.sample_rate, input_unit=window.unit); prep_ms = (time.perf_counter()-started)*1000
        if not np.isfinite(preprocessed.signal).all(): raise RuntimeError("Preprocessed 50M input contains NaN or Inf.")
        batch = self.tokenizer(preprocessed).as_batch(device=self.device); started = time.perf_counter(); embedding = self.backbone.extract_embeddings(batch=batch, return_layer_idx=self.config.output_layer_idx); feature = self.aggregator(token_embeddings=embedding, token_valid_mask=batch.token_valid_mask); backbone_ms = (time.perf_counter()-started)*1000
        if tuple(feature.shape) != (1, self.config.classifier_input_dim) or not torch.isfinite(feature).all(): raise RuntimeError("Shared feature is invalid.")
        started = time.perf_counter(); logits = {task: head.module(feature) for task, head in self._heads.items()}; heads_ms = (time.perf_counter()-started)*1000
        prediction = {task: self._prediction(task, logits[task], self._heads[task].info.class_names) for task in TASKS}
        missing = tuple(name for name, valid in zip(self.config.standard_channels, preprocessed.channel_valid_mask, strict=True) if valid < .5)
        self._last_diagnostics = ThreeMentalStateDiagnostics(1, 1, {task: 1 for task in TASKS}, tuple(preprocessed.signal.shape), tuple(embedding.shape), tuple(feature.shape), {task: tuple(value.shape) for task, value in logits.items()}, preprocessed.mapped_channel_count, preprocessed.missing_channel_count, missing, preprocessed.unknown_channel_names, preprocessed.duplicate_channel_count, prep_ms, backbone_ms, heads_ms)
        return ThreeMentalStatePrediction(**prediction)

MultiHeadPredictor = ThreeMentalStatePredictor
