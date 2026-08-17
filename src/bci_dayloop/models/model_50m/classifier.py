from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from .backbone import Model50MBackbone
from .config import Model50MConfig
from .tokenization import Model50MBatchedInput


# ======================================================================
# 特征聚合
# ======================================================================


class FeatureAggregator(nn.Module):
    """
    将 50M Backbone 输出的 Token Embedding 聚合成分类特征。

    输入：
        token_embeddings: [B, S, D]
        token_valid_mask:  [B, S]

    支持：
        flatten:
            [B, S, D] -> [B, S * D]

        mean:
            对有效 Token 求均值：
            [B, S, D] -> [B, D]

    阶段 0.5 默认使用 flatten，与当前 50M 微调代码保持一致。
    """

    def __init__(self, mode: str = "flatten") -> None:
        super().__init__()

        if mode not in {"flatten", "mean"}:
            raise ValueError(
                f"Unsupported aggregation mode: {mode!r}. "
                "Expected 'flatten' or 'mean'."
            )

        self.mode = mode

    def forward(
        self,
        token_embeddings: torch.Tensor,
        token_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if token_embeddings.ndim != 3:
            raise ValueError(
                "token_embeddings must have shape [B, S, D], "
                f"got {tuple(token_embeddings.shape)}."
            )

        if token_valid_mask.ndim != 2:
            raise ValueError(
                "token_valid_mask must have shape [B, S], "
                f"got {tuple(token_valid_mask.shape)}."
            )

        batch_size, num_tokens, _ = token_embeddings.shape

        expected_mask_shape = (batch_size, num_tokens)

        if tuple(token_valid_mask.shape) != expected_mask_shape:
            raise ValueError(
                "token_valid_mask shape does not match embeddings: "
                f"expected {expected_mask_shape}, "
                f"got {tuple(token_valid_mask.shape)}."
            )

        if not torch.isfinite(token_embeddings).all():
            raise ValueError(
                "token_embeddings contains NaN or Inf."
            )

        if not torch.isfinite(token_valid_mask).all():
            raise ValueError(
                "token_valid_mask contains NaN or Inf."
            )

        valid_mask = token_valid_mask.to(
            device=token_embeddings.device,
            dtype=token_embeddings.dtype,
        ).unsqueeze(-1)

        valid_token_count = valid_mask.sum(dim=1)

        if torch.any(valid_token_count <= 0):
            invalid_batch_indices = (
                valid_token_count.squeeze(-1) <= 0
            ).nonzero(as_tuple=False).flatten().tolist()

            raise ValueError(
                "At least one sample contains no valid EEG Token. "
                f"Invalid batch indices: {invalid_batch_indices}."
            )

        masked_embeddings = token_embeddings * valid_mask

        if self.mode == "flatten":
            # 缺失通道对应的 Token 保留位置，但特征值为 0。
            return masked_embeddings.flatten(start_dim=1)

        # mean pooling
        numerator = masked_embeddings.sum(dim=1)
        denominator = valid_token_count.clamp_min(1.0)

        return numerator / denominator


# ======================================================================
# 分类头
# ======================================================================


class LinearClassificationHead(nn.Module):
    """
    简单线性分类头。

    默认 50M 原始配置：

        num_tokens = 640
        d_model = 512
        aggregation = flatten

        input_dim = 640 * 512 = 327680

    输出：
        logits [B, num_classes]
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                f"input_dim must be positive, got {input_dim}."
            )

        if num_classes <= 1:
            raise ValueError(
                "num_classes must be greater than 1, "
                f"got {num_classes}."
            )

        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)

        self.linear = nn.Linear(
            self.input_dim,
            self.num_classes,
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(
                "Classification features must have shape [B, F], "
                f"got {tuple(features.shape)}."
            )

        if features.shape[1] != self.input_dim:
            raise ValueError(
                "Classification feature dimension mismatch: "
                f"expected {self.input_dim}, "
                f"got {features.shape[1]}."
            )

        if not torch.isfinite(features).all():
            raise ValueError(
                "Classification features contain NaN or Inf."
            )

        return self.linear(features)


class MLPClassificationHead(nn.Module):
    """One-hidden-layer MLP classification head for frozen 50M features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        norm: str,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if num_classes <= 1:
            raise ValueError(
                "num_classes must be greater than 1, "
                f"got {num_classes}."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.dropout_probability = float(dropout)
        self.norm_type = str(norm)

        if norm == "none":
            self.normalization: nn.Module = nn.Identity()
        elif norm == "layernorm":
            self.normalization = nn.LayerNorm(self.input_dim)
        elif norm == "batchnorm":
            self.normalization = nn.BatchNorm1d(self.input_dim)
        else:
            raise ValueError(
                f"Unsupported MLP head norm: {norm!r}. "
                "Expected 'none', 'layernorm', or 'batchnorm'."
            )

        self.hidden = nn.Linear(self.input_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_probability)
        self.output = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(
                "Classification features must have shape [B, F], "
                f"got {tuple(features.shape)}."
            )
        if features.shape[1] != self.input_dim:
            raise ValueError(
                "Classification feature dimension mismatch: "
                f"expected {self.input_dim}, got {features.shape[1]}."
            )
        if not torch.isfinite(features).all():
            raise ValueError("Classification features contain NaN or Inf.")
        normalized = self.normalization(features)
        hidden = self.hidden(normalized)
        activated = self.activation(hidden)
        return self.output(self.dropout(activated))


def build_classification_head(config: Model50MConfig) -> nn.Module:
    """Construct the configured task head in one shared location."""
    if config.head_type == "linear":
        return LinearClassificationHead(
            input_dim=config.classifier_input_dim,
            num_classes=config.num_classes,
        )
    if config.head_type == "mlp":
        return MLPClassificationHead(
            input_dim=config.classifier_input_dim,
            hidden_dim=config.head_hidden_dim,
            num_classes=config.num_classes,
            dropout=config.head_dropout,
            norm=config.head_norm,
        )
    raise ValueError(f"Unsupported head_type: {config.head_type!r}.")


# ======================================================================
# 预测结果
# ======================================================================


@dataclass(frozen=True, slots=True)
class ClassificationOutput:
    """
    批量预测结果。

    tensors 均保持在模型当前设备上。
    """

    logits: torch.Tensor
    probabilities: torch.Tensor
    predictions: torch.Tensor
    confidences: torch.Tensor

    feature_shape: tuple[int, ...]
    backbone_ms: float
    aggregation_ms: float
    classifier_ms: float
    total_ms: float

    def to_cpu(self) -> "ClassificationOutput":
        return ClassificationOutput(
            logits=self.logits.detach().cpu(),
            probabilities=self.probabilities.detach().cpu(),
            predictions=self.predictions.detach().cpu(),
            confidences=self.confidences.detach().cpu(),
            feature_shape=self.feature_shape,
            backbone_ms=self.backbone_ms,
            aggregation_ms=self.aggregation_ms,
            classifier_ms=self.classifier_ms,
            total_ms=self.total_ms,
        )

    def first_as_dict(self) -> dict[str, Any]:
        """
        将 batch 中第一个样本转换成 Pipeline 易用格式。
        """
        cpu_output = self.to_cpu()

        if cpu_output.predictions.numel() == 0:
            raise ValueError("Classification output is empty.")

        return {
            "prediction": int(
                cpu_output.predictions[0].item()
            ),
            "confidence": float(
                cpu_output.confidences[0].item()
            ),
            "probabilities": (
                cpu_output.probabilities[0].tolist()
            ),
            "logits": cpu_output.logits[0].tolist(),
            "feature_shape": list(
                cpu_output.feature_shape
            ),
            "backbone_ms": float(cpu_output.backbone_ms),
            "aggregation_ms": float(
                cpu_output.aggregation_ms
            ),
            "classifier_ms": float(
                cpu_output.classifier_ms
            ),
            "total_ms": float(cpu_output.total_ms),
        }


# ======================================================================
# Backbone + 聚合器 + 分类头
# ======================================================================


def _synchronize_device(device: torch.device) -> None:
    """
    在测量耗时时同步异步计算设备。
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elif device.type == "mps":
        if hasattr(torch, "mps"):
            torch.mps.synchronize()


class Model50MClassifier(nn.Module):
    """
    50M 下游分类模型。

    流程：

        Model50MBatchedInput
                ↓
        Model50MBackbone
                ↓
        Token Embedding [B, S, D]
                ↓
        FeatureAggregator
                ↓
        Classification Features
                ↓
        Configured classification head
                ↓
        logits [B, num_classes]

    当前阶段默认冻结 Backbone，只训练分类头。
    """

    def __init__(
        self,
        config: Model50MConfig,
        backbone: Model50MBackbone,
    ) -> None:
        super().__init__()

        self.config = config
        self.backbone = backbone

        self.aggregator = FeatureAggregator(
            mode=config.aggregation,
        )

        self.head = build_classification_head(config)

        self.head.to(self.backbone.device)
        self.aggregator.to(self.backbone.device)

        # 阶段 0.5 默认冻结 Backbone。
        self.backbone.freeze()

    @property
    def device(self) -> torch.device:
        return self.backbone.device

    @property
    def feature_dim(self) -> int:
        return int(self.config.classifier_input_dim)

    @property
    def num_classes(self) -> int:
        return int(self.config.num_classes)

    @property
    def trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def train(
        self,
        mode: bool = True,
    ) -> "Model50MClassifier":
        super().train(mode)

        # The Backbone owns its own freeze/partial-finetune mode policy.
        # In the frozen baseline this still keeps every backbone module in
        # eval; partial fine-tuning enables train mode only for selected
        # encoder blocks.
        self.backbone.train(mode)

        # 只有分类头跟随训练状态。
        self.head.train(mode)

        return self

    def extract_features(
        self,
        batch: Model50MBatchedInput,
    ) -> torch.Tensor:
        """
        提取分类头之前的固定长度特征。

        默认：
            Token Embedding: [B, 640, 512]
            Features:        [B, 327680]
        """
        batch = batch.to(self.device)

        token_embeddings = self.backbone.extract_embeddings(
            batch=batch,
            return_layer_idx=self.config.output_layer_idx,
        )

        features = self.aggregator(
            token_embeddings=token_embeddings,
            token_valid_mask=batch.token_valid_mask,
        )

        expected_shape = (
            batch.batch_size,
            self.config.classifier_input_dim,
        )

        if tuple(features.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected aggregated feature shape: "
                f"expected {expected_shape}, "
                f"got {tuple(features.shape)}."
            )

        if not torch.isfinite(features).all():
            raise RuntimeError(
                "Aggregated classification features contain "
                "NaN or Inf."
            )

        return features

    def forward(
        self,
        batch: Model50MBatchedInput,
    ) -> torch.Tensor:
        """
        返回 logits [B, num_classes]。
        """
        features = self.extract_features(batch)
        logits = self.head(features)

        expected_shape = (
            batch.batch_size,
            self.config.num_classes,
        )

        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected classifier logits shape: "
                f"expected {expected_shape}, "
                f"got {tuple(logits.shape)}."
            )

        return logits

    @torch.no_grad()
    def predict_proba(
        self,
        batch: Model50MBatchedInput,
    ) -> torch.Tensor:
        """
        返回类别概率 [B, num_classes]。
        """
        self.eval()

        logits = self.forward(batch)
        probabilities = torch.softmax(logits, dim=-1)

        if not torch.isfinite(probabilities).all():
            raise RuntimeError(
                "Classifier probabilities contain NaN or Inf."
            )

        probability_sum = probabilities.sum(dim=-1)

        if not torch.allclose(
            probability_sum,
            torch.ones_like(probability_sum),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise RuntimeError(
                "Classifier probabilities do not sum to 1."
            )

        return probabilities

    @torch.no_grad()
    def predict_batch(
        self,
        batch: Model50MBatchedInput,
    ) -> ClassificationOutput:
        """
        批量预测，并分别记录各阶段耗时。
        """
        self.eval()
        batch = batch.to(self.device)

        _synchronize_device(self.device)
        total_start = time.perf_counter()

        # --------------------------------------------------------------
        # Backbone
        # --------------------------------------------------------------
        _synchronize_device(self.device)
        backbone_start = time.perf_counter()

        token_embeddings = self.backbone.extract_embeddings(
            batch=batch,
            return_layer_idx=self.config.output_layer_idx,
        )

        _synchronize_device(self.device)
        backbone_ms = (
            time.perf_counter() - backbone_start
        ) * 1000.0

        # --------------------------------------------------------------
        # Aggregation
        # --------------------------------------------------------------
        aggregation_start = time.perf_counter()

        features = self.aggregator(
            token_embeddings=token_embeddings,
            token_valid_mask=batch.token_valid_mask,
        )

        _synchronize_device(self.device)
        aggregation_ms = (
            time.perf_counter() - aggregation_start
        ) * 1000.0

        # --------------------------------------------------------------
        # Classifier
        # --------------------------------------------------------------
        classifier_start = time.perf_counter()

        logits = self.head(features)
        probabilities = torch.softmax(logits, dim=-1)

        confidences, predictions = torch.max(
            probabilities,
            dim=-1,
        )

        _synchronize_device(self.device)
        classifier_ms = (
            time.perf_counter() - classifier_start
        ) * 1000.0

        total_ms = (
            time.perf_counter() - total_start
        ) * 1000.0

        return ClassificationOutput(
            logits=logits,
            probabilities=probabilities,
            predictions=predictions,
            confidences=confidences,
            feature_shape=tuple(features.shape),
            backbone_ms=float(backbone_ms),
            aggregation_ms=float(aggregation_ms),
            classifier_ms=float(classifier_ms),
            total_ms=float(total_ms),
        )


# ======================================================================
# 分类头 checkpoint
# ======================================================================


@dataclass(frozen=True, slots=True)
class ClassifierLoadReport:
    checkpoint_path: str
    format_version: int
    metadata: dict[str, Any]
    load_seconds: float
    device: str


def build_classifier_metadata(
    config: Model50MConfig,
) -> dict[str, Any]:
    """
    只保存普通 Python 数据，避免再次出现自定义 Config
    对象导致的 Pickle 加载问题。
    """
    return {
        "base_model": "50M",
        "aggregation": config.aggregation,
        "output_layer_idx": int(
            config.output_layer_idx
        ),
        "d_model": int(config.d_model),
        "num_tokens": int(config.num_tokens),
        "feature_dim": int(
            config.classifier_input_dim
        ),
        "num_classes": int(config.num_classes),
        "n_channels": int(config.n_channels),
        "target_sample_rate": float(
            config.target_sample_rate
        ),
        "window_seconds": float(
            config.window_seconds
        ),
        "patch_seconds": float(
            config.patch_seconds
        ),
        "patch_stride_seconds": float(
            config.patch_stride_seconds
        ),
        "target_num_points": int(
            config.target_num_points
        ),
        "patch_num_points": int(
            config.patch_num_points
        ),
        "num_time_patches": int(
            config.num_time_patches
        ),
        "channel_template": list(
            config.standard_channels
        ),
        "missing_channel_strategy": (
            "zero_with_valid_mask"
        ),
        "model_n_time_patches": int(
            config.model_n_time_patches
        ),
        "head_type": config.head_type,
        "head_hidden_dim": int(config.head_hidden_dim),
        "head_dropout": float(config.head_dropout),
        "head_norm": config.head_norm,
    }


def classifier_head_config_from_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Return architecture metadata, defaulting absent legacy fields to Linear."""
    return {
        "head_type": str(metadata.get("head_type", "linear")),
        "head_hidden_dim": int(metadata.get("head_hidden_dim", 512)),
        "head_dropout": float(metadata.get("head_dropout", 0.0)),
        "head_norm": str(metadata.get("head_norm", "none")),
    }


def save_classifier_checkpoint(
    classifier: Model50MClassifier,
    checkpoint_path: Path | str,
    *,
    extra_metadata: Mapping[str, Any] | None = None,
    backbone_state_dict: Mapping[str, torch.Tensor] | None = None,
) -> Path:
    """
    默认只保存任务分类头，不重复保存 50M Backbone。partial fine-tuning
    调用方可显式提供 ``backbone_state_dict``，使 checkpoint 能完整恢复。

    文件内容：
        format_version
        head_state_dict
        metadata

    这样每个任务只需保存一个较小的任务头文件。
    """
    checkpoint_path = (
        Path(checkpoint_path)
        .expanduser()
        .resolve()
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = build_classifier_metadata(
        classifier.config
    )

    if extra_metadata is not None:
        metadata.update(dict(extra_metadata))

    payload = {
        "format_version": 3 if backbone_state_dict is not None else 2,
        "head_state_dict": {
            key: value.detach().cpu()
            for key, value in (
                classifier.head.state_dict().items()
            )
        },
        "metadata": metadata,
    }
    if backbone_state_dict is not None:
        payload["backbone_state_dict"] = {
            key: value.detach().cpu()
            for key, value in backbone_state_dict.items()
        }

    temporary_path = Path(
        f"{checkpoint_path}.tmp"
    )

    torch.save(payload, temporary_path)
    temporary_path.replace(checkpoint_path)

    return checkpoint_path


def _safe_load_classifier_file(
    checkpoint_path: Path,
    device: torch.device,
) -> Any:
    """
    分类头文件只包含 Tensor 和普通字典，优先安全加载。
    """
    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # 兼容不支持 weights_only 的旧版 PyTorch。
        return torch.load(
            checkpoint_path,
            map_location=device,
        )


def read_classifier_head_config(
    checkpoint_path: Path | str,
) -> dict[str, Any]:
    """Read self-described head architecture, including legacy Linear defaults."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"50M classifier checkpoint was not found: {path}")
    checkpoint = _safe_load_classifier_file(path, torch.device("cpu"))
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Classifier checkpoint must be a mapping, "
            f"got {type(checkpoint)!r}."
        )
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("Classifier checkpoint metadata must be a mapping.")
    return classifier_head_config_from_metadata(metadata)


def _validate_checkpoint_metadata(
    metadata: Mapping[str, Any],
    config: Model50MConfig,
) -> None:
    """
    防止加载与当前窗口长度、聚合方式或输出层不匹配的任务头。
    """
    expected = {
        "aggregation": config.aggregation,
        "output_layer_idx": int(
            config.output_layer_idx
        ),
        "d_model": int(config.d_model),
        "num_tokens": int(config.num_tokens),
        "feature_dim": int(
            config.classifier_input_dim
        ),
        "num_classes": int(config.num_classes),
        "window_seconds": float(
            config.window_seconds
        ),
        "target_sample_rate": float(
            config.target_sample_rate
        ),
        "head_type": config.head_type,
        "head_hidden_dim": int(config.head_hidden_dim),
        "head_dropout": float(config.head_dropout),
        "head_norm": config.head_norm,
    }

    normalized_metadata = dict(metadata)
    normalized_metadata.update(classifier_head_config_from_metadata(metadata))

    mismatches: list[str] = []

    for key, expected_value in expected.items():
        if key not in normalized_metadata:
            mismatches.append(
                f"{key}: missing in checkpoint"
            )
            continue

        actual_value = normalized_metadata[key]

        if actual_value != expected_value:
            mismatches.append(
                f"{key}: checkpoint={actual_value!r}, "
                f"current={expected_value!r}"
            )

    if "model_n_time_patches" in metadata:
        actual_model_patches = int(
            metadata["model_n_time_patches"]
        )
        expected_model_patches = int(
            config.model_n_time_patches
        )

        if actual_model_patches != expected_model_patches:
            mismatches.append(
                "model_n_time_patches: "
                f"checkpoint={actual_model_patches}, "
                f"current={expected_model_patches}"
            )

    if mismatches:
        mismatch_text = "\n".join(
            f"  - {item}" for item in mismatches
        )

        raise ValueError(
            "Classifier checkpoint is incompatible with "
            "the current Model50MConfig:\n"
            f"{mismatch_text}"
        )


def load_classifier_checkpoint(
    classifier: Model50MClassifier,
    checkpoint_path: Path | str,
    *,
    strict_metadata: bool = True,
) -> ClassifierLoadReport:
    """
    加载已经训练好的任务分类头。
    """
    checkpoint_path = (
        Path(checkpoint_path)
        .expanduser()
        .resolve()
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "50M classifier checkpoint was not found: "
            f"{checkpoint_path}"
        )

    start_time = time.perf_counter()

    checkpoint = _safe_load_classifier_file(
        checkpoint_path=checkpoint_path,
        device=classifier.device,
    )

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Classifier checkpoint must be a mapping, "
            f"got {type(checkpoint)!r}."
        )

    state_dict = checkpoint.get(
        "head_state_dict"
    )

    if not isinstance(state_dict, Mapping):
        raise KeyError(
            "Classifier checkpoint does not contain "
            "'head_state_dict'."
        )

    metadata = checkpoint.get("metadata", {})

    if not isinstance(metadata, Mapping):
        raise TypeError(
            "Classifier checkpoint metadata must be a mapping."
        )

    if strict_metadata:
        _validate_checkpoint_metadata(
            metadata=metadata,
            config=classifier.config,
        )

    saved_backbone_state = checkpoint.get("backbone_state_dict")
    if saved_backbone_state is not None:
        if not isinstance(saved_backbone_state, Mapping):
            raise TypeError(
                "Classifier checkpoint backbone_state_dict must be a mapping."
            )
        try:
            classifier.backbone.model.load_state_dict(
                saved_backbone_state,
                strict=True,
            )
        except RuntimeError as error:
            raise RuntimeError(
                "Failed to load the fine-tuned 50M backbone state from "
                "the classifier checkpoint."
            ) from error

    try:
        classifier.head.load_state_dict(
            state_dict,
            strict=True,
        )
    except RuntimeError as error:
        raise RuntimeError(
            "Failed to load the 50M classifier head. "
            "The saved task head shape does not match the "
            "current feature dimension or number of classes."
        ) from error

    classifier.head.to(classifier.device)
    classifier.eval()

    elapsed = time.perf_counter() - start_time

    return ClassifierLoadReport(
        checkpoint_path=str(checkpoint_path),
        format_version=int(
            checkpoint.get("format_version", 0)
        ),
        metadata=dict(metadata),
        load_seconds=float(elapsed),
        device=str(classifier.device),
    )


# ======================================================================
# 构建入口
# ======================================================================


def build_model50m_classifier(
    config: Model50MConfig,
    *,
    load_head: bool = True,
) -> tuple[
    Model50MClassifier,
    ClassifierLoadReport | None,
]:
    """
    构建完整 50M 分类器。

    过程：
        1. 加载并冻结 Backbone；
        2. 构建 Flatten/Mean 聚合器；
        3. 构建配置指定的任务头；
        4. 可选加载已有分类头。
    """
    backbone = Model50MBackbone(
        config=config,
        load_checkpoint=True,
        freeze=True,
    )

    classifier = Model50MClassifier(
        config=config,
        backbone=backbone,
    )

    load_report: ClassifierLoadReport | None = None

    if load_head:
        if config.classifier_path is None:
            raise ValueError(
                "load_head=True, but "
                "config.classifier_path is None."
            )

        load_report = load_classifier_checkpoint(
            classifier=classifier,
            checkpoint_path=config.classifier_path,
        )

    return classifier, load_report
