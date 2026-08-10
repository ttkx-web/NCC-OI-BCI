from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.utils.config import dump_json, dump_yaml

import torch

from bci_dayloop.data.preprocessing import (
    PreprocessingConfig,
)
from bci_dayloop.models.labram_linear import (
    LaBraMLinearAdapter,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()

def export_labram_runtime_package(
    *,
    output_dir: str | Path,
    adapter: LaBraMLinearAdapter,
    backbone_checkpoint: str | Path,
    classifier_checkpoint: str | Path,
    preprocessing_config: PreprocessingConfig,
    class_names: Sequence[str],
    command_map: Mapping[str, str],
    dataset_name: str,
    package_id: str,
    package_version: str,
    task: str = "motor_imagery",
    architecture: str = "labram_base_patch200_200",
    step_sec: float = 0.5,
    confidence_threshold: float = 0.55,
    metrics: Mapping[str, Any] | None = None,
    adaptation: Mapping[str, Any] | None = None,
) -> Path:
    """
    导出自包含的 schema-v2 LaBraM Runtime Model Package。

    backbone_checkpoint 和 classifier_checkpoint 用于：
    1. 校验导出的源产物确实存在；
    2. 在 package.yaml 中记录源文件哈希；
    3. 保证模型包的来源可追溯。

    包内的 backbone.pt 和 classifier.pt 保存 adapter 当前的真实状态，
    因而也兼容未来经过个体化微调或 Rest-Tune 的模型。
    """

    package_dir = (
        Path(output_dir)
        .expanduser()
        .resolve()
    )

    source_backbone_path = (
        Path(backbone_checkpoint)
        .expanduser()
        .resolve()
    )

    source_classifier_path = (
        Path(classifier_checkpoint)
        .expanduser()
        .resolve()
    )

    # ----------------------------------------------------------
    # 1. 导出前校验
    # ----------------------------------------------------------

    for logical_name, path in (
        (
            "source LaBraM backbone checkpoint",
            source_backbone_path,
        ),
        (
            "source LaBraM classifier checkpoint",
            source_classifier_path,
        ),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"{logical_name} was not found: {path}"
            )

    if package_dir.exists():
        raise FileExistsError(
            "Runtime Model Package directory already exists: "
            f"{package_dir}"
        )

    if step_sec <= 0:
        raise ValueError(
            f"step_sec must be positive, got {step_sec}."
        )

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError(
            "confidence_threshold must be in [0, 1], "
            f"got {confidence_threshold}."
        )

    normalized_classes = tuple(
        str(name)
        for name in class_names
    )

    if not normalized_classes:
        raise ValueError(
            "class_names cannot be empty."
        )

    if (
        len(normalized_classes)
        != int(adapter.n_classes)
    ):
        raise ValueError(
            "class_names length does not match "
            "LaBraM n_classes: "
            f"{len(normalized_classes)} != "
            f"{adapter.n_classes}."
        )

    if int(adapter.n_patches) <= 0:
        raise ValueError(
            "adapter.n_patches must be positive, "
            f"got {adapter.n_patches}."
        )

    if int(preprocessing_config.patch_samples) <= 0:
        raise ValueError(
            "preprocessing_config.patch_samples "
            "must be positive."
        )

    if (
        float(
            preprocessing_config.target_sample_rate
        )
        <= 0
    ):
        raise ValueError(
            "preprocessing_config.target_sample_rate "
            "must be positive."
        )

    if not adapter.channel_names:
        raise ValueError(
            "LaBraM adapter.channel_names cannot be empty."
        )

    unknown_commands = (
        set(command_map)
        - set(normalized_classes)
    )

    if unknown_commands:
        raise ValueError(
            "command_map contains classes that are not "
            "present in class_names: "
            f"{sorted(unknown_commands)}."
        )

    package_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    packaged_backbone = (
        package_dir / "backbone.pt"
    )

    packaged_classifier = (
        package_dir / "classifier.pt"
    )

    # ----------------------------------------------------------
    # 2. 保存规范化后的包内模型权重
    # ----------------------------------------------------------

    torch.save(
        {
            "format_version": 1,
            # 保持 LaBraM checkpoint 常用的 model 键，
            # Loader 可以从这里恢复 Encoder。
            "model": (
                adapter.encoder.state_dict()
            ),
            "architecture": architecture,
        },
        packaged_backbone,
    )

    torch.save(
        {
            "format_version": 1,
            "state_dict": (
                adapter.head.state_dict()
            ),
            "embedding_dim": int(
                adapter.embedding_dim
            ),
            "n_classes": int(
                adapter.n_classes
            ),
            "class_names": list(
                normalized_classes
            ),
        },
        packaged_classifier,
    )

    # ----------------------------------------------------------
    # 3. 保存预处理配置
    # ----------------------------------------------------------

    preprocessing_payload = {
        "schema_version": 1,
        "canonicalizer": {
            "target_unit": "uV",
        },
        "transform": {
            "type": "labram",
            "bandpass_hz": [
                float(
                    preprocessing_config
                    .bandpass_hz[0]
                ),
                float(
                    preprocessing_config
                    .bandpass_hz[1]
                ),
            ],
            "notch_hz": float(
                preprocessing_config.notch_hz
            ),
            "target_sample_rate": float(
                preprocessing_config
                .target_sample_rate
            ),
            "zscore_epsilon": float(
                preprocessing_config
                .zscore_epsilon
            ),
            "patch_samples": int(
                preprocessing_config.patch_samples
            ),
        },
    }

    dump_yaml(
        preprocessing_payload,
        package_dir / "preprocessing.yaml",
    )

    target_num_samples = (
        int(adapter.n_patches)
        * int(
            preprocessing_config.patch_samples
        )
    )

    window_sec = (
        target_num_samples
        / float(
            preprocessing_config
            .target_sample_rate
        )
    )

    adaptation_payload = dict(
        adaptation
        or {
            "offline": {
                "type": "none",
                "subject_id": None,
            },
            "online": {
                "type": "none",
            },
        }
    )

    # ----------------------------------------------------------
    # 4. 保存统一 Package 描述
    # ----------------------------------------------------------

    package_payload = {
        "schema_version": 2,
        "package": {
            "id": str(package_id),
            "version": str(package_version),
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "is_test_head": False,
            "warning_message": None,
        },
        "model": {
            "type": "labram",
            "name": "labram-linear",
            "architecture": architecture,
            "task": str(task),
            "dataset": str(dataset_name),
            "num_classes": int(
                adapter.n_classes
            ),
            "class_names": list(
                normalized_classes
            ),
            "embedding_dim": int(
                adapter.embedding_dim
            ),
            "n_patches": int(
                adapter.n_patches
            ),
            "freeze_encoder": bool(
                adapter.freeze_encoder
            ),
            "amp": bool(adapter.amp),
            "embedding_batch_size": int(
                adapter.embedding_batch_size
            ),
        },
        "files": {
            "backbone": "backbone.pt",
            "classifier": "classifier.pt",
            "preprocessing": (
                "preprocessing.yaml"
            ),
            "metrics": "metrics.json",
            # 这里是包内文件的哈希。
            "sha256": {
                "backbone": sha256_file(
                    packaged_backbone
                ),
                "classifier": sha256_file(
                    packaged_classifier
                ),
            },
        },
        "input_contract": {
            "channel_names": [
                str(name)
                for name
                in adapter.channel_names
            ],
            "sample_rate": float(
                preprocessing_config
                .target_sample_rate
            ),
            "window_sec": float(
                window_sec
            ),
            "num_samples": int(
                target_num_samples
            ),
            "input_unit": "uV",
            "tensor_layout": "BCTP",
            "strict_window_duration": True,
            "model_input_keys": [
                "signal",
            ],
        },
        "runtime": {
            "step_sec": float(step_sec),
            "confidence_threshold": float(
                confidence_threshold
            ),
            "command_map": {
                str(key): str(value)
                for key, value
                in command_map.items()
            },
        },
        "adaptation": adaptation_payload,
        "provenance": {
            "classifier_type": (
                "trained_linear_probe"
            ),
            "source_model": "LaBraM",
            # 这里只保存来源文件名和哈希。
            # Runtime 不依赖这些外部路径。
            "source_artifacts": {
                "backbone": {
                    "filename": (
                        source_backbone_path.name
                    ),
                    "sha256": sha256_file(
                        source_backbone_path
                    ),
                },
                "classifier": {
                    "filename": (
                        source_classifier_path.name
                    ),
                    "sha256": sha256_file(
                        source_classifier_path
                    ),
                },
            },
        },
    }

    dump_yaml(
        package_payload,
        package_dir / "package.yaml",
    )

    # ----------------------------------------------------------
    # 5. 保存训练与评估指标
    # ----------------------------------------------------------

    metrics_payload: dict[str, Any] = {
        "schema_version": 1,
        "model_selection": {},
        "final_test": {},
        "export_smoke_test": None,
    }

    if metrics is not None:
        metrics_payload.update(
            dict(metrics)
        )

    dump_json(
        metrics_payload,
        package_dir / "metrics.json",
    )

    return package_dir


def export_50m_runtime_package(
    *,
    output_dir: str | Path,
    config: Model50MConfig,
    class_names: Sequence[str],
    command_map: Mapping[str, str],
    dataset_name: str,
    task: str = "motor_imagery",
    step_sec: float = 0.5,
    confidence_threshold: float = 0.55,
    package_id: str,
    package_version: str,
    metrics: Mapping[str, Any] | None = None,
    adaptation: Mapping[str, Any] | None = None,
) -> Path:
    package_dir = Path(output_dir).expanduser().resolve()

    checkpoint_path = Path(
        config.checkpoint_path
    ).expanduser().resolve()

    if config.classifier_path is None:
        raise ValueError(
            "Cannot export a package without classifier_path."
        )

    classifier_path = Path(
        config.classifier_path
    ).expanduser().resolve()

    for name, path in (
        ("backbone checkpoint", checkpoint_path),
        ("classifier checkpoint", classifier_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"{name} was not found: {path}"
            )

    if step_sec <= 0:
        raise ValueError("step_sec must be positive.")

    if confidence_threshold < 0 or confidence_threshold > 1:
        raise ValueError(
            "confidence_threshold must be in [0, 1]."
        )

    normalized_classes = tuple(
        str(name) for name in class_names
    )

    if len(normalized_classes) != config.num_classes:
        raise ValueError(
            "class_names length does not match num_classes: "
            f"{len(normalized_classes)} != "
            f"{config.num_classes}."
        )

    package_dir.mkdir(parents=True, exist_ok=False)

    packaged_backbone = package_dir / "backbone.pt"
    packaged_classifier = package_dir / "classifier.pt"

    # 真正复制权重，让 Package 不再依赖仓库外部路径。
    shutil.copy2(
        checkpoint_path,
        packaged_backbone,
    )
    shutil.copy2(
        classifier_path,
        packaged_classifier,
    )

    preprocessing_payload = {
        "schema_version": 1,
        "canonicalizer": {
            "target_unit": "uV",
        },
        "transform": {
            "type": "model_50m",
            "filter_enabled": bool(
                config.filter_enabled
            ),
            "filter_low_hz": float(
                config.filter_low_hz
            ),
            "filter_high_hz": float(
                config.filter_high_hz
            ),
            "filter_order": int(
                config.filter_order
            ),
            "reference_mode": (
                config.reference_mode
            ),
            "zscore_enabled": bool(
                config.zscore_enabled
            ),
            "zscore_eps": float(
                config.zscore_eps
            ),
            "missing_channel_fill_value": float(
                config.missing_channel_fill_value
            ),
            "window_tolerance_seconds": float(
                config.window_tolerance_seconds
            ),
        },
    }

    dump_yaml(
        preprocessing_payload,
        package_dir / "preprocessing.yaml",
    )

    adaptation_payload = dict(
        adaptation
        or {
            "offline": {
                "type": "none",
                "subject_id": None,
            },
            "online": {
                "type": "none",
            },
        }
    )

    exported_at = datetime.now(
        timezone.utc
    ).isoformat()

    package_payload = {
        "schema_version": 2,
        "package": {
            "id": package_id,
            "version": package_version,
            "created_at_utc": exported_at,
            "is_test_head": False,
            "warning_message": None,
        },
        "model": {
            "type": "model_50m",
            "name": "50m-linear",
            "task": task,
            "dataset": dataset_name,
            "num_classes": int(
                config.num_classes
            ),
            "class_names": list(
                normalized_classes
            ),
            "aggregation": config.aggregation,
            "output_layer_idx": int(
                config.output_layer_idx
            ),
            "model_n_time_patches": int(
                config.model_n_time_patches
            ),
            "patch_seconds": float(
                config.patch_seconds
            ),
            "patch_stride_seconds": float(
                config.patch_stride_seconds
            ),
            "d_model": int(config.d_model),
            "n_heads": int(config.n_heads),
            "depth": int(config.depth),
            "mlp_ratio": float(
                config.mlp_ratio
            ),
            "dropout": float(config.dropout),
        },
        "files": {
            "backbone": "backbone.pt",
            "classifier": "classifier.pt",
            "preprocessing": (
                "preprocessing.yaml"
            ),
            "metrics": "metrics.json",
            "sha256": {
                "backbone": sha256_file(
                    packaged_backbone
                ),
                "classifier": sha256_file(
                    packaged_classifier
                ),
            },
        },
        "input_contract": {
            "channel_names": list(
                config.standard_channels
            ),
            "sample_rate": float(
                config.target_sample_rate
            ),
            "window_sec": float(
                config.window_seconds
            ),
            "num_samples": int(
                config.target_num_points
            ),
            "input_unit": "uV",
            "tensor_layout": "BCT",
            "strict_window_duration": bool(
                config.strict_window_duration
            ),
            "model_input_keys": [
                "signal",
                "channel_valid_mask",
            ],
        },
        "runtime": {
            "step_sec": float(step_sec),
            "confidence_threshold": float(
                confidence_threshold
            ),
            "command_map": {
                str(key): str(value)
                for key, value
                in command_map.items()
            },
        },
        "adaptation": adaptation_payload,
        "provenance": {
            "classifier_type": (
                "trained_linear_probe"
            ),
        },
    }

    dump_yaml(
        package_payload,
        package_dir / "package.yaml",
    )

    metrics_payload = {
        "schema_version": 1,
        "model_selection": {},
        "final_test": {},
        "export_smoke_test": None,
    }

    if metrics is not None:
        metrics_payload.update(dict(metrics))

    dump_json(
        metrics_payload,
        package_dir / "metrics.json",
    )

    return package_dir