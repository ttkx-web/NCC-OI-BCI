from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.utils.config import dump_json, dump_yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


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