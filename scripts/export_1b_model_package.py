"""Export one fixed-window frozen-1B linear head as a Runtime Model Package."""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from _bootstrap import ROOT  # noqa: F401

from bci_dayloop.models.model_1b.classifier import classifier_input_dim
from bci_dayloop.models.model_1b.config import Model1BConfig
from bci_dayloop.training.model_1b.population import (
    load_1b_head_checkpoint,
    preprocessing_contract,
    validate_head_checkpoint_compatibility,
)
from bci_dayloop.packages.common import sha256_file
from bci_dayloop.utils.config import dump_json, dump_yaml


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--head-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--package-id", default="1b-frozen-linear")
    parser.add_argument("--package-version", default="1")
    parser.add_argument("--step-sec", type=float, default=0.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    return parser


def _required_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"1B head metadata field {key!r} must be a mapping")
    return dict(value)


def config_from_head_metadata(
    *, head_metadata: Mapping[str, Any], backbone_checkpoint: Path, device: str,
) -> Model1BConfig:
    """Build the fixed Package contract solely from validated head metadata."""
    contract = _required_mapping(head_metadata, "preprocessing_contract")
    window_seconds = float(head_metadata["window_seconds"])
    config = Model1BConfig(
        checkpoint_path=backbone_checkpoint,
        device=device,
        target_sample_rate=float(contract["target_sample_rate"]),
        window_seconds=window_seconds,
        patch_seconds=float(contract["patch_seconds"]),
        patch_stride_seconds=float(contract["patch_stride_seconds"]),
        filter_enabled=bool(contract["filter_enabled"]),
        filter_low_hz=float(contract["filter_low_hz"]),
        filter_high_hz=float(contract["filter_high_hz"]),
        filter_order=int(contract["filter_order"]),
        reference_mode=str(contract["reference_mode"]),
        zscore_enabled=bool(contract["zscore_enabled"]),
        zscore_eps=float(contract["zscore_eps"]),
        # First-version heads written before this field was added used the
        # immutable 1B default (zero-fill); packages always emit it explicitly.
        missing_channel_fill_value=float(contract.get("missing_channel_fill_value", 0.0)),
        window_tolerance_seconds=float(contract.get("window_tolerance_seconds", 0.02)),
    )
    if int(head_metadata["num_time_patches"]) != config.num_time_patches:
        raise ValueError("head num_time_patches does not match window_seconds")
    if int(head_metadata["classifier_input_dim"]) != classifier_input_dim(config):
        raise ValueError("head classifier_input_dim does not match the 1B window contract")
    if int(contract["num_tokens"]) != config.num_tokens or int(contract["patch_num_points"]) != config.patch_num_points:
        raise ValueError("head preprocessing token contract does not match Model1BConfig")
    if list(contract["standard_channels"]) != list(config.standard_channels):
        raise ValueError("head preprocessing standard channel order does not match 1B contract")
    return config


def validate_export_inputs(
    *, backbone_checkpoint: Path, head_checkpoint: Path, device: str,
) -> tuple[Model1BConfig, dict[str, Any]]:
    if not backbone_checkpoint.is_file():
        raise FileNotFoundError(f"1B backbone checkpoint was not found: {backbone_checkpoint}")
    if not head_checkpoint.is_file():
        raise FileNotFoundError(f"1B head checkpoint was not found: {head_checkpoint}")
    backbone_sha256 = sha256_file(backbone_checkpoint)
    _, metadata = load_1b_head_checkpoint(
        head_checkpoint, backbone_sha256=backbone_sha256, device="cpu"
    )
    validate_head_checkpoint_compatibility(metadata, backbone_sha256=backbone_sha256)
    architecture = _required_mapping(metadata, "backbone_architecture")
    expected_architecture = {"d_model": 2048, "n_heads": 16, "depth": 20, "output_layer_idx": 19}
    if {key: architecture.get(key) for key in expected_architecture} != expected_architecture:
        raise ValueError("head backbone architecture does not match the formal 1B checkpoint")
    config = config_from_head_metadata(
        head_metadata=metadata, backbone_checkpoint=backbone_checkpoint, device=device
    )
    # ``load_1b_head_checkpoint`` already validates state shape; this makes
    # the dynamic expected width explicit at the export boundary as well.
    state = metadata["head_state_dict"]
    if tuple(state["linear.weight"].shape) != (int(metadata["num_classes"]), classifier_input_dim(config)):
        raise ValueError("head linear.weight shape does not match the export window contract")
    return config, metadata


def export_1b_runtime_package(
    *, backbone_checkpoint: str | Path, head_checkpoint: str | Path,
    output_dir: str | Path, device: str = "cpu", package_id: str = "1b-frozen-linear",
    package_version: str = "1", step_sec: float = 0.5, confidence_threshold: float = 0.55,
) -> Path:
    if step_sec <= 0:
        raise ValueError("step_sec must be positive")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    backbone_path = Path(backbone_checkpoint).expanduser().resolve()
    head_path = Path(head_checkpoint).expanduser().resolve()
    package_dir = Path(output_dir).expanduser().resolve()
    if package_dir.exists():
        raise FileExistsError(f"1B package output already exists: {package_dir}")
    config, head_metadata = validate_export_inputs(
        backbone_checkpoint=backbone_path, head_checkpoint=head_path, device=device
    )
    class_names = [str(name) for name in head_metadata["class_names"]]
    label_mapping = {str(index): name for index, name in enumerate(class_names)}
    if head_metadata["label_mapping"] != label_mapping:
        raise ValueError("head label_mapping does not match class_names order")

    package_dir.mkdir(parents=True, exist_ok=False)
    packaged_backbone = package_dir / "backbone.pt"
    packaged_head = package_dir / "head.pt"
    shutil.copy2(backbone_path, packaged_backbone)
    shutil.copy2(head_path, packaged_head)
    preprocessing = preprocessing_contract(config)
    dump_yaml(
        {
            "schema_version": 1,
            "canonicalizer": {"target_unit": preprocessing["input_unit"]},
            "transform": {"type": "model_1b", **preprocessing},
        },
        package_dir / "preprocessing.yaml",
    )
    package_payload = {
        "schema_version": 2,
        "package": {
            "id": package_id, "version": package_version,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "is_test_head": False, "warning_message": None,
        },
        "model": {
            "type": "model_1b", "name": "1b-frozen-linear", "task": "motor_imagery",
            "num_classes": int(head_metadata["num_classes"]), "class_names": class_names,
            "label_mapping": label_mapping, "head_type": "linear", "aggregation": "flatten",
            "window_seconds": config.window_seconds, "num_time_patches": config.num_time_patches,
            "token_count": config.num_tokens, "token_length": config.patch_num_points,
            "classifier_input_dim": classifier_input_dim(config),
            "standard_channels": list(config.standard_channels),
            "model_n_time_patches": config.model_n_time_patches, "d_model": config.d_model,
            "n_heads": config.n_heads, "depth": config.depth, "mlp_ratio": config.mlp_ratio,
            "dropout": config.dropout, "output_layer_idx": config.output_layer_idx,
            "patch_seconds": config.patch_seconds, "patch_stride_seconds": config.patch_stride_seconds,
        },
        "files": {
            "backbone": "backbone.pt", "head": "head.pt", "preprocessing": "preprocessing.yaml",
            "metrics": "metrics.json",
            "sha256": {"backbone": sha256_file(packaged_backbone), "head": sha256_file(packaged_head)},
        },
        "input_contract": {
            "channel_names": list(config.standard_channels), "sample_rate": config.target_sample_rate,
            "window_sec": config.window_seconds, "num_samples": config.target_num_points,
            "input_unit": "uV", "tensor_layout": "BCT", "strict_window_duration": True,
            "model_input_keys": ["signal", "channel_valid_mask"],
        },
        "runtime": {
            "step_sec": step_sec, "confidence_threshold": confidence_threshold,
            "command_map": {name: name for name in class_names},
        },
        "adaptation": {"offline": {"type": "none", "subject_id": None}, "online": {"type": "none"}},
        "provenance": {
            "classifier_type": "trained_linear_probe", "head_training_metadata": {
                key: value for key, value in head_metadata.items() if key != "head_state_dict"
            },
        },
    }
    dump_yaml(package_payload, package_dir / "package.yaml")
    dump_json(
        {
            "schema_version": 1,
            "model_selection": head_metadata["best_validation_metrics"],
            "final_test": head_metadata["final_test_metrics"],
            "best_validation_epoch": head_metadata["best_validation_epoch"],
            "export_smoke_test": None,
        },
        package_dir / "metrics.json",
    )
    return package_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    output = export_1b_runtime_package(
        backbone_checkpoint=args.backbone_checkpoint, head_checkpoint=args.head_checkpoint,
        output_dir=args.output_dir, device=args.device, package_id=args.package_id,
        package_version=args.package_version, step_sec=args.step_sec,
        confidence_threshold=args.confidence_threshold,
    )
    print(f"exported generic-window 1B Runtime Model Package: {output}")
    return 0


if __name__ == "__main__":
    main()
