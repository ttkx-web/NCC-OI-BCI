from __future__ import annotations

"""Export one shared 50M backbone and three Linear heads as a runtime package."""

import argparse
from pathlib import Path

from _bootstrap import ROOT
from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.packages.multi_head import export_50m_multi_head_runtime_package
from bci_dayloop.applications.three_mental_states.contract import DEFAULT_PATHS


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a self-contained 50M multi-head runtime package.")
    parser.add_argument("--backbone-checkpoint", default=DEFAULT_PATHS["backbone_checkpoint"])
    parser.add_argument("--workload-head", default=DEFAULT_PATHS["workload_head"])
    parser.add_argument("--attention-head", default=DEFAULT_PATHS["attention_head"])
    parser.add_argument("--emotion-head", default=DEFAULT_PATHS["emotion_head"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--step-sec", type=float, default=2.0)
    parser.add_argument("--package-id", default="50m-three-mental-states")
    parser.add_argument("--package-version", default="1")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = Model50MConfig(
        checkpoint_path=_path(args.backbone_checkpoint),
        device="cpu",
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
    output = export_50m_multi_head_runtime_package(
        output_dir=_path(args.output_dir),
        config=config,
        workload_head=_path(args.workload_head),
        attention_head=_path(args.attention_head),
        emotion_head=_path(args.emotion_head),
        package_id=args.package_id,
        package_version=args.package_version,
        step_sec=args.step_sec,
        overwrite=args.overwrite,
    )
    print(f"Exported multi-head runtime package: {output}")


if __name__ == "__main__":
    main()
