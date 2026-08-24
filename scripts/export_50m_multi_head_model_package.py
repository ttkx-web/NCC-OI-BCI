from __future__ import annotations

"""Export the three-state Package from validated YAML with optional CLI overrides."""

import argparse
from pathlib import Path

from _bootstrap import ROOT
from bci_dayloop.applications.three_mental_states.export_config import (
    DEFAULT_EXPORT_CONFIG_PATH,
    load_three_mental_state_export_config,
)
from bci_dayloop.packages import export_50m_multi_head_runtime_package


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a self-contained 50M three-state runtime package.")
    parser.add_argument("--config", default=DEFAULT_EXPORT_CONFIG_PATH)
    # None deliberately means no explicit CLI override; YAML remains authoritative.
    parser.add_argument("--backbone-checkpoint")
    parser.add_argument("--workload-head")
    parser.add_argument("--attention-head")
    parser.add_argument("--emotion-head")
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--step-sec", type=float)
    parser.add_argument("--package-id")
    parser.add_argument("--package-version")
    return parser


def resolved_export_arguments(args: argparse.Namespace) -> dict[str, object]:
    """Apply explicit CLI values over a validated YAML export configuration."""
    export = load_three_mental_state_export_config(args.config)
    sources, runtime, package = export.sources, export.runtime, export.package
    return {
        "config": runtime.model_config(checkpoint_path=_path(args.backbone_checkpoint) if args.backbone_checkpoint else sources.backbone_checkpoint),
        "workload_head": _path(args.workload_head) if args.workload_head else sources.workload_head,
        "attention_head": _path(args.attention_head) if args.attention_head else sources.attention_head,
        "emotion_head": _path(args.emotion_head) if args.emotion_head else sources.emotion_head,
        "output_dir": _path(args.output_dir) if args.output_dir else package.output_dir,
        "package_id": args.package_id if args.package_id is not None else package.package_id,
        "package_version": args.package_version if args.package_version is not None else package.package_version,
        "step_sec": args.step_sec if args.step_sec is not None else runtime.step_sec,
    }


def main() -> None:
    args = build_parser().parse_args()
    resolved = resolved_export_arguments(args)
    output = export_50m_multi_head_runtime_package(
        output_dir=resolved["output_dir"], config=resolved["config"],
        workload_head=resolved["workload_head"], attention_head=resolved["attention_head"],
        emotion_head=resolved["emotion_head"], package_id=resolved["package_id"],
        package_version=resolved["package_version"], step_sec=resolved["step_sec"],
        overwrite=args.overwrite,
    )
    print(f"Exported multi-head runtime package: {output}")


if __name__ == "__main__":
    main()
