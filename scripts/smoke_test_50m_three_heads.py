from __future__ import annotations

"""Run the formal MultiHeadPredictor against one real canonical EEG trial."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _bootstrap import ROOT
from bci_dayloop.data.trial_reader import open_trial_reader
from bci_dayloop.inference.multi_head import MultiHeadPredictor
from bci_dayloop.runtime.types import RawEEGWindow


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One shared 50M forward followed by three task heads."
    )
    parser.add_argument(
        "--input-h5",
        default="data/processed/yaxin/smr_control_yaxin_0819_combined.h5",
    )
    parser.add_argument("--session", default="S6")
    parser.add_argument("--trial-index", type=int, default=0)
    parser.add_argument(
        "--backbone-checkpoint", default="checkpoints/backbones/50m/model_deploy.pt"
    )
    parser.add_argument(
        "--workload-head",
        default=(
            "checkpoints/heads/stage1/bnci2014_001/subject_01/Workload/"
            "subject_01/population/2s_flatten/head.pt"
        ),
    )
    parser.add_argument(
        "--attention-head",
        default=(
            "checkpoints/heads/stage1/bnci2014_001/subject_01/MEMA/"
            "subject_01/population/2s_flatten/head.pt"
        ),
    )
    parser.add_argument(
        "--emotion-head",
        default=(
            "checkpoints/heads/stage1/bnci2014_001/subject_01/SEED/"
            "subject_01/population/2s_flatten/head.pt"
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--workload-class-names", nargs="+", default=None)
    parser.add_argument("--attention-class-names", nargs="+", default=None)
    parser.add_argument("--emotion-class-names", nargs="+", default=None)
    return parser


def _prediction_dict(prediction: object) -> dict[str, object]:
    return {
        "label_id": prediction.label_id,  # type: ignore[attr-defined]
        "label": prediction.label,  # type: ignore[attr-defined]
        "confidence": prediction.confidence,  # type: ignore[attr-defined]
        "probabilities": list(prediction.probabilities),  # type: ignore[attr-defined]
    }


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.trial_index < 0:
        raise ValueError("--trial-index must be non-negative.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps was requested, but MPS is unavailable.")

    input_h5 = _path(args.input_h5)
    reader = open_trial_reader(data_reader="eeg", path=input_h5, canonical_subject_id=1)
    available_sessions = reader.available_sessions()
    if args.session not in available_sessions:
        raise ValueError(
            f"Requested session {args.session!r} is unavailable; "
            f"available={available_sessions}."
        )
    source = reader.load(session=args.session)
    if args.trial_index >= len(source["data"]):
        raise IndexError(
            f"Trial {args.trial_index} is outside {args.session} with "
            f"{len(source['data'])} trials."
        )
    raw_eeg = np.asarray(source["data"][args.trial_index], dtype=np.float32)
    window = RawEEGWindow(
        data=raw_eeg,
        channel_names=list(reader.metadata.channel_names),
        sample_rate=float(reader.metadata.sample_rate),
        unit=str(reader.metadata.unit),
        trial_id=str(source["trial_ids"][args.trial_index]),
        metadata={"source_h5": str(input_h5), "session": args.session},
    )
    predictor = MultiHeadPredictor.from_checkpoints(
        backbone_checkpoint=_path(args.backbone_checkpoint),
        workload_head=_path(args.workload_head),
        attention_head=_path(args.attention_head),
        emotion_head=_path(args.emotion_head),
        device=args.device,
        workload_class_names=args.workload_class_names,
        attention_class_names=args.attention_class_names,
        emotion_class_names=args.emotion_class_names,
    )
    result = predictor.predict(window)
    diagnostics = predictor.last_diagnostics
    if diagnostics is None:
        raise RuntimeError("MultiHeadPredictor did not return diagnostics.")
    result_json = {
        "input": {
            "h5": str(input_h5),
            "session": args.session,
            "trial_index": int(args.trial_index),
            "source_trial_id": int(source["trial_ids"][args.trial_index]),
            "raw_shape": list(raw_eeg.shape),
            "raw_sample_rate": float(reader.metadata.sample_rate),
            "raw_channel_count": len(reader.metadata.channel_names),
            "unit": reader.metadata.unit,
        },
        "contract": {
            "window_seconds": predictor.config.window_seconds,
            "target_sample_rate": predictor.config.target_sample_rate,
            "embedding_layer": 9,
            "internal_output_layer_idx": predictor.config.output_layer_idx,
            "aggregation": predictor.config.aggregation,
            "feature_dim": predictor.config.classifier_input_dim,
            "normalization": "per-valid-channel z-score",
        },
        "channel_mapping": {
            "raw_channels": len(reader.metadata.channel_names),
            "mapped_channels": diagnostics.mapped_channel_count,
            "unknown_or_ignored": list(diagnostics.unknown_channel_names),
            "missing_standard_channels": list(diagnostics.missing_standard_channel_names),
            "duplicate_mappings": diagnostics.duplicate_channel_count,
        },
        "execution_counts": {
            "preprocessing_calls": diagnostics.preprocessing_calls,
            "backbone_forwards": diagnostics.backbone_forwards,
            "head_forwards": diagnostics.head_forwards,
        },
        "shapes": {
            "preprocessed": list(diagnostics.preprocessed_shape),
            "selected_embedding": list(diagnostics.selected_embedding_shape),
            "shared_feature": list(diagnostics.shared_feature_shape),
            **{f"{task}_logits": list(shape) for task, shape in diagnostics.logit_shapes.items()},
        },
        "checkpoints": {
            task: {
                "path": str(info.checkpoint_path),
                "head_type": info.metadata.get("head_type", "linear"),
                "input_dim": info.input_dim,
                "output_dim": info.output_dim,
                "class_names": list(info.class_names),
            }
            for task, info in predictor.head_info.items()
        },
        "predictions": {
            "workload": _prediction_dict(result.workload),
            "attention": _prediction_dict(result.attention),
            "emotion": _prediction_dict(result.emotion),
        },
    }
    print(json.dumps(result_json, ensure_ascii=False, indent=2))
    print("Three-head offline inference smoke test: PASS")


if __name__ == "__main__":
    main()
