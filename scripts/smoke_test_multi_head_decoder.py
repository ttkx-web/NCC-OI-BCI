from __future__ import annotations

"""Compare direct and SlidingWindowDecoder multi-head inference once."""

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT
from bci_dayloop.data.trial_reader import open_trial_reader
from bci_dayloop.inference import MultiHeadDecodeResult, MultiHeadPredictor, SlidingWindowDecoder
from bci_dayloop.runtime.types import RawEEGWindow


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MultiHeadPredictor decoder smoke test.")
    parser.add_argument("--input-h5", default="data/processed/yaxin/smr_control_yaxin_0819_combined.h5")
    parser.add_argument("--session", default="S6")
    parser.add_argument("--trial-index", type=int, default=0)
    parser.add_argument("--backbone-checkpoint", default="checkpoints/backbones/50m/model_deploy.pt")
    parser.add_argument("--workload-head", default="checkpoints/heads/stage1/bnci2014_001/subject_01/Workload/subject_01/population/2s_flatten/head.pt")
    parser.add_argument("--attention-head", default="checkpoints/heads/stage1/bnci2014_001/subject_01/MEMA/subject_01/population/2s_flatten/head.pt")
    parser.add_argument("--emotion-head", default="checkpoints/heads/stage1/bnci2014_001/subject_01/SEED/subject_01/population/2s_flatten/head.pt")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    return parser


def _as_dict(prediction: object) -> dict[str, object]:
    return {
        "label_id": prediction.label_id,  # type: ignore[attr-defined]
        "label": prediction.label,  # type: ignore[attr-defined]
        "confidence": prediction.confidence,  # type: ignore[attr-defined]
        "probabilities": list(prediction.probabilities),  # type: ignore[attr-defined]
    }


def main() -> None:
    args = _parser().parse_args()
    reader = open_trial_reader(data_reader="eeg", path=_path(args.input_h5), canonical_subject_id=1)
    source = reader.load(session=args.session)
    if not 0 <= args.trial_index < len(source["data"]):
        raise IndexError("--trial-index is outside the requested session.")
    raw = np.asarray(source["data"][args.trial_index], dtype=np.float32)
    window = RawEEGWindow(
        data=raw,
        channel_names=list(reader.metadata.channel_names),
        sample_rate=float(reader.metadata.sample_rate),
        unit=str(reader.metadata.unit),
        trial_id=str(source["trial_ids"][args.trial_index]),
    )
    predictor = MultiHeadPredictor.from_checkpoints(
        backbone_checkpoint=_path(args.backbone_checkpoint),
        workload_head=_path(args.workload_head),
        attention_head=_path(args.attention_head),
        emotion_head=_path(args.emotion_head),
        device=args.device,
    )
    direct = predictor.predict(window)
    decoder = SlidingWindowDecoder(
        predictor=predictor,
        channel_names=reader.metadata.channel_names,
        sample_rate=float(reader.metadata.sample_rate),
        input_unit=str(reader.metadata.unit),
        window_sec=predictor.window_seconds,
        step_sec=predictor.window_seconds,
    )
    decoded = decoder.push(raw, trial_id=int(source["trial_ids"][args.trial_index]))
    if not isinstance(decoded, MultiHeadDecodeResult):
        raise RuntimeError("SlidingWindowDecoder did not return MultiHeadDecodeResult.")
    for task in ("workload", "attention", "emotion"):
        before = getattr(direct, task)
        after = getattr(decoded.prediction, task)
        if before.label_id != after.label_id or before.label != after.label:
            raise RuntimeError(f"{task}: direct and decoder labels differ.")
        if not np.allclose(before.probabilities, after.probabilities):
            raise RuntimeError(f"{task}: direct and decoder probabilities differ.")
    diagnostics = predictor.last_diagnostics
    if diagnostics is None:
        raise RuntimeError("Predictor did not retain diagnostics.")
    print(json.dumps({
        "input": {"h5": str(_path(args.input_h5)), "session": args.session, "trial_index": args.trial_index, "raw_shape": list(raw.shape)},
        "execution_counts": {"preprocessing": diagnostics.preprocessing_calls, "backbone": diagnostics.backbone_forwards, "heads": diagnostics.head_forwards},
        "shapes": {"preprocessed": list(diagnostics.preprocessed_shape), "shared_feature": list(diagnostics.shared_feature_shape), "logits": {task: list(shape) for task, shape in diagnostics.logit_shapes.items()}},
        "prediction": {task: _as_dict(getattr(decoded.prediction, task)) for task in ("workload", "attention", "emotion")},
    }, ensure_ascii=False, indent=2))
    print("Multi-head decoder integration smoke test: PASS")


if __name__ == "__main__":
    main()
