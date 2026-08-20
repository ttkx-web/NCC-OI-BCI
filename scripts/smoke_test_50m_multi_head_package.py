from __future__ import annotations

"""Verify raw-checkpoint and package-loaded three-head predictions are identical."""

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT
from bci_dayloop.data.trial_reader import open_trial_reader
from bci_dayloop.inference import MultiHeadDecodeResult, MultiHeadPredictor, SlidingWindowDecoder
from bci_dayloop.packages import load_multi_head_runtime_package
from bci_dayloop.runtime.types import RawEEGWindow


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="50M multi-head runtime package smoke test.")
    parser.add_argument("--package", required=True)
    parser.add_argument("--input-h5", default="data/processed/yaxin/smr_control_yaxin_0819_combined.h5")
    parser.add_argument("--session", default="S6")
    parser.add_argument("--trial-index", type=int, default=0)
    parser.add_argument("--backbone-checkpoint", default="checkpoints/backbones/50m/model_deploy.pt")
    parser.add_argument("--workload-head", default="checkpoints/heads/stage1/bnci2014_001/subject_01/Workload/subject_01/population/2s_flatten/head.pt")
    parser.add_argument("--attention-head", default="checkpoints/heads/stage1/bnci2014_001/subject_01/MEMA/subject_01/population/2s_flatten/head.pt")
    parser.add_argument("--emotion-head", default="checkpoints/heads/stage1/bnci2014_001/subject_01/SEED/subject_01/population/2s_flatten/head.pt")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    return parser


def _assert_same(direct: object, packaged: object) -> None:
    for task in ("workload", "attention", "emotion"):
        before = getattr(direct, task)
        after = getattr(packaged, task)
        if before.label_id != after.label_id or before.label != after.label:
            raise RuntimeError(f"{task}: labels differ between direct and package paths.")
        if not np.allclose(before.probabilities, after.probabilities):
            raise RuntimeError(f"{task}: probabilities differ between direct and package paths.")


def main() -> None:
    args = _parser().parse_args()
    reader = open_trial_reader(data_reader="eeg", path=_path(args.input_h5), canonical_subject_id=1)
    source = reader.load(session=args.session)
    raw = np.asarray(source["data"][args.trial_index], dtype=np.float32)
    window = RawEEGWindow(
        data=raw,
        channel_names=list(reader.metadata.channel_names),
        sample_rate=float(reader.metadata.sample_rate),
        unit=str(reader.metadata.unit),
        trial_id=str(source["trial_ids"][args.trial_index]),
    )
    direct = MultiHeadPredictor.from_checkpoints(
        backbone_checkpoint=_path(args.backbone_checkpoint),
        workload_head=_path(args.workload_head),
        attention_head=_path(args.attention_head),
        emotion_head=_path(args.emotion_head),
        device=args.device,
    )
    packaged = load_multi_head_runtime_package(_path(args.package), device=args.device)
    direct_result = direct.predict(window)
    package_result = packaged.predict(window)
    _assert_same(direct_result, package_result)
    direct_decoder = SlidingWindowDecoder(
        predictor=direct,
        channel_names=reader.metadata.channel_names,
        sample_rate=float(reader.metadata.sample_rate),
        input_unit=str(reader.metadata.unit),
        window_sec=direct.window_seconds,
        step_sec=direct.window_seconds,
    )
    direct_decoded = direct_decoder.push(
        raw, trial_id=int(source["trial_ids"][args.trial_index])
    )
    if not isinstance(direct_decoded, MultiHeadDecodeResult):
        raise RuntimeError("Direct predictor did not produce MultiHeadDecodeResult.")
    _assert_same(direct_result, direct_decoded.prediction)

    package_decoder = SlidingWindowDecoder(
        predictor=packaged,
        channel_names=reader.metadata.channel_names,
        sample_rate=float(reader.metadata.sample_rate),
        input_unit=str(reader.metadata.unit),
        window_sec=packaged.window_seconds,
        step_sec=packaged.window_seconds,
    )
    package_decoded = package_decoder.push(
        raw, trial_id=int(source["trial_ids"][args.trial_index])
    )
    if not isinstance(package_decoded, MultiHeadDecodeResult):
        raise RuntimeError("Package predictor did not produce MultiHeadDecodeResult.")
    _assert_same(package_result, package_decoded.prediction)
    _assert_same(direct_decoded.prediction, package_decoded.prediction)
    diagnostics = packaged.last_diagnostics
    if diagnostics is None:
        raise RuntimeError("Package predictor did not retain diagnostics.")
    print(json.dumps({
        "package": str(_path(args.package)),
        "execution_counts": {"preprocessing": diagnostics.preprocessing_calls, "backbone": diagnostics.backbone_forwards, "heads": diagnostics.head_forwards},
        "prediction": {task: {"label": getattr(package_result, task).label, "probabilities": list(getattr(package_result, task).probabilities)} for task in ("workload", "attention", "emotion")},
    }, ensure_ascii=False, indent=2))
    print("50M multi-head runtime package smoke test: PASS")


if __name__ == "__main__":
    main()
