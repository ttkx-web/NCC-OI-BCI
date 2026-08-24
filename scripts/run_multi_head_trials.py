from __future__ import annotations

"""Print one multi-head prediction per independent canonical H5 trial."""

import argparse
import sys
from pathlib import Path

import numpy as np

from _bootstrap import ROOT
from bci_dayloop.data.trial_reader import open_trial_reader
from bci_dayloop.inference import MultiHeadDecodeResult, SlidingWindowDecoder
from bci_dayloop.packages import load_inference_package
from bci_dayloop.applications.three_mental_states.contract import DEFAULT_PATHS, TASKS
from bci_dayloop.packages.inference import THREE_MENTAL_STATES_PREDICTION_MODE


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a packaged 50M multi-head predictor once per H5 trial."
    )
    parser.add_argument("--package", required=True)
    parser.add_argument(
        "--input-h5",
        default=DEFAULT_PATHS["input_h5"],
    )
    parser.add_argument("--session", default=DEFAULT_PATHS["session"])
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--max-trials", type=int)
    return parser


def _format_result(result: MultiHeadDecodeResult) -> str:
    prediction = result.prediction
    if result.trial_id is None:
        raise RuntimeError("Decoder result is missing the source trial_id.")
    return (
        f"trial={result.trial_id} | "
        f"workload={prediction.workload.label} ({prediction.workload.confidence:.4f}) | "
        f"attention={prediction.attention.label} ({prediction.attention.confidence:.4f}) | "
        f"emotion={prediction.emotion.label} ({prediction.emotion.confidence:.4f})"
    )


def main() -> None:
    args = _parser().parse_args()
    package_path = _path(args.package)
    input_h5 = _path(args.input_h5)
    if not package_path.is_dir():
        raise FileNotFoundError(f"Runtime package directory was not found: {package_path}")
    if not input_h5.is_file():
        raise FileNotFoundError(f"Input H5 file was not found: {input_h5}")
    if args.max_trials is not None and args.max_trials <= 0:
        raise ValueError("--max-trials must be a positive integer.")

    reader = open_trial_reader(
        data_reader="eeg", path=input_h5, canonical_subject_id=1
    )
    available_sessions = reader.available_sessions()
    if args.session not in available_sessions:
        raise ValueError(
            f"Requested session {args.session!r} was not found; "
            f"available sessions: {available_sessions}."
        )
    source = reader.load(session=args.session)
    data = np.asarray(source["data"], dtype=np.float32)
    trial_ids = np.asarray(source["trial_ids"])
    if data.ndim != 3 or len(data) != len(trial_ids):
        raise RuntimeError(
            "Session data/trial_ids are not aligned: "
            f"data={data.shape}, trial_ids={trial_ids.shape}."
        )
    if len(data) == 0:
        raise ValueError(f"Requested session {args.session!r} is empty.")
    if args.max_trials is not None:
        data = data[:args.max_trials]
        trial_ids = trial_ids[:args.max_trials]

    loaded = load_inference_package(package_path, device=args.device)
    if loaded.prediction_mode != THREE_MENTAL_STATES_PREDICTION_MODE or tuple(task.task_id for task in loaded.tasks) != TASKS:
        raise ValueError("run_multi_head_trials.py requires workload, attention, emotion tasks.")
    predictor = loaded.predictor
    decoder = SlidingWindowDecoder(
        predictor=predictor,
        channel_names=reader.metadata.channel_names,
        sample_rate=float(reader.metadata.sample_rate),
        input_unit=str(reader.metadata.unit),
        window_sec=predictor.window_seconds,
        step_sec=predictor.window_seconds,
    )
    print(
        f"Input: {input_h5.name} | session={args.session} | "
        f"trials={len(data)} | device={args.device}",
        file=sys.stderr,
    )

    for raw, trial_id in zip(data, trial_ids, strict=True):
        # Each canonical H5 epoch is a complete two-second window. Resetting
        # prevents stateful decoder buffering from carrying data across trials.
        decoder.reset()
        decoded = decoder.push(raw, trial_id=int(trial_id))
        if not isinstance(decoded, MultiHeadDecodeResult):
            raise RuntimeError(
                f"trial_id={int(trial_id)} did not produce a multi-head decode result."
            )
        print(_format_result(decoded))


if __name__ == "__main__":
    main()
