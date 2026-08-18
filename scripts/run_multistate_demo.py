"""Decode one real HDF5 EEG window with the independent demo decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT  # noqa: F401
from bci_dayloop.demo.input import list_demo_sessions, load_demo_trial, trial_window, window_count
from bci_dayloop.demo.state_decoder import DemoStateDecoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one EEG multi-state demo window")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data/processed/bnci2014_001/subject_01.h5")
    parser.add_argument("--session", help="HDF5 session; defaults to the first available session")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--step-sec", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    session = args.session or list_demo_sessions(args.data_path)[0]
    trial = load_demo_trial(args.data_path, session=session, trial_index=args.trial)
    count = window_count(trial, args.window_sec, args.step_sec)
    if count == 0:
        raise ValueError("The selected trial is shorter than the requested window")
    samples = trial_window(trial, min(args.window_index, count - 1), window_sec=args.window_sec, step_sec=args.step_sec)
    result = DemoStateDecoder(device=args.device).decode(
        samples,
        sample_rate=trial.sample_rate,
        channel_names=trial.channel_names,
        unit=trial.unit,
        timestamp=float(args.window_index * args.step_sec),
    )
    payload = {
        "states": result.states,
        "motor_intent": result.motor_intent,
        "band_power": result.band_power,
        "signal_quality": result.signal_quality,
        "latency_ms": result.latency_ms,
        "interpretation": result.interpretation,
        "topomap_channels": len(result.topomap_channel_names or []),
        "psd_points": int(0 if result.psd_frequencies is None else len(result.psd_frequencies)),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)))


if __name__ == "__main__":
    main()
