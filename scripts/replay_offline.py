from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from bci_dayloop.acquisition.factory import AcquirerFactory
from bci_dayloop.data.hdf5_dataset import EEGHDF5
from bci_dayloop.data.preprocessing import EEGPreprocessor
from bci_dayloop.inference.realtime import SlidingWindowDecoder
from bci_dayloop.models.factory import ModelFactory
from bci_dayloop.utils.config import load_yaml, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pseudo-realtime HDF5 replay through a model package")
    parser.add_argument("--config", default="configs/day1_bnci_s01.yaml")
    parser.add_argument("--data")
    parser.add_argument("--model-package")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--max-windows", type=int)
    args = parser.parse_args()
    config = load_yaml(resolve_path(args.config))
    replay = config["replay"]
    data_path = resolve_path(args.data or config["data"]["output_hdf5"])
    package = resolve_path(args.model_package or Path(config["project"]["run_dir"]) / "model_package")
    device = args.device or config["model"].get("device", "cuda")
    dataset = EEGHDF5(data_path)
    metadata = dataset.metadata
    model = ModelFactory.load_package(package, device=device)
    preprocessor = EEGPreprocessor(load_yaml(package / "preprocessing.yaml"))
    with (package / "command_map.json").open("r", encoding="utf-8") as handle:
        command_map = json.load(handle)
    acquirer = AcquirerFactory.create(
        replay["acquirer"],
        data_path=data_path,
        session=str(replay["session"]),
        speed=float(replay["speed"]),
        loop=bool(replay["loop"]),
        window_sec=float(replay["window_sec"]),
        step_sec=float(replay["step_sec"]),
    )
    decoder = SlidingWindowDecoder(
        model,
        preprocessor,
        metadata.class_names,
        sample_rate=metadata.sample_rate,
        input_unit=metadata.unit,
        window_sec=float(replay["window_sec"]),
        step_sec=float(replay["step_sec"]),
        confidence_threshold=float(replay["confidence_threshold"]),
        command_map=command_map,
    )
    expected_window_samples = round(
        float(replay["window_sec"]) * metadata.sample_rate
    )

    print("Replay window:", float(replay["window_sec"]), "seconds")
    print("Raw EEG sample rate:", metadata.sample_rate, "Hz")
    print("Decoder window samples:", decoder.window_samples)
    print("Expected window samples:", expected_window_samples)

    assert decoder.window_samples == expected_window_samples, (
        "Decoder window size does not match the replay configuration: "
        f"decoder={decoder.window_samples}, "
        f"expected={expected_window_samples}"
    )
    maximum = args.max_windows or int(replay["max_windows"])
    for result in decoder.run(acquirer, max_windows=maximum):
        print(json.dumps(result.to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()

