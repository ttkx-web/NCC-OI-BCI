from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _bootstrap import ROOT

from bci_dayloop.data.hdf5_dataset import (
    EEGHDF5,
)
from bci_dayloop.data.channel_selection import (
    select_named_channels,
    strict_channel_indices,
)
from bci_dayloop.data.preprocessing import (
    PreprocessingConfig,
)
from bci_dayloop.models.labram_linear import (
    LaBraMLinearAdapter,
)
from bci_dayloop.packages.exporter import (
    export_labram_runtime_package,
)
from bci_dayloop.packages.loader import (
    load_runtime_package,
)
from bci_dayloop.runtime.types import (
    RawEEGWindow,
)


DEFAULT_COMMANDS = {
    "left_hand": "LEFT",
    "right_hand": "RIGHT",
    "feet": "FORWARD",
    "tongue": "STOP",
}


def resolve_repo_path(
    value: str | Path,
) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def safe_torch_load(
    path: Path,
) -> dict[str, Any]:
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        payload = torch.load(
            path,
            map_location="cpu",
        )

    if not isinstance(payload, dict):
        raise TypeError(
            "LaBraM head checkpoint must "
            f"contain a dictionary: {path}"
        )

    return payload


def replace_directory_atomically(
    *,
    temporary_path: Path,
    output_path: Path,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Pass --overwrite to replace it."
        )

    backup = output_path.with_name(
        f"{output_path.name}.backup-"
        f"{int(time.time())}"
    )

    if output_path.exists():
        output_path.replace(backup)

    try:
        temporary_path.replace(
            output_path
        )
    except Exception:
        if (
            backup.exists()
            and not output_path.exists()
        ):
            backup.replace(output_path)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def verify_package(
    *,
    package_path: Path,
    dataset: EEGHDF5,
    session_name: str,
    device: str,
) -> dict[str, Any]:
    loaded = load_runtime_package(
        package_path,
        device=device,
        verify_hashes=True,
    )

    if loaded.model_type != "labram":
        raise ValueError(
            "Expected a LaBraM package, "
            f"got {loaded.model_type!r}."
        )

    session = dataset.load(session_name)

    trials = np.asarray(
        session["data"],
        dtype=np.float32,
    )

    raw_points = int(
        round(
            loaded.window_sec
            * dataset.metadata.sample_rate
        )
    )

    if trials.shape[-1] < raw_points:
        raise ValueError(
            "Test trial is shorter than the "
            "package window."
        )

    required_channel_names = tuple(
        loaded.runtime_model
        .input_contract.channel_names
    )
    selected_data, selected_channel_names = (
        select_named_channels(
            trials[0, :, :raw_points],
            source_channel_names=(
                dataset.metadata.channel_names
            ),
            requested_channel_names=(
                required_channel_names
            ),
            channel_axis=0,
        )
    )

    raw_window = RawEEGWindow(
        data=selected_data,
        channel_names=list(
            selected_channel_names
        ),
        sample_rate=float(
            dataset.metadata.sample_rate
        ),
        unit=str(dataset.metadata.unit),
        layout="CT",
        window_id="export_smoke_test",
    )

    output = loaded.runtime_model.predict(
        raw_window
    )

    probabilities = (
        output.probabilities
        .detach()
        .cpu()
        .numpy()
    )

    if probabilities.shape != (
        1,
        len(loaded.class_names),
    ):
        raise RuntimeError(
            "Unexpected package probability shape: "
            f"{probabilities.shape}."
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise RuntimeError(
            "Package produced NaN or Inf."
        )

    if not np.isclose(
        float(probabilities.sum()),
        1.0,
        rtol=0.0,
        atol=1e-5,
    ):
        raise RuntimeError(
            "Package probabilities do not sum to 1."
        )

    return {
        "status": "passed",
        "prediction": int(
            output.predicted_class
        ),
        "confidence": float(
            output.confidence
        ),
        "probabilities": (
            probabilities[0].tolist()
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a trained LaBraM linear head "
            "as a schema-v2 Runtime Model Package."
        )
    )

    parser.add_argument(
        "--data",
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="LaBraM backbone checkpoint.",
    )
    parser.add_argument(
        "--classifier",
        required=True,
        help="Trained LaBraM head.pt.",
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    parser.add_argument(
        "--session",
        default="1test",
    )
    parser.add_argument(
        "--step-sec",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.55,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    data_path = resolve_repo_path(
        args.data
    )
    backbone_path = resolve_repo_path(
        args.checkpoint
    )
    classifier_path = resolve_repo_path(
        args.classifier
    )
    output_path = resolve_repo_path(
        args.output
    )

    for name, path in (
        ("HDF5 data", data_path),
        ("LaBraM backbone", backbone_path),
        ("LaBraM head", classifier_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"{name} was not found: {path}"
            )

    head_payload = safe_torch_load(
        classifier_path
    )

    metadata = head_payload.get(
        "metadata"
    )

    if not isinstance(metadata, dict):
        raise ValueError(
            "LaBraM head checkpoint does not "
            "contain metadata."
        )

    state_dict = head_payload.get(
        "state_dict"
    )

    if not isinstance(state_dict, dict):
        raise ValueError(
            "LaBraM head checkpoint does not "
            "contain state_dict."
        )

    dataset = EEGHDF5(data_path)
    data_metadata = dataset.metadata

    class_names = tuple(
        str(name)
        for name in data_metadata.class_names
    )

    if tuple(
        metadata["class_names"]
    ) != class_names:
        raise ValueError(
            "Head class order does not match "
            "the HDF5 dataset."
        )

    channel_names = tuple(
        str(name)
        for name in metadata["channel_names"]
    )

    source_channel_names = tuple(
        str(name)
        for name in data_metadata.channel_names
    )
    strict_channel_indices(
        source_channel_names,
        channel_names,
    )

    source_channel_count = metadata.get(
        "source_channel_count"
    )
    if (
        source_channel_count is not None
        and int(source_channel_count)
        != len(source_channel_names)
    ):
        raise ValueError(
            "Head source_channel_count does not "
            "match the HDF5 dataset."
        )

    selected_channel_count = metadata.get(
        "selected_channel_count"
    )
    if (
        selected_channel_count is not None
        and int(selected_channel_count)
        != len(channel_names)
    ):
        raise ValueError(
            "Head selected_channel_count does not "
            "match head channel_names."
        )

    if (
        len(channel_names)
        < len(source_channel_names)
        and metadata.get(
            "channel_selection_policy"
        ) != "explicit_live_intersection"
    ):
        raise ValueError(
            "A subset head must declare "
            "channel_selection_policy="
            "'explicit_live_intersection'."
        )

    preprocessing_config = (
        PreprocessingConfig.from_dict(
            dict(metadata["preprocessing"])
        )
    )

    adapter = LaBraMLinearAdapter(
        channel_names=list(channel_names),
        n_classes=int(
            metadata["num_classes"]
        ),
        checkpoint=backbone_path,
        device=args.device,
        amp=False,
        freeze_encoder=True,
        embedding_batch_size=4,
        random_init=False,
        n_patches=int(
            metadata["n_patches"]
        ),
    )

    adapter.head.load_state_dict(
        state_dict,
        strict=True,
    )
    adapter.head.eval()

    command_map = {
        name: DEFAULT_COMMANDS[name]
        for name in class_names
        if name in DEFAULT_COMMANDS
    }

    temporary_path = (
        output_path.with_name(
            f".{output_path.name}.tmp-"
            f"{os.getpid()}"
        )
    )

    if temporary_path.exists():
        shutil.rmtree(temporary_path)

    try:
        saved_package = (
            export_labram_runtime_package(
                output_dir=temporary_path,
                adapter=adapter,
                backbone_checkpoint=(
                    backbone_path
                ),
                classifier_checkpoint=(
                    classifier_path
                ),
                preprocessing_config=(
                    preprocessing_config
                ),
                class_names=class_names,
                command_map=command_map,
                dataset_name=str(
                    data_metadata.dataset_name
                ),
                package_id=(
                    "labram_bnci2014_001_"
                    "subject_01_population_4s"
                    + (
                        f"_live{len(channel_names)}"
                        if metadata.get(
                            "channel_selection_policy"
                        )
                        == "explicit_live_intersection"
                        else ""
                    )
                ),
                package_version=(
                    output_path.name
                ),
                step_sec=float(
                    args.step_sec
                ),
                confidence_threshold=float(
                    args.confidence_threshold
                ),
                metrics={
                    "model_selection": (
                        metadata.get(
                            "model_selection",
                            {},
                        )
                    ),
                    "final_test": (
                        metadata.get(
                            "final_test",
                            {},
                        )
                    ),
                },
                adaptation={
                    "offline": {
                        "type": "none",
                        "head_type": "population",
                    },
                    "online": {
                        "type": "none",
                    },
                },
            )
        )

        verification = None

        if not args.skip_smoke_test:
            verification = verify_package(
                package_path=saved_package,
                dataset=dataset,
                session_name=args.session,
                device=args.device,
            )

        replace_directory_atomically(
            temporary_path=temporary_path,
            output_path=output_path,
            overwrite=args.overwrite,
        )

        if not args.skip_smoke_test:
            verify_package(
                package_path=output_path,
                dataset=dataset,
                session_name=args.session,
                device=args.device,
            )

    except Exception:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise

    print("LaBraM package exported:", output_path)
    print("Smoke test:", verification)


if __name__ == "__main__":
    main()
