from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _bootstrap import ROOT

from bci_dayloop.data.sequential_dataset import SequentialDataset, load_sequential_dataset
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


def _safe_slug(value: object) -> str:
    slug = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")
    if not slug:
        raise ValueError("Package identity field cannot be empty.")
    return slug


def build_labram_package_id(
    *,
    dataset_name: str,
    metadata: Mapping[str, Any],
) -> str:
    subject = metadata.get(
        "target_subject",
        metadata.get("subject"),
    )
    if subject is None:
        raise ValueError(
            "LaBraM head metadata must contain target_subject or subject."
        )

    subject_text = str(subject).strip()
    subject_match = re.fullmatch(
        r"(?:subject[_-]?|p)?0*(\d+)",
        subject_text,
        flags=re.IGNORECASE,
    )
    subject_slug = (
        f"{int(subject_match.group(1)):02d}"
        if subject_match is not None
        else _safe_slug(subject_text)
    )

    window_seconds = float(metadata["window_seconds"])
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError(
            "LaBraM head metadata window_seconds must be finite and positive."
        )
    window_slug = (
        str(int(round(window_seconds)))
        if np.isclose(window_seconds, round(window_seconds))
        else format(window_seconds, "g").replace(".", "p")
    )

    return (
        f"labram_{_safe_slug(dataset_name)}_"
        f"subject_{subject_slug}_population_{window_slug}s"
    )


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
    dataset: SequentialDataset,
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

    trials = np.asarray(dataset.data, dtype=np.float32)

    raw_points = int(
        round(
            loaded.window_sec
            * dataset.metadata.sample_rate
        )
    )

    if trials.shape[-1] != raw_points:
        raise ValueError(
            "Export smoke source trial must exactly match the package window; "
            "no crop, padding, or cross-trial concatenation is allowed."
        )

    raw_window = RawEEGWindow(
        data=trials[0, :, :raw_points],
        channel_names=[
            str(name)
            for name
            in dataset.metadata.channel_names
        ],
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

    dataset = load_sequential_dataset(data_path, session=args.session)
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

    if tuple(
        str(name)
        for name in data_metadata.channel_names
    ) != channel_names:
        raise ValueError(
            "Head channel order does not match "
            "the HDF5 dataset."
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
                package_id=build_labram_package_id(
                    dataset_name=str(
                        data_metadata.dataset_name
                    ),
                    metadata=metadata,
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
