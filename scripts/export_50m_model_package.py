from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _bootstrap import ROOT

from bci_dayloop.data.hdf5_dataset import EEGHDF5
from bci_dayloop.models.base import add_batch_dimension
from bci_dayloop.models.model_50m.runtime import (
    build_50m_runtime_from_metadata,
)
from bci_dayloop.models.runtime_package import (
    load_50m_runtime_package,
)
from bci_dayloop.utils.config import dump_json, dump_yaml, load_yaml


DEFAULT_COMMANDS = {
    "left_hand": "LEFT",
    "right_hand": "RIGHT",
    "feet": "FORWARD",
    "tongue": "STOP",
}


def resolve_repo_path(value: str | Path) -> Path:
    """Resolve a relative path from the repository root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_torch_load(path: Path) -> dict[str, Any]:
    """Load a dependency-free classifier checkpoint."""
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if not isinstance(payload, dict):
        raise TypeError(
            f"Checkpoint must contain a dictionary, got {type(payload)!r}: {path}"
        )
    return payload


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_json_mapping(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")

    return {str(key): str(item) for key, item in value.items()}


def build_label_map(class_names: tuple[str, ...]) -> dict[str, str]:
    return {
        str(index): class_name
        for index, class_name in enumerate(class_names)
    }


def build_default_command_map(
    class_names: tuple[str, ...],
) -> dict[str, str]:
    return {
        class_name: DEFAULT_COMMANDS[class_name]
        for class_name in class_names
        if class_name in DEFAULT_COMMANDS
    }


def preserve_classifier_metadata(
    *,
    source_classifier: Path,
    packaged_classifier: Path,
    class_names: tuple[str, ...],
    export_time: str,
) -> dict[str, Any]:
    """
    adapter.save() writes a valid task head, but its default export metadata
    only contains the runtime contract. Merge the formal training metadata
    from the source classifier into the packaged classifier.
    """
    source_payload = safe_torch_load(source_classifier)
    packaged_payload = safe_torch_load(packaged_classifier)

    source_metadata = source_payload.get("metadata", {})
    packaged_metadata = packaged_payload.get("metadata", {})

    if not isinstance(source_metadata, dict):
        raise TypeError(
            f"Source classifier metadata must be a dictionary: {source_classifier}"
        )
    if not isinstance(packaged_metadata, dict):
        raise TypeError(
            f"Packaged classifier metadata must be a dictionary: {packaged_classifier}"
        )

    merged_metadata = dict(source_metadata)
    merged_metadata.update(packaged_metadata)
    merged_metadata.update(
        {
            "class_names": list(class_names),
            "is_test_head": False,
            "trained_head": True,
            "source_classifier_path": str(source_classifier),
            "source_classifier_sha256": sha256_file(source_classifier),
            "package_exported_at_utc": export_time,
        }
    )

    packaged_payload["metadata"] = merged_metadata
    atomic_torch_save(packaged_payload, packaged_classifier)
    return merged_metadata


def update_package_metadata(
    *,
    package_path: Path,
    final_package_path: Path,
    checkpoint_path: Path,
    classifier_path: Path,
    class_names: tuple[str, ...],
    step_sec: float,
    dataset_name: str,
    export_time: str,
) -> None:
    model_path = package_path / "model.yaml"
    model_payload = load_yaml(model_path)
    model_payload.update(
        {
            "step_sec": float(step_sec),
            "task": "motor_imagery",
            "dataset": dataset_name,
            "classifier_type": "trained_linear_probe",
        }
    )
    dump_yaml(model_payload, model_path)

    base_model_path = package_path / "base_model.json"
    with base_model_path.open("r", encoding="utf-8") as handle:
        base_model = dict(json.load(handle))

    # Use a repository-relative reference instead of a developer-machine
    # absolute path. The temp export directory is a sibling of the final
    # directory, so the relative path is valid before and after the rename.
    relative_checkpoint = os.path.relpath(
        checkpoint_path,
        start=final_package_path,
    )

    base_model.pop("checkpoint_path_absolute", None)
    base_model.update(
        {
            "checkpoint_path": relative_checkpoint,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "classifier_path": "classifier.pt",
            "classifier_source_path": str(classifier_path),
            "classifier_source_sha256": sha256_file(classifier_path),
            "is_test_head": False,
            "trained_head": True,
            "warning_message": None,
            "task": "motor_imagery",
            "dataset": dataset_name,
            "class_names": list(class_names),
            "package_exported_at_utc": export_time,
        }
    )
    dump_json(base_model, base_model_path)


def extract_first_real_window(
    *,
    dataset: EEGHDF5,
    session_name: str,
    window_seconds: float,
) -> np.ndarray:
    """
    Build one real-length window using the same trial concatenation used by
    replay. This smoke test checks package loading and inference only; the
    window can cross original trial labels.
    """
    session = dataset.load(session_name)
    trials = np.asarray(session["data"], dtype=np.float32)

    if trials.ndim != 3:
        raise ValueError(
            f"Expected HDF5 trial data [N,C,T], got {trials.shape}."
        )

    stream = (
        trials.transpose(1, 0, 2)
        .reshape(trials.shape[1], -1)
        .astype(np.float32, copy=False)
    )

    window_samples = int(
        round(window_seconds * float(dataset.metadata.sample_rate))
    )

    if stream.shape[-1] < window_samples:
        raise ValueError(
            f"Session {session_name!r} has only "
            f"{stream.shape[-1] / dataset.metadata.sample_rate:.3f}s, "
            f"less than the required {window_seconds:.3f}s."
        )

    return stream[:, :window_samples].copy()


def verify_package(
    *,
    package_path: Path,
    dataset: EEGHDF5,
    device: str,
    session_name: str,
) -> dict[str, Any]:
    runtime = load_50m_runtime_package(
        package_path,
        dataset.metadata,
        device=device,
    )

    if runtime.is_test_head:
        raise RuntimeError(
            "The exported runtime package is still marked as a test head."
        )

    raw_window = extract_first_real_window(
        dataset=dataset,
        session_name=session_name,
        window_seconds=runtime.window_sec,
    )

    model_input = runtime.preprocessor.transform(
        raw_window,
        dataset.metadata.sample_rate,
        dataset.metadata.unit,
    )
    probabilities = runtime.model.predict_proba(
        add_batch_dimension(model_input)
    )

    expected_shape = (1, len(runtime.class_names))
    if probabilities.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected probability shape: expected {expected_shape}, "
            f"got {probabilities.shape}."
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Package inference produced NaN or Inf.")
    if not np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-5):
        raise RuntimeError("Package probabilities do not sum to 1.")

    return {
        "model_name": runtime.model_name,
        "class_names": list(runtime.class_names),
        "window_sec": float(runtime.window_sec),
        "step_sec": float(runtime.step_sec),
        "target_sample_rate": float(runtime.target_sample_rate),
        "is_test_head": bool(runtime.is_test_head),
        "probability_shape": list(probabilities.shape),
        "prediction": int(probabilities[0].argmax()),
        "confidence": float(probabilities[0].max()),
        "probabilities": probabilities[0].tolist(),
        "warning": (
            "The smoke-test window can cross original 4-second trial labels; "
            "this verifies runtime integrity, not accuracy."
        ),
    }


def replace_directory_atomically(
    *,
    temporary_path: Path,
    output_path: Path,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output package already exists: {output_path}. "
            "Pass --overwrite to replace it."
        )

    backup_path = output_path.with_name(
        f"{output_path.name}.backup-{int(time.time())}"
    )

    if output_path.exists():
        output_path.replace(backup_path)

    try:
        temporary_path.replace(output_path)
    except Exception:
        if backup_path.exists() and not output_path.exists():
            backup_path.replace(output_path)
        raise
    else:
        if backup_path.exists():
            shutil.rmtree(backup_path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a deployable 50M runtime model package using the "
            "formally trained BNCI2014_001 Subject 1 linear head."
        )
    )
    parser.add_argument(
        "--data",
        default="data/processed/bnci2014_001_s01.h5",
        help="HDF5 data used to resolve channels, units and class order.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/50m/model_deploy.pt",
        help="Dependency-free 50M backbone checkpoint.",
    )
    parser.add_argument(
        "--classifier",
        default="checkpoints/50m_bnci2014_001_s01_linear_head.pt",
        help="Formally trained linear-head checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="runs/stage05_50m/model_package",
        help="Output runtime package directory.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "mps", "auto"),
        help="Device used while loading and verifying the package.",
    )
    parser.add_argument(
        "--session",
        default="1test",
        help="HDF5 session used for the post-export smoke test.",
    )
    parser.add_argument(
        "--step-sec",
        type=float,
        default=0.5,
        help="Runtime sliding-window step stored in model.yaml.",
    )
    parser.add_argument(
        "--command-map-json",
        default=None,
        help=(
            "Optional JSON object mapping class names to commands. "
            "When omitted, known motor-imagery defaults are used."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output package after verification.",
    )
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Export files without loading the package and running one window.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    if args.step_sec <= 0:
        raise ValueError("--step-sec must be positive.")

    data_path = resolve_repo_path(args.data)
    checkpoint_path = resolve_repo_path(args.checkpoint)
    classifier_path = resolve_repo_path(args.classifier)
    output_path = resolve_repo_path(args.output)
    command_map_path = (
        resolve_repo_path(args.command_map_json)
        if args.command_map_json is not None
        else None
    )

    for name, path in (
        ("HDF5 data", data_path),
        ("50M backbone checkpoint", checkpoint_path),
        ("trained classifier", classifier_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} was not found: {path}")

    if command_map_path is not None and not command_map_path.is_file():
        raise FileNotFoundError(
            f"Command-map JSON was not found: {command_map_path}"
        )

    dataset = EEGHDF5(data_path)
    metadata = dataset.metadata
    class_names = tuple(str(name) for name in metadata.class_names)

    label_map = build_label_map(class_names)
    command_map = load_json_mapping(command_map_path)
    if command_map is None:
        command_map = build_default_command_map(class_names)

    export_time = datetime.now(timezone.utc).isoformat()

    temporary_path = output_path.with_name(
        f".{output_path.name}.tmp-{os.getpid()}"
    )
    if temporary_path.exists():
        shutil.rmtree(temporary_path)

    print("=" * 78)
    print("Export 50M runtime model package")
    print("=" * 78)
    print("data:", data_path)
    print("checkpoint:", checkpoint_path)
    print("classifier:", classifier_path)
    print("output:", output_path)
    print("class_names:", class_names)
    print("command_map:", command_map)
    print()

    try:
        runtime = build_50m_runtime_from_metadata(
            checkpoint_path=checkpoint_path,
            classifier_path=classifier_path,
            metadata=metadata,
            device=args.device,
        )

        saved_package = runtime.adapter.save(
            temporary_path,
            label_map=label_map,
            command_map=command_map,
        )

        preserve_classifier_metadata(
            source_classifier=classifier_path,
            packaged_classifier=saved_package / "classifier.pt",
            class_names=class_names,
            export_time=export_time,
        )

        update_package_metadata(
            package_path=saved_package,
            final_package_path=output_path,
            checkpoint_path=checkpoint_path,
            classifier_path=classifier_path,
            class_names=class_names,
            step_sec=args.step_sec,
            dataset_name=str(metadata.dataset_name),
            export_time=export_time,
        )

        pre_export_verification: dict[str, Any] | None = None
        if not args.skip_smoke_test:
            pre_export_verification = verify_package(
                package_path=saved_package,
                dataset=dataset,
                device=args.device,
                session_name=args.session,
            )
            print("Temporary package smoke test passed.")

        replace_directory_atomically(
            temporary_path=temporary_path,
            output_path=output_path,
            overwrite=args.overwrite,
        )

        final_verification: dict[str, Any] | None = None
        if not args.skip_smoke_test:
            final_verification = verify_package(
                package_path=output_path,
                dataset=dataset,
                device=args.device,
                session_name=args.session,
            )
            print("Final package smoke test passed.")

        manifest = {
            "status": "exported",
            "exported_at_utc": export_time,
            "package_path": str(output_path),
            "data_path": str(data_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "classifier_path": str(classifier_path),
            "classifier_sha256": sha256_file(classifier_path),
            "class_names": list(class_names),
            "label_map": label_map,
            "command_map": command_map,
            "step_sec": float(args.step_sec),
            "is_test_head": False,
            "trained_head": True,
            "temporary_verification": pre_export_verification,
            "final_verification": final_verification,
        }
        dump_json(manifest, output_path / "export_manifest.json")

    except Exception:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise

    print()
    print("Export completed:", output_path)
    print("Package files:")
    for path in sorted(output_path.iterdir()):
        if path.is_file():
            print(" -", path.name)
    print()
    print(
        "The package uses the trained head and is_test_head=false. "
        "It is ready for CLI Replay and Streamlit validation."
    )


if __name__ == "__main__":
    main()
