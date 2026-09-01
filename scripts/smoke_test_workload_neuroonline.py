"""Run one Workload HDF5 trial through an existing Runtime package."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    from _bootstrap import ROOT  # noqa: F401
except ModuleNotFoundError:  # Imported as a module by tests.
    from scripts._bootstrap import ROOT  # noqa: F401

from bci_dayloop.data.dataset_adapter_registry import (
    DEFAULT_DATASET_ADAPTER_REGISTRY,
    inspect_hdf5_dataset,
)
from bci_dayloop.data.sequential_dataset import (
    SequentialDataset,
    load_sequential_dataset,
    validate_package_window_contract,
)
from bci_dayloop.packages.loader import load_runtime_package
from bci_dayloop.runtime.types import RawEEGWindow


DEFAULT_WORKLOAD_DATA = Path("data/processed/workload/subject_01.h5")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_WORKLOAD_DATA)
    parser.add_argument("--session", default="S1")
    parser.add_argument("--model-package", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trial-index", type=int, default=0)
    return parser


def raw_window_from_dataset(
    dataset: SequentialDataset,
    *,
    trial_index: int,
) -> RawEEGWindow:
    """Build one Runtime input without changing the persisted trial payload."""
    index = int(trial_index)
    if index < 0 or index >= dataset.num_trials:
        raise ValueError(
            f"trial_index must be in [0, {dataset.num_trials}), got {index}."
        )
    return RawEEGWindow(
        data=np.asarray(dataset.data[index], dtype=np.float32),
        channel_names=list(dataset.metadata.channel_names),
        sample_rate=float(dataset.metadata.sample_rate),
        unit=str(dataset.metadata.unit),
        layout="CT",
        trial_id=str(dataset.trial_ids[index]),
        window_id=str(dataset.window_ids[index]),
        label=int(dataset.labels[index]),
        metadata={
            "dataset_name": dataset.metadata.dataset_name,
            "subject_id": str(dataset.subject_ids[index]),
            "session_id": str(dataset.session_ids[index]),
            "trial_ordinal": int(dataset.trial_ordinals[index]),
        },
    )


def _probability_row(
    probabilities: object,
    *,
    class_count: int,
) -> np.ndarray:
    detached = getattr(probabilities, "detach", None)
    values = detached() if callable(detached) else probabilities
    to_cpu = getattr(values, "cpu", None)
    values = to_cpu() if callable(to_cpu) else values
    numeric = np.asarray(values, dtype=np.float64)
    if numeric.shape != (1, class_count):
        raise ValueError(
            "Runtime probability shape does not match the package class names: "
            f"shape={numeric.shape}, class_count={class_count}."
        )
    if not np.isfinite(numeric).all():
        raise ValueError("Runtime probabilities contain NaN or Inf.")
    if not math.isclose(float(numeric.sum()), 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError(
            "Runtime probabilities must sum to one, got "
            f"{float(numeric.sum()):.9f}."
        )
    return numeric[0]


def run_smoke(
    *,
    data_path: Path,
    session: str,
    model_package: Path,
    device: str,
    trial_index: int,
) -> dict[str, Any]:
    """Verify adapter selection and one end-to-end Runtime prediction."""
    descriptor = inspect_hdf5_dataset(data_path)
    adapter = DEFAULT_DATASET_ADAPTER_REGISTRY.resolve(descriptor)
    if adapter.name != "workload_hdf5":
        raise ValueError(
            "Workload smoke test requires the workload_hdf5 adapter, got "
            f"{adapter.name!r}."
        )

    dataset = load_sequential_dataset(data_path, session=session)
    if dataset.metadata.dataset_name != "workload_pbci_hackathon":
        raise ValueError(
            "Workload smoke test loaded an unexpected dataset_name: "
            f"{dataset.metadata.dataset_name!r}."
        )
    raw_window = raw_window_from_dataset(dataset, trial_index=trial_index)
    loaded = load_runtime_package(model_package, device=device)
    validate_package_window_contract(
        dataset,
        package_window_sec=loaded.window_sec,
    )
    output = loaded.runtime_model.predict(raw_window)
    probabilities = _probability_row(
        output.probabilities,
        class_count=len(loaded.class_names),
    )
    prediction = int(output.predicted_class)
    if prediction < 0 or prediction >= len(loaded.class_names):
        raise ValueError(
            "Runtime prediction is outside the package class range: "
            f"prediction={prediction}, class_count={len(loaded.class_names)}."
        )

    class_names_match = tuple(dataset.metadata.class_names) == loaded.class_names
    return {
        "schema_version": 1,
        "adapter": adapter.name,
        "dataset": {
            "name": dataset.metadata.dataset_name,
            "class_names": list(dataset.metadata.class_names),
            "sample_rate": dataset.metadata.sample_rate,
            "channel_names": list(dataset.metadata.channel_names),
            "num_trials": dataset.num_trials,
        },
        "trial": {
            "index": int(trial_index),
            "subject_id": raw_window.metadata["subject_id"],
            "session_id": raw_window.metadata["session_id"],
            "trial_id": raw_window.trial_id,
            "window_id": raw_window.window_id,
            "label": raw_window.label,
            "trial_ordinal": raw_window.metadata["trial_ordinal"],
        },
        "runtime": {
            "model_type": loaded.model_type,
            "model_name": loaded.model_name,
            "class_names": list(loaded.class_names),
            "class_names_match_dataset": class_names_match,
            "window_sec": loaded.window_sec,
        },
        "prediction": {
            "class_index": prediction,
            "class_name": loaded.class_names[prediction],
            "confidence": float(output.confidence),
            "probabilities": probabilities.tolist(),
        },
        "warning": (
            None
            if class_names_match
            else (
                "Runtime package class_names differ from Workload labels; "
                "this validates input/inference compatibility only, not "
                "Workload accuracy."
            )
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_smoke(
            data_path=args.data,
            session=args.session,
            model_package=args.model_package,
            device=args.device,
            trial_index=args.trial_index,
        )
    except (FileNotFoundError, ValueError) as error:
        build_parser().error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
