from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT  # noqa: F401

from bci_dayloop.data.hdf5_dataset import (
    EEGHDF5,
)
from bci_dayloop.evaluation import (
    RuntimeEvaluator,
)
from bci_dayloop.packages.loader import (
    load_runtime_package,
)
from bci_dayloop.utils.config import (
    resolve_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Runtime Model Package "
            "with trial-aligned EEG windows."
        )
    )

    parser.add_argument(
        "--data",
        required=True,
    )

    parser.add_argument(
        "--model-package",
        required=True,
    )

    parser.add_argument(
        "--session",
        default="1test",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=(
            "cpu",
            "cuda",
            "mps",
        ),
    )

    parser.add_argument(
        "--step-sec",
        type=float,
        default=None,
        help=(
            "Window step. Defaults to the "
            "Runtime Model Package step_sec."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help=(
            "Smoke-test only. Formal evaluation "
            "should evaluate all trials."
        ),
    )

    parser.add_argument(
        "--skip-short-trials",
        action="store_true",
    )

    parser.add_argument(
        "--allow-missing-classes",
        action="store_true",
        help=(
            "Smoke-test only. Formal evaluation "
            "should contain every class."
        ),
    )

    parser.add_argument(
        "--include-preprocessing-details",
        action="store_true",
    )

    return parser


def print_metrics(
    title: str,
    metrics: dict,
) -> None:
    print()
    print(title)
    print("=" * len(title))

    print(
        f"Accuracy:          "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro-F1:          "
        f"{metrics['macro_f1']:.4f}"
    )

    print("Confusion Matrix:")

    matrix = np.asarray(
        metrics["confusion_matrix"],
        dtype=np.int64,
    )

    print(matrix)

    print(
        "Labels:",
        metrics[
            "confusion_matrix_labels"
        ],
    )


def main() -> None:
    args = build_parser().parse_args()

    data_path = resolve_path(
        args.data
    )

    package_path = resolve_path(
        args.model_package
    )

    output_path = resolve_path(
        args.output
    )

    dataset = EEGHDF5(data_path)
    metadata = dataset.metadata

    loaded = load_runtime_package(
        package_path,
        device=args.device,
        verify_hashes=True,
    )

    dataset_classes = tuple(
        str(name)
        for name in metadata.class_names
    )

    if (
        dataset_classes
        != loaded.class_names
    ):
        raise ValueError(
            "Dataset class order does not "
            "match Runtime Package: "
            f"dataset={dataset_classes}, "
            f"package={loaded.class_names}."
        )

    step_sec = (
        loaded.step_sec
        if args.step_sec is None
        else float(args.step_sec)
    )

    evaluator = RuntimeEvaluator(
        runtime_model=(
            loaded.runtime_model
        ),
        class_names=(
            loaded.class_names
        ),
        step_sec=step_sec,
        window_sec=loaded.window_sec,
        short_trial_policy=(
            "skip"
            if args.skip_short_trials
            else "error"
        ),
        require_all_classes=(
            not args.allow_missing_classes
        ),
        include_preprocessing_details=(
            args
            .include_preprocessing_details
        ),
    )

    result = evaluator.evaluate_hdf5(
        data_path,
        session=args.session,
        max_trials=args.max_trials,
    )

    result.save_json(
        output_path
    )

    print()
    print("Runtime evaluation completed.")
    print(f"Model:          {loaded.model_name}")
    print(f"Package:        {package_path}")
    print(f"Dataset:        {data_path}")
    print(f"Session:        {args.session}")
    print(f"Window seconds: {result.window_sec}")
    print(f"Step seconds:   {result.step_sec}")
    print(f"Trials:         {result.num_evaluated_trials}")
    print(f"Windows:        {result.num_windows}")

    print_metrics(
        "Window-level Metrics",
        result.window_metrics,
    )

    print_metrics(
        "Trial-level Metrics",
        result.trial_metrics,
    )

    print()
    print("Latency:")
    print(
        json.dumps(
            result.latency,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()