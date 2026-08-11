from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _bootstrap import ROOT

from bci_dayloop.data.hdf5_dataset import EEGHDF5
from bci_dayloop.models.cbramod.runtime import (
    build_cbramod_runtime,
)
from bci_dayloop.runtime.types import RawEEGWindow


def resolve_repo_path(
    value: str | Path,
) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def required_file(
    path: Path,
    *,
    name: str,
) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{name} was not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"{name} is not a file: {path}"
        )

    return path


def load_json_mapping(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    if not isinstance(payload, Mapping):
        raise TypeError(
            f"Expected JSON mapping in {path}."
        )

    return dict(payload)


def required_mapping(
    payload: Mapping[str, Any],
    key: str,
    *,
    source: Path,
) -> dict[str, Any]:
    value = payload.get(key)

    if not isinstance(value, Mapping):
        raise ValueError(
            f"{source}: field {key!r} must be a mapping."
        )

    return dict(value)


def runtime_kwargs_from_training_report(
    report: Mapping[str, Any],
    *,
    source: Path,
) -> dict[str, Any]:
    preprocessing = required_mapping(
        report,
        "preprocessing",
        source=source,
    )

    return {
        "target_sample_rate": float(
            preprocessing["target_sample_rate"]
        ),
        "window_seconds": float(
            preprocessing["window_seconds"]
        ),
        "n_channels": len(
            preprocessing["standard_channels"]
        ),
        "time_segments": int(
            preprocessing["time_segments"]
        ),
        "points_per_patch": int(
            preprocessing["points_per_patch"]
        ),
        "input_unit": str(
            preprocessing["input_unit"]
        ),
        "strict_window_duration": bool(
            preprocessing.get(
                "strict_window_duration",
                True,
            )
        ),
        "window_tolerance_seconds": float(
            preprocessing.get(
                "window_tolerance_seconds",
                0.02,
            )
        ),
        "filter_enabled": bool(
            preprocessing.get(
                "filter_enabled",
                False,
            )
        ),
        "filter_low_hz": float(
            preprocessing.get(
                "filter_low_hz",
                0.1,
            )
        ),
        "filter_high_hz": float(
            preprocessing.get(
                "filter_high_hz",
                75.0,
            )
        ),
        "filter_order": int(
            preprocessing.get(
                "filter_order",
                4,
            )
        ),
        "reference_mode": str(
            preprocessing.get(
                "reference_mode",
                "none",
            )
        ),
        "normalization": str(
            preprocessing.get(
                "normalization",
                "none",
            )
        ),
        "head_type": str(
            report.get(
                "head_type",
                "official_mlp",
            )
        ),
    }


def validate_frozen_backbone(
    runtime: Any,
) -> None:
    trainable = [
        name
        for name, parameter in (
            runtime.backbone.named_parameters()
        )
        if parameter.requires_grad
    ]

    if trainable:
        raise AssertionError(
            "CBraMod backbone must be completely frozen, "
            f"but these parameters are trainable: "
            f"{trainable[:10]}."
        )

    if runtime.backbone.model.training:
        raise AssertionError(
            "Frozen CBRaMod backbone must remain in eval mode."
        )


def assert_probabilities(
    probabilities: torch.Tensor,
    *,
    num_classes: int,
    source: str,
) -> None:
    expected_shape = (1, num_classes)

    if tuple(probabilities.shape) != expected_shape:
        raise AssertionError(
            f"{source}: expected probability shape "
            f"{expected_shape}, got "
            f"{tuple(probabilities.shape)}."
        )

    if not torch.isfinite(probabilities).all():
        raise AssertionError(
            f"{source}: probabilities contain NaN or Inf."
        )

    probability_sum = probabilities.sum(
        dim=-1
    )

    if not torch.allclose(
        probability_sum,
        torch.ones_like(probability_sum),
        atol=1e-5,
        rtol=0.0,
    ):
        raise AssertionError(
            f"{source}: probabilities do not sum to 1. "
            f"Got {probability_sum.detach().cpu().tolist()}."
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a CBRaMod RuntimeModel smoke test on one "
            "real BNCI2014_001 4-second trial."
        )
    )

    parser.add_argument(
        "--data",
        default=(
            "data/processed/bnci2014_001/"
            "subject_01.h5"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/backbones/cbramod/"
            "pretrained_weights.pth"
        ),
    )

    parser.add_argument(
        "--classifier",
        default=(
            "checkpoints/heads/stage1/"
            "bnci2014_001/subject_01/"
            "cbramod/4s_flatten/head.pt"
        ),
    )

    parser.add_argument(
        "--training-report",
        default=(
            "runs/stage1/bnci2014_001/"
            "subject_01/cbramod/4s_flatten/"
            "training_report.json"
        ),
    )

    parser.add_argument(
        "--session",
        default="1test",
    )

    parser.add_argument(
        "--trial-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "mps"),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional JSON result path, relative to "
            "repository root if not absolute."
        ),
    )

    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    data_path = resolve_repo_path(args.data)
    checkpoint_path = resolve_repo_path(args.checkpoint)
    classifier_path = resolve_repo_path(args.classifier)
    report_path = resolve_repo_path(
        args.training_report
    )

    for name, path in (
        ("HDF5 data", data_path),
        ("CBraMod backbone checkpoint", checkpoint_path),
        ("CBraMod classifier checkpoint", classifier_path),
        ("CBraMod training report", report_path),
    ):
        required_file(path, name=name)

    dataset = EEGHDF5(data_path)
    metadata = dataset.metadata

    class_names = tuple(
        str(name)
        for name in metadata.class_names
    )

    if len(class_names) != 4:
        raise ValueError(
            "CBraMod smoke test currently expects "
            "four-class BNCI2014_001, got "
            f"{list(class_names)}."
        )

    report = load_json_mapping(report_path)

    report_class_names = tuple(
        str(name)
        for name in report.get(
            "class_names",
            [],
        )
    )

    if report_class_names != class_names:
        raise ValueError(
            "training_report.json class_names does not "
            "match HDF5 metadata class_names: "
            f"report={report_class_names}, "
            f"dataset={class_names}."
        )

    runtime_kwargs = runtime_kwargs_from_training_report(
        report,
        source=report_path,
    )

    expected_standard_channels = tuple(
        str(name)
        for name in report["preprocessing"][
            "standard_channels"
        ]
    )

    runtime = build_cbramod_runtime(
        checkpoint_path=checkpoint_path,
        classifier_path=classifier_path,
        class_names=class_names,
        device=args.device,
        **runtime_kwargs,
    )

    if (
        tuple(runtime.config.standard_channels)
        != expected_standard_channels
    ):
        raise AssertionError(
            "Code-level CBRaMod channel order does not match "
            "the channel order recorded during head training. "
            f"runtime={runtime.config.standard_channels}, "
            f"report={expected_standard_channels}."
        )

    validate_frozen_backbone(runtime)

    health = runtime.health_check()

    session = dataset.load(args.session)

    trials = np.asarray(
        session["data"],
        dtype=np.float32,
    )

    labels = np.asarray(
        session["labels"],
        dtype=np.int64,
    )

    trial_ids = np.asarray(
        session["trial_ids"],
        dtype=np.int64,
    )

    if trials.ndim != 3:
        raise ValueError(
            "Expected HDF5 trial data [N, C, T], got "
            f"{trials.shape}."
        )

    if not 0 <= args.trial_index < len(trials):
        raise IndexError(
            "--trial-index is out of range: "
            f"{args.trial_index}; session has "
            f"{len(trials)} trials."
        )

    raw_window = RawEEGWindow(
        data=trials[args.trial_index],
        channel_names=list(
            metadata.channel_names
        ),
        sample_rate=float(metadata.sample_rate),
        unit=str(metadata.unit),
        layout="CT",
        trial_id=str(
            int(trial_ids[args.trial_index])
        ),
        window_id="cbramod_runtime_smoke_test",
        label=int(labels[args.trial_index]),
        metadata={
            "dataset": metadata.dataset_name,
            "session": args.session,
            "source": (
                "smoke_test_cbramod_runtime"
            ),
        },
    )

    expected_raw_samples = int(
        round(
            runtime.config.window_seconds
            * float(metadata.sample_rate)
        )
    )

    if raw_window.data.shape[-1] != expected_raw_samples:
        raise AssertionError(
            "Selected HDF5 trial is not a complete CBRaMod "
            "window. Expected "
            f"{expected_raw_samples} samples at "
            f"{metadata.sample_rate} Hz, got "
            f"{raw_window.data.shape[-1]}."
        )

    started = time.perf_counter()

    prepared = runtime.runtime_model.prepare(
        raw_window
    )

    preprocessing_ms = (
        time.perf_counter() - started
    ) * 1000.0

    signal = prepared.model_input["signal"]

    if not isinstance(signal, torch.Tensor):
        raise AssertionError(
            "Prepared model input must contain a Tensor "
            "under key 'signal'."
        )

    expected_prepared_shape = (
        1,
        runtime.config.n_channels,
        runtime.config.time_segments,
        runtime.config.points_per_patch,
    )

    if tuple(signal.shape) != expected_prepared_shape:
        raise AssertionError(
            "Prepared CBRaMod input shape mismatch. "
            f"Expected {expected_prepared_shape}, got "
            f"{tuple(signal.shape)}."
        )

    if signal.dtype != torch.float32:
        raise AssertionError(
            "Prepared CBRaMod input must be float32, got "
            f"{signal.dtype}."
        )

    model_started = time.perf_counter()

    prepared_output = (
        runtime.runtime_model.predict_prepared(
            prepared,
            return_features=True,
        )
    )

    model_ms = (
        time.perf_counter() - model_started
    ) * 1000.0

    assert_probabilities(
        prepared_output.probabilities,
        num_classes=len(class_names),
        source="predict_prepared",
    )

    if prepared_output.features is None:
        raise AssertionError(
            "return_features=True did not return features."
        )

    expected_feature_shape = (
        1,
        runtime.config.n_channels,
        runtime.config.time_segments,
        runtime.config.backbone_output_dim,
    )

    if tuple(
        prepared_output.features.shape
    ) != expected_feature_shape:
        raise AssertionError(
            "CBraMod feature shape mismatch. Expected "
            f"{expected_feature_shape}, got "
            f"{tuple(prepared_output.features.shape)}."
        )

    direct_output = runtime.runtime_model.predict(
        raw_window,
        return_features=False,
    )

    assert_probabilities(
        direct_output.probabilities,
        num_classes=len(class_names),
        source="predict",
    )

    if not torch.allclose(
        prepared_output.probabilities,
        direct_output.probabilities,
        atol=1e-6,
        rtol=1e-5,
    ):
        raise AssertionError(
            "predict_prepared() and predict() produced "
            "different probabilities. "
            f"prepared="
            f"{prepared_output.probabilities.detach().cpu().tolist()}, "
            f"direct="
            f"{direct_output.probabilities.detach().cpu().tolist()}."
        )

    result = {
        "status": "passed",
        "model_name": "cbramod-frozen-head",
        "device": str(runtime.backend.device),
        "data": str(data_path),
        "session": args.session,
        "trial_index": args.trial_index,
        "trial_id": int(
            trial_ids[args.trial_index]
        ),
        "expected_label_id": int(
            labels[args.trial_index]
        ),
        "expected_label_name": class_names[
            int(labels[args.trial_index])
        ],
        "class_names": list(class_names),
        "raw_shape": list(raw_window.data.shape),
        "prepared_shape": list(signal.shape),
        "feature_shape": list(
            prepared_output.features.shape
        ),
        "probability_shape": list(
            prepared_output.probabilities.shape
        ),
        "prediction_id": (
            prepared_output.predicted_class
        ),
        "prediction_name": class_names[
            prepared_output.predicted_class
        ],
        "confidence": prepared_output.confidence,
        "probabilities": (
            prepared_output.probabilities[0]
            .detach()
            .cpu()
            .tolist()
        ),
        "preprocessing_ms": preprocessing_ms,
        "model_ms": model_ms,
        "preprocessing_trace": (
            prepared.preprocessing_trace
        ),
        "preprocessing_diagnostics": (
            prepared.diagnostics
        ),
        "model_diagnostics": (
            prepared_output.diagnostics
        ),
        "health_check": health,
        "warning": (
            "This smoke test validates package components and "
            "RuntimeModel consistency only; one prediction does "
            "not indicate classification accuracy."
        ),
    }

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    if args.output is not None:
        output_path = resolve_repo_path(args.output)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )

        print()
        print("saved result:", output_path)

    print()
    print("CBraMod RuntimeModel smoke test passed.")


if __name__ == "__main__":
    main()