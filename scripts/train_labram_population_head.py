from __future__ import annotations

"""
Train a frozen-LaBraM LOSO population linear head.

Protocol
--------
For one target subject:

- Population training:
    all non-target subjects / 0train

- Population validation:
    all non-target subjects / 1test

- Final held-out test:
    target subject / 1test

The target subject is not loaded for training or model selection.
Only the linear classification head is trained.
"""

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from _bootstrap import ROOT

from bci_dayloop.data.hdf5_dataset import (
    EEGHDF5,
    HDF5Metadata,
)
from bci_dayloop.data.preprocessing import (
    EEGPreprocessor,
    PreprocessingConfig,
)
from bci_dayloop.models.factory import ModelFactory
from bci_dayloop.training.labram_linear_head import (
    atomic_torch_save,
    sha256_file,
)
from bci_dayloop.utils.config import (
    dump_json,
    seed_everything,
)
from bci_dayloop.utils.metrics import (
    classification_metrics,
)
from bci_dayloop.utils.paths import (
    population_head_path,
    population_run_dir,
)


def resolve_repo_path(
    value: str | Path,
) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return None

    value = result.stdout.strip()
    return value or None


def normalize_subjects(
    subjects: Sequence[int],
) -> list[int]:
    normalized = sorted(
        set(int(subject) for subject in subjects)
    )

    if not normalized:
        raise ValueError(
            "At least one subject is required."
        )

    invalid = [
        subject
        for subject in normalized
        if subject <= 0
    ]

    if invalid:
        raise ValueError(
            "Subject IDs must be positive, "
            f"got {invalid}."
        )

    return normalized


def resolve_subject_file(
    *,
    data_root: Path,
    data_pattern: str,
    subject_id: int,
) -> Path:
    try:
        relative_name = data_pattern.format(
            subject=subject_id
        )
    except (KeyError, ValueError) as error:
        raise ValueError(
            "--data-pattern must be a valid Python "
            "format string containing {subject}, for "
            "example subject_{subject:02d}.h5."
        ) from error

    candidates = [
        data_root / relative_name,
        data_root / (
            f"subject_{subject_id:02d}.h5"
        ),
        data_root / (
            f"bnci2014_001_s"
            f"{subject_id:02d}.h5"
        ),
    ]

    resolved_candidates: list[Path] = []
    seen: set[Path] = set()

    for candidate in candidates:
        resolved = (
            candidate.expanduser().resolve()
        )

        if resolved not in seen:
            resolved_candidates.append(
                resolved
            )
            seen.add(resolved)

    for candidate in resolved_candidates:
        if candidate.is_file():
            return candidate

    attempted = "\n".join(
        f"  - {path}"
        for path in resolved_candidates
    )

    raise FileNotFoundError(
        "Could not find HDF5 data for "
        f"subject {subject_id}. Tried:\n"
        f"{attempted}"
    )


def validate_metadata_compatibility(
    reference: HDF5Metadata,
    candidate: HDF5Metadata,
    *,
    subject_id: int,
    path: Path,
) -> None:
    mismatches: list[str] = []

    if not np.isclose(
        reference.sample_rate,
        candidate.sample_rate,
    ):
        mismatches.append(
            "sample_rate differs: "
            f"{candidate.sample_rate} != "
            f"{reference.sample_rate}"
        )

    if (
        list(reference.channel_names)
        != list(candidate.channel_names)
    ):
        mismatches.append(
            "channel_names or channel order differ"
        )

    if (
        list(reference.class_names)
        != list(candidate.class_names)
    ):
        mismatches.append(
            "class_names or class order differ"
        )

    if str(reference.unit) != str(
        candidate.unit
    ):
        mismatches.append(
            "unit differs: "
            f"{candidate.unit!r} != "
            f"{reference.unit!r}"
        )

    if (
        str(reference.dataset_name)
        != str(candidate.dataset_name)
    ):
        mismatches.append(
            "dataset_name differs: "
            f"{candidate.dataset_name!r} != "
            f"{reference.dataset_name!r}"
        )

    if mismatches:
        raise ValueError(
            "Metadata mismatch for subject "
            f"{subject_id} at {path}: "
            + "; ".join(mismatches)
        )


def validate_loaded_session(
    payload: dict[str, np.ndarray],
    *,
    subject_id: int,
    session_name: str,
    path: Path,
) -> None:
    required_keys = {
        "data",
        "labels",
        "subject_ids",
        "session_ids",
        "trial_ids",
    }

    missing = required_keys - set(payload)

    if missing:
        raise KeyError(
            f"{path}: session payload is missing "
            f"{sorted(missing)}."
        )

    data = np.asarray(payload["data"])
    labels = np.asarray(
        payload["labels"],
        dtype=np.int64,
    )

    if data.ndim != 3:
        raise ValueError(
            f"{path}: expected EEG data [N,C,T], "
            f"got {data.shape}."
        )

    if len(data) == 0:
        raise ValueError(
            f"{path}: session {session_name!r} "
            "contains no trials."
        )

    if labels.shape != (len(data),):
        raise ValueError(
            f"{path}: labels shape {labels.shape} "
            f"does not match trial count "
            f"{len(data)}."
        )

    if not np.isfinite(data).all():
        raise ValueError(
            f"{path}: data contains NaN or Inf."
        )

    subject_values = sorted(
        set(
            np.asarray(
                payload["subject_ids"],
                dtype=np.int64,
            ).tolist()
        )
    )

    if subject_values != [subject_id]:
        raise ValueError(
            f"{path}: expected only subject "
            f"{subject_id}, found "
            f"{subject_values}."
        )

    session_values = sorted(
        set(
            np.asarray(
                payload["session_ids"]
            ).astype(str).tolist()
        )
    )

    if session_values != [session_name]:
        raise ValueError(
            f"{path}: expected session "
            f"{session_name!r}, found "
            f"{session_values}."
        )

    trial_ids = np.asarray(
        payload["trial_ids"],
        dtype=np.int64,
    )

    if len(np.unique(trial_ids)) != len(
        trial_ids
    ):
        raise ValueError(
            f"{path}: duplicate trial IDs found "
            f"in session {session_name!r}."
        )


def limit_trials_per_class(
    X: np.ndarray,
    y: np.ndarray,
    *,
    maximum_per_class: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Optional balanced subset for smoke tests.

    Formal experiments should omit
    --max-trials-per-class-per-subject.
    """

    if maximum_per_class is None:
        return X, y

    if maximum_per_class <= 0:
        raise ValueError(
            "maximum_per_class must be positive."
        )

    rng = np.random.default_rng(seed)
    selected: list[int] = []

    for label in sorted(
        set(
            np.asarray(
                y,
                dtype=np.int64,
            ).tolist()
        )
    ):
        class_indices = np.flatnonzero(
            y == label
        )

        rng.shuffle(class_indices)

        selected.extend(
            class_indices[
                :maximum_per_class
            ].tolist()
        )

    selected_array = np.asarray(
        sorted(selected),
        dtype=np.int64,
    )

    return (
        np.asarray(
            X[selected_array],
            dtype=np.float32,
        ),
        np.asarray(
            y[selected_array],
            dtype=np.int64,
        ),
    )


def load_preprocessed_subject_session(
    *,
    subject_id: int,
    path: Path,
    session_name: str,
    preprocessor: EEGPreprocessor,
    reference_metadata: (
        HDF5Metadata | None
    ),
    expected_window_sec: float,
    maximum_per_class: int | None,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    HDF5Metadata,
    dict[str, Any],
]:
    dataset = EEGHDF5(path)
    metadata = dataset.metadata

    if reference_metadata is not None:
        validate_metadata_compatibility(
            reference_metadata,
            metadata,
            subject_id=subject_id,
            path=path,
        )

    payload = dataset.load(session_name)

    validate_loaded_session(
        payload,
        subject_id=subject_id,
        session_name=session_name,
        path=path,
    )

    raw_data = np.asarray(
        payload["data"],
        dtype=np.float32,
    )

    labels = np.asarray(
        payload["labels"],
        dtype=np.int64,
    )

    raw_window_sec = (
        raw_data.shape[-1]
        / float(metadata.sample_rate)
    )

    if not np.isclose(
        raw_window_sec,
        expected_window_sec,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            f"{path}: source trials are "
            f"{raw_window_sec:.6f}s, but this "
            f"experiment requires "
            f"{expected_window_sec:.6f}s direct "
            "trials. Do not silently crop or "
            "concatenate trials."
        )

    X = preprocessor.transform(
        raw_data,
        metadata.sample_rate,
        metadata.unit,
        reshape=True,
    )

    if X.ndim != 4:
        raise ValueError(
            "LaBraM preprocessing must produce "
            f"[N,C,A,200], got {X.shape}."
        )

    if X.shape[0] != len(labels):
        raise ValueError(
            "Preprocessed sample count does not "
            "match labels."
        )

    if X.shape[1] != len(
        metadata.channel_names
    ):
        raise ValueError(
            "Preprocessed channel count does not "
            "match metadata."
        )

    if (
        X.shape[-1]
        != preprocessor.config.patch_samples
    ):
        raise ValueError(
            "Unexpected LaBraM patch length: "
            f"{X.shape[-1]}."
        )

    expected_patches = int(
        round(
            expected_window_sec
            * preprocessor.config
            .target_sample_rate
            / preprocessor.config.patch_samples
        )
    )

    if X.shape[2] != expected_patches:
        raise ValueError(
            "Unexpected number of LaBraM patches: "
            f"expected={expected_patches}, "
            f"actual={X.shape[2]}."
        )

    X, labels = limit_trials_per_class(
        X,
        labels,
        maximum_per_class=(
            maximum_per_class
        ),
        seed=seed,
    )

    class_counts = {
        str(class_index): int(
            np.sum(labels == class_index)
        )
        for class_index in range(
            len(metadata.class_names)
        )
    }

    summary = {
        "subject_id": int(subject_id),
        "path": str(path),
        "session": session_name,
        "raw_shape": list(
            raw_data.shape
        ),
        "preprocessed_shape": list(
            X.shape
        ),
        "num_trials": int(len(X)),
        "class_counts": class_counts,
    }

    return (
        np.asarray(
            X,
            dtype=np.float32,
        ),
        labels,
        metadata,
        summary,
    )


def combine_subject_splits(
    arrays: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not arrays:
        raise ValueError(
            "No subject arrays were provided."
        )

    X = np.concatenate(
        arrays,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    y = np.concatenate(
        labels,
        axis=0,
    ).astype(
        np.int64,
        copy=False,
    )

    if len(X) != len(y):
        raise ValueError(
            "Combined data and labels have "
            "different lengths."
        )

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(X))

    return (
        X[permutation],
        y[permutation],
    )


def predict_from_embeddings(
    *,
    adapter: Any,
    X: np.ndarray,
    cache_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    embeddings = (
        adapter.extract_embeddings(
            X,
            cache_path,
        )
    )

    adapter.head.eval()

    with torch.inference_mode():
        features = torch.from_numpy(
            embeddings
        ).to(adapter.device)

        logits = adapter.head(features)

        probabilities = (
            torch.softmax(
                logits,
                dim=-1,
            )
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

    predictions = np.argmax(
        probabilities,
        axis=1,
    ).astype(
        np.int64,
        copy=False,
    )

    return predictions, probabilities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a frozen-LaBraM LOSO "
            "population linear head on "
            "multi-subject BNCI2014_001 data."
        )
    )

    parser.add_argument(
        "--data-root",
        default=(
            "data/processed/bnci2014_001"
        ),
    )

    parser.add_argument(
        "--data-pattern",
        default=(
            "subject_{subject:02d}.h5"
        ),
    )

    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=list(range(1, 10)),
    )

    parser.add_argument(
        "--target-subject",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--train-session",
        default="0train",
    )

    parser.add_argument(
        "--validation-session",
        default="1test",
    )

    parser.add_argument(
        "--final-test-session",
        default="1test",
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/backbones/labram/"
            "labram_base.pth"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output head path. "
            "A standard population-head path "
            "is used when omitted."
        ),
    )

    parser.add_argument(
        "--run-dir",
        default=None,
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
        "--amp",
        action="store_true",
    )

    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Smoke testing only.",
    )

    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--window-sec",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--target-sample-rate",
        type=float,
        default=200.0,
    )

    parser.add_argument(
        "--patch-samples",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--bandpass-low-hz",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--bandpass-high-hz",
        type=float,
        default=75.0,
    )

    parser.add_argument(
        "--notch-hz",
        type=float,
        default=50.0,
    )

    parser.add_argument(
        "--zscore-epsilon",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max-trials-per-class-per-subject",
        type=int,
        default=None,
        help=(
            "Smoke-test only. Formal experiments "
            "should leave this unset."
        ),
    )

    parser.add_argument(
        "--dataset-name",
        default="bnci2014_001",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.window_sec <= 0:
        raise ValueError(
            "--window-sec must be positive."
        )

    if args.target_sample_rate <= 0:
        raise ValueError(
            "--target-sample-rate must be "
            "positive."
        )

    if args.patch_samples <= 0:
        raise ValueError(
            "--patch-samples must be positive."
        )

    subjects = normalize_subjects(
        args.subjects
    )

    target_subject = int(
        args.target_subject
    )

    if target_subject not in subjects:
        raise ValueError(
            "target subject must be included "
            "in --subjects."
        )

    population_subjects = [
        subject
        for subject in subjects
        if subject != target_subject
    ]

    if not population_subjects:
        raise ValueError(
            "At least one non-target population "
            "subject is required."
        )

    seed_everything(args.seed)

    data_root = resolve_repo_path(
        args.data_root
    )

    checkpoint_path = resolve_repo_path(
        args.checkpoint
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "LaBraM backbone was not found: "
            f"{checkpoint_path}"
        )

    subject_paths = {
        subject: resolve_subject_file(
            data_root=data_root,
            data_pattern=args.data_pattern,
            subject_id=subject,
        )
        for subject in subjects
    }

    if args.output is None:
        output_path = population_head_path(
            stage="stage0",
            dataset=args.dataset_name,
            subject_id=target_subject,
            window_seconds=(
                args.window_sec
            ),
            aggregation="labram",
        )
    else:
        output_path = resolve_repo_path(
            args.output
        )

    if args.run_dir is None:
        run_dir = population_run_dir(
            stage="stage0",
            dataset=args.dataset_name,
            subject_id=target_subject,
            window_seconds=(
                args.window_sec
            ),
            aggregation="labram",
        )
    else:
        run_dir = resolve_repo_path(
            args.run_dir
        )

    if output_path.exists() and not (
        args.overwrite
    ):
        raise FileExistsError(
            f"Output already exists: "
            f"{output_path}. Pass --overwrite "
            "to replace it."
        )

    if run_dir.exists() and any(
        run_dir.iterdir()
    ):
        raise FileExistsError(
            "Run directory already exists and "
            f"is not empty: {run_dir}"
        )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_dir = (
        run_dir / "embedding_cache"
    )

    preprocessing_config = (
        PreprocessingConfig(
            bandpass_hz=(
                args.bandpass_low_hz,
                args.bandpass_high_hz,
            ),
            notch_hz=args.notch_hz,
            target_sample_rate=(
                args.target_sample_rate
            ),
            output_unit="uV",
            zscore_epsilon=(
                args.zscore_epsilon
            ),
            patch_samples=(
                args.patch_samples
            ),
        )
    )

    preprocessor = EEGPreprocessor(
        preprocessing_config
    )

    run_config   = {
        "status": "started",
        "created_at": (
            datetime.now().isoformat()
        ),
        "git_commit": (
            current_git_commit()
        ),
        "protocol": "loso_population",
        "target_subject": target_subject,
        "population_subjects": (
            population_subjects
        ),
        "sessions": {
            "population_train": (
                args.train_session
            ),
            "population_validation": (
                args.validation_session
            ),
            "final_target_test": (
                args.final_test_session
            ),
        },
        "subject_paths": {
            str(subject): str(path)
            for subject, path
            in subject_paths.items()
        },
        "backbone_checkpoint": str(
            checkpoint_path
        ),
        "backbone_sha256": sha256_file(
            checkpoint_path
        ),
        "output": str(output_path),
        "arguments": vars(args),
    }

    dump_json(
        run_config,
        run_dir / "run_config.json",
    )

    train_arrays: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    validation_arrays: list[
        np.ndarray
    ] = []
    validation_labels: list[
        np.ndarray
    ] = []

    split_summaries: dict[
        str,
        Any,
    ] = {
        "population_train": {},
        "population_validation": {},
    }

    reference_metadata: (
        HDF5Metadata | None
    ) = None

    print(
        "Loading population subjects:",
        population_subjects,
    )

    for offset, subject_id in enumerate(
        population_subjects
    ):
        subject_path = subject_paths[
            subject_id
        ]

        X_train, y_train, metadata, summary = (
            load_preprocessed_subject_session(
                subject_id=subject_id,
                path=subject_path,
                session_name=(
                    args.train_session
                ),
                preprocessor=preprocessor,
                reference_metadata=(
                    reference_metadata
                ),
                expected_window_sec=(
                    args.window_sec
                ),
                maximum_per_class=(
                    args
                    .max_trials_per_class_per_subject
                ),
                seed=(
                    args.seed
                    + offset * 100
                ),
            )
        )

        if reference_metadata is None:
            reference_metadata = metadata

        X_val, y_val, _, val_summary = (
            load_preprocessed_subject_session(
                subject_id=subject_id,
                path=subject_path,
                session_name=(
                    args.validation_session
                ),
                preprocessor=preprocessor,
                reference_metadata=(
                    reference_metadata
                ),
                expected_window_sec=(
                    args.window_sec
                ),
                maximum_per_class=(
                    args
                    .max_trials_per_class_per_subject
                ),
                seed=(
                    args.seed
                    + 10_000
                    + offset * 100
                ),
            )
        )

        train_arrays.append(X_train)
        train_labels.append(y_train)

        validation_arrays.append(X_val)
        validation_labels.append(y_val)

        split_summaries[
            "population_train"
        ][
            f"subject_{subject_id:02d}"
        ] = summary

        split_summaries[
            "population_validation"
        ][
            f"subject_{subject_id:02d}"
        ] = val_summary

    assert reference_metadata is not None

    X_train, y_train = (
        combine_subject_splits(
            train_arrays,
            train_labels,
            seed=args.seed + 20_000,
        )
    )

    X_val, y_val = (
        combine_subject_splits(
            validation_arrays,
            validation_labels,
            seed=args.seed + 30_000,
        )
    )

    class_names = tuple(
        str(name)
        for name
        in reference_metadata.class_names
    )

    n_patches = int(
        X_train.shape[2]
    )

    print(
        "Population train shape:",
        X_train.shape,
    )
    print(
        "Population validation shape:",
        X_val.shape,
    )
    print(
        "Target subject has not been "
        "loaded yet."
    )

    adapter = ModelFactory.create(
        "labram-linear",
        channel_names=list(
            reference_metadata.channel_names
        ),
        n_classes=len(class_names),
        checkpoint=checkpoint_path,
        device=args.device,
        amp=args.amp,
        freeze_encoder=True,
        embedding_batch_size=(
            args.embedding_batch_size
        ),
        random_init=args.random_init,
        n_patches=n_patches,
    )

    fit_metrics = adapter.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val,
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=(
            args.learning_rate
        ),
        weight_decay=args.weight_decay,
        patience=args.patience,
        cache_dir=cache_dir,
    )

    validation_predictions, _ = (
        predict_from_embeddings(
            adapter=adapter,
            X=X_val,
            cache_path=(
                cache_dir
                / "population_validation_eval.npz"
            ),
        )
    )

    validation_metrics = (
        classification_metrics(
            y_val,
            validation_predictions,
        )
    )

    # The target subject is loaded only
    # after training and model selection.
    print(
        "Training and model selection "
        "completed. Loading target subject "
        "for final evaluation..."
    )

    target_path = subject_paths[
        target_subject
    ]

    (
        X_target,
        y_target,
        _,
        target_summary,
    ) = load_preprocessed_subject_session(
        subject_id=target_subject,
        path=target_path,
        session_name=(
            args.final_test_session
        ),
        preprocessor=preprocessor,
        reference_metadata=(
            reference_metadata
        ),
        expected_window_sec=(
            args.window_sec
        ),
        maximum_per_class=(
            args
            .max_trials_per_class_per_subject
        ),
        seed=args.seed + 40_000,
    )

    target_predictions, _ = (
        predict_from_embeddings(
            adapter=adapter,
            X=X_target,
            cache_path=(
                cache_dir
                / "final_target_test_embeddings.npz"
            ),
        )
    )

    final_metrics = classification_metrics(
        y_target,
        target_predictions,
    )

    window_seconds = (
        n_patches
        * preprocessing_config.patch_samples
        / preprocessing_config
        .target_sample_rate
    )

    checkpoint_report = getattr(
        adapter,
        "_checkpoint_report",
        {},
    )

    head_metadata = {
        "model_type": "labram",
        "model_name": "labram-linear",
        "architecture": (
            "labram_base_patch200_200"
        ),
        "protocol": "loso_population",
        "embedding_dim": int(
            adapter.embedding_dim
        ),
        "num_classes": int(
            adapter.n_classes
        ),
        "class_names": list(
            class_names
        ),
        "channel_names": [
            str(name)
            for name
            in reference_metadata.channel_names
        ],
        "n_patches": n_patches,
        "patch_samples": int(
            preprocessing_config.patch_samples
        ),
        "target_sample_rate": float(
            preprocessing_config
            .target_sample_rate
        ),
        "window_seconds": float(
            window_seconds
        ),
        "preprocessing": (
            preprocessing_config.to_dict()
        ),
        "dataset": str(
            reference_metadata.dataset_name
        ),
        # Retained for compatibility with
        # the current exporter.
        "subject": target_subject,
        "target_subject": target_subject,
        "population_subjects": (
            population_subjects
        ),
        "train_session": (
            args.train_session
        ),
        "validation_session": (
            args.validation_session
        ),
        "test_session": (
            args.final_test_session
        ),
        "seed": args.seed,
        "backbone_checkpoint": str(
            checkpoint_path
        ),
        "backbone_sha256": sha256_file(
            checkpoint_path
        ),
        "checkpoint_report": (
            checkpoint_report
        ),
        "freeze_encoder": True,
        "trained_head": True,
        "head_type": "population",
        "is_test_head": False,
        "model_selection": fit_metrics,
        "population_validation": (
            validation_metrics
        ),
        # Retained because the current
        # exporter reads final_test.
        "final_test": final_metrics,
        "final_target_test": final_metrics,
        "split_summaries": {
            **split_summaries,
            "final_target_test": (
                target_summary
            ),
        },
    }

    checkpoint_payload = {
        "format_version": 1,
        "state_dict": (
            adapter.head.state_dict()
        ),
        "metadata": head_metadata,
    }

    atomic_torch_save(
        checkpoint_payload,
        output_path,
    )

    metrics_payload = {
        "protocol": "loso_population",
        "target_subject": target_subject,
        "population_subjects": (
            population_subjects
        ),
        "n_train": int(len(X_train)),
        "n_validation": int(len(X_val)),
        "n_final_test": int(
            len(X_target)
        ),
        "model_selection": fit_metrics,
        "population_validation": (
            validation_metrics
        ),
        "final_test": final_metrics,
        "final_target_test": final_metrics,
        "split_summaries": {
            **split_summaries,
            "final_target_test": (
                target_summary
            ),
        },
    }

    dump_json(
        metrics_payload,
        run_dir
        / "training_metrics.json",
    )

    run_config["status"] = "completed"
    run_config["completed_at"] = (
        datetime.now().isoformat()
    )
    run_config["head_checkpoint"] = str(
        output_path
    )
    run_config["metrics"] = {
        "population_validation": (
            validation_metrics
        ),
        "final_target_test": (
            final_metrics
        ),
    }

    dump_json(
        run_config,
        run_dir / "run_config.json",
    )

    print()
    print(
        "LaBraM population-head "
        "training completed."
    )
    print("Head:", output_path)
    print("Run directory:", run_dir)
    print(
        "Population validation:",
        validation_metrics,
    )
    print(
        "Final held-out target test:",
        final_metrics,
    )


if __name__ == "__main__":
    main()