from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _bootstrap import ROOT  # noqa: F401

from bci_dayloop.data.hdf5_dataset import EEGHDF5
from bci_dayloop.data.preprocessing import (
    EEGPreprocessor,
    PreprocessingConfig,
)
from bci_dayloop.models.labram_backend import (
    LaBraMBackend,
)
from bci_dayloop.models.labram_linear import (
    LaBraMLinearAdapter,
)
from bci_dayloop.packages.loader import (
    load_runtime_package,
)
from bci_dayloop.preprocessing.canonical import (
    SignalCanonicalizer,
)
from bci_dayloop.preprocessing.labram import (
    LaBraMInputTransform,
)
from bci_dayloop.runtime.model import RuntimeModel
from bci_dayloop.runtime.types import RawEEGWindow


def resolve_path(value: str | Path) -> Path:
    """将相对路径解析为项目根目录下的绝对路径。"""

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare legacy LaBraM preprocessing/inference "
            "with the unified RuntimeModel path, and optionally "
            "with a schema-v2 Runtime Model Package."
        )
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Processed EEG HDF5 path.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="LaBraM backbone checkpoint.",
    )
    parser.add_argument(
        "--classifier",
        required=True,
        help=(
            "LaBraM classification head. It may be a direct "
            "head.pt/classifier.pt file or a directory containing one."
        ),
    )
    parser.add_argument(
        "--model-package",
        help=(
            "Optional schema-v2 LaBraM Runtime Model Package. "
            "When provided, compare package-loaded inference too."
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
        "--start-sec",
        type=float,
        default=0.0,
        help=(
            "Start time inside the selected trial. "
            "Useful when the HDF5 trial is longer than the model window."
        ),
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA AMP when running on CUDA.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--n-patches",
        type=int,
        default=4,
        help="Number of one-second LaBraM patches.",
    )
    parser.add_argument(
        "--patch-samples",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--target-sample-rate",
        type=float,
        default=200.0,
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
        "--channels-json",
        help=(
            "Optional JSON file containing the required LaBraM "
            "channel order. When omitted, use the HDF5 channel order."
        ),
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--embedding-rtol",
        type=float,
        default=1e-5,
        help=(
            "Relative tolerance for end-to-end embedding "
            "comparison. This comparison includes tiny "
            "preprocessing differences."
        ),
    )

    parser.add_argument(
        "--embedding-atol",
        type=float,
        default=1e-5,
        help=(
            "Absolute tolerance for end-to-end embedding "
            "comparison."
        ),
    )

    parser.add_argument(
        "--backend-rtol",
        type=float,
        default=1e-6,
        help=(
            "Relative tolerance when legacy and Backend "
            "receive exactly the same model input."
        ),
    )

    parser.add_argument(
        "--backend-atol",
        type=float,
        default=1e-7,
        help=(
            "Absolute tolerance when legacy and Backend "
            "receive exactly the same model input."
        ),
    )

    return parser.parse_args()


def load_target_channels(
    *,
    channels_json: str | None,
    fallback: list[str],
) -> tuple[str, ...]:
    if channels_json is None:
        channels = tuple(
            str(name)
            for name in fallback
        )
    else:
        path = resolve_path(channels_json)

        if not path.is_file():
            raise FileNotFoundError(
                f"Channel-order JSON was not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            payload = json.load(handle)

        if isinstance(payload, dict):
            # 同时兼容：
            # {"channels": [...]}
            # {"channel_names": [...]}
            payload = payload.get(
                "channels",
                payload.get("channel_names"),
            )

        if not isinstance(payload, list):
            raise ValueError(
                "Channel-order JSON must contain a list, "
                "or a mapping with 'channels'/'channel_names'."
            )

        channels = tuple(
            str(name)
            for name in payload
        )

    if not channels:
        raise ValueError(
            "LaBraM channel list cannot be empty."
        )

    normalized = [
        name.strip().upper()
        for name in channels
    ]

    if len(normalized) != len(set(normalized)):
        raise ValueError(
            "LaBraM channel list contains duplicates."
        )

    return channels


def build_preprocessing_config(
    args: argparse.Namespace,
) -> PreprocessingConfig:
    return PreprocessingConfig(
        bandpass_hz=(
            float(args.bandpass_low_hz),
            float(args.bandpass_high_hz),
        ),
        notch_hz=float(args.notch_hz),
        target_sample_rate=float(
            args.target_sample_rate
        ),
        output_unit="uV",
        zscore_epsilon=float(
            args.zscore_epsilon
        ),
        patch_samples=int(
            args.patch_samples
        ),
    )


def resolve_classifier_file(
    value: str | Path,
) -> Path:
    path = resolve_path(value)

    if path.is_file():
        return path

    if not path.is_dir():
        raise FileNotFoundError(
            f"LaBraM classifier path was not found: {path}"
        )

    candidates = (
        path / "classifier.pt",
        path / "head.pt",
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Classifier directory contains neither "
        f"classifier.pt nor head.pt: {path}"
    )


def load_torch_payload(path: Path) -> Any:
    """
    兼容新旧 PyTorch。

    部分较旧版本的 torch.load 不支持 weights_only 参数。
    """

    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def load_classifier_head(
    *,
    adapter: LaBraMLinearAdapter,
    classifier_path: Path,
) -> None:
    payload = load_torch_payload(
        classifier_path
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "Unsupported LaBraM classifier checkpoint: "
            f"{classifier_path}"
        )

    # 新旧文件通常均为：
    # {
    #     "state_dict": ...,
    #     "embedding_dim": ...,
    #     "n_classes": ...,
    # }
    if "state_dict" in payload:
        state_dict = payload["state_dict"]

        if "embedding_dim" in payload:
            classifier_embedding_dim = int(
                payload["embedding_dim"]
            )

            if (
                classifier_embedding_dim
                != adapter.embedding_dim
            ):
                raise ValueError(
                    "Classifier embedding_dim does not "
                    "match LaBraM backbone: "
                    f"classifier={classifier_embedding_dim}, "
                    f"backbone={adapter.embedding_dim}."
                )

        if "n_classes" in payload:
            classifier_classes = int(
                payload["n_classes"]
            )

            if classifier_classes != adapter.n_classes:
                raise ValueError(
                    "Classifier class count does not match "
                    "the requested adapter: "
                    f"classifier={classifier_classes}, "
                    f"adapter={adapter.n_classes}."
                )
    else:
        # 兼容直接保存的 Linear state_dict。
        state_dict = payload

    if not isinstance(state_dict, dict):
        raise ValueError(
            "LaBraM classifier state_dict is invalid."
        )

    adapter.head.load_state_dict(
        state_dict,
        strict=True,
    )
    adapter.head.eval()


def build_adapter(
    *,
    args: argparse.Namespace,
    channel_names: tuple[str, ...],
    num_classes: int,
    classifier_path: Path,
) -> LaBraMLinearAdapter:
    adapter = LaBraMLinearAdapter(
        channel_names=list(channel_names),
        n_classes=int(num_classes),
        checkpoint=resolve_path(
            args.checkpoint
        ),
        device=str(args.device),
        amp=bool(args.amp),
        freeze_encoder=True,
        embedding_batch_size=int(
            args.embedding_batch_size
        ),
        random_init=False,
        n_patches=int(args.n_patches),
    )

    load_classifier_head(
        adapter=adapter,
        classifier_path=classifier_path,
    )

    return adapter


def align_channels(
    *,
    data: np.ndarray,
    source_names: list[str],
    target_names: tuple[str, ...],
) -> np.ndarray:
    """
    将旧链路输入重排为与新 LaBraMInputTransform 相同的通道顺序。

    这里不补缺失通道；缺失时直接报错。
    """

    if data.ndim != 2:
        raise ValueError(
            f"Expected raw EEG [C,T], got {data.shape}."
        )

    if data.shape[0] != len(source_names):
        raise ValueError(
            "Raw EEG channel count does not match "
            "source channel names."
        )

    source_index: dict[str, int] = {}

    for index, name in enumerate(source_names):
        key = str(name).strip().upper()

        if key in source_index:
            raise ValueError(
                "Duplicate source channel after "
                f"case normalization: {name!r}."
            )

        source_index[key] = index

    missing = [
        name
        for name in target_names
        if name.strip().upper()
        not in source_index
    ]

    if missing:
        raise ValueError(
            "Raw EEG is missing required LaBraM channels: "
            f"{missing}."
        )

    indexes = [
        source_index[
            name.strip().upper()
        ]
        for name in target_names
    ]

    return np.asarray(
        data[indexes, :],
        dtype=np.float32,
    )


def extract_raw_window(
    *,
    trial: np.ndarray,
    source_sample_rate: float,
    window_sec: float,
    start_sec: float,
) -> np.ndarray:
    if trial.ndim != 2:
        raise ValueError(
            f"Expected trial [C,T], got {trial.shape}."
        )

    if source_sample_rate <= 0:
        raise ValueError(
            "source_sample_rate must be positive."
        )

    if window_sec <= 0:
        raise ValueError(
            "window_sec must be positive."
        )

    if start_sec < 0:
        raise ValueError(
            "start_sec cannot be negative."
        )

    start_point = int(
        round(
            start_sec
            * source_sample_rate
        )
    )

    required_points = int(
        round(
            window_sec
            * source_sample_rate
        )
    )

    stop_point = (
        start_point
        + required_points
    )

    if stop_point > trial.shape[-1]:
        available_sec = (
            trial.shape[-1]
            / source_sample_rate
        )

        raise ValueError(
            "Selected trial is too short for the "
            "requested LaBraM window: "
            f"trial_duration={available_sec:.3f}s, "
            f"start_sec={start_sec:.3f}, "
            f"window_sec={window_sec:.3f}."
        )

    return np.asarray(
        trial[:, start_point:stop_point],
        dtype=np.float32,
    )


def tensor_to_numpy(
    value: torch.Tensor,
) -> np.ndarray:
    return (
        value
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )


def maximum_absolute_difference(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    if left.shape != right.shape:
        return float("inf")

    return float(
        np.max(
            np.abs(left - right)
        )
    )


def assert_close(
    *,
    name: str,
    old: np.ndarray,
    new: np.ndarray,
    rtol: float,
    atol: float,
) -> None:
    if old.shape != new.shape:
        raise AssertionError(
            f"{name} shape mismatch: "
            f"old={old.shape}, new={new.shape}."
        )

    np.testing.assert_allclose(
        old,
        new,
        rtol=rtol,
        atol=atol,
        err_msg=f"{name} values are inconsistent.",
    )


def release_model_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "empty_cache")
    ):
        try:
            torch.mps.empty_cache()
        except RuntimeError:
            pass


def main() -> None:
    args = parse_args()

    data_path = resolve_path(args.data)
    checkpoint_path = resolve_path(
        args.checkpoint
    )
    classifier_path = resolve_classifier_file(
        args.classifier
    )

    for logical_name, path in (
        ("HDF5 data", data_path),
        ("LaBraM backbone", checkpoint_path),
        ("LaBraM classifier", classifier_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"{logical_name} was not found: {path}"
            )

    if args.n_patches <= 0:
        raise ValueError(
            "n-patches must be positive."
        )

    if args.patch_samples <= 0:
        raise ValueError(
            "patch-samples must be positive."
        )

    if args.target_sample_rate <= 0:
        raise ValueError(
            "target-sample-rate must be positive."
        )

    expected_patch_samples = int(
        round(
            args.target_sample_rate
        )
    )

    if args.patch_samples != expected_patch_samples:
        raise ValueError(
            "This repository's LaBraM backbone expects "
            "one-second, 200-point patches. "
            f"Got patch_samples={args.patch_samples}, "
            f"target_sample_rate="
            f"{args.target_sample_rate}."
        )

    window_sec = (
        args.n_patches
        * args.patch_samples
        / args.target_sample_rate
    )

    dataset = EEGHDF5(data_path)
    metadata = dataset.metadata

    session_data = dataset.load(
        args.session
    )

    trials = session_data["data"]
    labels = session_data["labels"]
    trial_ids = session_data["trial_ids"]

    if not 0 <= args.trial_index < len(trials):
        raise IndexError(
            f"trial-index={args.trial_index} is outside "
            f"[0, {len(trials) - 1}]."
        )

    class_names = tuple(
        str(name)
        for name in metadata.class_names
    )

    if not class_names:
        raise ValueError(
            "HDF5 metadata has no class_names."
        )

    target_channels = load_target_channels(
        channels_json=args.channels_json,
        fallback=metadata.channel_names,
    )

    full_trial = np.asarray(
        trials[args.trial_index],
        dtype=np.float32,
    )

    raw_window = extract_raw_window(
        trial=full_trial,
        source_sample_rate=float(
            metadata.sample_rate
        ),
        window_sec=float(window_sec),
        start_sec=float(args.start_sec),
    )

    preprocessing_config = (
        build_preprocessing_config(args)
    )

    print("=" * 76)
    print("LaBraM Runtime Regression Input")
    print("=" * 76)
    print(f"Data:                  {data_path}")
    print(f"Session:               {args.session}")
    print(f"Trial index:           {args.trial_index}")
    print(
        f"Trial ID:              "
        f"{int(trial_ids[args.trial_index])}"
    )
    print(
        f"True label:            "
        f"{int(labels[args.trial_index])}"
    )
    print(f"Source shape:          {full_trial.shape}")
    print(f"Selected raw shape:    {raw_window.shape}")
    print(
        f"Source sample rate:    "
        f"{metadata.sample_rate}"
    )
    print(f"Source unit:           {metadata.unit}")
    print(f"Target channels:       {len(target_channels)}")
    print(f"Target sample rate:    {args.target_sample_rate}")
    print(f"Patch samples:         {args.patch_samples}")
    print(f"Number of patches:     {args.n_patches}")
    print(f"Window seconds:        {window_sec}")
    print(f"Device:                {args.device}")

    # ==========================================================
    # 1. 旧链路
    #
    # Raw EEG
    #   -> 通道对齐
    #   -> EEGPreprocessor
    #   -> LaBraMLinearAdapter.extract_embeddings()
    #   -> Linear head
    #   -> softmax
    # ==========================================================

    print("\nRunning legacy LaBraM path...")

    old_aligned_window = align_channels(
        data=raw_window,
        source_names=metadata.channel_names,
        target_names=target_channels,
    )

    old_preprocessor = EEGPreprocessor(
        preprocessing_config
    )

    old_unbatched_signal = (
        old_preprocessor.transform(
            old_aligned_window,
            sample_rate=float(
                metadata.sample_rate
            ),
            input_unit=str(metadata.unit),
            reshape=True,
        )
    )

    expected_unbatched_shape = (
        len(target_channels),
        int(args.n_patches),
        int(args.patch_samples),
    )

    if (
        old_unbatched_signal.shape
        != expected_unbatched_shape
    ):
        raise RuntimeError(
            "Legacy preprocessing produced an unexpected "
            "LaBraM shape: "
            f"expected={expected_unbatched_shape}, "
            f"actual={old_unbatched_signal.shape}."
        )

    old_signal = (
        old_unbatched_signal[None, ...]
        .astype(
            np.float32,
            copy=False,
        )
    )

    old_adapter = build_adapter(
        args=args,
        channel_names=target_channels,
        num_classes=len(class_names),
        classifier_path=classifier_path,
    )

    checkpoint_report = getattr(
        old_adapter,
        "_checkpoint_report",
        {},
    )

    print("\nLaBraM checkpoint loading report:")
    print(
        f"Loaded tensors: "
        f"{checkpoint_report.get('loaded_tensors')}"
    )
    print(
        f"Model tensors:  "
        f"{checkpoint_report.get('model_tensors')}"
    )
    print(
        f"Missing keys:   "
        f"{len(checkpoint_report.get('missing_keys', []))}"
    )
    print(
        f"Skipped keys:   "
        f"{len(checkpoint_report.get('skipped_keys', []))}"
    )

    missing_keys = checkpoint_report.get(
        "missing_keys",
        [],
    )

    if missing_keys:
        print(
            "First missing keys:",
            missing_keys[:10],
        )

    # 只执行一次 Encoder，随后直接复现 predict_proba
    # 中的 Linear + softmax。
    old_features = (
        old_adapter.extract_embeddings(
            old_signal
        )
    )

    with torch.inference_mode():
        old_feature_tensor = (
            torch.from_numpy(
                old_features
            )
            .to(old_adapter.device)
        )

        old_logits_tensor = (
            old_adapter.head(
                old_feature_tensor
            )
        )

        old_probabilities = (
            torch.softmax(
                old_logits_tensor,
                dim=-1,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

    old_prediction = int(
        np.argmax(
            old_probabilities[0]
        )
    )

    old_confidence = float(
        old_probabilities[
            0,
            old_prediction,
        ]
    )

    old_features = old_features.copy()
    old_probabilities = (
        old_probabilities.copy()
    )

    del old_logits_tensor
    del old_feature_tensor
    del old_preprocessor

    # 不要删除 old_adapter。
    # 接下来新 RuntimeModel 复用同一个 Encoder 和分类头，
    # 从而只比较调用链，不比较两次模型初始化。
    shared_adapter = old_adapter

    # ==========================================================
    # 2. 新统一 RuntimeModel 链路
    #
    # RawEEGWindow
    #   -> SignalCanonicalizer
    #   -> LaBraMInputTransform
    #   -> LaBraMBackend
    #   -> ModelOutput
    # ==========================================================

    print(
        "Running unified LaBraM RuntimeModel "
        "path with the same adapter..."
    )

    direct_runtime = RuntimeModel(
        canonicalizer=SignalCanonicalizer(
            target_unit="uV"
        ),
        input_transform=LaBraMInputTransform(
            channel_names=target_channels,
            preprocessing_config=(
                preprocessing_config
            ),
            n_patches=int(args.n_patches),
            strict_window_duration=True,
        ),
        backend=LaBraMBackend(
            shared_adapter
        ),
    )

    runtime_window = RawEEGWindow(
        data=raw_window,
        channel_names=[
            str(name)
            for name in metadata.channel_names
        ],
        sample_rate=float(
            metadata.sample_rate
        ),
        unit=str(metadata.unit),
        layout="CT",
        start_time_sec=float(
            args.start_sec
        ),
        trial_id=str(
            int(
                trial_ids[
                    args.trial_index
                ]
            )
        ),
        window_id=(
            f"{args.session}:"
            f"{args.trial_index}:"
            f"{args.start_sec:.3f}"
        ),
        label=int(
            labels[
                args.trial_index
            ]
        ),
        metadata={
            "dataset": metadata.dataset_name,
            "session": args.session,
            "source": (
                "compare_old_new_labram_runtime"
            ),
        },
    )

    prepared = direct_runtime.prepare(
        runtime_window
    )

    new_signal_tensor = prepared.get_tensor(
        "signal"
    )

    new_signal = tensor_to_numpy(
        new_signal_tensor
    )

    # ==========================================================
    # Backend 等价性测试
    #
    # 必须让旧 Adapter 和新 Backend 接收完全相同的输入。
    # 这里统一使用 new_signal。
    # ==========================================================

    legacy_features_on_new_signal = (
        shared_adapter.extract_embeddings(
            new_signal
        )
    )

    backend_features_tensor = (
        direct_runtime.backend.encode_tensor(
            prepared.model_input
        )
    )

    backend_features = tensor_to_numpy(
        backend_features_tensor
    )

    same_input_backend_max_diff = (
        maximum_absolute_difference(
            legacy_features_on_new_signal,
            backend_features,
        )
    )

    print(
        "Same-input legacy/backend embedding "
        "max abs diff: "
        f"{same_input_backend_max_diff:.12g}"
    )

    assert_close(
        name=(
            "Same-input legacy/backend "
            "LaBraM embeddings"
        ),
        old=legacy_features_on_new_signal,
        new=backend_features,
        rtol=args.backend_rtol,
        atol=args.backend_atol,
    )

    new_output = (
        direct_runtime.predict_prepared(
            prepared,
            return_features=True,
        )
    )

    new_probabilities = tensor_to_numpy(
        new_output.probabilities
    )

    if new_output.features is None:
        raise RuntimeError(
            "LaBraMBackend did not return features "
            "although return_features=True."
        )

    new_features = tensor_to_numpy(
        new_output.features
    )

    assert_close(
        name=(
            "LaBraM encode_tensor/predict_tensor "
            "features"
        ),
        old=backend_features,
        new=new_features,
        rtol=0.0,
        atol=0.0,
    )

    new_prediction = int(
        new_output.predicted_class
    )

    new_confidence = float(
        new_output.confidence
    )

    # 保留结果，随后释放直接构建的模型。
    direct_probabilities = (
        new_probabilities.copy()
    )
    direct_features = new_features.copy()
    direct_prediction = new_prediction
    direct_confidence = new_confidence

    # ==========================================================
    # 3. 比较旧链路和新 RuntimeModel
    # ==========================================================

    signal_max_diff = (
        maximum_absolute_difference(
            old_signal,
            new_signal,
        )
    )

    feature_max_diff = (
        maximum_absolute_difference(
            old_features,
            new_features,
        )
    )

    legacy_features_on_new_signal_max_diff = (
        maximum_absolute_difference(
            old_features,
            legacy_features_on_new_signal,
        )
    )

    print(
        "Embedding diff caused by preprocessing: "
        f"{legacy_features_on_new_signal_max_diff:.12g}"
    )

    probability_max_diff = (
        maximum_absolute_difference(
            old_probabilities,
            new_probabilities,
        )
    )

    print("\n" + "=" * 76)
    print("Legacy LaBraM vs Unified RuntimeModel")
    print("=" * 76)

    print(
        "Preprocessed signal shapes: "
        f"old={old_signal.shape}, "
        f"new={new_signal.shape}"
    )
    print(
        "Feature shapes:             "
        f"old={old_features.shape}, "
        f"new={new_features.shape}"
    )
    print(
        "Probability shapes:         "
        f"old={old_probabilities.shape}, "
        f"new={new_probabilities.shape}"
    )

    print(
        "Signal max abs diff:        "
        f"{signal_max_diff:.10g}"
    )
    print(
        "Feature max abs diff:       "
        f"{feature_max_diff:.10g}"
    )
    print(
        "Probability max abs diff:   "
        f"{probability_max_diff:.10g}"
    )

    print(
        "Old prediction/confidence:  "
        f"{old_prediction} / "
        f"{old_confidence:.10f}"
    )
    print(
        "New prediction/confidence:  "
        f"{new_prediction} / "
        f"{new_confidence:.10f}"
    )

    print(
        "Old probabilities:         "
        f"{old_probabilities[0].tolist()}"
    )
    print(
        "New probabilities:         "
        f"{new_probabilities[0].tolist()}"
    )

    print(
        "New preprocessing trace:   "
        f"{prepared.preprocessing_trace}"
    )
    print(
        "New preprocessing details: "
        f"{prepared.diagnostics}"
    )

    assert_close(
        name="Preprocessed LaBraM signal",
        old=old_signal,
        new=new_signal,
        rtol=args.rtol,
        atol=args.atol,
    )

    assert_close(
        name=(
            "End-to-end LaBraM embeddings "
            "including preprocessing"
        ),
        old=old_features,
        new=new_features,
        rtol=args.embedding_rtol,
        atol=args.embedding_atol,
    )

    assert_close(
        name="LaBraM probabilities",
        old=old_probabilities,
        new=new_probabilities,
        rtol=args.rtol,
        atol=args.atol,
    )

    if old_prediction != new_prediction:
        raise AssertionError(
            "Legacy and RuntimeModel predicted classes "
            "are inconsistent: "
            f"old={old_prediction}, "
            f"new={new_prediction}."
        )

    if not np.isclose(
        old_confidence,
        new_confidence,
        rtol=args.rtol,
        atol=args.atol,
    ):
        raise AssertionError(
            "Legacy and RuntimeModel confidence values "
            "are inconsistent: "
            f"old={old_confidence}, "
            f"new={new_confidence}."
        )

    print(
        "\nPASS: legacy and unified LaBraM "
        "RuntimeModel predictions are consistent."
    )

    del new_output
    del prepared
    del direct_runtime
    del old_adapter

    release_model_memory()

    # ==========================================================
    # 4. 可选：直接 Runtime vs Package Runtime
    # ==========================================================

    if args.model_package is not None:
        package_path = resolve_path(
            args.model_package
        )

        if not package_path.is_dir():
            raise FileNotFoundError(
                "LaBraM Runtime Model Package was not found: "
                f"{package_path}"
            )

        print(
            "\nRunning package-loaded LaBraM RuntimeModel..."
        )
        print(f"Package: {package_path}")

        loaded_package = load_runtime_package(
            package_path,
            device=str(args.device),
            verify_hashes=True,
        )

        if loaded_package.model_type != "labram":
            raise ValueError(
                "Expected a LaBraM Runtime Package, got "
                f"{loaded_package.model_type!r}."
            )

        if tuple(
            loaded_package.class_names
        ) != class_names:
            raise ValueError(
                "Package class order does not match "
                "the HDF5 dataset: "
                f"package={loaded_package.class_names}, "
                f"dataset={class_names}."
            )

        package_contract = (
            loaded_package
            .runtime_model
            .input_contract
        )

        if (
            package_contract.channel_names
            != target_channels
        ):
            raise ValueError(
                "Package channel order does not match "
                "the direct Runtime setting."
            )

        if not np.isclose(
            package_contract.window_sec,
            window_sec,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                "Package window_sec does not match "
                "the direct Runtime setting: "
                f"package="
                f"{package_contract.window_sec}, "
                f"direct={window_sec}."
            )

        if not np.isclose(
            package_contract.sample_rate,
            args.target_sample_rate,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                "Package target sample rate does not "
                "match the direct Runtime setting."
            )

        package_output = (
            loaded_package
            .runtime_model
            .predict(
                runtime_window,
                return_features=True,
            )
        )

        package_probabilities = tensor_to_numpy(
            package_output.probabilities
        )

        if package_output.features is None:
            raise RuntimeError(
                "Package-loaded LaBraM backend did not "
                "return features."
            )

        package_features = tensor_to_numpy(
            package_output.features
        )

        package_prediction = int(
            package_output.predicted_class
        )

        package_confidence = float(
            package_output.confidence
        )

        print("\n" + "=" * 76)
        print("Direct RuntimeModel vs Package RuntimeModel")
        print("=" * 76)

        print(
            "Feature max abs diff:       "
            f"{maximum_absolute_difference(direct_features,package_features):.10g}"
        )
        print(
            "Probability max abs diff:   "
            f"{maximum_absolute_difference(direct_probabilities,package_probabilities):.10g}"
        )
        print(
            "Direct prediction/confidence: "
            f"{direct_prediction} / "
            f"{direct_confidence:.10f}"
        )
        print(
            "Package prediction/confidence:"
            f" {package_prediction} / "
            f"{package_confidence:.10f}"
        )

        assert_close(
            name="Direct/package LaBraM embeddings",
            old=direct_features,
            new=package_features,
            rtol=args.rtol,
            atol=args.atol,
        )

        assert_close(
            name="Direct/package LaBraM probabilities",
            old=direct_probabilities,
            new=package_probabilities,
            rtol=args.rtol,
            atol=args.atol,
        )

        if direct_prediction != package_prediction:
            raise AssertionError(
                "Direct and packaged LaBraM predictions "
                "are inconsistent: "
                f"direct={direct_prediction}, "
                f"package={package_prediction}."
            )

        if not np.isclose(
            direct_confidence,
            package_confidence,
            rtol=args.rtol,
            atol=args.atol,
        ):
            raise AssertionError(
                "Direct and packaged LaBraM confidence "
                "values are inconsistent."
            )

        print(
            "\nPASS: direct and package-loaded LaBraM "
            "RuntimeModel predictions are consistent."
        )

        del package_output
        del loaded_package

        release_model_memory()


if __name__ == "__main__":
    main()