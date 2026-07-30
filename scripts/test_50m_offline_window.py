from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# 与仓库其他 scripts 一样，将项目根目录和 src 加入 Python 路径。
from _bootstrap import ROOT  # noqa: F401

from bci_dayloop.models.base import add_batch_dimension
from bci_dayloop.data.hdf5_dataset import EEGHDF5
from bci_dayloop.models.base import add_batch_dimension
from bci_dayloop.models.model_50m.adapter import Model50MAdapter
from bci_dayloop.data.hdf5_dataset import EEGHDF5
from bci_dayloop.models.model_50m.adapter import Model50MAdapter
from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.models.model_50m.pipeline_preprocessor import (
    Model50MPipelinePreprocessor,
)


# ======================================================================
# 路径和 JSON 辅助函数
# ======================================================================


def resolve_repo_path(value: str | Path) -> Path:
    """
    将相对路径解释为相对于项目根目录的路径。

    例如：
        data/processed/example.h5
    会解析为：
        <NCC-OI-BCI>/data/processed/example.h5
    """
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def json_default(value: Any) -> Any:
    """让 json.dumps 支持 Path 和 NumPy 数据类型。"""
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, tuple):
        return list(value)

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable."
    )


# ======================================================================
# HDF5 trial → 连续 EEG 流
# ======================================================================


def build_continuous_stream(
    trials: np.ndarray,
    labels: np.ndarray,
    trial_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    按照 ReplayAcquirer 的方式，将 trial 数据拼成连续 EEG 流。

    Args:
        trials:
            [N, C, T]

        labels:
            [N]

        trial_ids:
            [N]

    Returns:
        stream:
            [C, N*T]

        sample_labels:
            [N*T]，每个采样点对应的类别标签。

        sample_trial_ids:
            [N*T]，每个采样点对应的 trial ID。

    注意：
        trial 是首尾直接拼接的，中间不会自动加入休息段。
    """
    trials = np.asarray(trials)
    labels = np.asarray(labels)
    trial_ids = np.asarray(trial_ids)

    if trials.ndim != 3:
        raise ValueError(
            f"trials must have shape [N,C,T], got {trials.shape}."
        )

    num_trials = trials.shape[0]

    if labels.shape != (num_trials,):
        raise ValueError(
            "labels shape does not match trials: "
            f"expected {(num_trials,)}, got {labels.shape}."
        )

    if trial_ids.shape != (num_trials,):
        raise ValueError(
            "trial_ids shape does not match trials: "
            f"expected {(num_trials,)}, got {trial_ids.shape}."
        )

    if not np.isfinite(trials).all():
        raise ValueError("HDF5 EEG trials contain NaN or Inf.")

    samples_per_trial = int(trials.shape[-1])

    # 与 ReplayAcquirer 中的实现保持一致：
    # [N,C,T] → [C,N,T] → [C,N*T]
    stream = (
        trials
        .transpose(1, 0, 2)
        .reshape(trials.shape[1], -1)
        .astype(np.float32, copy=False)
    )

    sample_labels = np.repeat(
        labels.astype(np.int64, copy=False),
        samples_per_trial,
    )

    sample_trial_ids = np.repeat(
        trial_ids.astype(np.int64, copy=False),
        samples_per_trial,
    )

    expected_samples = num_trials * samples_per_trial

    if stream.shape[1] != expected_samples:
        raise RuntimeError(
            "Continuous stream sample count is incorrect: "
            f"expected {expected_samples}, got {stream.shape[1]}."
        )

    return stream, sample_labels, sample_trial_ids


def extract_window(
    stream: np.ndarray,
    sample_labels: np.ndarray,
    sample_trial_ids: np.ndarray,
    *,
    sample_rate: float,
    start_sec: float,
    window_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    从连续流中截取一个真实长度的 EEG 窗口。

    不做补零。数据不够时直接报错。
    """
    if sample_rate <= 0:
        raise ValueError(
            f"sample_rate must be positive, got {sample_rate}."
        )

    if start_sec < 0:
        raise ValueError(
            f"start_sec cannot be negative, got {start_sec}."
        )

    if window_sec <= 0:
        raise ValueError(
            f"window_sec must be positive, got {window_sec}."
        )

    start_sample = int(round(start_sec * sample_rate))
    window_samples = int(round(window_sec * sample_rate))
    end_sample = start_sample + window_samples

    if start_sample >= stream.shape[1]:
        total_duration = stream.shape[1] / sample_rate

        raise ValueError(
            f"start_sec={start_sec} exceeds the stream duration "
            f"{total_duration:.3f}s."
        )

    if end_sample > stream.shape[1]:
        available_samples = stream.shape[1] - start_sample
        available_seconds = available_samples / sample_rate

        raise ValueError(
            "The selected stream does not contain a complete "
            f"{window_sec:.3f}s window. "
            f"Only {available_seconds:.3f}s remain after "
            f"start_sec={start_sec:.3f}. "
            "This script does not pad short windows."
        )

    raw_window = stream[:, start_sample:end_sample].copy()
    window_labels = sample_labels[start_sample:end_sample].copy()
    window_trial_ids = sample_trial_ids[start_sample:end_sample].copy()

    expected_shape = (
        stream.shape[0],
        window_samples,
    )

    if raw_window.shape != expected_shape:
        raise RuntimeError(
            "Unexpected raw window shape: "
            f"expected {expected_shape}, got {raw_window.shape}."
        )

    return (
        raw_window.astype(np.float32, copy=False),
        window_labels.astype(np.int64, copy=False),
        window_trial_ids.astype(np.int64, copy=False),
        start_sample,
        end_sample,
    )


# ======================================================================
# 窗口标签和 trial 信息
# ======================================================================


def get_class_name(
    label: int,
    class_names: Sequence[str],
) -> str:
    if 0 <= label < len(class_names):
        return str(class_names[label])

    return f"unknown_label_{label}"


def summarize_label_composition(
    window_labels: np.ndarray,
    *,
    class_names: Sequence[str],
    sample_rate: float,
) -> list[dict[str, Any]]:
    """统计 10 秒窗口中各标签占据的时间。"""
    unique_labels, counts = np.unique(
        window_labels,
        return_counts=True,
    )

    result: list[dict[str, Any]] = []

    for label, count in zip(unique_labels, counts):
        label_int = int(label)
        count_int = int(count)

        result.append(
            {
                "label": label_int,
                "class_name": get_class_name(
                    label_int,
                    class_names,
                ),
                "samples": count_int,
                "duration_sec": count_int / sample_rate,
                "ratio": count_int / len(window_labels),
            }
        )

    return result


def summarize_segments(
    window_labels: np.ndarray,
    window_trial_ids: np.ndarray,
    *,
    class_names: Sequence[str],
    sample_rate: float,
) -> list[dict[str, Any]]:
    """
    将窗口按连续的 trial ID 和标签切分，便于查看是否跨 trial。
    """
    if len(window_labels) != len(window_trial_ids):
        raise ValueError(
            "window_labels and window_trial_ids must have "
            "the same length."
        )

    if len(window_labels) == 0:
        return []

    segments: list[dict[str, Any]] = []
    segment_start = 0

    for index in range(1, len(window_labels) + 1):
        reached_end = index == len(window_labels)

        changed = (
            not reached_end
            and (
                window_labels[index]
                != window_labels[segment_start]
                or window_trial_ids[index]
                != window_trial_ids[segment_start]
            )
        )

        if not reached_end and not changed:
            continue

        label = int(window_labels[segment_start])
        trial_id = int(window_trial_ids[segment_start])
        length = index - segment_start

        segments.append(
            {
                "trial_id": trial_id,
                "label": label,
                "class_name": get_class_name(
                    label,
                    class_names,
                ),
                "start_sec_in_window": (
                    segment_start / sample_rate
                ),
                "end_sec_in_window": index / sample_rate,
                "duration_sec": length / sample_rate,
                "samples": length,
            }
        )

        segment_start = index

    return segments


# ======================================================================
# 参数
# ======================================================================


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test one real 10-second BNCI EEG window through "
            "the 50M preprocessor, backbone and test classifier."
        )
    )

    parser.add_argument(
        "--data",
        default="data/processed/bnci2014_001_s01.h5",
        help="Input HDF5 path.",
    )

    parser.add_argument(
        "--session",
        default="1test",
        help="HDF5 session to use, for example 1test.",
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/50m/model_deploy.pt",
        help="Dependency-free 50M backbone checkpoint.",
    )

    parser.add_argument(
        "--classifier",
        default="checkpoints/50m/test_linear_head.pt",
        help="50M linear classifier checkpoint.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "mps", "auto"),
        help="Inference device.",
    )

    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="Window start position in the concatenated session.",
    )

    parser.add_argument(
        "--window-sec",
        type=float,
        default=10.0,
        help=(
            "Window duration. During the current stage this must "
            "remain 10 seconds."
        ),
    )

    parser.add_argument(
        "--filter-low-hz",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--filter-high-hz",
        type=float,
        default=75.0,
    )

    parser.add_argument(
        "--reference-mode",
        choices=("none", "average"),
        default="none",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Number of repeated predictions used to test "
            "deterministic output."
        ),
    )

    parser.add_argument(
        "--strict-head-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Verify that the classifier checkpoint metadata matches "
            "the current 50M configuration."
        ),
    )

    parser.add_argument(
        "--check-raw-interface",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also call adapter.predict_raw() and verify that it "
            "matches the Pipeline-preprocessed result."
        ),
    )

    parser.add_argument(
        "--save-npz",
        default=None,
        help=(
            "Optional output path used to save the raw window, "
            "preprocessed input, mask and probabilities."
        ),
    )

    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path for saving the JSON test report.",
    )

    return parser


# ======================================================================
# 主流程
# ======================================================================


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if abs(args.window_sec - 10.0) > 1e-6:
        parser.error(
            "The current 50M flow test must use a real 10-second "
            "window. Set --window-sec 10.0."
        )

    if args.repeat <= 0:
        parser.error("--repeat must be greater than zero.")

    data_path = resolve_repo_path(args.data)
    checkpoint_path = resolve_repo_path(args.checkpoint)
    classifier_path = resolve_repo_path(args.classifier)

    for name, path in (
        ("HDF5 data", data_path),
        ("50M checkpoint", checkpoint_path),
        ("classifier checkpoint", classifier_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"{name} was not found: {path}"
            )

    print("=" * 72)
    print("50M offline window test")
    print("=" * 72)
    print("Data:", data_path)
    print("Session:", args.session)
    print("Backbone:", checkpoint_path)
    print("Classifier:", classifier_path)
    print("Device:", args.device)
    print()

    # ------------------------------------------------------------------
    # 1. 读取 HDF5
    # ------------------------------------------------------------------

    dataset = EEGHDF5(data_path)
    metadata = dataset.metadata
    loaded = dataset.load(args.session)

    trials = loaded["data"]
    labels = loaded["labels"]
    trial_ids = loaded["trial_ids"]

    print("[1/6] HDF5 loaded")
    print("  Dataset:", metadata.dataset_name)
    print("  Trial data shape:", trials.shape)
    print("  Trial data dtype:", trials.dtype)
    print("  Sample rate:", metadata.sample_rate, "Hz")
    print("  Unit:", metadata.unit)
    print("  Channel count:", len(metadata.channel_names))
    print("  Class names:", metadata.class_names)
    print()

    if trials.shape[1] != len(metadata.channel_names):
        raise RuntimeError(
            "HDF5 channel dimension does not match metadata: "
            f"data={trials.shape[1]}, "
            f"names={len(metadata.channel_names)}."
        )

    # ------------------------------------------------------------------
    # 2. 构建 Replay 风格连续流并截取 10 秒
    # ------------------------------------------------------------------

    stream, sample_labels, sample_trial_ids = (
        build_continuous_stream(
            trials=trials,
            labels=labels,
            trial_ids=trial_ids,
        )
    )

    (
        raw_window,
        window_labels,
        window_trial_ids,
        start_sample,
        end_sample,
    ) = extract_window(
        stream=stream,
        sample_labels=sample_labels,
        sample_trial_ids=sample_trial_ids,
        sample_rate=metadata.sample_rate,
        start_sec=args.start_sec,
        window_sec=args.window_sec,
    )

    label_composition = summarize_label_composition(
        window_labels,
        class_names=metadata.class_names,
        sample_rate=metadata.sample_rate,
    )

    segments = summarize_segments(
        window_labels,
        window_trial_ids,
        class_names=metadata.class_names,
        sample_rate=metadata.sample_rate,
    )

    print("[2/6] Real 10-second window extracted")
    print("  Continuous stream shape:", stream.shape)
    print("  Raw window shape:", raw_window.shape)
    print(
        "  Window samples:",
        raw_window.shape[-1],
    )
    print(
        "  Window range:",
        f"{start_sample} -> {end_sample}",
    )
    print(
        "  Window time:",
        f"{args.start_sec:.3f}s -> "
        f"{args.start_sec + args.window_sec:.3f}s",
    )
    print(
        "  Included trial IDs:",
        np.unique(window_trial_ids).tolist(),
    )
    print("  Label composition:")

    for item in label_composition:
        print(
            "   ",
            item["class_name"],
            f"{item['duration_sec']:.3f}s",
            f"({item['ratio']:.1%})",
        )

    print()
    print(
        "  Warning: the 10-second window may cross multiple "
        "original 4-second trials. Its prediction is only used "
        "to test the Pipeline, not to evaluate accuracy."
    )
    print()

    # ------------------------------------------------------------------
    # 3. 构建 50M 配置
    # ------------------------------------------------------------------

    config = Model50MConfig(
        checkpoint_path=checkpoint_path,
        classifier_path=classifier_path,
        device=args.device,

        # 当前严格使用 50M 原始输入配置。
        target_sample_rate=100.0,
        window_seconds=10.0,
        patch_seconds=1.0,
        patch_stride_seconds=1.0,

        filter_enabled=True,
        filter_low_hz=args.filter_low_hz,
        filter_high_hz=args.filter_high_hz,
        reference_mode=args.reference_mode,

        strict_window_duration=True,
        aggregation="flatten",
        num_classes=len(metadata.class_names),
    )

    print("[3/6] Model50MConfig created")
    print(
        "  Target shape:",
        (config.n_channels, config.target_num_points),
    )
    print(
        "  Token shape:",
        (config.num_tokens, config.patch_num_points),
    )
    print(
        "  Classifier input dim:",
        config.classifier_input_dim,
    )
    print()

    # ------------------------------------------------------------------
    # 4. 加载 Adapter
    # ------------------------------------------------------------------

    adapter_load_start = time.perf_counter()

    adapter = Model50MAdapter(
        config=config,
        class_names=metadata.class_names,
        strict_head_metadata=args.strict_head_metadata,
    )

    adapter_load_ms = (
        time.perf_counter() - adapter_load_start
    ) * 1000.0

    print("[4/6] 50M Adapter loaded")
    print("  Adapter load time:", f"{adapter_load_ms:.2f} ms")
    print("  Runtime device:", adapter.device)
    print(
        "  Backbone tensors loaded:",
        (
            adapter.backbone.load_report.loaded_tensor_count
            if adapter.backbone.load_report is not None
            else None
        ),
    )
    print(
        "  Classifier metadata:",
        adapter.classifier_load_report.metadata,
    )
    print()

    # ------------------------------------------------------------------
    # 5. Pipeline 预处理
    # ------------------------------------------------------------------

    pipeline_preprocessor = Model50MPipelinePreprocessor(
        config=config,
        channel_names=metadata.channel_names,
        sample_rate=metadata.sample_rate,
        input_unit=metadata.unit,
    )

    preprocess_start = time.perf_counter()

    model_input = pipeline_preprocessor.transform(
        samples=raw_window,
        sample_rate=metadata.sample_rate,
        input_unit=metadata.unit,
        reshape=True,
    )

    preprocess_ms = (
                            time.perf_counter() - preprocess_start
                    ) * 1000.0

    # 新版通用 Pipeline 返回一个字典：
    # {
    #     "signal": [64, 1000],
    #     "channel_valid_mask": [64],
    # }
    if not isinstance(model_input, dict):
        raise RuntimeError(
            "Model50MPipelinePreprocessor should return "
            "dict[str, np.ndarray]."
        )

    if "signal" not in model_input:
        raise RuntimeError(
            "50M preprocessing output is missing 'signal'."
        )

    if "channel_valid_mask" not in model_input:
        raise RuntimeError(
            "50M preprocessing output is missing "
            "'channel_valid_mask'."
        )

    model_signal = model_input["signal"]
    channel_valid_mask = model_input["channel_valid_mask"]

    preprocess_result = pipeline_preprocessor.last_result

    if preprocess_result is None:
        raise RuntimeError(
            "Pipeline preprocessor did not retain last_result."
        )

    if channel_valid_mask is None:
        raise RuntimeError(
            "Pipeline preprocessor did not retain "
            "last_channel_valid_mask."
        )

    expected_signal_shape = (
        config.n_channels,
        config.target_num_points,
    )

    if model_signal.shape != expected_signal_shape:
        raise RuntimeError(
            "Unexpected preprocessed signal shape: "
            f"expected {expected_signal_shape}, "
            f"got {model_signal.shape}."
        )

    if model_signal.dtype != np.float32:
        raise RuntimeError(
            "Preprocessed signal must be float32, "
            f"got {model_signal.dtype}."
        )

    if not np.isfinite(model_signal).all():
        raise RuntimeError(
            "Preprocessed signal contains NaN or Inf."
        )

    expected_mask_shape = (config.n_channels,)

    if channel_valid_mask.shape != expected_mask_shape:
        raise RuntimeError(
            "Unexpected channel_valid_mask shape: "
            f"expected {expected_mask_shape}, "
            f"got {channel_valid_mask.shape}."
        )

    if channel_valid_mask.dtype != np.float32:
        raise RuntimeError(
            "channel_valid_mask must be float32, "
            f"got {channel_valid_mask.dtype}."
        )

    if not np.isfinite(channel_valid_mask).all():
        raise RuntimeError(
            "channel_valid_mask contains NaN or Inf."
        )

    print("[5/6] 50M preprocessing completed")
    print("  Preprocessed signal shape:", model_signal.shape)
    print("  Preprocessed signal dtype:", model_signal.dtype)
    print("  Channel mask shape:", channel_valid_mask.shape)
    print()

    # ------------------------------------------------------------------
    # 6. Adapter 推理
    # ------------------------------------------------------------------

    prediction_probabilities: list[np.ndarray] = []
    prediction_timings: list[dict[str, float]] = []

    batched_model_input = add_batch_dimension(model_input)

    prediction_probabilities: list[np.ndarray] = []
    prediction_timings: list[dict[str, float]] = []

    for repeat_index in range(args.repeat):
        batched_model_input = add_batch_dimension(model_input)

        probabilities = adapter.predict_proba(
            batched_model_input
        )

        if probabilities.shape != (
            1,
            len(metadata.class_names),
        ):
            raise RuntimeError(
                "Unexpected probability shape: "
                f"{probabilities.shape}."
            )

        if not np.isfinite(probabilities).all():
            raise RuntimeError(
                "Adapter probabilities contain NaN or Inf."
            )

        if not np.allclose(
            probabilities.sum(axis=-1),
            1.0,
            atol=1e-5,
        ):
            raise RuntimeError(
                "Adapter probabilities do not sum to 1."
            )

        prediction_probabilities.append(
            probabilities[0].copy()
        )

        if adapter.last_timing is None:
            raise RuntimeError(
                "Adapter did not record inference timing."
            )

        prediction_timings.append(
            adapter.last_timing.to_dict()
        )

        print(
            f"  Repeat {repeat_index + 1}/{args.repeat}:",
            probabilities[0],
        )

    reference_probabilities = prediction_probabilities[0]

    for index, probabilities in enumerate(
        prediction_probabilities[1:],
        start=2,
    ):
        if not np.allclose(
            reference_probabilities,
            probabilities,
            atol=1e-6,
            rtol=1e-6,
        ):
            raise RuntimeError(
                "Repeated inference produced different results: "
                f"run 1 vs run {index}."
            )

    prediction = int(
        np.argmax(reference_probabilities)
    )
    confidence = float(
        reference_probabilities[prediction]
    )
    predicted_class = get_class_name(
        prediction,
        metadata.class_names,
    )

    raw_interface_consistent: bool | None = None
    raw_interface_probabilities: list[float] | None = None

    if args.check_raw_interface:
        raw_result = adapter.predict_raw(
            signal=raw_window,
            channel_names=metadata.channel_names,
            original_sample_rate=metadata.sample_rate,
            input_unit=metadata.unit,
        )

        raw_interface_array = np.asarray(
            raw_result.probabilities,
            dtype=np.float32,
        )

        raw_interface_consistent = bool(
            np.allclose(
                reference_probabilities,
                raw_interface_array,
                atol=1e-5,
                rtol=1e-5,
            )
        )

        raw_interface_probabilities = (
            raw_interface_array.tolist()
        )

        if not raw_interface_consistent:
            raise RuntimeError(
                "Pipeline-preprocessed inference and "
                "adapter.predict_raw() produced different "
                "probabilities."
            )

    print()
    print("[6/6] Prediction completed")
    print(
        "  Probabilities:",
        reference_probabilities.tolist(),
    )
    print("  Prediction:", prediction)
    print("  Predicted class:", predicted_class)
    print("  Confidence:", confidence)
    print(
        "  Interface consistency:",
        raw_interface_consistent,
    )
    print(
        "  Warning: test_linear_head.pt is not trained. "
        "The prediction has no accuracy meaning."
    )
    print()

    # ------------------------------------------------------------------
    # 生成报告
    # ------------------------------------------------------------------

    total_times = [
        item["total_ms"]
        for item in prediction_timings
    ]

    report: dict[str, Any] = {
        "status": "passed",
        "warning": (
            "test_linear_head.pt is an untrained test head. "
            "Prediction and confidence are only used to verify "
            "the inference Pipeline."
        ),
        "files": {
            "data": data_path,
            "checkpoint": checkpoint_path,
            "classifier": classifier_path,
        },
        "dataset": {
            "dataset_name": metadata.dataset_name,
            "session": args.session,
            "sample_rate": metadata.sample_rate,
            "unit": metadata.unit,
            "channel_count": len(metadata.channel_names),
            "channel_names": metadata.channel_names,
            "class_names": metadata.class_names,
            "trial_data_shape": list(trials.shape),
        },
        "window": {
            "start_sec": args.start_sec,
            "end_sec": args.start_sec + args.window_sec,
            "window_sec": args.window_sec,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "raw_shape": list(raw_window.shape),
            "trial_ids": np.unique(
                window_trial_ids
            ).tolist(),
            "label_composition": label_composition,
            "segments": segments,
        },
        "preprocessing": {
            "output_shape": list(model_signal.shape),
            "output_dtype": str(model_signal.dtype),
            "channel_valid_mask_shape": list(
                channel_valid_mask.shape
            ),
            "valid_channel_count": int(
                channel_valid_mask.sum()
            ),
            "mapped_channel_count": (
                preprocess_result.mapped_channel_count
            ),
            "missing_channel_count": (
                preprocess_result.missing_channel_count
            ),
            "unknown_channel_names": (
                preprocess_result.unknown_channel_names
            ),
            "padded_points": preprocess_result.padded_points,
            "cropped_points": preprocess_result.cropped_points,
            "preprocess_ms": preprocess_ms,
            "notes": preprocess_result.notes,
        },
        "model": {
            "device": str(adapter.device),
            "adapter_load_ms": adapter_load_ms,
            "num_tokens": config.num_tokens,
            "patch_num_points": config.patch_num_points,
            "output_layer_idx": config.output_layer_idx,
            "classifier_input_dim": (
                config.classifier_input_dim
            ),
            "num_classes": config.num_classes,
        },
        "prediction": {
            "prediction": prediction,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": (
                reference_probabilities.tolist()
            ),
        },
        "timing": {
            "runs": prediction_timings,
            "mean_total_ms": float(
                np.mean(total_times)
            ),
            "max_total_ms": float(
                np.max(total_times)
            ),
        },
        "interface_consistency": {
            "checked": args.check_raw_interface,
            "passed": raw_interface_consistent,
            "raw_interface_probabilities": (
                raw_interface_probabilities
            ),
        },
    }

    # ------------------------------------------------------------------
    # 可选保存 NPZ
    # ------------------------------------------------------------------

    if args.save_npz is not None:
        save_npz_path = resolve_repo_path(
            args.save_npz
        )
        save_npz_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            save_npz_path,
            raw_window=raw_window,
            model_signal=model_signal,
            channel_valid_mask=channel_valid_mask,
            probabilities=reference_probabilities,
            window_labels=window_labels,
            window_trial_ids=window_trial_ids,
            channel_names=np.asarray(
                metadata.channel_names,
                dtype=str,
            ),
            metadata_json=np.asarray(
                json.dumps(
                    report,
                    ensure_ascii=False,
                    default=json_default,
                )
            ),
        )

        print("Saved NPZ:", save_npz_path)

    # ------------------------------------------------------------------
    # 可选保存 JSON
    # ------------------------------------------------------------------

    if args.json_output is not None:
        json_output_path = resolve_repo_path(
            args.json_output
        )
        json_output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with json_output_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                report,
                handle,
                ensure_ascii=False,
                indent=2,
                default=json_default,
            )

        print("Saved JSON report:", json_output_path)

    print()
    print("=" * 72)
    print("50M offline window test passed.")
    print("=" * 72)


if __name__ == "__main__":
    main()