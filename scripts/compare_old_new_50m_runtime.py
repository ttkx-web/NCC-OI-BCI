from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _bootstrap import ROOT  # noqa: F401

from bci_dayloop.data.hdf5_dataset import EEGHDF5
from bci_dayloop.models.base import add_batch_dimension
from bci_dayloop.models.model_50m.adapter import Model50MAdapter
from bci_dayloop.models.model_50m.backend import Model50MBackend
from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.models.model_50m.pipeline_preprocessor import (
    Model50MPipelinePreprocessor,
)
from bci_dayloop.preprocessing.canonical import SignalCanonicalizer
from bci_dayloop.preprocessing.model_50m import Model50MInputTransform
from bci_dayloop.runtime.model import RuntimeModel
from bci_dayloop.runtime.types import RawEEGWindow
from bci_dayloop.packages.loader import (
    load_runtime_package,
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the legacy Model50MAdapter + "
            "Model50MPipelinePreprocessor path with the new RuntimeModel path."
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
        help="50M backbone checkpoint.",
    )
    parser.add_argument(
        "--classifier",
        required=True,
        help="50M classification head checkpoint.",
    )
    parser.add_argument(
        "--session",
        default="1test",
    )
    parser.add_argument(
        "--trial-index",
        type=int,
        default=0,
        help="Trial index inside the selected session.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--target-sample-rate",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--patch-sec",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--patch-stride-sec",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--model-n-time-patches",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--output-layer-idx",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--aggregation",
        choices=("flatten", "mean"),
        default="flatten",
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
        "--model-package",
        help=(
            "Optional schema-v2 Runtime Model Package. "
            "When provided, also compare package-loaded inference."
        ),
    )

    return parser.parse_args()


def build_config(
    args: argparse.Namespace,
    *,
    num_classes: int,
) -> Model50MConfig:
    return Model50MConfig(
        checkpoint_path=resolve_path(args.checkpoint),
        classifier_path=resolve_path(args.classifier),
        device=args.device,

        target_sample_rate=float(
            args.target_sample_rate
        ),
        window_seconds=float(
            args.window_sec
        ),
        patch_seconds=float(
            args.patch_sec
        ),
        patch_stride_seconds=float(
            args.patch_stride_sec
        ),
        model_n_time_patches=int(
            args.model_n_time_patches
        ),

        output_layer_idx=int(
            args.output_layer_idx
        ),
        aggregation=str(
            args.aggregation
        ),
        num_classes=int(
            num_classes
        ),

        strict_window_duration=True,
    )


def require_numpy_input_dict(
    value: Any,
    *,
    source: str,
) -> dict[str, np.ndarray]:
    if not isinstance(value, dict):
        raise TypeError(
            f"{source} must produce a dictionary, "
            f"got {type(value).__name__}."
        )

    required = {
        "signal",
        "channel_valid_mask",
    }

    missing = required - set(value)

    if missing:
        raise KeyError(
            f"{source} is missing keys: "
            f"{sorted(missing)}."
        )

    result: dict[str, np.ndarray] = {}

    for key in required:
        item = value[key]

        if not isinstance(item, np.ndarray):
            raise TypeError(
                f"{source}[{key!r}] must be "
                f"numpy.ndarray, got "
                f"{type(item).__name__}."
            )

        result[key] = np.asarray(
            item,
            dtype=np.float32,
        )

    return result


def require_tensor_input_dict(
    value: Any,
    *,
    source: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(value, dict):
        raise TypeError(
            f"{source} must produce a dictionary, "
            f"got {type(value).__name__}."
        )

    required = {
        "signal",
        "channel_valid_mask",
    }

    missing = required - set(value)

    if missing:
        raise KeyError(
            f"{source} is missing keys: "
            f"{sorted(missing)}."
        )

    result: dict[str, torch.Tensor] = {}

    for key in required:
        item = value[key]

        if not isinstance(item, torch.Tensor):
            raise TypeError(
                f"{source}[{key!r}] must be "
                f"torch.Tensor, got "
                f"{type(item).__name__}."
            )

        result[key] = item

    return result


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
    classifier_path = resolve_path(
        args.classifier
    )

    for logical_name, path in (
        ("HDF5 data", data_path),
        ("50M backbone", checkpoint_path),
        ("classification head", classifier_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"{logical_name} was not found: "
                f"{path}"
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
            f"trial-index={args.trial_index} "
            f"is outside [0, {len(trials) - 1}]."
        )

    raw_window = np.asarray(
        trials[args.trial_index],
        dtype=np.float32,
    )

    expected_raw_points = int(
        round(
            args.window_sec
            * metadata.sample_rate
        )
    )

    if raw_window.shape != (
        len(metadata.channel_names),
        expected_raw_points,
    ):
        raise ValueError(
            "Selected trial does not match the requested "
            "raw-window contract: "
            f"expected "
            f"({len(metadata.channel_names)}, "
            f"{expected_raw_points}), "
            f"got {raw_window.shape}."
        )

    class_names = tuple(
        str(name)
        for name in metadata.class_names
    )

    print("=" * 72)
    print("Regression input")
    print("=" * 72)
    print(f"Data:          {data_path}")
    print(f"Session:       {args.session}")
    print(f"Trial index:   {args.trial_index}")
    print(f"Trial ID:      {int(trial_ids[args.trial_index])}")
    print(f"True label:    {int(labels[args.trial_index])}")
    print(f"Raw shape:     {raw_window.shape}")
    print(f"Sample rate:   {metadata.sample_rate}")
    print(f"Unit:          {metadata.unit}")
    print(f"Device:        {args.device}")

    # ==========================================================
    # 旧链路
    #
    # Raw EEG
    #   -> Model50MPipelinePreprocessor
    #   -> add_batch_dimension
    #   -> Model50MAdapter.predict_proba
    # ==========================================================

    print("\nRunning legacy path...")

    old_config = build_config(
        args,
        num_classes=len(class_names),
    )

    old_preprocessor = (
        Model50MPipelinePreprocessor(
            config=old_config,
            channel_names=metadata.channel_names,
            sample_rate=float(
                metadata.sample_rate
            ),
            input_unit=str(
                metadata.unit
            ),
        )
    )

    old_adapter = Model50MAdapter(
        config=old_config,
        class_names=class_names,
        strict_head_metadata=True,
    )

    old_unbatched_input = (
        old_preprocessor.transform(
            raw_window,
            float(metadata.sample_rate),
            str(metadata.unit),
            reshape=True,
        )
    )

    old_batched_input_raw = (
        add_batch_dimension(
            old_unbatched_input
        )
    )

    old_batched_input = (
        require_numpy_input_dict(
            old_batched_input_raw,
            source="legacy path",
        )
    )

    old_probabilities = (
        old_adapter.predict_proba(
            old_batched_input
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

    # 先复制结果，再释放旧模型，避免 Mac 同时加载两套 50M。
    old_signal = (
        old_batched_input["signal"]
        .copy()
    )
    old_mask = (
        old_batched_input[
            "channel_valid_mask"
        ].copy()
    )
    old_probabilities = (
        old_probabilities.copy()
    )




    del old_adapter
    del old_preprocessor
    del old_config
    release_model_memory()

    # ==========================================================
    # 新链路
    #
    # RawEEGWindow
    #   -> SignalCanonicalizer
    #   -> Model50MInputTransform
    #   -> Model50MBackend
    #   -> RuntimeModel
    # ==========================================================

    print("Running unified RuntimeModel path...")

    new_config = build_config(
        args,
        num_classes=len(class_names),
    )

    new_adapter = Model50MAdapter(
        config=new_config,
        class_names=class_names,
        strict_head_metadata=True,
    )

    new_runtime = RuntimeModel(
        canonicalizer=SignalCanonicalizer(
            target_unit="uV"
        ),
        input_transform=(
            Model50MInputTransform(
                new_config
            )
        ),
        backend=Model50MBackend(
            new_adapter
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
        unit=str(
            metadata.unit
        ),
        layout="CT",
        trial_id=str(
            int(
                trial_ids[
                    args.trial_index
                ]
            )
        ),
        window_id=(
            f"{args.session}:"
            f"{args.trial_index}"
        ),
        label=int(
            labels[
                args.trial_index
            ]
        ),
        metadata={
            "dataset": (
                metadata.dataset_name
            ),
            "session": args.session,
        },
    )

    prepared = new_runtime.prepare(
        runtime_window
    )

    new_tensor_input = (
        require_tensor_input_dict(
            prepared.model_input,
            source="RuntimeModel path",
        )
    )

    new_signal = tensor_to_numpy(
        new_tensor_input["signal"]
    )

    new_mask = tensor_to_numpy(
        new_tensor_input[
            "channel_valid_mask"
        ]
    )

    new_output = (
        new_runtime.predict_prepared(
            prepared,
            return_features=False,
        )
    )

    new_probabilities = tensor_to_numpy(
        new_output.probabilities
    )

    new_prediction = int(
        new_output.predicted_class
    )

    new_confidence = float(
        new_output.confidence
    )

    direct_probabilities = (
        new_probabilities.copy()
    )
    direct_prediction = new_prediction
    direct_confidence = new_confidence

    del new_output
    del prepared
    del new_runtime
    del new_adapter
    del new_config

    release_model_memory()

    # ==========================================================
    # 分层比较
    # ==========================================================

    signal_max_diff = (
        maximum_absolute_difference(
            old_signal,
            new_signal,
        )
    )

    mask_max_diff = (
        maximum_absolute_difference(
            old_mask,
            new_mask,
        )
    )

    probability_max_diff = (
        maximum_absolute_difference(
            old_probabilities,
            new_probabilities,
        )
    )

    print("\n" + "=" * 72)
    print("Comparison")
    print("=" * 72)

    print(
        "Preprocessed signal shapes: "
        f"old={old_signal.shape}, "
        f"new={new_signal.shape}"
    )
    print(
        "Channel mask shapes:         "
        f"old={old_mask.shape}, "
        f"new={new_mask.shape}"
    )
    print(
        "Probability shapes:          "
        f"old={old_probabilities.shape}, "
        f"new={new_probabilities.shape}"
    )

    print(
        f"Signal max abs diff:          "
        f"{signal_max_diff:.10g}"
    )
    print(
        f"Mask max abs diff:            "
        f"{mask_max_diff:.10g}"
    )
    print(
        f"Probability max abs diff:     "
        f"{probability_max_diff:.10g}"
    )

    print(
        f"Old prediction/confidence:    "
        f"{old_prediction} / "
        f"{old_confidence:.10f}"
    )
    print(
        f"New prediction/confidence:    "
        f"{new_prediction} / "
        f"{new_confidence:.10f}"
    )

    print(
        "Old probabilities:           "
        f"{old_probabilities[0].tolist()}"
    )
    print(
        "New probabilities:           "
        f"{new_probabilities[0].tolist()}"
    )

    # 第一层：预处理结果一致。
    np.testing.assert_allclose(
        old_signal,
        new_signal,
        rtol=args.rtol,
        atol=args.atol,
        err_msg=(
            "Legacy and RuntimeModel preprocessed "
            "signals are inconsistent."
        ),
    )

    np.testing.assert_allclose(
        old_mask,
        new_mask,
        rtol=0.0,
        atol=0.0,
        err_msg=(
            "Legacy and RuntimeModel channel masks "
            "are inconsistent."
        ),
    )

    # 第二层：最终预测概率一致。
    np.testing.assert_allclose(
        old_probabilities,
        new_probabilities,
        rtol=args.rtol,
        atol=args.atol,
        err_msg=(
            "Legacy and RuntimeModel probabilities "
            "are inconsistent."
        ),
    )

    if old_prediction != new_prediction:
        raise AssertionError(
            "Predicted classes are inconsistent: "
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
            "Prediction confidences are inconsistent: "
            f"old={old_confidence}, "
            f"new={new_confidence}."
        )

    print("\nPASS: legacy and RuntimeModel predictions are consistent.")

    # ==========================================================
    # 第三层：直接构建的 RuntimeModel
    #         vs 从 Runtime Model Package 加载的 RuntimeModel
    # ==========================================================

    if args.model_package is not None:
        model_package_path = resolve_path(
            args.model_package
        )

        if not model_package_path.is_dir():
            raise FileNotFoundError(
                "Runtime Model Package directory "
                f"was not found: {model_package_path}"
            )

        print(
            "\nRunning packaged RuntimeModel path..."
        )
        print(
            f"Model package: {model_package_path}"
        )

        loaded_package = load_runtime_package(
            model_package_path,
            device=args.device,
            verify_hashes=True,
        )

        # ------------------------------------------------------
        # 1. 检查 Package 的实验配置
        # ------------------------------------------------------

        if not np.isclose(
            loaded_package.window_sec,
            args.window_sec,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                "Package window_sec does not match "
                "the direct RuntimeModel setting: "
                f"package={loaded_package.window_sec}, "
                f"direct={args.window_sec}."
            )

        if not np.isclose(
            loaded_package.target_sample_rate,
            args.target_sample_rate,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                "Package target sample rate does not "
                "match the direct RuntimeModel setting: "
                f"package="
                f"{loaded_package.target_sample_rate}, "
                f"direct={args.target_sample_rate}."
            )

        if tuple(
            loaded_package.class_names
        ) != tuple(class_names):
            raise ValueError(
                "Package class order does not match "
                "the dataset class order: "
                f"package="
                f"{loaded_package.class_names}, "
                f"dataset={class_names}."
            )

        # ------------------------------------------------------
        # 2. 使用同一个 RawEEGWindow 预测
        #
        # runtime_window 就是上面直接 Runtime 使用的那个窗口，
        # 不要重新读取数据或重新构造另一个窗口。
        # ------------------------------------------------------

        package_output = (
            loaded_package.runtime_model.predict(
                runtime_window,
                return_features=False,
            )
        )

        package_probabilities = tensor_to_numpy(
            package_output.probabilities
        )

        package_prediction = int(
            package_output.predicted_class
        )

        package_confidence = float(
            package_output.confidence
        )

        # ------------------------------------------------------
        # 3. 输出对比信息
        # ------------------------------------------------------

        package_probability_max_diff = (
            maximum_absolute_difference(
                direct_probabilities,
                package_probabilities,
            )
        )

        print("\n" + "=" * 72)
        print("Direct Runtime vs Package Runtime")
        print("=" * 72)

        print(
            "Direct probability shape:    "
            f"{direct_probabilities.shape}"
        )
        print(
            "Package probability shape:   "
            f"{package_probabilities.shape}"
        )
        print(
            "Probability max abs diff:     "
            f"{package_probability_max_diff:.10g}"
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

        print(
            "Direct probabilities:         "
            f"{direct_probabilities[0].tolist()}"
        )
        print(
            "Package probabilities:        "
            f"{package_probabilities[0].tolist()}"
        )

        # ------------------------------------------------------
        # 4. 正式断言
        # ------------------------------------------------------

        np.testing.assert_allclose(
            direct_probabilities,
            package_probabilities,
            rtol=args.rtol,
            atol=args.atol,
            err_msg=(
                "Direct RuntimeModel and package-loaded "
                "RuntimeModel probabilities are inconsistent."
            ),
        )

        if (
            direct_prediction
            != package_prediction
        ):
            raise AssertionError(
                "Direct and packaged predictions "
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
                "Direct and packaged confidence values "
                "are inconsistent: "
                f"direct={direct_confidence}, "
                f"package={package_confidence}."
            )

        print(
            "\nPASS: direct and package-loaded "
            "RuntimeModel predictions are consistent."
        )

        del package_output
        del loaded_package
        release_model_memory()



if __name__ == "__main__":
    main()