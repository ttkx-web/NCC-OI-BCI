from pathlib import Path

import numpy as np

from _bootstrap import ROOT

from bci_dayloop.data.hdf5_dataset import EEGHDF5
from bci_dayloop.models.model_50m.runtime import (
    build_50m_runtime_from_metadata,
)


def main() -> None:
    data_path = (
        ROOT
        / "data/processed/bnci2014_001_s01.h5"
    )

    dataset = EEGHDF5(data_path)
    metadata = dataset.metadata
    session = dataset.load("1test")

    runtime = build_50m_runtime_from_metadata(
        checkpoint_path=(
            ROOT / "checkpoints/50m/model_deploy.pt"
        ),
        classifier_path=(
            ROOT / "checkpoints/test_linear_head.pt"
        ),
        metadata=metadata,
        device="cpu",
    )

    print("Runtime health check:")

    health = runtime.health_check()

    for key, value in health.items():
        print(key, ":", value)

    # session["data"] 为 [N,C,T]，
    # 当前每个 trial 只有 4 秒，因此这里拼接出真实 10 秒。
    trials = np.asarray(
        session["data"],
        dtype=np.float32,
    )

    continuous_stream = (
        trials
        .transpose(1, 0, 2)
        .reshape(trials.shape[1], -1)
    )

    raw_window_samples = int(
        round(10.0 * metadata.sample_rate)
    )

    raw_window = continuous_stream[
        :,
        :raw_window_samples,
    ]

    print("Raw window shape:", raw_window.shape)

    result = runtime.predict_raw_window(raw_window)

    print("Runtime prediction:")
    print(result.to_dict())

    assert raw_window.shape[-1] == raw_window_samples
    assert len(result.probabilities) == len(
        metadata.class_names
    )
    assert np.isclose(
        sum(result.probabilities),
        1.0,
        atol=1e-5,
    )

    print("50M runtime smoke test passed.")


if __name__ == "__main__":
    main()