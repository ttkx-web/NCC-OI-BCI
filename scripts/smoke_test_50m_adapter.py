from pathlib import Path

import numpy as np

from bci_dayloop.models.model_50m.adapter import (
    Model50MAdapter,
)
from bci_dayloop.models.model_50m.config import (
    Model50MConfig,
)

from _bootstrap import ROOT


def main() -> None:
    config = Model50MConfig(
        checkpoint_path=(
            ROOT / "checkpoints/model.pt"
        ),
        classifier_path=(
            ROOT / "checkpoints/test_linear_head.pt"
        ),
        device="cpu",
        aggregation="flatten",
        num_classes=4,
    )

    adapter = Model50MAdapter(
        config=config,
        class_names=(
            "left_hand",
            "right_hand",
            "feet",
            "tongue",
        ),
    )

    print("Classifier report:")
    print(adapter.classifier_load_report)

    health = adapter.health_check()

    print("\nAdapter health check:")
    for key, value in health.items():
        print(f"{key}: {value}")

    # 模拟已经完成 50M 预处理的批量输入。
    X = np.random.randn(
        2,
        64,
        1000,
    ).astype(np.float32)

    channel_masks = np.ones(
        (2, 64),
        dtype=np.float32,
    )

    # 模拟部分缺失通道。
    X[:, -10:, :] = 0.0
    channel_masks[:, -10:] = 0.0

    probabilities = adapter.predict_proba(
        X,
        channel_valid_masks=channel_masks,
    )

    predictions = adapter.predict(
        X,
        channel_valid_masks=channel_masks,
    )

    embeddings = adapter.extract_embeddings(
        X,
        channel_valid_masks=channel_masks,
    )

    print("\nProbabilities shape:", probabilities.shape)
    print("Predictions:", predictions)
    print("Embeddings shape:", embeddings.shape)
    print("Last timing:", adapter.last_timing)

    assert probabilities.shape == (2, 4)
    assert predictions.shape == (2,)
    assert embeddings.shape == (2, 327680)

    assert np.allclose(
        probabilities.sum(axis=-1),
        1.0,
        atol=1e-5,
    )

    print("\nAdapter smoke test passed.")


if __name__ == "__main__":
    main()