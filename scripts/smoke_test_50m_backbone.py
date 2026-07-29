from pathlib import Path

import numpy as np

from bci_dayloop.models.model_50m.backbone import (
    Model50MBackbone,
)
from bci_dayloop.models.model_50m.config import (
    Model50MConfig,
)
from bci_dayloop.models.model_50m.tokenization import (
    Model50MTokenizer,
)
from _bootstrap import ROOT

def main() -> None:
    config = Model50MConfig(
        checkpoint_path=(
            ROOT / "checkpoints/model.pt"
        ),
        device="cpu",
    )

    # 1. 构建并加载 checkpoint。
    backbone = Model50MBackbone(
        config=config,
        load_checkpoint=True,
        freeze=True,
    )

    print("Checkpoint report:")
    print(backbone.load_report)

    # 2. 先运行内置结构检查。
    health = backbone.health_check()

    print("\nHealth check:")
    for key, value in health.items():
        print(f"{key}: {value}")

    # 3. 构造一个 Tokenizer 输出。
    signal = np.random.randn(
        config.n_channels,
        config.target_num_points,
    ).astype(np.float32)

    channel_valid_mask = np.ones(
        config.n_channels,
        dtype=np.float32,
    )

    # 模拟部分缺失通道。
    signal[-10:] = 0.0
    channel_valid_mask[-10:] = 0.0

    tokenizer = Model50MTokenizer(config)

    tokens = tokenizer.tokenize(
        signal=signal,
        channel_valid_mask=channel_valid_mask,
    )

    batch = tokens.as_batch(
        device=backbone.device,
    )

    token_embeddings = backbone.extract_embeddings(batch)

    print(
        "\nToken embeddings:",
        token_embeddings.shape,
    )

    assert token_embeddings.shape == (
        1,
        640,
        512,
    )


if __name__ == "__main__":
    main()