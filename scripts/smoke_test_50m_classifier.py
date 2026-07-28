from pathlib import Path

import numpy as np
import torch

from bci_dayloop.models.model_50m.backbone import (
    Model50MBackbone,
)
from bci_dayloop.models.model_50m.classifier import (
    Model50MClassifier,
    load_classifier_checkpoint,
    save_classifier_checkpoint,
)
from bci_dayloop.models.model_50m.config import (
    Model50MConfig,
)
from bci_dayloop.models.model_50m.tokenization import (
    Model50MTokenizer,
)


def main() -> None:
    config = Model50MConfig(
        checkpoint_path=Path(
            "/Volumes/file/NCC-OI-BCI/checkpoints/model.pt"
        ),
        classifier_path=None,
        device="cpu",
        aggregation="flatten",
        num_classes=4,
    )

    backbone = Model50MBackbone(
        config=config,
        load_checkpoint=True,
        freeze=True,
    )

    classifier = Model50MClassifier(
        config=config,
        backbone=backbone,
    )

    print("Feature dim:", classifier.feature_dim)
    print(
        "Trainable parameters:",
        classifier.trainable_parameters,
    )

    # 模拟预处理后的 [64, 1000] 数据。
    signal = np.random.randn(
        config.n_channels,
        config.target_num_points,
    ).astype(np.float32)

    channel_valid_mask = np.ones(
        config.n_channels,
        dtype=np.float32,
    )

    # 模拟缺失 10 个通道。
    signal[-10:] = 0.0
    channel_valid_mask[-10:] = 0.0

    tokenizer = Model50MTokenizer(config)

    tokens = tokenizer.tokenize(
        signal=signal,
        channel_valid_mask=channel_valid_mask,
    )

    batch = tokens.as_batch(
        device=classifier.device,
    )

    # 当前分类头还没有训练，所以概率没有业务意义，
    # 这里只检查完整前向链路和张量形状。
    output = classifier.predict_batch(batch)

    print("Logits:", output.logits.shape)
    print(
        "Probabilities:",
        output.probabilities.shape,
    )
    print(
        "Predictions:",
        output.predictions.shape,
    )
    print(
        "Feature shape:",
        output.feature_shape,
    )
    print("Timing:", output.first_as_dict())

    assert output.logits.shape == (1, 4)
    assert output.probabilities.shape == (1, 4)
    assert output.feature_shape == (1, 327680)

    assert torch.allclose(
        output.probabilities.sum(dim=-1),
        torch.ones(1, device=output.probabilities.device),
        atol=1e-5,
    )

    # 测试任务头保存与重新加载。
    head_path = Path(
        "/Volumes/file/NCC-OI-BCI/checkpoints/test_linear_head.pt"
    )

    save_classifier_checkpoint(
        classifier=classifier,
        checkpoint_path=head_path,
        extra_metadata={
            "task": "smoke_test",
        },
    )

    second_backbone = Model50MBackbone(
        config=config,
        load_checkpoint=True,
        freeze=True,
    )

    second_classifier = Model50MClassifier(
        config=config,
        backbone=second_backbone,
    )

    report = load_classifier_checkpoint(
        classifier=second_classifier,
        checkpoint_path=head_path,
    )

    print("Classifier load report:", report)

    second_output = second_classifier.predict_batch(
        batch
    )

    assert torch.allclose(
        output.logits,
        second_output.logits,
        atol=1e-6,
        rtol=1e-6,
    )

    print("Classifier smoke test passed.")


if __name__ == "__main__":
    main()