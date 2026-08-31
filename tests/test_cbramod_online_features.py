from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from bci_dayloop.models.cbramod.backend import (
    CBraModBackend,
)
from bci_dayloop.models.cbramod.backbone import (
    CBraModBackbone,
)
from bci_dayloop.models.cbramod.classifier import (
    build_cbramod_classifier,
)
from bci_dayloop.models.cbramod.config import (
    BCICIV2A_22_CHANNELS,
    CBraModConfig,
)
from bci_dayloop.models.online_features import (
    OnlineTrainableFeatureBackend,
)


class TinyCBraModBackbone(
    CBraModBackbone
):
    """
    用于单元测试的轻量 CBrAmod backbone。

    它保留正式 CBrAmodBackbone 的输入输出契约：

        输入：[B,C,S,P]
        输出：[B,C,S,D]

    但是不会加载正式 checkpoint，也不会创建完整
    Transformer，只使用一个 Linear 模拟每个 patch
    从 P 维映射到 D 维。
    """

    def __init__(
        self,
        config: CBraModConfig,
    ) -> None:
        # 不调用 CBraModBackbone.__init__()，
        # 因为正式构造函数会创建完整模型并加载 checkpoint。
        nn.Module.__init__(self)

        self.config = config

        # 模拟每个 EEG patch 的特征映射：
        #
        # [B,C,S,P] -> [B,C,S,D]
        self.model = nn.Linear(
            self.config.points_per_patch,
            self.config.backbone_output_dim,
        )

        self.to(
            torch.device(
                self.config.device
            )
        )

        self.freeze()


@pytest.fixture
def cbramod_config(
    tmp_path: Path,
) -> CBraModConfig:
    """
    构造轻量测试配置。

    通道数和 patch 结构保持正式约定：
        22 channels
        4 time segments
        200 points/segment

    只把 backbone_output_dim 从 200 缩小到 8，
    以减少测试计算量。
    """

    return CBraModConfig(
        checkpoint_path=(
            tmp_path
            / "unused_cbramod_checkpoint.pt"
        ),
        classifier_path=None,
        device="cpu",
        target_sample_rate=200.0,
        window_seconds=4.0,
        n_channels=22,
        standard_channels=(
            BCICIV2A_22_CHANNELS
        ),
        time_segments=4,
        points_per_patch=200,
        input_unit="uV",
        num_classes=4,
        backbone_output_dim=8,

        # 测试中使用线性头，避免构造较大的
        # official_mlp；不影响路径等价性验证。
        head_type="linear",
    )


@pytest.fixture
def cbramod_backend(
    cbramod_config: CBraModConfig,
) -> CBraModBackend:
    """
    创建不依赖正式 checkpoint 的 Backend。
    """

    torch.manual_seed(42)

    backbone = TinyCBraModBackbone(
        cbramod_config
    )

    classifier = (
        build_cbramod_classifier(
            cbramod_config
        )
    )

    backend = CBraModBackend(
        backbone=backbone,
        classifier=classifier,
        config=cbramod_config,
    )

    backend.backbone.freeze()
    backend.classifier.eval()

    return backend


def make_model_input(
    *,
    batch_size: int,
    config: CBraModConfig,
) -> torch.Tensor:
    """
    构造固定随机输入：

        [B,22,4,200]
    """

    random_generator = (
        torch.Generator()
    )

    random_generator.manual_seed(123)

    return torch.randn(
        batch_size,
        config.n_channels,
        config.time_segments,
        config.points_per_patch,
        dtype=torch.float32,
        generator=random_generator,
    )


def test_cbramod_backend_implements_online_interface(
    cbramod_backend: CBraModBackend,
) -> None:
    """
    验证 CBraModBackend 已实现
    OnlineTrainableFeatureBackend 协议。
    """

    assert isinstance(
        cbramod_backend,
        OnlineTrainableFeatureBackend,
    )

    spec = (
        cbramod_backend
        .online_feature_spec
    )

    assert spec.model_name == "cbramod"

    # 22 channels × 4 time segments
    assert spec.token_count == 88

    assert spec.embedding_dim == 8


def test_cbramod_online_token_shape(
    cbramod_backend: CBraModBackend,
    cbramod_config: CBraModConfig,
) -> None:
    """
    验证在线特征的 shape 转换：

        [B,22,4,200]
        → backbone
        → [B,22,4,8]
        → flatten
        → [B,88,8]
    """

    model_input = make_model_input(
        batch_size=1,
        config=cbramod_config,
    )

    tokens = (
        cbramod_backend
        .encode_online_tokens(
            model_input,
            train_backbone=False,
        )
    )

    assert tokens.shape == (
        1,
        88,
        8,
    )

    assert tokens.dtype == (
        torch.float32
    )

    assert torch.isfinite(
        tokens
    ).all()

    # V1 冻结 backbone，因此 token 本身
    # 不应要求计算 backbone 梯度。
    assert tokens.requires_grad is False


def test_cbramod_static_and_online_paths_are_equivalent(
    cbramod_backend: CBraModBackend,
    cbramod_config: CBraModConfig,
) -> None:
    """
    验证无 Generator 时，新旧路径完全等价。

    原路径：
        signal
        → backbone
        → [B,C,S,D]
        → classifier

    新路径：
        signal
        → backbone
        → [B,C,S,D]
        → flatten [B,C*S,D]
        → reshape [B,C,S,D]
        → classifier
    """

    model_input = make_model_input(
        batch_size=1,
        config=cbramod_config,
    )

    cbramod_backend.backbone.freeze()
    cbramod_backend.classifier.eval()

    # -------------------------------------------------
    # 1. 原有静态 Runtime 路径
    # -------------------------------------------------

    static_output = (
        cbramod_backend
        .predict_tensor(
            model_input,
            return_features=True,
        )
    )

    assert static_output.logits.shape == (
        1,
        cbramod_config.num_classes,
    )

    assert static_output.features is not None

    assert static_output.features.shape == (
        1,
        cbramod_config.n_channels,
        cbramod_config.time_segments,
        cbramod_config.backbone_output_dim,
    )

    # -------------------------------------------------
    # 2. 新增在线 token 路径
    # -------------------------------------------------

    tokens = (
        cbramod_backend
        .encode_online_tokens(
            model_input,
            train_backbone=False,
        )
    )

    assert tokens.shape == (
        1,
        (
            cbramod_config.n_channels
            * cbramod_config.time_segments
        ),
        cbramod_config.backbone_output_dim,
    )

    with torch.no_grad():
        online_logits = (
            cbramod_backend
            .classify_online_tokens(
                tokens
            )
        )

    assert online_logits.shape == (
        1,
        cbramod_config.num_classes,
    )

    # -------------------------------------------------
    # 3. 验证 flatten/reshape 没有改变 feature
    # -------------------------------------------------

    restored_features = tokens.reshape(
        tokens.shape[0],
        cbramod_config.n_channels,
        cbramod_config.time_segments,
        cbramod_config.backbone_output_dim,
    )

    torch.testing.assert_close(
        static_output.features,
        restored_features,
        rtol=0.0,
        atol=0.0,
    )

    # -------------------------------------------------
    # 4. 验证两条路径的 logits 相同
    # -------------------------------------------------

    torch.testing.assert_close(
        static_output.logits,
        online_logits,
        rtol=1e-5,
        atol=1e-6,
    )

    # -------------------------------------------------
    # 5. 验证 probabilities 也相同
    # -------------------------------------------------

    online_probabilities = torch.softmax(
        online_logits,
        dim=-1,
    )

    torch.testing.assert_close(
        static_output.probabilities,
        online_probabilities,
        rtol=1e-5,
        atol=1e-6,
    )

    # -------------------------------------------------
    # 6. 验证预测类别相同
    # -------------------------------------------------

    online_prediction = int(
        online_probabilities
        .argmax(dim=-1)[0]
        .item()
    )

    assert (
        static_output.predicted_class
        == online_prediction
    )


def test_cbramod_online_path_supports_batch(
    cbramod_backend: CBraModBackend,
    cbramod_config: CBraModConfig,
) -> None:
    """
    普通 Runtime 每次只预测一个窗口，
    但 NeuroOnline 更新需要支持 batch。
    """

    model_input = make_model_input(
        batch_size=16,
        config=cbramod_config,
    )

    tokens = (
        cbramod_backend
        .encode_online_tokens(
            model_input,
            train_backbone=False,
        )
    )

    assert tokens.shape == (
        16,
        88,
        8,
    )

    with torch.no_grad():
        logits = (
            cbramod_backend
            .classify_online_tokens(
                tokens
            )
        )

    assert logits.shape == (
        16,
        cbramod_config.num_classes,
    )


def test_cbramod_static_predict_still_requires_one_window(
    cbramod_backend: CBraModBackend,
    cbramod_config: CBraModConfig,
) -> None:
    """
    在线接口支持 batch，不代表静态 Runtime
    的单窗口约束被取消。
    """

    model_input = make_model_input(
        batch_size=2,
        config=cbramod_config,
    )

    with pytest.raises(
        ValueError,
        match=(
            "expects one window per prediction"
        ),
    ):
        cbramod_backend.predict_tensor(
            model_input
        )


def test_cbramod_online_path_accepts_signal_dictionary(
    cbramod_backend: CBraModBackend,
    cbramod_config: CBraModConfig,
) -> None:
    """
    ModelTensor 可以是 Tensor，也可以是：
        {"signal": Tensor}

    两种输入形式应得到相同 token。
    """

    signal = make_model_input(
        batch_size=1,
        config=cbramod_config,
    )

    tensor_tokens = (
        cbramod_backend
        .encode_online_tokens(
            signal,
            train_backbone=False,
        )
    )

    dictionary_tokens = (
        cbramod_backend
        .encode_online_tokens(
            {
                "signal": signal,
            },
            train_backbone=False,
        )
    )

    torch.testing.assert_close(
        tensor_tokens,
        dictionary_tokens,
        rtol=0.0,
        atol=0.0,
    )