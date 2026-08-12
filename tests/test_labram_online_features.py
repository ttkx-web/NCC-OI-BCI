from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from bci_dayloop.models.labram_backend import (
    LaBraMBackend,
)
from bci_dayloop.models.labram_linear import (
    LaBraMLinearAdapter,
)
from bci_dayloop.models.online_features import (
    OnlineTrainableFeatureBackend,
)


class TinyPatchEncoder(nn.Module):
    """
    用于单元测试的极简 LaBraM Encoder。

    它模拟真实 LaBraM 的 forward_features() 行为：

    - return_patch_tokens=False:
        返回 [B,D]，即 token mean pooling 后的特征；

    - return_patch_tokens=True:
        返回 [B,C*S,D]，即未池化 patch tokens。
    """

    embed_dim = 8

    def __init__(self) -> None:
        super().__init__()

        self.projection = nn.Linear(
            200,
            self.embed_dim,
        )

    def forward_features(
        self,
        x: torch.Tensor,
        input_chans: torch.Tensor | None = None,
        return_patch_tokens: bool = False,
        return_all_tokens: bool = False,
    ) -> torch.Tensor:
        del input_chans

        if x.ndim != 4:
            raise ValueError(
                "TinyPatchEncoder expects "
                f"[B,C,S,200], got {tuple(x.shape)}."
            )

        batch_size, channels, patches, patch_size = (
            x.shape
        )

        if patch_size != 200:
            raise ValueError(
                "TinyPatchEncoder requires "
                f"patch_size=200, got {patch_size}."
            )

        # [B,C,S,200] -> [B,C*S,200]
        flattened_patches = x.reshape(
            batch_size,
            channels * patches,
            patch_size,
        )

        # [B,C*S,200] -> [B,C*S,D]
        tokens = self.projection(
            flattened_patches
        )

        if return_all_tokens:
            # 只为模拟真实接口；当前测试不会使用。
            cls_token = tokens.mean(
                dim=1,
                keepdim=True,
            )

            return torch.cat(
                (cls_token, tokens),
                dim=1,
            )

        if return_patch_tokens:
            return tokens

        # 模拟真实 LaBraM 默认的 mean pooling。
        return tokens.mean(dim=1)

    def forward(
        self,
        x: torch.Tensor,
        input_chans: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_features(
            x,
            input_chans=input_chans,
        )


class TinyGenerator(nn.Module):
    """
    用于验证梯度路径的极简 Generator。

    初始时：
        alpha = 1
        beta = 0

    因此初始 adapted_tokens 与输入 tokens 完全相同。
    """

    def __init__(
        self,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.alpha_head = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.beta_head = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        # identity 初始化：
        # adapted = tokens * 1 + 0
        nn.init.zeros_(
            self.alpha_head.weight
        )
        nn.init.zeros_(
            self.alpha_head.bias
        )
        nn.init.zeros_(
            self.beta_head.weight
        )
        nn.init.zeros_(
            self.beta_head.bias
        )

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        alpha = (
            1.0
            + self.alpha_head(tokens)
        )

        beta = self.beta_head(tokens)

        return tokens * alpha + beta


@pytest.fixture
def labram_backend() -> LaBraMBackend:
    """
    构建不依赖正式 checkpoint 的测试 Backend。
    """

    torch.manual_seed(42)

    encoder = TinyPatchEncoder()

    adapter = LaBraMLinearAdapter(
        channel_names=[
            "C3",
            "C4",
        ],
        n_classes=4,
        device="cpu",
        amp=False,
        random_init=True,
        n_patches=2,
        encoder=encoder,
        embedding_batch_size=4,
    )

    adapter.encoder.eval()
    adapter.head.eval()

    return LaBraMBackend(
        adapter=adapter
    )


def make_signal(
    *,
    batch_size: int,
) -> torch.Tensor:
    """
    构造 [B,2,2,200] 测试输入。

    2 channels × 2 patches = 4 tokens。
    """

    generator = torch.Generator()
    generator.manual_seed(123)

    return torch.randn(
        batch_size,
        2,
        2,
        200,
        generator=generator,
        dtype=torch.float32,
    )


def parameters_changed(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> bool:
    """
    判断至少一个参数是否发生变化。
    """

    return any(
        not torch.equal(
            before[name],
            after[name],
        )
        for name in before
    )


def assert_state_dict_equal(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> None:
    """
    检查两个 state_dict 完全一致。
    """

    assert before.keys() == after.keys()

    for name in before:
        assert torch.equal(
            before[name],
            after[name],
        ), f"Parameter changed unexpectedly: {name}"


def test_labram_backend_implements_online_interface(
    labram_backend: LaBraMBackend,
) -> None:
    assert isinstance(
        labram_backend,
        OnlineTrainableFeatureBackend,
    )

    spec = (
        labram_backend.online_feature_spec
    )

    assert spec.model_name == "labram"

    # 2 channels × 2 patches
    assert spec.token_count == 4

    assert spec.embedding_dim == 8


def test_labram_online_token_shape(
    labram_backend: LaBraMBackend,
) -> None:
    signal = make_signal(
        batch_size=1
    )

    tokens = (
        labram_backend
        .encode_online_tokens(
            signal,
            train_backbone=False,
        )
    )

    assert tokens.shape == (
        1,
        4,
        8,
    )

    assert tokens.dtype == torch.float32

    assert torch.isfinite(
        tokens
    ).all()

    assert tokens.requires_grad is False

    # 这里应该是普通 no_grad Tensor，
    # 而不是 inference_mode 生成的 Tensor。
    assert not torch.is_inference(
        tokens
    )


def test_labram_static_and_online_paths_are_equivalent(
    labram_backend: LaBraMBackend,
) -> None:
    """
    验证：

    原路径：
        signal
        -> encoder 内部 mean pooling
        -> head

    新路径：
        signal
        -> patch tokens
        -> 外部 mean pooling
        -> head

    在没有 Generator 修改 token 时，
    两条路径必须得到相同 logits。
    """

    signal = make_signal(
        batch_size=1
    )

    labram_backend.adapter.encoder.eval()
    labram_backend.adapter.head.eval()

    static_output = (
        labram_backend.predict_tensor(
            signal,
            return_features=True,
        )
    )

    tokens = (
        labram_backend
        .encode_online_tokens(
            signal,
            train_backbone=False,
        )
    )

    with torch.no_grad():
        online_logits = (
            labram_backend
            .classify_online_tokens(
                tokens
            )
        )

    assert static_output.logits.shape == (
        1,
        4,
    )

    assert online_logits.shape == (
        1,
        4,
    )

    torch.testing.assert_close(
        static_output.logits,
        online_logits,
        rtol=1e-5,
        atol=1e-6,
    )

    # 同时验证 pooled feature 也等价。
    online_pooled_features = (
        tokens.mean(dim=1)
    )

    assert static_output.features is not None

    torch.testing.assert_close(
        static_output.features,
        online_pooled_features,
        rtol=1e-5,
        atol=1e-6,
    )


def test_identity_generator_preserves_logits(
    labram_backend: LaBraMBackend,
) -> None:
    """
    验证 identity 初始化的 Generator
    不会改变当前模型输出。
    """

    signal = make_signal(
        batch_size=1
    )

    tokens = (
        labram_backend
        .encode_online_tokens(
            signal,
            train_backbone=False,
        )
    )

    generator = TinyGenerator(
        embedding_dim=(
            labram_backend
            .online_feature_spec
            .embedding_dim
        )
    )

    generator.eval()
    labram_backend.adapter.head.eval()

    with torch.no_grad():
        logits_without_generator = (
            labram_backend
            .classify_online_tokens(
                tokens
            )
        )

        adapted_tokens = generator(
            tokens
        )

        logits_with_generator = (
            labram_backend
            .classify_online_tokens(
                adapted_tokens
            )
        )

    torch.testing.assert_close(
        adapted_tokens,
        tokens,
        rtol=0.0,
        atol=0.0,
    )

    torch.testing.assert_close(
        logits_with_generator,
        logits_without_generator,
        rtol=0.0,
        atol=0.0,
    )


def test_labram_online_path_supports_training_batch(
    labram_backend: LaBraMBackend,
) -> None:
    """
    NeuroOnline 更新通常不是 B=1，
    因此在线接口需要支持 batch。
    """

    signal = make_signal(
        batch_size=16
    )

    tokens = (
        labram_backend
        .encode_online_tokens(
            signal,
            train_backbone=False,
        )
    )

    assert tokens.shape == (
        16,
        4,
        8,
    )

    logits = (
        labram_backend
        .classify_online_tokens(
            tokens
        )
    )

    assert logits.shape == (
        16,
        4,
    )


def test_static_predict_still_rejects_batch_greater_than_one(
    labram_backend: LaBraMBackend,
) -> None:
    """
    解除 _unpack_signal 的 B=1 限制后，
    普通 Runtime 单窗口推理仍必须保持 B=1。
    """

    signal = make_signal(
        batch_size=2
    )

    with pytest.raises(
        ValueError,
        match=(
            "expects one LaBraM window"
        ),
    ):
        labram_backend.predict_tensor(
            signal
        )


def test_online_update_changes_generator_and_head_only(
    labram_backend: LaBraMBackend,
) -> None:
    """
    V1 梯度范围：

    - backbone：冻结；
    - Generator：更新；
    - classification head：更新。
    """

    torch.manual_seed(7)

    signal = make_signal(
        batch_size=8
    )

    labels = torch.tensor(
        [
            0,
            1,
            2,
            3,
            0,
            1,
            2,
            3,
        ],
        dtype=torch.int64,
    )

    generator = TinyGenerator(
        embedding_dim=(
            labram_backend
            .online_feature_spec
            .embedding_dim
        )
    )

    # scope="head" 会：
    # - 冻结 encoder；
    # - 将 head 设为可训练。
    head_parameters = (
        labram_backend
        .get_trainable_parameters(
            scope="head"
        )
    )

    generator.train()
    labram_backend.adapter.head.train()
    labram_backend.adapter.encoder.eval()

    optimizer = torch.optim.AdamW(
        [
            *generator.parameters(),
            *head_parameters,
        ],
        lr=1e-2,
        weight_decay=0.0,
    )

    criterion = nn.CrossEntropyLoss()

    backbone_before = deepcopy(
        labram_backend
        .adapter.encoder.state_dict()
    )

    generator_before = deepcopy(
        generator.state_dict()
    )

    head_before = deepcopy(
        labram_backend
        .adapter.head.state_dict()
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    tokens = (
        labram_backend
        .encode_online_tokens(
            signal,
            train_backbone=False,
        )
    )

    adapted_tokens = generator(
        tokens
    )

    logits = (
        labram_backend
        .classify_online_tokens(
            adapted_tokens
        )
    )

    loss = criterion(
        logits,
        labels,
    )

    loss.backward()

    # Backbone 必须没有梯度。
    assert all(
        parameter.grad is None
        for parameter
        in labram_backend
        .adapter.encoder.parameters()
    )

    # Generator 至少一个参数具有非零梯度。
    assert any(
        parameter.grad is not None
        and torch.count_nonzero(
            parameter.grad
        ).item() > 0
        for parameter
        in generator.parameters()
    )

    # Head 至少一个参数具有非零梯度。
    assert any(
        parameter.grad is not None
        and torch.count_nonzero(
            parameter.grad
        ).item() > 0
        for parameter
        in labram_backend
        .adapter.head.parameters()
    )

    optimizer.step()

    backbone_after = (
        labram_backend
        .adapter.encoder.state_dict()
    )

    generator_after = (
        generator.state_dict()
    )

    head_after = (
        labram_backend
        .adapter.head.state_dict()
    )

    # Backbone 必须完全不变。
    assert_state_dict_equal(
        backbone_before,
        backbone_after,
    )

    # Generator 和 head 应发生变化。
    assert parameters_changed(
        generator_before,
        generator_after,
    )

    assert parameters_changed(
        head_before,
        head_after,
    )