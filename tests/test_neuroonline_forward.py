from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from bci_dayloop.inference.neuroonline_forward import (
    NeuroOnlineForward,
)
from bci_dayloop.models.base import (
    ModelBackend,
)
from bci_dayloop.models.neuroonline import (
    NeuroOnlineGenerator,
)
from bci_dayloop.models.online_features import (
    OnlineFeatureSpec,
    OnlineTrainableFeatureBackend,
)
from bci_dayloop.runtime.model import (
    RuntimeModel,
)
from bci_dayloop.runtime.types import (
    CanonicalEEGWindow,
    ModelOutput,
    ModelTensor,
    PreparedModelInput,
)


class TinyOnlineBackend(ModelBackend):
    """
    用于测试 NeuroOnlineForward 的轻量 Backend。

    这里不加载真实 LaBraM/CBraMod。

    输入直接使用：
        tokens: [B,N,D]

    分类路径：
        tokens
        -> mean pooling
        -> Linear head
        -> logits
    """

    def __init__(
        self,
        *,
        token_count: int = 6,
        embedding_dim: int = 8,
        num_classes: int = 4,
    ) -> None:
        self._device = torch.device(
            "cpu"
        )

        self._num_classes = int(
            num_classes
        )

        self._feature_spec = (
            OnlineFeatureSpec(
                model_name="tiny-online-backend",
                token_count=token_count,
                embedding_dim=embedding_dim,
            )
        )

        self.head = nn.Linear(
            embedding_dim,
            num_classes,
        ).to(
            self._device
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def online_feature_spec(
        self,
    ) -> OnlineFeatureSpec:
        return self._feature_spec

    @staticmethod
    def _unpack_tokens(
        model_input: ModelTensor,
    ) -> torch.Tensor:
        if isinstance(
            model_input,
            torch.Tensor,
        ):
            tokens = model_input

        elif isinstance(
            model_input,
            dict,
        ):
            if "signal" not in model_input:
                raise ValueError(
                    "TinyOnlineBackend requires "
                    "model_input['signal']."
                )

            tokens = model_input["signal"]

        else:
            raise TypeError(
                "model_input must be Tensor or "
                "dict[str, Tensor]."
            )

        if not isinstance(
            tokens,
            torch.Tensor,
        ):
            raise TypeError(
                "TinyOnlineBackend tokens must be "
                "torch.Tensor."
            )

        return tokens

    def _validate_tokens(
        self,
        tokens: torch.Tensor,
    ) -> None:
        expected_tail = (
            self.online_feature_spec.token_count,
            self.online_feature_spec.embedding_dim,
        )

        if tokens.ndim != 3:
            raise ValueError(
                "TinyOnlineBackend expects "
                "[B,N,D], got "
                f"{tuple(tokens.shape)}."
            )

        if tuple(
            tokens.shape[1:]
        ) != expected_tail:
            raise ValueError(
                "TinyOnlineBackend token shape "
                "mismatch: expected "
                f"[B,{expected_tail[0]},"
                f"{expected_tail[1]}], got "
                f"{tuple(tokens.shape)}."
            )

        if tokens.shape[0] <= 0:
            raise ValueError(
                "Batch size must be positive."
            )

        if not torch.isfinite(
            tokens
        ).all():
            raise ValueError(
                "Tokens contain NaN or Inf."
            )

    def encode_online_tokens(
        self,
        model_input: ModelTensor,
        *,
        train_backbone: bool = False,
    ) -> torch.Tensor:
        """
        测试 Backend 没有真正的 backbone，
        所以直接把输入当作 tokens。
        """

        del train_backbone

        tokens = self._unpack_tokens(
            model_input
        ).to(
            self.device,
            dtype=torch.float32,
        )

        self._validate_tokens(tokens)

        return tokens

    def classify_online_tokens(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_tokens(tokens)

        pooled = tokens.mean(
            dim=1
        )

        return self.head(pooled)

    def set_online_mode(
        self,
        *,
        training: bool,
        train_backbone: bool = False,
    ) -> None:
        del train_backbone

        self.head.train(
            mode=training
        )

    def predict_tensor(
        self,
        model_input: ModelTensor,
        return_features: bool = False,
    ) -> ModelOutput:
        """
        不经过 Generator 的静态基线预测。
        """

        tokens = self.encode_online_tokens(
            model_input,
            train_backbone=False,
        )

        if tokens.shape[0] != 1:
            raise ValueError(
                "Static prediction expects "
                "batch_size=1."
            )

        self.set_online_mode(
            training=False,
            train_backbone=False,
        )

        with torch.no_grad():
            logits = (
                self.classify_online_tokens(
                    tokens
                )
            )

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            confidence, prediction = (
                probabilities.max(dim=-1)
            )

        return ModelOutput(
            logits=logits,
            probabilities=probabilities,
            predicted_class=int(
                prediction[0].item()
            ),
            confidence=float(
                confidence[0].item()
            ),
            features=(
                tokens.mean(dim=1)
                if return_features
                else None
            ),
            diagnostics={
                "backend": (
                    "tiny-online-backend"
                ),
                "online_strategy": "none",
            },
        )

    def encode_tensor(
        self,
        model_input: ModelTensor,
    ) -> torch.Tensor:
        tokens = self.encode_online_tokens(
            model_input,
            train_backbone=False,
        )

        return tokens.mean(dim=1)

    def get_trainable_parameters(
        self,
        scope: str,
    ) -> list[torch.nn.Parameter]:
        if scope.strip().lower() != "head":
            raise ValueError(
                "TinyOnlineBackend only supports "
                "scope='head'."
            )

        for parameter in (
            self.head.parameters()
        ):
            parameter.requires_grad = True

        self.head.train()

        return list(
            self.head.parameters()
        )


@pytest.fixture
def backend() -> TinyOnlineBackend:
    """
    每个测试使用全新 Backend，
    避免测试之间共享参数状态。
    """

    torch.manual_seed(42)

    return TinyOnlineBackend(
        token_count=6,
        embedding_dim=8,
        num_classes=4,
    )


@pytest.fixture
def runtime_model(
    backend: TinyOnlineBackend,
) -> RuntimeModel:
    """
    NeuroOnlineForward 这里只会使用
    runtime_model.backend。

    canonicalizer 和 input_transform
    在本测试中不会被调用。
    """

    return RuntimeModel(
        canonicalizer=None,  # type: ignore[arg-type]
        input_transform=None,  # type: ignore[arg-type]
        backend=backend,
    )


@pytest.fixture
def generator(
    backend: TinyOnlineBackend,
) -> NeuroOnlineGenerator:
    torch.manual_seed(123)

    model = NeuroOnlineGenerator(
        feature_spec=(
            backend.online_feature_spec
        ),
        num_subject_codes=4,
        num_attention_heads=2,
        dropout=0.0,
    ).to(
        backend.device
    )

    model.eval()

    return model


@pytest.fixture
def forward_model(
    runtime_model: RuntimeModel,
    generator: NeuroOnlineGenerator,
) -> NeuroOnlineForward:
    return NeuroOnlineForward(
        runtime_model=runtime_model,
        generator=generator,
    )


def make_tokens(
    *,
    batch_size: int,
) -> torch.Tensor:
    """
    构造确定性的 [B,6,8] tokens。
    """

    random_generator = (
        torch.Generator()
    )

    random_generator.manual_seed(999)

    return torch.randn(
        batch_size,
        6,
        8,
        generator=random_generator,
        dtype=torch.float32,
    )


def make_prepared_input(
    tokens: torch.Tensor,
) -> PreparedModelInput:
    """
    构造 predict_prepared() 需要的
    PreparedModelInput。

    canonical_window 在当前测试中不会参与计算，
    但为了满足正式数据结构仍然提供。
    """

    canonical_window = (
        CanonicalEEGWindow(
            data=np.zeros(
                (1, 1),
                dtype=np.float32,
            ),
            channel_names=["CZ"],
            sample_rate=1.0,
            unit="uV",
        )
    )

    return PreparedModelInput(
        model_input=tokens,
        canonical_window=(
            canonical_window
        ),
        preprocessing_trace=[
            "unit-test"
        ],
        diagnostics={
            "source": (
                "test_neuroonline_forward"
            )
        },
    )


def test_backend_implements_online_protocol(
    backend: TinyOnlineBackend,
) -> None:
    """
    先确认测试 Backend 本身满足接口，
    避免后面的失败来自测试替身写错。
    """

    assert isinstance(
        backend,
        OnlineTrainableFeatureBackend,
    )

    assert (
        backend
        .online_feature_spec
        .token_count
        == 6
    )

    assert (
        backend
        .online_feature_spec
        .embedding_dim
        == 8
    )


def test_generator_identity_initialization(
    generator: NeuroOnlineGenerator,
) -> None:
    """
    验证新创建的 Generator 严格满足：

        alpha = 1
        beta = 0
        adapted_tokens = tokens
    """

    tokens = make_tokens(
        batch_size=3
    )

    with torch.no_grad():
        (
            alpha,
            beta,
            router_probs,
        ) = generator(
            tokens,
            return_aux=True,
        )

        adapted_tokens = (
            tokens * alpha + beta
        )

    assert alpha.shape == (
        3,
        6,
        8,
    )

    assert beta.shape == (
        3,
        6,
        8,
    )

    assert router_probs.shape == (
        3,
        4,
    )

    assert (
        generator
        .gate_alpha
        .detach()
        .item()
        == 0.0
    )

    assert (
        generator
        .gate_beta
        .detach()
        .item()
        == 0.0
    )

    torch.testing.assert_close(
        alpha,
        torch.ones_like(alpha),
        rtol=0.0,
        atol=0.0,
    )

    torch.testing.assert_close(
        beta,
        torch.zeros_like(beta),
        rtol=0.0,
        atol=0.0,
    )

    torch.testing.assert_close(
        adapted_tokens,
        tokens,
        rtol=0.0,
        atol=0.0,
    )

    torch.testing.assert_close(
        router_probs.sum(dim=-1),
        torch.ones(3),
        rtol=1e-6,
        atol=1e-6,
    )


def test_forward_batch_identity_preserves_logits(
    backend: TinyOnlineBackend,
    generator: NeuroOnlineGenerator,
    forward_model: NeuroOnlineForward,
) -> None:
    """
    验证经过完整 Forward 后：

        tokens
        -> Generator
        -> adapted_tokens
        -> head

    得到的 logits 与不经过 Generator 相同。
    """

    tokens = make_tokens(
        batch_size=5
    )

    backend.set_online_mode(
        training=False,
        train_backbone=False,
    )

    generator.eval()

    with torch.no_grad():
        baseline_logits = (
            backend
            .classify_online_tokens(
                tokens
            )
        )

        result = (
            forward_model
            .forward_batch(
                tokens,
                train_backbone=False,
            )
        )

    assert result.logits.shape == (
        5,
        4,
    )

    assert (
        result.original_tokens.shape
        == (5, 6, 8)
    )

    assert (
        result.adapted_tokens.shape
        == (5, 6, 8)
    )

    torch.testing.assert_close(
        result.original_tokens,
        tokens,
        rtol=0.0,
        atol=0.0,
    )

    torch.testing.assert_close(
        result.alpha,
        torch.ones_like(
            result.alpha
        ),
        rtol=0.0,
        atol=0.0,
    )

    torch.testing.assert_close(
        result.beta,
        torch.zeros_like(
            result.beta
        ),
        rtol=0.0,
        atol=0.0,
    )

    torch.testing.assert_close(
        result.adapted_tokens,
        result.original_tokens,
        rtol=0.0,
        atol=0.0,
    )

    torch.testing.assert_close(
        result.logits,
        baseline_logits,
        rtol=1e-6,
        atol=1e-6,
    )


def test_predict_prepared_identity_matches_static_prediction(
    backend: TinyOnlineBackend,
    forward_model: NeuroOnlineForward,
) -> None:
    """
    验证 Runtime 单窗口预测层面的等价性：

        backend.predict_tensor()
        ==
        NeuroOnlineForward.predict_prepared()
    """

    tokens = make_tokens(
        batch_size=1
    )

    prepared = make_prepared_input(
        tokens
    )

    baseline_output = (
        backend.predict_tensor(
            tokens,
            return_features=False,
        )
    )

    neuroonline_output = (
        forward_model
        .predict_prepared(
            prepared,
            return_features=True,
        )
    )

    torch.testing.assert_close(
        neuroonline_output.logits,
        baseline_output.logits,
        rtol=1e-6,
        atol=1e-6,
    )

    torch.testing.assert_close(
        neuroonline_output.probabilities,
        baseline_output.probabilities,
        rtol=1e-6,
        atol=1e-6,
    )

    assert (
        neuroonline_output
        .predicted_class
        == baseline_output
        .predicted_class
    )

    assert (
        neuroonline_output.confidence
        == pytest.approx(
            baseline_output.confidence,
            abs=1e-7,
        )
    )

    assert (
        neuroonline_output.features
        is not None
    )

    # NeuroOnlineForward 的 return_features
    # 返回的是适配后的全部 tokens。
    torch.testing.assert_close(
        neuroonline_output.features,
        tokens,
        rtol=0.0,
        atol=0.0,
    )

    assert (
        neuroonline_output
        .diagnostics[
            "online_strategy"
        ]
        == "neuroonline"
    )

    assert (
        neuroonline_output
        .diagnostics["backend"]
        == "tiny-online-backend"
    )

    assert (
        neuroonline_output
        .diagnostics["token_shape"]
        == [1, 6, 8]
    )

    assert (
        neuroonline_output
        .diagnostics["gate_alpha"]
        == 0.0
    )

    assert (
        neuroonline_output
        .diagnostics["gate_beta"]
        == 0.0
    )


def test_forward_rejects_mismatched_generator_spec(
    runtime_model: RuntimeModel,
) -> None:
    """
    Generator 与 backend 的 N/D 不匹配时，
    应在构建 Forward 时立即失败。
    """

    wrong_generator = (
        NeuroOnlineGenerator(
            feature_spec=(
                OnlineFeatureSpec(
                    model_name=(
                        "wrong-backend"
                    ),
                    token_count=5,
                    embedding_dim=8,
                )
            ),
            num_subject_codes=4,
            num_attention_heads=2,
            dropout=0.0,
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "does not match Runtime backend"
        ),
    ):
        NeuroOnlineForward(
            runtime_model=runtime_model,
            generator=wrong_generator,
        )