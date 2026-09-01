from __future__ import annotations

import numpy as np
import pytest
import torch

from bci_dayloop.inference.neuroonline_forward import NeuroOnlineForward
from bci_dayloop.models.model_50m.backend import Model50MBackend
from bci_dayloop.models.model_50m.backbone import Model50MBackbone
from bci_dayloop.models.model_50m.classifier import Model50MClassifier
from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.models.model_50m.tokenization import (
    Model50MTokenizer,
    stack_model50m_tokens,
)
from bci_dayloop.models.online_features import OnlineTrainableFeatureBackend
from bci_dayloop.models.neuroonline import NeuroOnlineGenerator
from bci_dayloop.runtime.model import RuntimeModel


class Tiny50MAdapter:
    """使用正式 50M tokenization/backbone/head 的轻量 adapter。"""

    def __init__(self, config: Model50MConfig) -> None:
        self.config = config
        self.tokenizer = Model50MTokenizer(config)
        self.backbone = Model50MBackbone(
            config=config,
            load_checkpoint=False,
            freeze=True,
        )
        self.classifier = Model50MClassifier(
            config=config,
            backbone=self.backbone,
        )

    @property
    def device(self) -> torch.device:
        return self.backbone.device

    @property
    def num_classes(self) -> int:
        return self.config.num_classes

    def _build_model_batch(
        self,
        *,
        X: np.ndarray,
        channel_valid_masks: np.ndarray | None,
    ):
        if channel_valid_masks is None:
            raise ValueError("Test adapter requires channel_valid_masks.")
        signals = np.asarray(X, dtype=np.float32)
        masks = np.asarray(channel_valid_masks, dtype=np.float32)
        if signals.ndim != 3:
            raise ValueError(
                "signals must have shape [B,C,T], got "
                f"{signals.shape}."
            )
        expected_mask_shape = (signals.shape[0], self.config.n_channels)
        if tuple(masks.shape) != expected_mask_shape:
            raise ValueError(
                "channel_valid_masks shape mismatch: expected "
                f"{expected_mask_shape}, got {masks.shape}."
            )
        samples = [
            self.tokenizer.tokenize(
                signal=signals[index],
                channel_valid_mask=masks[index],
            )
            for index in range(signals.shape[0])
        ]
        return stack_model50m_tokens(samples, device=self.device), 0.0, 0.0


@pytest.fixture
def backend() -> Model50MBackend:
    return _make_backend()


def _make_backend(
    *,
    aggregation: str = "flatten",
) -> Model50MBackend:
    torch.manual_seed(42)
    config = Model50MConfig(
        checkpoint_path="unused.pt",
        device="cpu",
        target_sample_rate=2.0,
        window_seconds=2.0,
        patch_seconds=1.0,
        patch_stride_seconds=1.0,
        n_channels=2,
        standard_channels=("C3", "C4"),
        filter_enabled=False,
        zscore_enabled=False,
        d_model=4,
        n_heads=2,
        depth=1,
        mlp_ratio=1.0,
        dropout=0.0,
        model_n_time_patches=2,
        output_layer_idx=0,
        aggregation=aggregation,
        num_classes=3,
    )
    return Model50MBackend(Tiny50MAdapter(config))


def make_input(
    batch_size: int,
    *,
    mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    signal = torch.arange(
        batch_size * 2 * 4,
        dtype=torch.float32,
    ).reshape(batch_size, 2, 4)
    if mask is None:
        mask = torch.ones(batch_size, 2, dtype=torch.float32)
    else:
        signal = signal * mask.unsqueeze(-1)
    return {"signal": signal, "channel_valid_mask": mask}


def _reference_logits(
    backend: Model50MBackend,
    model_input: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """独立保留 50M 原始 backbone -> aggregator -> head 公式。"""
    model_batch = backend._build_batch(model_input)
    backend.adapter.backbone.freeze()
    backend.adapter.classifier.eval()
    with torch.no_grad():
        tokens = backend.adapter.backbone.extract_embeddings(
            batch=model_batch,
            return_layer_idx=backend.adapter.config.output_layer_idx,
        )
        features = backend.adapter.classifier.aggregator(
            token_embeddings=tokens,
            token_valid_mask=model_batch.token_valid_mask,
        )
        return backend.adapter.classifier.head(features), features


def test_50m_backend_implements_online_interface(
    backend: Model50MBackend,
) -> None:
    assert isinstance(backend, OnlineTrainableFeatureBackend)
    assert backend.online_feature_spec.model_name == "50m-linear"
    assert backend.online_feature_spec.token_count == 4
    assert backend.online_feature_spec.embedding_dim == 4


@pytest.mark.parametrize("batch_size", [1, 3])
def test_online_tokens_have_real_batch_token_shape(
    backend: Model50MBackend,
    batch_size: int,
) -> None:
    tokens = backend.encode_online_tokens(make_input(batch_size))
    assert tokens.shape == (batch_size, 4, 4)
    assert tokens.requires_grad is False
    assert not torch.is_inference(tokens)


def test_token_order_is_channel_major_then_time_patch(
    backend: Model50MBackend,
) -> None:
    model_batch = backend._build_batch(make_input(1))
    assert model_batch.token_inputs[0].tolist() == [
        [0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]
    ]
    assert model_batch.token_channel_indices[0].tolist() == [0, 0, 1, 1]
    assert model_batch.token_time_indices[0].tolist() == [0, 1, 0, 1]


def test_static_path_matches_independent_original_formula(
    backend: Model50MBackend,
) -> None:
    model_input = make_input(1, mask=torch.tensor([[1.0, 0.0]]))
    reference_logits, reference_features = _reference_logits(backend, model_input)
    output = backend.predict_tensor(model_input, return_features=True)
    context = backend.encode_online_token_context(model_input)
    online_logits = backend.classify_online_tokens(
        context.tokens,
        token_valid_mask=context.token_valid_mask,
    )
    torch.testing.assert_close(output.logits, reference_logits)
    torch.testing.assert_close(online_logits, reference_logits)
    assert output.features is not None
    torch.testing.assert_close(output.features, reference_features)
    torch.testing.assert_close(output.probabilities, torch.softmax(reference_logits, dim=-1))
    assert output.predicted_class == int(reference_logits.argmax(dim=-1)[0])
    assert output.confidence == pytest.approx(float(torch.softmax(reference_logits, dim=-1).max().item()))


def test_mean_aggregation_remains_mask_aware_and_equivalent() -> None:
    backend = _make_backend(aggregation="mean")
    model_input = make_input(1, mask=torch.tensor([[1.0, 0.0]]))
    reference_logits, _ = _reference_logits(backend, model_input)
    context = backend.encode_online_token_context(model_input)
    online_logits = backend.classify_online_tokens(
        context.tokens,
        token_valid_mask=context.token_valid_mask,
    )
    torch.testing.assert_close(online_logits, reference_logits)


def test_mask_is_explicit_not_mutated_or_cached(
    backend: Model50MBackend,
) -> None:
    first_mask = torch.tensor([[1.0, 0.0]])
    first_context = backend.encode_online_token_context(make_input(1, mask=first_mask))
    assert torch.equal(first_mask, torch.tensor([[1.0, 0.0]]))
    assert first_context.token_valid_mask is not None
    assert first_context.token_valid_mask.tolist() == [[1.0, 1.0, 0.0, 0.0]]
    second_context = backend.encode_online_token_context(
        make_input(1, mask=torch.tensor([[0.0, 1.0]]))
    )
    assert second_context.token_valid_mask is not None
    assert second_context.token_valid_mask.tolist() == [[0.0, 0.0, 1.0, 1.0]]
    first_logits = backend.classify_online_tokens(first_context.tokens, token_valid_mask=first_context.token_valid_mask)
    second_logits = backend.classify_online_tokens(second_context.tokens, token_valid_mask=second_context.token_valid_mask)
    assert not torch.equal(first_logits, second_logits)


def test_head_scope_and_mode_control(backend: Model50MBackend) -> None:
    head_parameters = backend.get_trainable_parameters("head")
    assert {id(parameter) for parameter in head_parameters} == {id(parameter) for parameter in backend.adapter.classifier.head.parameters()}
    assert not {id(parameter) for parameter in head_parameters}.intersection(id(parameter) for parameter in backend.adapter.backbone.parameters())
    assert len(head_parameters) == len({id(parameter) for parameter in head_parameters})
    backend.set_online_mode(training=True, train_backbone=False)
    assert backend.adapter.backbone.model.training is False
    assert backend.adapter.classifier.head.training is True
    assert not any(parameter.requires_grad for parameter in backend.adapter.backbone.parameters())
    assert all(parameter.requires_grad for parameter in head_parameters)
    backend.set_online_mode(training=False, train_backbone=False)
    assert backend.adapter.backbone.model.training is False
    assert backend.adapter.classifier.head.training is False
    assert not any(parameter.requires_grad for parameter in backend.adapter.backbone.parameters())


def test_gradient_reaches_adapted_tokens_and_head_but_not_backbone(
    backend: Model50MBackend,
) -> None:
    backend.set_online_mode(training=True, train_backbone=False)
    context = backend.encode_online_token_context(make_input(2))
    assert context.token_valid_mask is not None
    scale = torch.ones(1, 1, 4, requires_grad=True)
    shift = torch.zeros(1, 1, 4, requires_grad=True)
    logits = backend.classify_online_tokens(
        context.tokens * scale + shift,
        token_valid_mask=context.token_valid_mask,
    )
    logits.sum().backward()
    assert scale.grad is not None
    assert shift.grad is not None
    assert all(parameter.grad is not None for parameter in backend.adapter.classifier.head.parameters())
    assert all(parameter.grad is None for parameter in backend.adapter.backbone.parameters())


def test_neuroonline_forward_uses_explicit_50m_token_context(
    backend: Model50MBackend,
) -> None:
    runtime_model = RuntimeModel(
        canonicalizer=None,  # type: ignore[arg-type]
        input_transform=None,  # type: ignore[arg-type]
        backend=backend,
    )
    generator = NeuroOnlineGenerator(
        feature_spec=backend.online_feature_spec,
        num_subject_codes=2,
        num_attention_heads=2,
        dropout=0.0,
    )
    forward_model = NeuroOnlineForward(
        runtime_model=runtime_model,
        generator=generator,
    )
    model_input = make_input(1, mask=torch.tensor([[1.0, 0.0]]))
    static_output = backend.predict_tensor(model_input)
    forward_result = forward_model.forward_batch(model_input)
    torch.testing.assert_close(forward_result.logits, static_output.logits)


def test_invalid_inputs_fail_with_clear_contract_errors(
    backend: Model50MBackend,
) -> None:
    with pytest.raises(ValueError, match="missing required key 'signal'"):
        backend.encode_online_tokens({"channel_valid_mask": torch.ones(1, 2)})
    with pytest.raises(ValueError, match="missing required key 'channel_valid_mask'"):
        backend.encode_online_tokens({"signal": torch.zeros(1, 2, 4)})
    with pytest.raises(ValueError, match="channel_valid_masks shape mismatch"):
        backend.encode_online_tokens({"signal": torch.zeros(2, 2, 4), "channel_valid_mask": torch.ones(1, 2)})
    with pytest.raises(ValueError, match=r"shape \[B,N,D\]"):
        backend.classify_online_tokens(torch.zeros(1, 4))
    with pytest.raises(ValueError, match="token shape mismatch"):
        backend.classify_online_tokens(torch.zeros(1, 3, 4))
    with pytest.raises(ValueError, match="token shape mismatch"):
        backend.classify_online_tokens(torch.zeros(1, 4, 3))
    with pytest.raises(ValueError, match="only supports trainable scope 'head'"):
        backend.get_trainable_parameters("full")
    with pytest.raises(NotImplementedError, match="train_backbone=True"):
        backend.set_online_mode(training=True, train_backbone=True)
    with pytest.raises(NotImplementedError, match="train_backbone=True"):
        backend.encode_online_tokens(make_input(1), train_backbone=True)
