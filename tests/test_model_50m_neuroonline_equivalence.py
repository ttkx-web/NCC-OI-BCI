from __future__ import annotations

import pytest
import torch

from bci_dayloop.inference.neuroonline_forward import (
    build_neuroonline_forward,
)

from model_50m_neuroonline_support import (
    build_backend,
    build_runtime_model,
    independent_static_reference,
    make_batched_model_input,
    make_model_input,
    make_prepared_input,
)


@pytest.mark.parametrize(
    ("model_input", "expected_mask"),
    [
        (
            make_model_input(
                values=(1.0, 10.0),
                mask=(1.0, 1.0),
            ),
            [[1.0, 1.0, 1.0, 1.0]],
        ),
        (
            make_model_input(
                values=(2.0, 20.0),
                mask=(1.0, 0.0),
            ),
            [[1.0, 1.0, 0.0, 0.0]],
        ),
    ],
)
def test_50m_online_split_matches_independent_static_reference(
    model_input: dict[str, torch.Tensor],
    expected_mask: list[list[float]],
) -> None:
    backend = build_backend()
    reference_logits, _, reference_mask = independent_static_reference(
        backend,
        model_input,
    )
    context = backend.encode_online_token_context(model_input)
    new_logits = backend.classify_online_tokens(
        context.tokens,
        token_valid_mask=context.token_valid_mask,
    )

    torch.testing.assert_close(
        new_logits,
        reference_logits,
        rtol=1e-5,
        atol=1e-6,
    )
    assert context.token_valid_mask is not None
    torch.testing.assert_close(
        context.token_valid_mask,
        reference_mask,
        rtol=0.0,
        atol=0.0,
    )
    assert context.token_valid_mask.tolist() == expected_mask

    reference_probabilities = torch.softmax(reference_logits, dim=-1)
    new_probabilities = torch.softmax(new_logits, dim=-1)
    torch.testing.assert_close(
        new_probabilities,
        reference_probabilities,
        rtol=1e-5,
        atol=1e-6,
    )
    assert int(new_logits.argmax(dim=-1)[0]) == int(
        reference_logits.argmax(dim=-1)[0]
    )
    assert float(new_probabilities.max()) == pytest.approx(
        float(reference_probabilities.max()),
        abs=1e-7,
    )


def test_50m_online_split_supports_batched_independent_reference() -> None:
    backend = build_backend()
    input_a = make_model_input(values=(1.0, 10.0), mask=(1.0, 0.0))
    input_b = make_model_input(values=(3.0, 30.0), mask=(0.0, 1.0))
    batched_input = make_batched_model_input(input_a, input_b)
    reference_logits, _, reference_mask = independent_static_reference(
        backend,
        batched_input,
    )
    context = backend.encode_online_token_context(batched_input)
    new_logits = backend.classify_online_tokens(
        context.tokens,
        token_valid_mask=context.token_valid_mask,
    )

    assert context.tokens.shape == (2, 4, 4)
    assert context.token_valid_mask is not None
    assert context.token_valid_mask.shape == (2, 4)
    torch.testing.assert_close(context.token_valid_mask, reference_mask)
    torch.testing.assert_close(
        new_logits,
        reference_logits,
        rtol=1e-5,
        atol=1e-6,
    )
    reference_probabilities = torch.softmax(reference_logits, dim=-1)
    new_probabilities = torch.softmax(new_logits, dim=-1)
    torch.testing.assert_close(
        new_probabilities,
        reference_probabilities,
        rtol=1e-5,
        atol=1e-6,
    )
    assert new_logits.argmax(dim=-1).tolist() == reference_logits.argmax(
        dim=-1
    ).tolist()
    torch.testing.assert_close(
        new_probabilities.max(dim=-1).values,
        reference_probabilities.max(dim=-1).values,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "model_input",
    [
        make_model_input(values=(1.0, 10.0), mask=(1.0, 1.0)),
        make_model_input(values=(2.0, 20.0), mask=(1.0, 0.0)),
    ],
)
def test_official_generator_identity_matches_static_prediction(
    model_input: dict[str, torch.Tensor],
) -> None:
    backend = build_backend()
    runtime_model = build_runtime_model(backend)
    forward_model = build_neuroonline_forward(
        runtime_model=runtime_model,
        num_subject_codes=2,
        num_attention_heads=2,
        dropout=0.0,
    )
    prepared = make_prepared_input(model_input, trial_id="identity")

    backend.set_online_mode(training=False, train_backbone=False)
    forward_model.generator.eval()
    static_output = runtime_model.predict_prepared(prepared)
    online_output = forward_model.predict_prepared(prepared)
    result = forward_model.forward_batch(
        prepared.model_input,
        train_backbone=False,
    )

    torch.testing.assert_close(
        online_output.logits,
        static_output.logits,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        online_output.probabilities,
        static_output.probabilities,
        rtol=1e-5,
        atol=1e-6,
    )
    assert online_output.predicted_class == static_output.predicted_class
    assert online_output.confidence == pytest.approx(
        static_output.confidence,
        abs=1e-7,
    )
    assert forward_model.generator.gate_alpha.item() == 0.0
    assert forward_model.generator.gate_beta.item() == 0.0
    torch.testing.assert_close(
        result.alpha,
        torch.ones_like(result.alpha),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.beta,
        torch.zeros_like(result.beta),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.original_tokens,
        result.adapted_tokens,
        rtol=0.0,
        atol=0.0,
    )


def test_identity_forward_preserves_batched_tokens_and_per_sample_masks() -> None:
    backend = build_backend()
    runtime_model = build_runtime_model(backend)
    forward_model = build_neuroonline_forward(
        runtime_model=runtime_model,
        num_subject_codes=2,
        num_attention_heads=2,
        dropout=0.0,
    )
    batched_input = make_batched_model_input(
        make_model_input(values=(1.0, 10.0), mask=(1.0, 0.0)),
        make_model_input(values=(3.0, 30.0), mask=(0.0, 1.0)),
    )
    result = forward_model.forward_batch(
        batched_input,
        train_backbone=False,
    )

    assert result.original_tokens.shape == (2, 4, 4)
    assert result.adapted_tokens.shape == (2, 4, 4)
    assert result.logits.shape == (2, 3)
    assert forward_model.generator.gate_alpha.item() == 0.0
    assert forward_model.generator.gate_beta.item() == 0.0
    torch.testing.assert_close(
        result.alpha,
        torch.ones_like(result.alpha),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.beta,
        torch.zeros_like(result.beta),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.original_tokens,
        result.adapted_tokens,
        rtol=0.0,
        atol=0.0,
    )
    reference_logits, _, _ = independent_static_reference(
        backend,
        batched_input,
    )
    torch.testing.assert_close(
        result.logits,
        reference_logits,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        torch.softmax(result.logits, dim=-1),
        torch.softmax(reference_logits, dim=-1),
        rtol=1e-5,
        atol=1e-6,
    )


def test_channel_valid_mask_is_batch_aligned_and_has_no_cross_call_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = build_backend()
    captured_masks: list[torch.Tensor] = []
    original_extract = backend.adapter.backbone.extract_embeddings

    def spy_extract(*args: object, **kwargs: object) -> torch.Tensor:
        batch = kwargs["batch"]
        captured_masks.append(batch.token_valid_mask.detach().clone())
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(
        backend.adapter.backbone,
        "extract_embeddings",
        spy_extract,
    )

    mask_a = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    first_input = make_batched_model_input(
        make_model_input(values=(1.0, 10.0), mask=(1.0, 1.0)),
        make_model_input(values=(2.0, 20.0), mask=(1.0, 0.0)),
    )
    input_mask_before = first_input["channel_valid_mask"].clone()
    first_context = backend.encode_online_token_context(first_input)

    second_context = backend.encode_online_token_context(
        make_model_input(values=(3.0, 30.0), mask=(0.0, 1.0))
    )

    torch.testing.assert_close(first_input["channel_valid_mask"], input_mask_before)
    torch.testing.assert_close(first_input["channel_valid_mask"], mask_a)
    assert len(captured_masks) == 2
    assert captured_masks[0].tolist() == [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
    ]
    assert captured_masks[1].tolist() == [[0.0, 0.0, 1.0, 1.0]]
    assert first_context.token_valid_mask is not None
    assert second_context.token_valid_mask is not None
    assert first_context.token_valid_mask.tolist() == captured_masks[0].tolist()
    assert second_context.token_valid_mask.tolist() == captured_masks[1].tolist()
