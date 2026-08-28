from __future__ import annotations

import math
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from bci_dayloop.inference.neuroonline_strategy import (
    NeuroOnlineConfig,
    NeuroOnlineStrategy,
)
from bci_dayloop.runtime.adaptation_types import (
    AdaptationContext,
    FeedbackEvent,
    OnlineObservation,
)

from model_50m_neuroonline_support import (
    build_backend,
    build_runtime_model,
    changed_parameter_names,
    clone_state_dict,
    make_batched_model_input,
    make_model_input,
    make_prepared_input,
)


def build_strategy(
    update_scope: str = "generator_and_head",
) -> tuple[NeuroOnlineStrategy, object, object]:
    backend = build_backend()
    runtime_model = build_runtime_model(backend)
    strategy = NeuroOnlineStrategy(
        NeuroOnlineConfig(
            update_scope=update_scope,
            num_subject_codes=2,
            num_attention_heads=2,
            dropout=0.0,
            learning_rate=1e-2,
            weight_decay=0.0,
            warmup_feedback=2,
            update_interval=1,
            recent_buffer_size=4,
            batch_size=2,
            epochs_per_update=1,
            max_pending_observations=8,
            seed=42,
        )
    )
    strategy.initialize(
        runtime_model=runtime_model,
        context=AdaptationContext(run_id="50m-stage-b"),
    )
    return strategy, runtime_model, backend


def apply_one_update(
    strategy: NeuroOnlineStrategy,
    runtime_model: object,
) -> object:
    for observation_id, values, mask, label in (
        ("scope-a", (1.0, 10.0), (1.0, 0.0), 0),
        ("scope-b", (3.0, 30.0), (0.0, 1.0), 1),
    ):
        prepared = make_prepared_input(
            make_model_input(values=values, mask=mask),
            trial_id=observation_id,
        )
        output = strategy.predict_prepared(prepared)
        strategy.observe(
            OnlineObservation(
                observation_id=observation_id,
                prepared_input=prepared,
                output=output,
                timestamp_sec=0.0,
            )
        )
        strategy.submit_feedback(
            FeedbackEvent(observation_id=observation_id, label=label)
        )
    return strategy.maybe_update(runtime_model=runtime_model)  # type: ignore[arg-type]


def assert_state_dict_unchanged(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> None:
    assert before.keys() == after.keys()
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0.0, atol=0.0)


def test_optimizer_and_manual_training_gradients_respect_50m_freeze_boundary() -> None:
    strategy, runtime_model, backend = build_strategy()
    optimizer = strategy._optimizer
    assert optimizer is not None

    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    generator_parameters = list(strategy.generator.parameters())
    head_parameters = list(backend.adapter.classifier.head.parameters())
    backbone_parameters = list(backend.adapter.backbone.parameters())
    expected_ids = {id(parameter) for parameter in [*generator_parameters, *head_parameters]}

    assert optimizer_ids == expected_ids
    assert len(optimizer_ids) == len(generator_parameters) + len(head_parameters)
    assert not optimizer_ids.intersection(id(parameter) for parameter in backbone_parameters)
    assert all(parameter.requires_grad for parameter in generator_parameters)
    assert all(parameter.requires_grad for parameter in head_parameters)
    assert not any(parameter.requires_grad for parameter in backbone_parameters)
    assert backend.adapter.backbone.training is False
    assert backend.adapter.backbone.model.training is False

    batched_input = make_batched_model_input(
        make_model_input(values=(1.0, 10.0), mask=(1.0, 0.0)),
        make_model_input(values=(3.0, 30.0), mask=(0.0, 1.0)),
    )
    strategy.forward_model.backend.set_online_mode(training=True, train_backbone=False)
    strategy.generator.train()
    optimizer.zero_grad(set_to_none=True)
    result = strategy.forward_model.forward_batch(batched_input, train_backbone=False)
    loss = F.cross_entropy(result.logits, torch.tensor([0, 1]))
    assert torch.isfinite(loss)
    loss.backward()

    generator_grads = [parameter.grad for parameter in generator_parameters]
    head_grads = [parameter.grad for parameter in head_parameters]
    assert any(gradient is not None and torch.isfinite(gradient).all() for gradient in generator_grads)
    assert any(gradient is not None and torch.isfinite(gradient).all() for gradient in head_grads)
    assert all(parameter.grad is None for parameter in backbone_parameters)
    assert backend.adapter.backbone.training is False
    assert backend.adapter.backbone.model.training is False

    strategy.forward_model.backend.set_online_mode(training=False, train_backbone=False)
    strategy.generator.eval()


def test_strategy_update_changes_generator_and_head_only_with_batched_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy, runtime_model, backend = build_strategy()
    prepared_a = make_prepared_input(
        make_model_input(values=(1.0, 10.0), mask=(1.0, 0.0)),
        trial_id="trial-a",
    )
    prepared_b = make_prepared_input(
        make_model_input(values=(3.0, 30.0), mask=(0.0, 1.0)),
        trial_id="trial-b",
    )

    generator_before = clone_state_dict(strategy.generator)
    head_before = clone_state_dict(backend.adapter.classifier.head)
    backbone_before = clone_state_dict(backend.adapter.backbone)
    old_revision = strategy.model_revision
    forwarded_batches: list[dict[str, torch.Tensor]] = []
    original_forward_batch = strategy.forward_model.forward_batch

    for observation_id, prepared, label in (
        ("obs-a", prepared_a, 0),
        ("obs-b", prepared_b, 1),
    ):
        output = strategy.predict_prepared(prepared)
        strategy.observe(
            OnlineObservation(
                observation_id=observation_id,
                prepared_input=prepared,
                output=output,
                timestamp_sec=0.0,
            )
        )
        strategy.submit_feedback(
            FeedbackEvent(observation_id=observation_id, label=label)
        )

    assert strategy.buffered_sample_count == 2
    assert strategy.pending_observation_count == 0

    def spy_forward_batch(
        model_input: dict[str, torch.Tensor],
        *,
        train_backbone: bool = False,
    ) -> Any:
        forwarded_batches.append(
            {
                key: value.detach().clone()
                for key, value in model_input.items()
            }
        )
        return original_forward_batch(
            model_input,
            train_backbone=train_backbone,
        )

    monkeypatch.setattr(
        strategy.forward_model,
        "forward_batch",
        spy_forward_batch,
    )
    update_result = strategy.maybe_update(runtime_model=runtime_model)

    assert update_result.applied is True
    assert update_result.update_step == 1
    assert strategy.update_step == 1
    assert update_result.model_revision == "neuroonline-1"
    assert strategy.model_revision != old_revision
    assert update_result.samples_used == 2
    assert math.isfinite(float(update_result.metrics["loss"]))
    assert math.isfinite(float(update_result.metrics["last_gradient_norm"]))
    assert update_result.metrics["batches"] == 1
    assert len(forwarded_batches) == 1
    assert forwarded_batches[0]["signal"].shape == (2, 2, 4)
    assert forwarded_batches[0]["channel_valid_mask"].tolist() == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    assert strategy.generator.training is False
    assert backend.adapter.classifier.head.training is False
    assert backend.adapter.backbone.training is False
    assert backend.adapter.backbone.model.training is False

    assert changed_parameter_names(strategy.generator, generator_before)
    assert changed_parameter_names(backend.adapter.classifier.head, head_before)
    assert_state_dict_unchanged(backbone_before, clone_state_dict(backend.adapter.backbone))
    assert all(parameter.grad is None for parameter in backend.adapter.backbone.parameters())

    updated_output = strategy.predict_prepared(prepared_a)
    assert strategy.update_step == 1
    assert strategy.model_revision == "neuroonline-1"
    assert torch.isfinite(updated_output.probabilities).all()
    torch.testing.assert_close(
        updated_output.probabilities.sum(dim=-1),
        torch.ones(1),
        rtol=1e-6,
        atol=1e-6,
    )


def test_default_update_scope_preserves_current_generator_and_head_behavior() -> None:
    assert NeuroOnlineConfig().update_scope == "generator_and_head"
    strategy, _, _ = build_strategy()
    assert strategy.parameter_audit["update_scope"] == "generator_and_head"
    assert strategy.parameter_audit["generator_trainable_param_count"] > 0
    assert strategy.parameter_audit["head_trainable_param_count"] > 0


@pytest.mark.parametrize(
    ("scope", "generator_changes", "head_changes"),
    (
        ("generator_and_head", True, True),
        ("generator_only", True, False),
        ("head_only", False, True),
    ),
)
def test_update_scope_optimizer_freezing_and_identity_gate(
    scope: str,
    generator_changes: bool,
    head_changes: bool,
) -> None:
    strategy, runtime_model, backend = build_strategy(scope)
    generator_parameters = list(strategy.generator.parameters())
    head_parameters = list(backend.adapter.classifier.head.parameters())
    backbone_parameters = list(backend.adapter.backbone.parameters())
    optimizer = strategy._optimizer
    assert optimizer is not None
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected_ids = {
        id(parameter)
        for parameter in (
            ([*generator_parameters] if generator_changes else [])
            + ([*head_parameters] if head_changes else [])
        )
    }
    assert optimizer_ids == expected_ids
    assert not any(parameter.requires_grad for parameter in backbone_parameters)
    audit = strategy.parameter_audit
    assert audit["backbone_trainable_param_count"] == 0
    assert (audit["generator_trainable_param_count"] > 0) is generator_changes
    assert (audit["head_trainable_param_count"] > 0) is head_changes

    prepared = make_prepared_input(
        make_model_input(values=(1.0, 10.0), mask=(1.0, 0.0)),
        trial_id="identity",
    )
    static_output = runtime_model.predict_prepared(prepared)
    online_output = strategy.predict_prepared(prepared)
    assert online_output.predicted_class == static_output.predicted_class
    torch.testing.assert_close(
        online_output.probabilities,
        static_output.probabilities,
        rtol=0.0,
        atol=0.0,
    )

    generator_before = clone_state_dict(strategy.generator)
    head_before = clone_state_dict(backend.adapter.classifier.head)
    backbone_before = clone_state_dict(backend.adapter.backbone)
    update = apply_one_update(strategy, runtime_model)
    assert update.applied is True
    assert bool(changed_parameter_names(strategy.generator, generator_before)) is generator_changes
    assert bool(
        changed_parameter_names(backend.adapter.classifier.head, head_before)
    ) is head_changes
    assert_state_dict_unchanged(
        backbone_before,
        clone_state_dict(backend.adapter.backbone),
    )
    assert update.metrics["update_scope"] == scope


def test_invalid_update_scope_fails_closed() -> None:
    with pytest.raises(ValueError, match="update_scope"):
        NeuroOnlineConfig(update_scope="everything")
