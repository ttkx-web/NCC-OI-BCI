from __future__ import annotations

from collections.abc import Sequence

import pytest

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
    make_model_input,
    make_prepared_input,
)


def build_strategy(
    *,
    num_classes: int = 3,
    update_trigger: str = "feedback_count",
    min_feedback_per_class: int = 1,
    memory_strategy: str = "recent_fifo",
    balanced_memory_per_class: int = 2,
) -> tuple[NeuroOnlineStrategy, object]:
    backend = build_backend(num_classes=num_classes)
    runtime = build_runtime_model(backend)
    strategy = NeuroOnlineStrategy(
        NeuroOnlineConfig(
            num_subject_codes=2,
            num_attention_heads=2,
            dropout=0.0,
            update_trigger=update_trigger,
            min_feedback_per_class=min_feedback_per_class,
            memory_strategy=memory_strategy,
            balanced_memory_per_class=balanced_memory_per_class,
            warmup_feedback=2,
            update_interval=1,
            recent_buffer_size=4,
            batch_size=2,
            epochs_per_update=1,
            max_pending_observations=16,
            seed=42,
        )
    )
    strategy.initialize(
        runtime_model=runtime,
        context=AdaptationContext(run_id="balanced-memory-test"),
    )
    return strategy, runtime


def reveal(
    strategy: NeuroOnlineStrategy,
    runtime: object,
    *,
    ordinal: int,
    label: int,
    submit: bool = True,
) -> None:
    observation_id = f"trial-{ordinal}-label-{label}"
    prepared = make_prepared_input(
        make_model_input(
            values=(float(ordinal), float(ordinal + 100)),
            mask=(1.0, 1.0),
        ),
        trial_id=observation_id,
    )
    output = strategy.predict_prepared(prepared)
    strategy.observe(
        OnlineObservation(
            observation_id=observation_id,
            prepared_input=prepared,
            output=output,
            timestamp_sec=float(ordinal),
        )
    )
    if submit:
        strategy.submit_feedback(
            FeedbackEvent(
                observation_id=observation_id,
                label=label,
                metadata={"trial_ordinal": ordinal},
            )
        )


def reveal_many(
    strategy: NeuroOnlineStrategy,
    runtime: object,
    labels: Sequence[int],
) -> None:
    for ordinal, label in enumerate(labels, start=1):
        reveal(strategy, runtime, ordinal=ordinal, label=label)


def test_default_config_exactly_preserves_v1_trigger_and_memory() -> None:
    config = NeuroOnlineConfig()
    assert config.update_trigger == "feedback_count"
    assert config.memory_strategy == "recent_fifo"
    assert config.warmup_feedback == 32
    assert config.recent_buffer_size == 64
    assert config.update_scope == "generator_and_head"


def test_class_coverage_waits_for_every_dynamic_class_and_revealed_labels_only() -> None:
    strategy, runtime = build_strategy(
        num_classes=4,
        update_trigger="class_coverage",
        min_feedback_per_class=2,
    )
    reveal_many(strategy, runtime, [0, 0, 0, 1, 1, 2, 2, 3])
    waiting = strategy.maybe_update(runtime_model=runtime)  # type: ignore[arg-type]
    assert waiting.applied is False
    assert waiting.metrics["class_coverage_satisfied"] is False
    assert waiting.metrics["label_counts_seen"] == {
        "0": 3,
        "1": 2,
        "2": 2,
        "3": 1,
    }

    # Merely observing a future trial cannot affect class coverage.
    reveal(strategy, runtime, ordinal=9, label=3, submit=False)
    still_waiting = strategy.maybe_update(runtime_model=runtime)  # type: ignore[arg-type]
    assert still_waiting.applied is False
    assert still_waiting.metrics["label_counts_seen"]["3"] == 1

    strategy.submit_feedback(
        FeedbackEvent(
            observation_id="trial-9-label-3",
            label=3,
            metadata={"trial_ordinal": 9},
        )
    )
    applied = strategy.maybe_update(runtime_model=runtime)  # type: ignore[arg-type]
    assert applied.applied is True
    assert applied.metrics["class_coverage_satisfied"] is True
    assert applied.metrics["first_update_evaluation_ordinal"] == 9
    assert applied.metrics["feedback_count_total"] == 9


def test_class_balanced_memory_is_independently_bounded_and_non_evicting() -> None:
    strategy, runtime = build_strategy(
        memory_strategy="class_balanced_history",
        balanced_memory_per_class=2,
    )
    reveal_many(strategy, runtime, [0, 1, 2, 0, 0, 0, 0])
    histogram = strategy._memory_label_histogram()
    assert histogram == {"0": 2, "1": 1, "2": 1}
    assert [
        sample.observation_id for sample in strategy._balanced_training_buffers[1]
    ] == ["trial-2-label-1"]
    assert [
        sample.observation_id for sample in strategy._balanced_training_buffers[2]
    ] == ["trial-3-label-2"]


def test_feedback_count_balanced_memory_updates_with_one_available_class() -> None:
    strategy, runtime = build_strategy(
        memory_strategy="class_balanced_history",
        balanced_memory_per_class=2,
    )
    reveal_many(strategy, runtime, [0, 0, 0, 0])
    result = strategy.maybe_update(runtime_model=runtime)  # type: ignore[arg-type]
    assert result.applied is True
    assert result.metrics["trigger_satisfied"] is True
    assert result.metrics["trigger_reason"] == "update_conditions_satisfied"
    assert result.metrics["model_class_count"] == 3
    assert result.metrics["available_memory_classes"] == [0]
    assert result.metrics["missing_memory_classes"] == [1, 2]
    assert result.metrics["selected_update_sample_count"] == 2
    assert result.metrics["selected_update_label_histogram"] == {
        "0": 2,
        "1": 0,
        "2": 0,
    }
    assert result.metrics["memory_label_histogram"] == {"0": 2, "1": 0, "2": 0}


def test_feedback_count_balanced_memory_balances_only_available_classes() -> None:
    strategy, runtime = build_strategy(
        memory_strategy="class_balanced_history",
        balanced_memory_per_class=32,
    )
    reveal_many(strategy, runtime, [0, 0, 0, 1, 1])
    result = strategy.maybe_update(runtime_model=runtime)  # type: ignore[arg-type]
    assert result.applied is True
    assert result.metrics["available_memory_classes"] == [0, 1]
    assert result.metrics["missing_memory_classes"] == [2]
    assert result.metrics["selected_update_label_histogram"] == {
        "0": 2,
        "1": 2,
        "2": 0,
    }
    assert result.metrics["selected_update_sample_count"] == 4


def test_class_coverage_balanced_memory_still_requires_every_model_class() -> None:
    strategy, runtime = build_strategy(
        update_trigger="class_coverage",
        min_feedback_per_class=2,
        memory_strategy="class_balanced_history",
        balanced_memory_per_class=32,
    )
    reveal_many(strategy, runtime, [0, 0, 1, 1])
    result = strategy.maybe_update(runtime_model=runtime)  # type: ignore[arg-type]
    assert result.applied is False
    assert result.metrics["trigger_satisfied"] is False
    assert result.metrics["available_memory_classes"] == [0, 1]
    assert result.metrics["missing_memory_classes"] == [2]
    assert "class coverage" in str(result.reason)


def test_balanced_update_set_is_equal_paired_and_fixed_seed_reproducible() -> None:
    selections = []
    for _ in range(2):
        strategy, runtime = build_strategy(
            memory_strategy="class_balanced_history",
            balanced_memory_per_class=3,
        )
        reveal_many(strategy, runtime, [0, 0, 0, 1, 1, 2, 2])
        selected = strategy._select_update_samples()
        selections.append([(sample.observation_id, sample.label) for sample in selected])
        assert strategy._label_histogram(selected) == {"0": 2, "1": 2, "2": 2}
        assert all(f"label-{sample.label}" in sample.observation_id for sample in selected)
        result = strategy.maybe_update(runtime_model=runtime)  # type: ignore[arg-type]
        assert result.applied is True
        assert result.metrics["selected_update_label_histogram"] == {
            "0": 2,
            "1": 2,
            "2": 2,
        }
        assert result.metrics["selected_update_sample_count"] == 6
    assert selections[0] == selections[1]


def test_two_class_workload_style_coverage_and_balanced_memory_are_generic() -> None:
    strategy, runtime = build_strategy(
        num_classes=2,
        update_trigger="class_coverage",
        min_feedback_per_class=2,
        memory_strategy="class_balanced_history",
    )
    reveal_many(strategy, runtime, [0, 0, 0, 1])
    waiting = strategy.maybe_update(runtime_model=runtime)  # type: ignore[arg-type]
    assert waiting.applied is False
    reveal(strategy, runtime, ordinal=5, label=1)
    applied = strategy.maybe_update(runtime_model=runtime)  # type: ignore[arg-type]
    assert applied.applied is True
    assert applied.metrics["selected_update_label_histogram"] == {"0": 2, "1": 2}


def test_balanced_memory_and_seen_label_telemetry_survive_state_roundtrip() -> None:
    strategy, runtime = build_strategy(
        memory_strategy="class_balanced_history",
        balanced_memory_per_class=2,
    )
    reveal_many(strategy, runtime, [0, 1, 2, 0])
    state = strategy.state_dict()

    restored, _ = build_strategy(
        memory_strategy="class_balanced_history",
        balanced_memory_per_class=2,
    )
    restored.load_state_dict(state)
    assert restored._memory_label_histogram() == {"0": 2, "1": 1, "2": 1}
    telemetry = restored._telemetry(
        trigger_reason="state_roundtrip",
        trigger_satisfied=False,
        update_applied=False,
    )
    assert telemetry["feedback_count_total"] == 4
    assert telemetry["label_counts_seen"] == {"0": 2, "1": 1, "2": 1}


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("update_trigger", "future_labels"),
        ("memory_strategy", "reservoir"),
        ("min_feedback_per_class", 0),
        ("balanced_memory_per_class", 0),
    ),
)
def test_invalid_trigger_and_memory_config_fail_closed(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        NeuroOnlineConfig(**{field: value})
