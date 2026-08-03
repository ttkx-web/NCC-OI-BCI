from __future__ import annotations

"""Metrics and result aggregation for Stage-1 personalization.

This module builds on ``trainer.ClassifierMetrics`` and
``trainer.EvaluationResult``.  It does not run model inference; it converts
population/personal evaluations into stable records that can be written to
JSON/CSV and aggregated over subjects, trial budgets, and random seeds.
"""

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence

from .trainer import ClassifierMetrics, EvaluationResult


@dataclass(frozen=True, slots=True)
class PersonalizationGain:
    """Absolute personal-minus-population metric differences."""

    accuracy: float
    balanced_accuracy: float
    macro_f1: float

    def to_dict(self) -> dict[str, float]:
        return {
            "accuracy": float(self.accuracy),
            "balanced_accuracy": float(self.balanced_accuracy),
            "macro_f1": float(self.macro_f1),
        }


@dataclass(frozen=True, slots=True)
class PersonalizationDecision:
    """Decision made on a personal validation set, never on final test data."""

    accepted: bool
    metric_name: str
    population_value: float
    personal_value: float
    gain: float
    minimum_gain: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": bool(self.accepted),
            "metric_name": self.metric_name,
            "population_value": float(self.population_value),
            "personal_value": float(self.personal_value),
            "gain": float(self.gain),
            "minimum_gain": float(self.minimum_gain),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ScalarSummary:
    count: int
    mean: float
    std: float
    minimum: float
    maximum: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": int(self.count),
            "mean": float(self.mean),
            "std": float(self.std),
            "min": float(self.minimum),
            "max": float(self.maximum),
        }


@dataclass(frozen=True, slots=True)
class PersonalizationRunRecord:
    """One population-vs-personal evaluation run."""

    run_id: str
    user_id: str
    target_subject: int | str
    task: str
    adaptation_type: str
    trials_per_class: int | None
    seed: int
    population: ClassifierMetrics
    personal: ClassifierMetrics
    gain: PersonalizationGain
    validation_decision: PersonalizationDecision | None = None
    training_seconds: float | None = None
    best_epoch: int | None = None
    model_package: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "user_id": self.user_id,
            "target_subject": self.target_subject,
            "task": self.task,
            "adaptation_type": self.adaptation_type,
            "trials_per_class": self.trials_per_class,
            "seed": int(self.seed),
            "population": self.population.to_dict(),
            "personal": self.personal.to_dict(),
            "gain": self.gain.to_dict(),
            "validation_decision": (
                self.validation_decision.to_dict()
                if self.validation_decision is not None
                else None
            ),
            "training_seconds": self.training_seconds,
            "best_epoch": self.best_epoch,
            "model_package": self.model_package,
            "extra": _json_safe(self.extra),
        }

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten the main scalar fields for CSV output."""

        return {
            "run_id": self.run_id,
            "user_id": self.user_id,
            "target_subject": self.target_subject,
            "task": self.task,
            "adaptation_type": self.adaptation_type,
            "trials_per_class": self.trials_per_class,
            "seed": int(self.seed),
            "population_loss": self.population.loss,
            "population_accuracy": self.population.accuracy,
            "population_balanced_accuracy": (
                self.population.balanced_accuracy
            ),
            "population_macro_f1": self.population.macro_f1,
            "personal_loss": self.personal.loss,
            "personal_accuracy": self.personal.accuracy,
            "personal_balanced_accuracy": (
                self.personal.balanced_accuracy
            ),
            "personal_macro_f1": self.personal.macro_f1,
            "accuracy_gain": self.gain.accuracy,
            "balanced_accuracy_gain": self.gain.balanced_accuracy,
            "macro_f1_gain": self.gain.macro_f1,
            "personalization_accepted": (
                self.validation_decision.accepted
                if self.validation_decision is not None
                else None
            ),
            "decision_metric": (
                self.validation_decision.metric_name
                if self.validation_decision is not None
                else None
            ),
            "decision_gain": (
                self.validation_decision.gain
                if self.validation_decision is not None
                else None
            ),
            "training_seconds": self.training_seconds,
            "best_epoch": self.best_epoch,
            "model_package": self.model_package,
            "extra_json": json.dumps(
                _json_safe(self.extra),
                ensure_ascii=False,
                sort_keys=True,
            ),
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return value


def compute_personalization_gain(
    population: ClassifierMetrics | EvaluationResult,
    personal: ClassifierMetrics | EvaluationResult,
) -> PersonalizationGain:
    population_metrics = (
        population.metrics
        if isinstance(population, EvaluationResult)
        else population
    )
    personal_metrics = (
        personal.metrics
        if isinstance(personal, EvaluationResult)
        else personal
    )
    return PersonalizationGain(
        accuracy=(
            personal_metrics.accuracy
            - population_metrics.accuracy
        ),
        balanced_accuracy=(
            personal_metrics.balanced_accuracy
            - population_metrics.balanced_accuracy
        ),
        macro_f1=(
            personal_metrics.macro_f1
            - population_metrics.macro_f1
        ),
    )


def compare_evaluations(
    *,
    population: EvaluationResult,
    personal: EvaluationResult,
) -> dict[str, Any]:
    """Compare two heads evaluated on the exact same ordered examples."""

    if population.labels != personal.labels:
        raise ValueError(
            "Population and personal evaluations have different labels."
        )
    return {
        "population": population.to_dict(),
        "personal": personal.to_dict(),
        "gain": compute_personalization_gain(
            population,
            personal,
        ).to_dict(),
    }


def _metric_value(
    metrics: ClassifierMetrics | EvaluationResult,
    metric_name: str,
) -> float:
    values = (
        metrics.metrics
        if isinstance(metrics, EvaluationResult)
        else metrics
    )
    mapping = {
        "accuracy": values.accuracy,
        "acc": values.accuracy,
        "balanced_accuracy": values.balanced_accuracy,
        "bacc": values.balanced_accuracy,
        "macro_f1": values.macro_f1,
        "f1": values.macro_f1,
        "loss": values.loss,
    }
    if metric_name not in mapping:
        raise ValueError(
            f"Unsupported metric_name={metric_name!r}. "
            f"Choose from {sorted(mapping)}."
        )
    return float(mapping[metric_name])


def decide_personalization(
    *,
    population_validation: ClassifierMetrics | EvaluationResult,
    personal_validation: ClassifierMetrics | EvaluationResult,
    metric_name: str = "balanced_accuracy",
    minimum_gain: float = 0.0,
) -> PersonalizationDecision:
    """Accept or reject a personal model using validation data only.

    For accuracy-like metrics, a positive personal-minus-population gain is
    better.  For loss, a decrease is better and the reported ``gain`` is
    population loss minus personal loss, so positive still means improvement.
    """

    population_value = _metric_value(
        population_validation,
        metric_name,
    )
    personal_value = _metric_value(
        personal_validation,
        metric_name,
    )

    if metric_name == "loss":
        gain = population_value - personal_value
    else:
        gain = personal_value - population_value

    accepted = bool(gain >= minimum_gain)
    reason = (
        f"accepted: validation {metric_name} gain "
        f"{gain:+.6f} >= {minimum_gain:+.6f}"
        if accepted
        else (
            f"rejected: validation {metric_name} gain "
            f"{gain:+.6f} < {minimum_gain:+.6f}"
        )
    )
    return PersonalizationDecision(
        accepted=accepted,
        metric_name=metric_name,
        population_value=population_value,
        personal_value=personal_value,
        gain=gain,
        minimum_gain=float(minimum_gain),
        reason=reason,
    )


def build_run_record(
    *,
    run_id: str,
    user_id: str,
    target_subject: int | str,
    task: str,
    adaptation_type: str,
    seed: int,
    population: ClassifierMetrics | EvaluationResult,
    personal: ClassifierMetrics | EvaluationResult,
    trials_per_class: int | None = None,
    validation_decision: PersonalizationDecision | None = None,
    training_seconds: float | None = None,
    best_epoch: int | None = None,
    model_package: str | Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> PersonalizationRunRecord:
    population_metrics = (
        population.metrics
        if isinstance(population, EvaluationResult)
        else population
    )
    personal_metrics = (
        personal.metrics
        if isinstance(personal, EvaluationResult)
        else personal
    )
    return PersonalizationRunRecord(
        run_id=str(run_id),
        user_id=str(user_id),
        target_subject=target_subject,
        task=str(task),
        adaptation_type=str(adaptation_type),
        trials_per_class=(
            int(trials_per_class)
            if trials_per_class is not None
            else None
        ),
        seed=int(seed),
        population=population_metrics,
        personal=personal_metrics,
        gain=compute_personalization_gain(
            population_metrics,
            personal_metrics,
        ),
        validation_decision=validation_decision,
        training_seconds=(
            float(training_seconds)
            if training_seconds is not None
            else None
        ),
        best_epoch=(
            int(best_epoch)
            if best_epoch is not None
            else None
        ),
        model_package=(
            str(model_package)
            if model_package is not None
            else None
        ),
        extra=dict(extra or {}),
    )


def summarize_values(values: Iterable[float]) -> ScalarSummary:
    finite = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]
    if not finite:
        raise ValueError("Cannot summarize an empty/non-finite value list.")
    return ScalarSummary(
        count=len(finite),
        mean=mean(finite),
        std=pstdev(finite) if len(finite) > 1 else 0.0,
        minimum=min(finite),
        maximum=max(finite),
    )


def aggregate_run_records(
    records: Sequence[PersonalizationRunRecord],
    *,
    group_by: Sequence[str] = (
        "task",
        "adaptation_type",
        "trials_per_class",
    ),
) -> list[dict[str, Any]]:
    """Aggregate records without requiring pandas."""

    if not records:
        return []

    allowed_group_fields = {
        "user_id",
        "target_subject",
        "task",
        "adaptation_type",
        "trials_per_class",
        "seed",
    }
    invalid = set(group_by) - allowed_group_fields
    if invalid:
        raise ValueError(
            f"Unsupported group_by fields: {sorted(invalid)}."
        )

    groups: dict[tuple[Any, ...], list[PersonalizationRunRecord]] = {}
    for record in records:
        key = tuple(getattr(record, field) for field in group_by)
        groups.setdefault(key, []).append(record)

    output: list[dict[str, Any]] = []
    for key, group in sorted(
        groups.items(),
        key=lambda item: tuple(str(value) for value in item[0]),
    ):
        row: dict[str, Any] = {
            field: value
            for field, value in zip(group_by, key)
        }
        row["runs"] = len(group)
        row["subjects"] = len(
            {str(record.target_subject) for record in group}
        )
        row["seeds"] = sorted({record.seed for record in group})

        metric_extractors = {
            "population_accuracy": (
                lambda record: record.population.accuracy
            ),
            "personal_accuracy": (
                lambda record: record.personal.accuracy
            ),
            "accuracy_gain": (
                lambda record: record.gain.accuracy
            ),
            "population_balanced_accuracy": (
                lambda record: record.population.balanced_accuracy
            ),
            "personal_balanced_accuracy": (
                lambda record: record.personal.balanced_accuracy
            ),
            "balanced_accuracy_gain": (
                lambda record: record.gain.balanced_accuracy
            ),
            "population_macro_f1": (
                lambda record: record.population.macro_f1
            ),
            "personal_macro_f1": (
                lambda record: record.personal.macro_f1
            ),
            "macro_f1_gain": (
                lambda record: record.gain.macro_f1
            ),
        }
        row["metrics"] = {
            name: summarize_values(
                extractor(record) for record in group
            ).to_dict()
            for name, extractor in metric_extractors.items()
        }

        decisions = [
            record.validation_decision
            for record in group
            if record.validation_decision is not None
        ]
        if decisions:
            accepted = sum(
                int(decision.accepted) for decision in decisions
            )
            row["acceptance"] = {
                "evaluated": len(decisions),
                "accepted": accepted,
                "acceptance_rate": accepted / len(decisions),
            }
        output.append(row)

    return output


def save_run_records_json(
    records: Sequence[PersonalizationRunRecord],
    path: str | Path,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{destination}.tmp")
    temporary.write_text(
        json.dumps(
            [record.to_dict() for record in records],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def save_run_records_csv(
    records: Sequence[PersonalizationRunRecord],
    path: str | Path,
) -> Path:
    if not records:
        raise ValueError("Cannot save an empty record list to CSV.")

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.to_flat_dict() for record in records]
    temporary = Path(f"{destination}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
    return destination


def save_aggregate_json(
    aggregates: Sequence[Mapping[str, Any]],
    path: str | Path,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{destination}.tmp")
    temporary.write_text(
        json.dumps(
            _json_safe(list(aggregates)),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
