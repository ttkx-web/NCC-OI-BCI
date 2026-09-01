from __future__ import annotations

from types import SimpleNamespace

import pytest

from bci_dayloop.training.model_50m import runner, workflows


@pytest.mark.parametrize(
    ("mode", "target"),
    [("loso", "loso"), ("within-subject", "within")],
)
def test_runner_dispatches_only_to_matching_workflow(monkeypatch, mode, target) -> None:
    calls: list[str] = []
    config = SimpleNamespace(split_mode=mode)
    monkeypatch.setattr(runner, "run_loso_workflow", lambda value: calls.append("loso") or value)
    monkeypatch.setattr(runner, "run_within_subject_workflow", lambda value: calls.append("within") or value)
    assert runner.run_training(config) is config
    assert calls == [target]


def test_workflow_wrappers_reuse_the_single_shared_implementation(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        workflows,
        "_run_training_workflow",
        lambda config: calls.append(config.split_mode) or "result",
    )
    assert workflows.run_loso_workflow(SimpleNamespace(split_mode="loso")) == "result"
    assert workflows.run_within_subject_workflow(
        SimpleNamespace(split_mode="within-subject")
    ) == "result"
    assert calls == ["loso", "within-subject"]


def test_workflow_wrappers_fail_before_shared_work_for_wrong_mode() -> None:
    with pytest.raises(ValueError):
        workflows.run_loso_workflow(SimpleNamespace(split_mode="within-subject"))
    with pytest.raises(ValueError):
        workflows.run_within_subject_workflow(SimpleNamespace(split_mode="loso"))
