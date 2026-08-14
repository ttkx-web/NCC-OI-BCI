from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.schemas.events import state_event
from app.schemas.runs import LiveCreate, ReplayCreate, RunState
from app.services.dataset_service import DatasetRegistry
from app.services.model_service import ModelRegistry
from app.services.run_service import RunService


class FakeStats:
    def snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(successful_windows=0, failed_windows=0, expected_windows=3)


class FakeController:
    def __init__(self, record: object) -> None:
        self.record = record
        self.stats = FakeStats()

    def start(self) -> int:
        self.record.state = RunState.RUNNING
        self.record.broker.publish(state_event(self.record.id, "running"))
        return 1

    def stop(self, *, wait: bool, timeout: float) -> bool:
        self.record.state = RunState.STOPPED
        self.record.broker.publish(state_event(self.record.id, "stopped"))
        return True

    def restart(self, *, timeout: float) -> int:
        self.record.state = RunState.RUNNING
        self.record.broker.publish(state_event(self.record.id, "running"))
        return 2


def _service(runtime_package: Path, dataset_file: Path) -> tuple[RunService, str]:
    models = ModelRegistry([runtime_package.parents[3]], runtime_verifier=lambda _path: True)
    datasets = DatasetRegistry(dataset_file.parents[2])
    model_id = models.list()[0].id
    datasets.list()
    return RunService(datasets, models, controller_factory=lambda record, _dataset, _model: FakeController(record)), model_id


def _request(model_id: str) -> ReplayCreate:
    return ReplayCreate(
        dataset_id="bnci2014_001",
        subject_id="S01",
        session="1test",
        model_id=model_id,
        compute_device="cpu",
        replay_speed=100,
        max_windows=3,
        confidence_threshold=0.55,
    )


def test_replay_create_stop_and_restart(runtime_package: Path, dataset_file: Path) -> None:
    service, model_id = _service(runtime_package, dataset_file)
    record = service.create_replay(_request(model_id))
    for _ in range(100):
        if record.state is RunState.RUNNING:
            break
        time.sleep(0.01)
    assert record.state is RunState.RUNNING
    assert service.stop(record.id).state is RunState.STOPPED
    assert service.restart(record.id).state is RunState.RUNNING
    assert [event["payload"]["state"] for event in record.broker.history if event["type"] == "state"] == [
        "starting",
        "running",
        "stopped",
        "starting",
        "running",
    ]


def test_replay_rejects_invalid_model(runtime_package: Path, dataset_file: Path) -> None:
    service, _ = _service(runtime_package, dataset_file)
    with pytest.raises(LookupError):
        service.create_replay(_request("model_missing"))


def test_live_rejects_runtime_package_without_formal_live_verification(
    runtime_package: Path,
    dataset_file: Path,
) -> None:
    models = ModelRegistry(
        [runtime_package.parents[3]],
        runtime_verifier=lambda _path: True,
        live_verifier=lambda _path: False,
    )
    model_id = models.list()[0].id
    service = RunService(DatasetRegistry(dataset_file.parents[2]), models)

    with pytest.raises(ValueError, match="not approved for formal Live"):
        service.create_live(LiveCreate(model_id=model_id, compute_device="cpu"))
