from __future__ import annotations

import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.schemas.events import (
    error_event,
    input_contract_event,
    latency_event,
    prediction_event,
    runtime_health_event,
    state_event,
)
from app.schemas.runs import LiveCreate, ReplayCreate, RunState, RunSummary
from app.services.dataset_service import DatasetEntry, DatasetRegistry
from app.services.model_service import ModelEntry, ModelRegistry


TERMINAL_STATES = {RunState.STOPPED, RunState.COMPLETED, RunState.FAILED}


class RunEventBroker:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._history: list[dict[str, Any]] = []
        self._subscribers: set[queue.Queue[dict[str, Any]]] = set()

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._history.append(event)
            if len(self._history) > 1000:
                del self._history[:-1000]
            for subscriber in tuple(self._subscribers):
                subscriber.put_nowait(event)

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        subscriber: queue.Queue[dict[str, Any]] = queue.Queue()
        with self._lock:
            for event in self._history:
                subscriber.put_nowait(event)
            self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)

    @property
    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history)


@dataclass(slots=True)
class RunRecord:
    id: str
    request: ReplayCreate | LiveCreate
    created_at: float
    run_type: str = "replay"
    state: RunState = RunState.STARTING
    controller: Any | None = None
    broker: RunEventBroker = field(default_factory=RunEventBroker)
    expected_windows: int | None = None
    successful_windows: int = 0
    failed_windows: int = 0
    window_id: int = 0
    stop_requested: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)

    def summary(self) -> RunSummary:
        with self.lock:
            if self.controller is not None and hasattr(self.controller, "stats"):
                stats = self.controller.stats.snapshot()
                successful = stats.successful_windows
                failed = stats.failed_windows
                expected = stats.expected_windows
            elif self.controller is not None:
                successful = int(getattr(self.controller, "successful_windows", self.successful_windows))
                failed = int(getattr(self.controller, "failed_windows", self.failed_windows))
                expected = getattr(self.controller, "expected_windows", self.expected_windows)
            else:
                successful = self.successful_windows
                failed = self.failed_windows
                expected = self.expected_windows
            return RunSummary(
                id=self.id,
                run_type=self.run_type,
                state=self.state,
                dataset_id=getattr(self.request, "dataset_id", None),
                subject_id=getattr(self.request, "subject_id", None),
                session=getattr(self.request, "session", None),
                model_id=self.request.model_id,
                created_at=self.created_at,
                successful_windows=successful,
                failed_windows=failed,
                expected_windows=expected,
            )


ControllerFactory = Callable[[RunRecord, DatasetEntry, ModelEntry], Any]
LiveControllerFactory = Callable[[RunRecord, ModelEntry], Any]


class RunService:
    def __init__(
        self,
        datasets: DatasetRegistry,
        models: ModelRegistry,
        *,
        controller_factory: ControllerFactory | None = None,
        live_controller_factory: LiveControllerFactory | None = None,
    ) -> None:
        self.datasets = datasets
        self.models = models
        self.controller_factory = controller_factory or self._build_controller
        self.live_controller_factory = live_controller_factory or self._build_live_controller
        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.RLock()

    def create_replay(self, request: ReplayCreate) -> RunRecord:
        dataset = self.datasets.get_entry(request.dataset_id, request.subject_id)
        if request.session not in dataset.summary.sessions:
            raise ValueError(f"Unknown session: {request.session}")
        model = self.models.get_entry(request.model_id)
        if not model.summary.runtime_verified:
            raise ValueError("Selected Runtime Package is not supported by the current runtime")
        package_classes = tuple(str(item) for item in model.payload["model"]["class_names"])
        if tuple(dataset.summary.class_names) != package_classes:
            raise ValueError("Dataset class order does not match the Runtime Package")

        run_id = f"run_{uuid.uuid4().hex[:12]}"
        record = RunRecord(id=run_id, request=request, created_at=time.time())
        record.broker.publish(state_event(run_id, RunState.STARTING.value))
        with self._lock:
            self._runs[run_id] = record
        threading.Thread(
            target=self._initialize,
            args=(record, dataset, model),
            name=f"console-replay-init-{run_id}",
            daemon=True,
        ).start()
        return record

    def create_live(self, request: LiveCreate) -> RunRecord:
        model = self.models.get_entry(request.model_id)
        if not model.summary.runtime_verified:
            raise ValueError("Selected Runtime Package has not passed runtime verification")
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        record = RunRecord(id=run_id, request=request, created_at=time.time(), run_type="live")
        record.broker.publish(state_event(run_id, RunState.STARTING.value))
        with self._lock:
            if any(item.run_type == "live" and item.state not in TERMINAL_STATES for item in self._runs.values()):
                raise RuntimeError("A Live run is already active")
            self._runs[run_id] = record
        threading.Thread(
            target=self._initialize_live,
            args=(record, model),
            name=f"console-live-init-{run_id}",
            daemon=True,
        ).start()
        return record

    def _initialize_live(self, record: RunRecord, model: ModelEntry) -> None:
        try:
            controller = self.live_controller_factory(record, model)
            with record.lock:
                if record.stop_requested:
                    record.state = RunState.STOPPED
                    record.broker.publish(state_event(record.id, record.state.value))
                    return
                record.controller = controller
            controller.start()
        except Exception:
            with record.lock:
                record.state = RunState.FAILED
            record.broker.publish(error_event(
                record.id, code="LIVE_INITIALIZATION_FAILED",
                message="Live Runtime 初始化失败，预测已阻断", fatal=True,
            ))
            record.broker.publish(state_event(record.id, RunState.FAILED.value))

    def _initialize(self, record: RunRecord, dataset: DatasetEntry, model: ModelEntry) -> None:
        try:
            controller = self.controller_factory(record, dataset, model)
            with record.lock:
                if record.stop_requested:
                    record.state = RunState.STOPPED
                    record.broker.publish(state_event(record.id, record.state.value))
                    return
                record.controller = controller
            controller.start()
        except Exception:
            with record.lock:
                record.state = RunState.FAILED
            record.broker.publish(
                error_event(
                    record.id,
                    code="RUNTIME_INITIALIZATION_FAILED",
                    message="Runtime 初始化失败，请检查模型包、数据集与计算设备。",
                    fatal=True,
                )
            )
            record.broker.publish(state_event(record.id, RunState.FAILED.value))

    def _build_controller(self, record: RunRecord, dataset: DatasetEntry, model: ModelEntry) -> Any:
        import torch

        from bci_dayloop.acquisition.factory import AcquirerFactory
        from bci_dayloop.data.hdf5_dataset import EEGHDF5
        from bci_dayloop.inference.observability import PipelineRunStats, calculate_expected_windows
        from bci_dayloop.inference.realtime import SlidingWindowDecoder
        from bci_dayloop.inference.runtime_control import PipelineController, PipelineState
        from bci_dayloop.packages.loader import load_runtime_package

        device = record.request.compute_device.value
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        runtime_package = load_runtime_package(model.package_path, device=device, verify_hashes=True)
        metadata = EEGHDF5(dataset.path).metadata
        with __import__("h5py").File(dataset.path, "r") as handle:
            session_values = handle["session_ids"].asstr()[:]
            session_mask = session_values == record.request.session
            trial_samples = int(handle["data"].shape[-1])
            trial_count = int(np.count_nonzero(session_mask))
        expected = calculate_expected_windows(
            trial_count * trial_samples,
            round(runtime_package.window_sec * metadata.sample_rate),
            round(runtime_package.step_sec * metadata.sample_rate),
        )
        expected = min(expected, record.request.max_windows)
        record.expected_windows = expected
        stats = PipelineRunStats()
        stats.set_expected_windows(expected)

        source_names = {name.strip().upper() for name in metadata.channel_names}
        target_names = {name.strip().upper() for name in runtime_package.runtime_model.input_contract.channel_names}
        record.broker.publish(
            input_contract_event(
                record.id,
                safe=True,
                source_channels=len(source_names),
                target_channels=len(target_names),
                valid_channels=len(source_names & target_names),
                window_sec=runtime_package.window_sec,
                target_sample_rate=runtime_package.target_sample_rate,
            )
        )

        decoder = SlidingWindowDecoder(
            runtime_model=runtime_package.runtime_model,
            class_names=runtime_package.class_names,
            channel_names=metadata.channel_names,
            sample_rate=metadata.sample_rate,
            input_unit=metadata.unit,
            window_sec=runtime_package.window_sec,
            step_sec=runtime_package.step_sec,
            confidence_threshold=record.request.confidence_threshold,
            command_map=runtime_package.command_map,
            run_stats=stats,
        )

        def on_result(result: Any) -> None:
            with record.lock:
                record.window_id += 1
                window_id = record.window_id
            expected_name = None
            if result.expected_class_id is not None and 0 <= result.expected_class_id < len(runtime_package.class_names):
                expected_name = runtime_package.class_names[result.expected_class_id]
            record.broker.publish(
                prediction_event(
                    record.id,
                    window_id=window_id,
                    trial_id=result.trial_id,
                    predicted_class=result.class_id,
                    predicted_name=result.prediction,
                    command=result.command,
                    confidence=result.confidence,
                    probabilities=list(result.probabilities),
                    expected_class_id=result.expected_class_id,
                    expected_class_name=expected_name,
                )
            )
            snapshot = stats.snapshot()
            totals = [event["payload"]["total_ms"] for event in record.broker.history if event["type"] == "latency"]
            p50 = float(np.percentile(totals + [result.total_latency_ms], 50))
            p95 = float(np.percentile(totals + [result.total_latency_ms], 95))
            record.broker.publish(
                latency_event(
                    record.id,
                    prepare_ms=result.preprocessing_latency_ms,
                    inference_ms=result.model_latency_ms,
                    total_ms=result.total_latency_ms,
                    p50_ms=p50,
                    p95_ms=p95,
                )
            )
            record.broker.publish(
                runtime_health_event(
                    record.id,
                    successful_windows=max(snapshot.successful_windows, window_id),
                    failed_windows=snapshot.failed_windows,
                    expected_windows=expected,
                )
            )

        def on_state(snapshot: Any) -> None:
            state_map = {
                PipelineState.IDLE: RunState.IDLE,
                PipelineState.RUNNING: RunState.RUNNING,
                PipelineState.STOPPING: RunState.STOPPING,
                PipelineState.STOPPED: RunState.STOPPED,
                PipelineState.COMPLETED: RunState.COMPLETED,
                PipelineState.FAILED: RunState.FAILED,
            }
            state = state_map[snapshot.state]
            with record.lock:
                record.state = state
            if state is RunState.FAILED:
                record.broker.publish(
                    error_event(
                        record.id,
                        code="PIPELINE_FAILED",
                        message="Pipeline 运行失败，推理输出已禁用。",
                        fatal=True,
                    )
                )
            record.broker.publish(state_event(record.id, state.value))

        return PipelineController(
            decoder,
            lambda: AcquirerFactory.create(
                "replay",
                data_path=dataset.path,
                session=record.request.session,
                speed=record.request.replay_speed,
                loop=False,
                window_sec=runtime_package.window_sec,
                step_sec=runtime_package.step_sec,
            ),
            max_windows=record.request.max_windows,
            on_result=on_result,
            on_state_change=on_state,
        )

    def _build_live_controller(self, record: RunRecord, model: ModelEntry) -> Any:
        from app.services.live_service import LiveRuntimeController

        return LiveRuntimeController(record, model)

    def get(self, run_id: str) -> RunRecord:
        with self._lock:
            try:
                return self._runs[run_id]
            except KeyError as error:
                raise LookupError(f"Unknown run_id: {run_id}") from error

    def list(self) -> list[RunSummary]:
        with self._lock:
            records = list(self._runs.values())
        return sorted((record.summary() for record in records), key=lambda item: item.created_at, reverse=True)

    def stop(self, run_id: str) -> RunRecord:
        record = self.get(run_id)
        with record.lock:
            if record.controller is None:
                record.stop_requested = True
                record.state = RunState.STOPPED
                record.broker.publish(state_event(record.id, record.state.value))
                return record
            controller = record.controller
        controller.stop(wait=True, timeout=5.0)
        return record

    def restart(self, run_id: str) -> RunRecord:
        record = self.get(run_id)
        with record.lock:
            if record.controller is None:
                raise RuntimeError("Run controller is not ready")
            record.window_id = 0
            record.state = RunState.STARTING
            record.broker.publish(state_event(record.id, record.state.value))
            controller = record.controller
        controller.restart(timeout=5.0)
        return record
