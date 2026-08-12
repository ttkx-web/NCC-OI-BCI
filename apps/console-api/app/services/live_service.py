"""Live-run orchestration assembled exclusively from the Stage 2B primitives."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from app.schemas.events import (
    device_health_event,
    error_event,
    input_contract_event,
    latency_event,
    prediction_event,
    runtime_health_event,
    state_event,
    trigger_event,
    window_event,
)
from app.schemas.runs import RunState


class LiveRuntimeController:
    """A fail-closed adapter around the verified realtime source/pipeline/bridge.

    It deliberately exposes only health, trigger, window, contract and model
    result metadata to the Console event broker.  EEG samples never cross this
    boundary.
    """

    def __init__(
        self,
        record: Any,
        model: Any,
        *,
        source_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.record = record
        self.model = model
        self.source_factory = source_factory
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._source: Any | None = None
        self.successful_windows = 0
        self.failed_windows = 0
        self.expected_windows: int | None = None
        self._latencies: list[float] = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"console-live-{self.record.id}", daemon=True)
        self._thread.start()

    def stop(self, *, wait: bool = True, timeout: float = 5.0) -> None:
        self._stop.set()
        if wait and self._thread is not None:
            self._thread.join(timeout=timeout)

    def restart(self, *, timeout: float = 5.0) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Live run is still stopping")
        self._stop.clear()
        self.start()

    def _emit_contract(self, package: Any, *, safe: bool, reason: str | None = None) -> None:
        contract = package.runtime_model.input_contract
        event = input_contract_event(
            self.record.id,
            safe=safe,
            source_channels=59,
            target_channels=len(contract.channel_names),
            valid_channels=(57 if getattr(package, "model_type", "") == "model_50m" else len(contract.channel_names)),
            window_sec=package.window_sec,
            target_sample_rate=package.target_sample_rate,
        )
        if reason:
            event["payload"]["reason"] = reason
        self.record.broker.publish(event)

    def _set_state(self, state: RunState) -> None:
        with self.record.lock:
            self.record.state = state
        self.record.broker.publish(state_event(self.record.id, state.value))

    def _block(self, package: Any, *, code: str, message: str) -> None:
        self.failed_windows += 1
        self._emit_contract(package, safe=False, reason=message)
        self.record.broker.publish(error_event(self.record.id, code=code, message=message, fatal=True))
        self.record.broker.publish(runtime_health_event(
            self.record.id, successful_windows=self.successful_windows,
            failed_windows=self.failed_windows, expected_windows=None,
        ))
        self._set_state(RunState.FAILED)
        self._stop.set()

    def _source_health(self) -> dict[str, object]:
        health = self._source.health() if self._source is not None else {}
        return dict(health) if isinstance(health, Mapping) else {}

    @staticmethod
    def _health_failure(health: Mapping[str, object]) -> tuple[str, str] | None:
        if health.get("connected") is False:
            return "DEVICE_DISCONNECTED", "设备连接已断开，预测已阻断"
        counters = (
            ("missing_packets", "PACKET_GAP", "检测到数据包缺失，预测已阻断"),
            ("duplicate_packets", "PACKET_GAP", "检测到重复数据包，预测已阻断"),
            ("out_of_order_packets", "PACKET_GAP", "检测到乱序数据包，预测已阻断"),
            ("malformed_packets", "PACKET_INVALID", "检测到无效数据包，预测已阻断"),
        )
        for key, code, message in counters:
            if int(health.get(key, 0) or 0) > 0:
                return code, message
        return None

    def _run(self) -> None:
        package: Any | None = None
        pipeline: Any | None = None
        try:
            import torch

            from bci_dayloop.packages.loader import load_runtime_package
            from bci_dayloop.realtime.channel_units import select_verified_eeg_channels
            from bci_dayloop.realtime.neuracle_jellyfish import (
                NeuracleJellyFishConfig,
                NeuracleJellyFishSource,
            )
            from bci_dayloop.realtime.pipeline import RealtimeEEGWindowPipeline
            from bci_dayloop.realtime.runtime_bridge import RealtimeRuntimeBridge
            from bci_dayloop.realtime.runtime_policy import RealtimeModelPolicyRegistry

            device = self.record.request.compute_device.value
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            package = load_runtime_package(self.model.package_path, device=device, verify_hashes=True)
            policy = RealtimeModelPolicyRegistry.create(package)
            pipeline = RealtimeEEGWindowPipeline.from_runtime_input_contract(package.runtime_model.input_contract)
            bridge = RealtimeRuntimeBridge(package.runtime_model, policy=policy)
            self._emit_contract(package, safe=False, reason="等待首个通过审计的模型输入窗口")
            self._source = self.source_factory() if self.source_factory else NeuracleJellyFishSource(
                NeuracleJellyFishConfig(expected_sampling_rate=1000.0)
            )
            self._source.connect()
            health = self._source_health()
            self.record.broker.publish(device_health_event(self.record.id, health=health))
            failure = self._health_failure(health)
            if failure:
                self._block(package, code=failure[0], message=failure[1])
                return
            self._set_state(RunState.RUNNING)

            while not self._stop.is_set():
                health = self._source_health()
                self.record.broker.publish(device_health_event(self.record.id, health=health))
                failure = self._health_failure(health)
                if failure:
                    self._block(package, code=failure[0], message=failure[1])
                    return
                raw_chunk = self._source.read_chunk()
                if raw_chunk is None:
                    time.sleep(0.005)
                    continue
                markers = []
                read_event = getattr(self._source, "read_event", None)
                while callable(read_event):
                    marker = read_event()
                    if marker is None:
                        break
                    markers.append(marker)
                    self.record.broker.publish(trigger_event(
                        self.record.id, event_type=marker.event_type, code=marker.code,
                    ))
                try:
                    eeg_chunk = select_verified_eeg_channels(raw_chunk)
                    results = pipeline.process(eeg_chunk, markers)
                except Exception as exc:
                    self._block(package, code="SOURCE_CONTRACT_FAILED", message=f"EEG 通道、单位或连续性校验失败：{type(exc).__name__}")
                    return
                for result in results:
                    if result.window is None:
                        self._block(package, code="WINDOW_FAILED", message="窗口连续性校验失败，预测已阻断")
                        return
                    self.record.broker.publish(window_event(
                        self.record.id, window_id=result.window.window_id, status=result.status,
                        continuous_segment_id=result.window.metadata.get("continuous_segment_id"),
                    ))
                    prepared = bridge.prepare(result.window)
                    if not prepared.model_input_safe or prepared.prepared_input is None:
                        self._block(package, code="MODEL_INPUT_FAILED", message="Model Input Contract 未通过，预测已阻断")
                        return
                    self._emit_contract(package, safe=True)
                    inference_started = time.perf_counter()
                    try:
                        output = package.runtime_model.predict_prepared(prepared.prepared_input)
                        probabilities = output.probabilities.detach().cpu().numpy().reshape(-1)
                        predicted = int(output.predicted_class)
                        confidence = float(output.confidence)
                    except Exception:
                        self._block(package, code="PREDICTION_FAILED", message="模型预测失败，预测已阻断")
                        return
                    inference_ms = (time.perf_counter() - inference_started) * 1000.0
                    prepare_ms = float(prepared.prepare_latency_ms or 0.0)
                    total_ms = prepare_ms + inference_ms
                    self._latencies.append(total_ms)
                    self.successful_windows += 1
                    name = package.class_names[predicted] if 0 <= predicted < len(package.class_names) else "UNKNOWN"
                    command = package.command_map.get(name, "STOP") if confidence >= self.record.request.confidence_threshold else "STOP"
                    self.record.broker.publish(prediction_event(
                        self.record.id, window_id=result.window.window_id, trial_id=None,
                        predicted_class=predicted, predicted_name=name, command=command,
                        confidence=confidence, probabilities=[float(value) for value in probabilities],
                        expected_class_id=None, expected_class_name=None,
                    ))
                    self.record.broker.publish(latency_event(
                        self.record.id, prepare_ms=prepare_ms, inference_ms=inference_ms,
                        total_ms=total_ms, p50_ms=float(np.percentile(self._latencies, 50)),
                        p95_ms=float(np.percentile(self._latencies, 95)),
                    ))
                    self.record.broker.publish(runtime_health_event(
                        self.record.id, successful_windows=self.successful_windows,
                        failed_windows=self.failed_windows, expected_windows=None,
                    ))
        except Exception:
            if package is None:
                self.record.broker.publish(error_event(
                    self.record.id, code="LIVE_INITIALIZATION_FAILED",
                    message="Live Runtime 初始化失败，预测已阻断", fatal=True,
                ))
                self._set_state(RunState.FAILED)
            elif not self._stop.is_set():
                self._block(package, code="LIVE_RUNTIME_FAILED", message="实时运行失败，预测已阻断")
        finally:
            if self._source is not None:
                try:
                    self._source.disconnect()
                except Exception:
                    pass
            if pipeline is not None:
                pipeline.close()
            if self.record.state not in {RunState.FAILED, RunState.STOPPED}:
                self._set_state(RunState.STOPPED)
