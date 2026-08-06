from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from bci_dayloop.acquisition.base import AbstractAcquirer, AcquirerMetadata
from bci_dayloop.inference.realtime import SlidingWindowDecoder
from bci_dayloop.inference.runtime_control import PipelineController, PipelineState
from bci_dayloop.models.base import (
    ModelBackend,
)
from tests.runtime_fakes import (
    build_fixed_runtime,
)


class FakeAcquirer(AbstractAcquirer):
    def __init__(self, chunks, *, delay: float = 0.0) -> None:
        self.metadata = AcquirerMetadata("fake", 20.0, ["C3", "C4"], "uV")
        self.chunks = list(chunks)
        self.delay = delay
        self.started = False
        self.stopped = False
        self.current_label = 2
        self.current_trial_id = 9

    def start_stream(self) -> None:
        self.started = True

    def stop_stream(self) -> None:
        self.stopped = True

    def get_chunk(self, window_sec=None):
        return np.empty((2, 0), dtype=np.float32), np.empty(0)

    def get_new_samples(self):
        if self.delay:
            time.sleep(self.delay)
        if not self.chunks:
            return np.empty((2, 0), dtype=np.float32), np.empty(0)
        return self.chunks.pop(0), np.empty(0)

def make_decoder(
    *,
    error_message: str | None = None,
) -> SlidingWindowDecoder:
    runtime_model = build_fixed_runtime(
        channel_names=("C3", "C4"),
        sample_rate=20.0,
        window_sec=1.0,
        probabilities=(
            0.05,
            0.05,
            0.85,
            0.05,
        ),
        error_message=error_message,
    )

    return SlidingWindowDecoder(
        runtime_model=runtime_model,
        class_names=(
            "left_hand",
            "right_hand",
            "feet",
            "tongue",
        ),
        channel_names=(
            "C3",
            "C4",
        ),
        sample_rate=20.0,
        input_unit="uV",
        window_sec=1.0,
        step_sec=0.5,
    )


def wait_for_state(controller, state, timeout=1.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if controller.snapshot().state == state:
            return
        time.sleep(0.002)
    raise AssertionError(f"Controller did not reach {state}")


def test_start_is_non_blocking_completion_callbacks_and_max_windows():
    chunks = [np.ones((2, 20), dtype=np.float32), np.ones((2, 10), dtype=np.float32), np.ones((2, 10), dtype=np.float32)]
    created = []
    results = []
    states = []
    controller = PipelineController(
        make_decoder(),
        lambda: created.append(FakeAcquirer(chunks)) or created[-1],
        max_windows=2,
        on_result=results.append,
        on_state_change=lambda snapshot: states.append(snapshot.state),
    )

    started = time.perf_counter()
    run_id = controller.start()
    assert time.perf_counter() - started < 0.1
    assert controller.wait(timeout=1.0)
    assert controller.snapshot().state == PipelineState.COMPLETED
    assert controller.snapshot().results_emitted == 2
    assert len(results) == 2
    assert created[0].stopped
    assert PipelineState.RUNNING in states and PipelineState.COMPLETED in states
    assert run_id == 1


def test_result_with_samples_callback_preserves_existing_result_callback():
    chunks = [np.ones((2, 20), dtype=np.float32)]
    results = []
    sample_shapes = []
    controller = PipelineController(
        make_decoder(),
        lambda: FakeAcquirer(chunks),
        on_result=results.append,
        on_result_with_samples=lambda result, samples: sample_shapes.append((result.prediction, samples.shape)),
    )

    controller.start()

    assert controller.wait(timeout=1.0)
    assert [result.prediction for result in results] == ["feet"]
    assert sample_shapes == [("feet", (2, 20))]


def test_duplicate_start_stop_and_short_window_stop_do_not_fail():
    controller = PipelineController(
        make_decoder(),
        lambda: FakeAcquirer([np.ones((2, 10), dtype=np.float32)] * 100, delay=0.002),
    )
    controller.start()
    wait_for_state(controller, PipelineState.RUNNING)
    with pytest.raises(RuntimeError, match="Cannot start"):
        controller.start()
    assert controller.stop(wait=True, timeout=1.0)
    snapshot = controller.snapshot()
    assert snapshot.state == PipelineState.STOPPED
    assert not snapshot.thread_alive
    assert controller.stats.snapshot().failed_windows == 0
    assert not controller.stop(wait=True, timeout=0.1)


def test_restart_uses_fresh_acquirer_and_clears_decoder_buffer():
    created = []

    def factory():
        acquirer = FakeAcquirer([np.ones((2, 10), dtype=np.float32)])
        created.append(acquirer)
        return acquirer

    controller = PipelineController(make_decoder(), factory)
    first_run = controller.start()
    assert controller.wait(timeout=1.0)
    assert controller.snapshot().state == PipelineState.COMPLETED
    second_run = controller.restart(timeout=1.0)
    assert controller.wait(timeout=1.0)
    assert second_run != first_run
    assert len(created) == 2
    assert created[0] is not created[1]
    assert controller.snapshot().results_emitted == 0


def test_worker_failure_is_exposed_and_wait_timeout_is_bounded():
    controller = PipelineController(
        make_decoder(
            error_message="model inference failed"
        ),
        lambda: FakeAcquirer(
            [
                np.ones(
                    (2, 20),
                    dtype=np.float32,
                )
            ]
        ),
    )
    controller.start()
    assert controller.wait(timeout=1.0)
    snapshot = controller.snapshot()
    assert snapshot.state == PipelineState.FAILED
    assert snapshot.last_error_type == "ValueError"
    assert snapshot.last_error_message == "model inference failed"
    with pytest.raises(ValueError, match="model inference failed"):
        controller.raise_if_failed()
    assert controller.wait(timeout=0.01)


def test_wait_timeout_and_result_callback_failure_are_bounded():
    slow_controller = PipelineController(
        make_decoder(),
        lambda: FakeAcquirer([np.ones((2, 10), dtype=np.float32)] * 100, delay=0.05),
    )
    slow_controller.start()
    assert not slow_controller.wait(timeout=0.001)
    assert slow_controller.stop(wait=True, timeout=1.0)
    assert slow_controller.snapshot().state == PipelineState.STOPPED

    def fail_callback(result):
        del result
        raise RuntimeError("result callback failed")

    callback_controller = PipelineController(
        make_decoder(),
        lambda: FakeAcquirer([np.ones((2, 20), dtype=np.float32)]),
        on_result=fail_callback,
    )
    callback_controller.start()
    assert callback_controller.wait(timeout=1.0)
    assert callback_controller.snapshot().state == PipelineState.FAILED
    with pytest.raises(RuntimeError, match="result callback failed"):
        callback_controller.raise_if_failed()
