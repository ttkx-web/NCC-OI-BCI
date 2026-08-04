from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bci_dayloop.acquisition.factory import AcquirerFactory  # noqa: E402
from bci_dayloop.data.hdf5_dataset import EEGHDF5  # noqa: E402
from bci_dayloop.inference.observability import (  # noqa: E402
    JsonlWindowLogger,
    PipelineRunStats,
    calculate_expected_windows,
)
from bci_dayloop.inference.realtime import SlidingWindowDecoder  # noqa: E402
from bci_dayloop.inference.runtime_control import PipelineController, PipelineState  # noqa: E402
from bci_dayloop.models.factory import ModelFactory  # noqa: E402
from bci_dayloop.models.runtime_package import validate_runtime_request  # noqa: E402
from bci_dayloop.utils.config import load_yaml  # noqa: E402
from web.ui_runtime import (  # noqa: E402
    UiEventQueue,
    apply_events,
    control_availability,
    target_window_count,
)

st.set_page_config(page_title="BCI DayLoop", page_icon=":material/neurology:", layout="wide")


@st.cache_data(max_entries=16)
def describe_data(path: str) -> tuple[list[str], list[str], float, str, list[str]]:
    dataset = EEGHDF5(path)
    metadata = dataset.metadata
    return dataset.sessions(), metadata.class_names, metadata.sample_rate, metadata.unit, metadata.channel_names


@st.cache_data(max_entries=32)
def describe_session(path: str, session: str) -> tuple[int, int]:
    trials = EEGHDF5(path).load(session)["data"]
    return int(trials.shape[0]), int(trials.shape[-1])


def discover_hdf5() -> list[str]:
    return [str(path) for path in sorted((ROOT / "data" / "processed").glob("*.h5"))]


def discover_packages() -> list[str]:
    packages = {
        str(model_yaml.parent.resolve())
        for model_yaml in (ROOT / "runs").rglob("model.yaml")
    }

    return sorted(packages)


def controller_snapshot() -> object | None:
    controller = st.session_state.controller
    return controller.snapshot() if controller is not None else None


def format_ms(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f} ms"


def clear_run_display() -> None:
    st.session_state.history = []
    st.session_state.waveform = None
    st.session_state.last_result = None
    st.session_state.runtime_error = None

def format_package_path(path_value: str | Path) -> str:
    """Display a package path relative to the repository root."""
    path = Path(path_value).expanduser().resolve()

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        # 路径不在仓库根目录下时，保留完整路径。
        return str(path)

for key, value in {
    "controller": None,
    "ui_event_queue": None,
    "history": [],
    "waveform": None,
    "last_result": None,
    "runtime_error": None,
    "active_configuration": None,
    "controller_snapshot": None,
}.items():
    st.session_state.setdefault(key, value)

st.title("BCI DayLoop")
st.caption("Model-agnostic pseudo-realtime EEG inference pipeline")

snapshot = controller_snapshot()
availability = control_availability(
    snapshot.state if snapshot is not None else None,
    thread_alive=bool(snapshot.thread_alive) if snapshot is not None else False,
)

data_files = discover_hdf5()
packages = discover_packages()
with st.sidebar:
    st.header("Replay setup")
    if not data_files:
        st.warning("No HDF5 file found under data/processed.")
        data_path = st.text_input(
            "Data file",
            str(ROOT / "data" / "processed" / "bnci2014_001_s01.h5"),
            disabled=not availability.configuration_enabled,
        )
    else:
        data_path = st.selectbox("Data file", data_files, disabled=not availability.configuration_enabled)
    if not packages:
        st.warning("No model package found under runs.")
        package_path = st.text_input(
            "Model package",
            str(ROOT / "runs" / "day1_bnci_s01" / "model_package"),
            disabled=not availability.configuration_enabled,
        )
    else:
        package_path = st.selectbox(
            "Model package",
            packages,
            format_func=format_package_path,
            disabled = not availability.configuration_enabled
        )
    try:
        selected_package_name = str(load_yaml(Path(package_path) / "model.yaml").get("name"))
    except Exception:  # noqa: BLE001
        selected_package_name = None
    try:
        package_model_config = load_yaml(
            Path(package_path) / "model.yaml"
        )

        selected_package_name = str(
            package_model_config.get("name")
        )

        package_window_default = float(
            package_model_config.get(
                "window_seconds",
                10.0,
            )
        )

        package_step_default = float(
            package_model_config.get(
                "step_sec",
                0.5,
            )
        )

    except Exception:
        selected_package_name = None
        package_window_default = 4.0
        package_step_default = 0.5
    package_step_default = 0.5
    if selected_package_name:
        st.caption(f"Package model: {selected_package_name}")
    acquirer_name = st.selectbox(
        "Acquirer", AcquirerFactory.list_acquirers(), disabled=not availability.configuration_enabled
    )
    model_name = st.selectbox("Model", ModelFactory.list_models(), disabled=not availability.configuration_enabled)
    device = st.segmented_control(
        "Compute device", ["cuda", "cpu"], default="cuda", disabled=not availability.configuration_enabled
    )
    threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, 0.55, 0.01, disabled=not availability.configuration_enabled
    )
    replay_speed = st.number_input(
        "Replay speed", min_value=0.1, max_value=100.0, value=1.0, step=0.1, disabled=not availability.configuration_enabled
    )
    max_windows = int(
        st.number_input(
            "Maximum windows", min_value=1, max_value=10000, value=100, step=10, disabled=not availability.configuration_enabled
        )
    )
    window_sec = float(
        st.number_input(
            "Window seconds", min_value=0.1, value=package_window_default, step=0.5, disabled=not availability.configuration_enabled
        )
    )
    step_sec = float(
        st.number_input(
            "Step seconds", min_value=0.1, value=package_step_default, step=0.1, disabled=not availability.configuration_enabled
        )
    )
    enable_jsonl = st.checkbox("Enable JSONL logging", disabled=not availability.configuration_enabled)
    jsonl_path = st.text_input(
        "JSONL log path",
        str(ROOT / "runs" / "ui" / "pipeline_windows.jsonl"),
        disabled=not availability.configuration_enabled,
    )

sessions: list[str] = []
class_names: list[str] = []
channel_names: list[str] = []
sample_rate = 0.0
input_unit = "V"
try:
    sessions, class_names, sample_rate, input_unit, channel_names = describe_data(data_path)
except Exception as exc:  # noqa: BLE001
    st.error(f"Cannot read data file: {exc}")

with st.sidebar:
    session = (
        st.selectbox("Session", sessions, index=max(0, len(sessions) - 1), disabled=not availability.configuration_enabled)
        if sessions
        else None
    )

trial_count: int | None = None
samples_per_trial: int | None = None
expected_windows: int | None = None
if session is not None:
    try:
        trial_count, samples_per_trial = describe_session(data_path, session)
        expected_windows = calculate_expected_windows(
            trial_count * samples_per_trial,
            round(window_sec * sample_rate),
            round(step_sec * sample_rate),
        )
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Cannot calculate expected windows: {exc}")

with st.sidebar.container(horizontal=True):
    start_clicked = st.button(
        "Start", type="primary", icon=":material/play_arrow:", disabled=not availability.start_enabled
    )
    stop_clicked = st.button("Stop", icon=":material/stop:", disabled=not availability.stop_enabled)
    restart_clicked = st.button(
        "Restart", icon=":material/restart_alt:", disabled=not availability.restart_enabled
    )

if start_clicked:
    try:
        if (
            session is None
            or expected_windows is None
            or trial_count is None
            or samples_per_trial is None
        ):
            raise ValueError(
                "Choose a readable data file and session before starting"
            )

        if window_sec <= 0:
            raise ValueError(
                "Window seconds must be greater than zero"
            )

        if step_sec <= 0 or step_sec > window_sec:
            raise ValueError(
                "Step seconds must be greater than zero "
                "and no greater than Window seconds"
            )

        package = Path(package_path)

        runtime_package = ModelFactory.load_runtime_package(
            package,
            EEGHDF5(data_path).metadata,
            device=str(device),
        )

        validate_runtime_request(
            runtime_package,
            window_sec=window_sec,
            step_sec=step_sec,
        )

        if runtime_package.is_test_head:
            st.warning(
                runtime_package.warning_message
                or (
                    "仅用于链路验证，预测和置信度"
                    "无准确率意义"
                )
            )

        target_windows = target_window_count(
            expected_windows,
            max_windows,
        )

        stats = PipelineRunStats()
        stats.set_expected_windows(target_windows)

        event_queue = UiEventQueue()

        logger = (
            JsonlWindowLogger(jsonl_path)
            if enable_jsonl
            else None
        )

        decoder = SlidingWindowDecoder(
            runtime_package.model,
            runtime_package.preprocessor,
            list(runtime_package.class_names),
            sample_rate=sample_rate,
            input_unit=input_unit,
            window_sec=window_sec,
            step_sec=step_sec,
            confidence_threshold=float(threshold),
            command_map=runtime_package.command_map,
            run_stats=stats,
            jsonl_logger=logger,
        )

        acquirer_factory = lambda: AcquirerFactory.create(
            acquirer_name,
            data_path=data_path,
            session=str(session),
            speed=float(replay_speed),
            loop=False,
            window_sec=window_sec,
            step_sec=step_sec,
        )

        controller = PipelineController(
            decoder,
            acquirer_factory,
            max_windows=max_windows,
            on_result_with_samples=(
                event_queue.publish_result
            ),
            on_state_change=(
                event_queue.publish_state
            ),
        )

        clear_run_display()

        st.session_state.controller = controller
        st.session_state.ui_event_queue = event_queue

        st.session_state.active_configuration = {
            "data_path": data_path,
            "session": session,
            "model_package": package_path,
            "model": runtime_package.model_name,
            "window_sec": runtime_package.window_sec,
            "step_sec": runtime_package.step_sec,
            "maximum_windows": max_windows,
            "expected_windows": target_windows,
            "channel_names": channel_names,
            "input_unit": input_unit,
            "jsonl_logging": enable_jsonl,
            "jsonl_path": (
                jsonl_path
                if enable_jsonl
                else None
            ),
            "model_warning": (
                runtime_package.warning_message
            ),
        }

        controller.start()

        # 关键：重新渲染整页。
        # 此时 Controller 已经是 RUNNING，
        # Stop 按钮会从 disabled 变成 enabled。
        st.rerun()

    except Exception as exc:  # noqa: BLE001
        st.session_state.runtime_error = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


if (
    stop_clicked
    and st.session_state.controller is not None
):
    try:
        controller = st.session_state.controller

        # 等待工作线程真正停止。
        controller.stop(
            wait=True,
            timeout=5.0,
        )

        # 处理工作线程最后提交的状态和结果事件。
        event_queue = st.session_state.ui_event_queue
        if event_queue is not None:
            apply_events(
                st.session_state,
                event_queue.drain(),
                history_limit=500,
            )

        # 关键：重新渲染整页。
        # Controller 变为 STOPPED 后，
        # Start 和配置项会重新启用。
        st.rerun()

    except Exception as exc:  # noqa: BLE001
        st.session_state.runtime_error = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

if (
    restart_clicked
    and st.session_state.controller is not None
):
    try:
        controller = st.session_state.controller

        controller.stop(
            wait=True,
            timeout=5.0,
        )

        event_queue = st.session_state.ui_event_queue
        if event_queue is not None:
            event_queue.drain()

        clear_run_display()

        controller.restart(
            timeout=5.0,
        )

        # Restart 后重新显示 RUNNING 状态和按钮。
        st.rerun()

    except Exception as exc:  # noqa: BLE001
        st.session_state.runtime_error = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


@st.fragment(run_every=0.2)
def live_dashboard() -> None:
    event_queue = st.session_state.ui_event_queue
    if event_queue is not None:
        apply_events(st.session_state, event_queue.drain(), history_limit=500)

    controller = st.session_state.controller
    snapshot = controller.snapshot() if controller is not None else None
    stats = controller.stats.snapshot() if controller is not None else None
    result = st.session_state.last_result
    history = st.session_state.history
    configuration = st.session_state.active_configuration or {}
    state = snapshot.state.value if snapshot is not None else PipelineState.IDLE.value

    with st.container(horizontal=True):
        st.metric("Pipeline state", state, border=True)
        st.metric("Current prediction", result["prediction"] if result else "—", border=True)
        st.metric("Confidence", f"{result['confidence']:.1%}" if result else "—", border=True)
        st.metric("Vehicle command", result["command"] if result else "STOP", border=True)
    with st.container(horizontal=True):
        st.metric("Current latency", format_ms(stats.current_latency_ms if stats else None), border=True)
        st.metric("Average latency", format_ms(stats.average_latency_ms if stats else None), border=True)
        st.metric("P95 latency", format_ms(stats.p95_latency_ms if stats else None), border=True)
        st.metric("Preprocessing average", format_ms(stats.preprocessing_average_ms if stats else None), border=True)
        st.metric("Model average", format_ms(stats.model_average_ms if stats else None), border=True)
    with st.container(horizontal=True):
        expected = stats.expected_windows if stats is not None else configuration.get("expected_windows")
        successful = stats.successful_windows if stats is not None else 0
        failed = stats.failed_windows if stats is not None else 0
        st.metric("Runtime", f"{stats.runtime_sec:.1f} s" if stats else "0.0 s", border=True)
        st.metric("Successful / expected windows", f"{successful} / {expected if expected is not None else '—'}", border=True)
        st.metric("Failed windows", failed, border=True)
        st.metric("Run ID", snapshot.run_id if snapshot is not None else 0, border=True)

    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            st.subheader("EEG waveform")
            waveform = st.session_state.waveform
            if waveform is None:
                st.info("Start replay to display EEG samples.")
            else:
                names = configuration.get("channel_names", [])[: min(8, waveform.shape[0])]
                names = names or [f"Ch {index + 1}" for index in range(min(8, waveform.shape[0]))]
                values = waveform[: len(names)].T
                if str(configuration.get("input_unit", "")).lower() == "v":
                    values = values * 1e6
                st.line_chart(pd.DataFrame(values, columns=names))
                st.caption(f"Showing {len(names)} channels · display unit μV")
    with right:
        with st.container(border=True):
            st.subheader("Prediction history")
            if history:
                frame = pd.DataFrame(history)
                columns = [
                    "prediction",
                    "confidence",
                    "command",
                    "preprocessing_latency_ms",
                    "model_latency_ms",
                    "total_latency_ms",
                    "trial_id",
                    "expected_class_id",
                ]
                st.dataframe(frame.reindex(columns=columns).tail(30), hide_index=True)
            else:
                st.info("No decoded window yet.")

    runtime_error = st.session_state.runtime_error
    if runtime_error:
        st.error(f"{runtime_error['error_type']}: {runtime_error['error_message']}")


live_dashboard()
