"""Streamlit EEG multi-state visualization demo, isolated from production UI."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from bci_dayloop.data.hdf5_dataset import EEGHDF5  # noqa: E402
from bci_dayloop.demo.input import MIN_STEP_SEC, list_demo_sessions, load_demo_trial, trial_window, window_count  # noqa: E402
from bci_dayloop.demo.model_package_discovery import discover_motor_intent_packages  # noqa: E402
from bci_dayloop.demo.motor_decoder import DemoMotorIntentDecoder, ModelPackageMotorIntentDecoder, MotorIntentDecoder  # noqa: E402
from bci_dayloop.demo.schemas import DemoEEGWindow, MOTOR_LABELS_CN, STATE_LABELS_CN  # noqa: E402
from bci_dayloop.demo.state_decoder import DemoStateDecoder  # noqa: E402
from bci_dayloop.demo.utils import score_level  # noqa: E402
from bci_dayloop.demo.visual_state import VISUAL_INTERVAL_SEC, VisualState  # noqa: E402


st.set_page_config(page_title="Omni Neural Decoder", page_icon=":material/neurology:", layout="wide")

# Keep the motor package controls and result contract ready for future use while
# keeping the current video-focused dashboard centered on neural-state indices.
SHOW_MOTOR_INTENT = False
MAX_DECODE_CATCHUP = 3

DEMO_CSS = """
<style>
.block-container {max-width: 1680px; padding-top: 1.3rem; padding-bottom: 0.45rem;}
div[data-testid="stSidebar"] .block-container {padding-top: 0.7rem;}
h1 {font-size: 1.45rem !important; margin: 0 0 0.08rem !important; line-height: 1.28 !important;}
.demo-subtitle {color: #AAB7C4; font-size: 0.78rem; margin: 0.10rem 0 0.42rem; line-height:1.3;}
.demo-status {display:flex; align-items:center; gap:0.55rem; flex-wrap:wrap; padding:0.34rem 0.65rem; margin:0 0 0.42rem; background:#17212B; border:1px solid #263849; border-radius:0.42rem; color:#DCE6ED; font-size:0.76rem; line-height:1.15;}
.demo-status .live {color:#39D98A; font-weight:600;}
.demo-status .done {color:#F5B642; font-weight:600;}
.demo-status .item {white-space:nowrap;}
.core-card {height:100%; box-sizing:border-box; min-height:118px; padding:0.55rem 0.7rem; background:#17212B; border:1px solid #263849; border-radius:0.5rem;}
.core-title {font-size:0.78rem; color:#AAB7C4; margin-bottom:0.26rem;}
.motor-head {display:flex; gap:0.55rem; align-items:baseline; margin-bottom:0.25rem;}
.motor-label {font-size:1.28rem; font-weight:700; color:#F3F6F9;}.motor-confidence {font-size:0.82rem; color:#58C7C7;}
.prob-row {display:grid; grid-template-columns:3.2rem 1fr 2.4rem; align-items:center; gap:0.32rem; margin:0.16rem 0; font-size:0.70rem; color:#C7D3DB;}
.prob-track,.state-track {height:0.28rem; border-radius:99px; overflow:hidden; background:#263849;}.prob-fill,.state-fill {height:100%; border-radius:99px; background:#00A6A6;}
.interpretation {font-size:0.82rem; line-height:1.38; color:#E7EFF4; margin-top:0.22rem;}.disclaimer {font-size:0.63rem; color:#8293A1; margin-top:0.42rem;}
.section-label {font-size:0.80rem; font-weight:600; color:#DCE6ED; margin:0.40rem 0 0.23rem;}
.state-card {height:76px; box-sizing:border-box; padding:0.38rem 0.46rem; background:#17212B; border:1px solid #263849; border-radius:0.42rem; overflow:hidden;}
.state-top,.state-bottom {display:flex; align-items:baseline; justify-content:space-between; gap:0.22rem;}.state-name {font-size:0.64rem; color:#B6C4CE; white-space:nowrap;}.state-value {font-size:1.02rem; color:#F3F6F9; font-weight:700;}.state-level {font-size:0.60rem; color:#7ECCD1;}.state-spark {width:48px; height:16px;}.state-track {margin-top:0.30rem; height:0.20rem;}
.emotion-emoji {font-size:0.96rem; line-height:1;}.emotion-label {font-size:0.72rem; color:#7ECCD1; white-space:nowrap;}
.signal-section {font-size:0.80rem; font-weight:600; color:#DCE6ED; margin:1.35rem 0 0.23rem;}
.signal-title {font-size:0.80rem; font-weight:600; color:#DCE6ED; margin:0 0 0.05rem;}.signal-caption {font-size:0.62rem; color:#8293A1; margin:0 0 0.08rem;}
.visual-signal-grid {display:grid; grid-template-columns:1.40fr 1fr; gap:0.55rem;}.visual-signal-pane {min-width:0;}
.demo-debug-counter {display:none;}
div[data-testid="stProgress"] {height:0.34rem; margin:0.08rem 0;}
div[data-testid="stSidebar"] label, div[data-testid="stSidebar"] p {font-size:0.74rem !important;}
div[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {margin-bottom:-0.18rem;}
</style>
"""
st.markdown(DEMO_CSS, unsafe_allow_html=True)


def command_line_defaults() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-path")
    parser.add_argument("--device", default="cpu")
    return parser.parse_known_args()[0]


def discover_hdf5() -> list[str]:
    root = ROOT / "data" / "processed"
    return sorted(str(path.resolve()) for pattern in ("*.h5", "*.hdf5") for path in root.rglob(pattern)) if root.exists() else []


@st.cache_data(max_entries=1)
def discover_motor_packages() -> list[dict[str, object]]:
    return [
        {
            "path": option.path,
            "label": option.label,
            "model_name": option.model_name,
            "model_type": option.model_type,
            "window_sec": option.window_sec,
        }
        for option in discover_motor_intent_packages(ROOT)
    ]


@st.cache_data(max_entries=32)
def trial_count(data_path: str, session: str) -> int:
    return int(EEGHDF5(data_path).load(session)["data"].shape[0])


@st.cache_data(max_entries=32)
def cached_trial(data_path: str, session: str, trial_index: int):
    return load_demo_trial(data_path, session=session, trial_index=trial_index)


def reset_replay(start_trial_index: int) -> None:
    reset_motor_decoder = getattr(st.session_state.get("demo_motor_decoder"), "reset", None)
    if callable(reset_motor_decoder):
        reset_motor_decoder()
    st.session_state.demo_decoder = None
    st.session_state.demo_result = None
    st.session_state.demo_history = {name: [] for name in STATE_LABELS_CN}
    st.session_state.demo_playing = False
    st.session_state.demo_current_trial_index = start_trial_index
    st.session_state.demo_current_window_index = 0
    st.session_state.demo_last_trial_index = None
    st.session_state.demo_last_window_index = None
    st.session_state.demo_finished = False
    st.session_state.demo_next_decode_stream_time = 0.0
    st.session_state.demo_trial_decode_complete = False
    st.session_state.demo_static_revision = 0
    st.session_state.demo_static_rendered_revision = -1
    st.session_state.demo_cortical_revision = 0
    st.session_state.demo_cortical_rendered_revision = -1
    st.session_state.demo_visual_state.reset()


def ensure_decoder(
    configuration: tuple[object, ...],
    start_trial_index: int,
    motor_decoder: MotorIntentDecoder,
) -> None:
    if st.session_state.get("demo_configuration") != configuration:
        reset_replay(start_trial_index)
        st.session_state.demo_configuration = configuration
        st.session_state.demo_decoder = DemoStateDecoder(
            device=str(configuration[-1]), motor_decoder=motor_decoder, compute_motor_intent=SHOW_MOTOR_INTENT
        )
    elif st.session_state.demo_decoder is None:
        st.session_state.demo_decoder = DemoStateDecoder(
            device=str(configuration[-1]), motor_decoder=motor_decoder, compute_motor_intent=SHOW_MOTOR_INTENT
        )
    else:
        st.session_state.demo_decoder.set_motor_decoder(motor_decoder)
        st.session_state.demo_decoder.set_compute_motor_intent(SHOW_MOTOR_INTENT)


def resolve_device(requested_device: str) -> tuple[str, str | None]:
    if requested_device == "cpu":
        return "cpu", None
    import torch

    if requested_device == "cuda" and torch.cuda.is_available():
        return "cuda", None
    if requested_device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", None
    return "cpu", f"{requested_device.upper()} 不可用，已使用 CPU。"


def ensure_motor_decoder(
    *,
    mode: str,
    package_path: str | None,
    device: str,
) -> tuple[MotorIntentDecoder | None, str | None]:
    configuration = (mode, package_path, device)
    if st.session_state.get("demo_motor_configuration") == configuration:
        return st.session_state.demo_motor_decoder, st.session_state.demo_motor_error

    # A model change is a safe boundary: stop data flow but retain neural-state history.
    st.session_state.demo_playing = False
    st.session_state.demo_motor_configuration = configuration
    st.session_state.demo_motor_error = None
    try:
        if mode == "Demo Decoder":
            decoder: MotorIntentDecoder = DemoMotorIntentDecoder()
        else:
            if not package_path:
                raise ValueError("请选择一个 Model Package 路径。")
            with st.spinner("正在加载运动意图模型..."):
                decoder = ModelPackageMotorIntentDecoder(package_path, device=device)
        st.session_state.demo_motor_decoder = decoder
    except Exception as exc:  # noqa: BLE001 - package errors must stay in the sidebar.
        st.session_state.demo_motor_decoder = None
        st.session_state.demo_motor_error = str(exc)
    if st.session_state.demo_decoder is not None and st.session_state.demo_motor_decoder is not None:
        st.session_state.demo_decoder.set_motor_decoder(st.session_state.demo_motor_decoder)
    return st.session_state.demo_motor_decoder, st.session_state.demo_motor_error


for key, value in {
    "demo_decoder": None,
    "demo_result": None,
    "demo_history": {name: [] for name in STATE_LABELS_CN},
    "demo_playing": False,
    "demo_configuration": None,
    "demo_current_trial_index": 0,
    "demo_current_window_index": 0,
    "demo_last_trial_index": None,
    "demo_last_window_index": None,
    "demo_finished": False,
    "demo_motor_decoder": None,
    "demo_motor_configuration": None,
    "demo_motor_error": None,
    "demo_next_decode_stream_time": 0.0,
    "demo_trial_decode_complete": False,
    "demo_static_revision": 0,
    "demo_static_rendered_revision": -1,
    "demo_cortical_revision": 0,
    "demo_cortical_rendered_revision": -1,
    # Development-only execution counters. They make it possible to verify
    # that automatic data flow stays inside fragments rather than rerunning the
    # entire Streamlit script.
    "demo_app_run_count": 0,
    "demo_visual_fragment_run_count": 0,
    "demo_decode_fragment_run_count": 0,
}.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("demo_visual_state", VisualState())
st.session_state.demo_app_run_count += 1

defaults = command_line_defaults()
available_files = discover_hdf5()
default_data = defaults.data_path or (available_files[0] if available_files else str(ROOT / "data/processed/bnci2014_001/subject_01.h5"))

st.title("Omni Neural Decoder · 多维脑状态实时解码")
st.markdown('<div class="demo-subtitle">一个 EEG 输入，多维神经状态同步解析</div>', unsafe_allow_html=True)
st.markdown(
    f'<span class="demo-debug-counter demo-debug-app" data-app-runs="{st.session_state.demo_app_run_count}"></span>',
    unsafe_allow_html=True,
)
motor_package_options = discover_motor_packages()

with st.sidebar:
    st.header("数据流设置")
    data_path = st.text_input("EEG HDF5 数据路径", default_data)
    try:
        sessions = list_demo_sessions(data_path)
        data_error = None
    except Exception as exc:  # noqa: BLE001
        sessions, data_error = [], str(exc)
    if data_error:
        st.error(f"无法读取数据：{data_error}")
        st.stop()
    session = st.selectbox("Session", sessions)
    count = trial_count(data_path, session)
    trial_options: list[str | int] = ["全部 Trial", *range(count)]
    trial_selection = st.selectbox(
        "Trial",
        trial_options,
        format_func=lambda value: value if isinstance(value, str) else f"Trial {value + 1}",
    )
    selected_window_sec = st.select_slider("窗口长度", options=[1.0, 1.5, 2.0, 3.0, 4.0], value=2.0)
    step_sec = st.select_slider("推进步长", options=[0.1, 0.2, 0.25, 0.5, 0.75, 1.0], value=0.5)
    speed = st.select_slider("数据流速度", options=[0.5, 1.0, 2.0, 4.0], value=1.0, format_func=lambda value: f"{value:g}x")

    st.divider()
    st.subheader("运动意图模型")
    decoder_modes = ["Model Package", "Demo Decoder"] if motor_package_options else ["Demo Decoder"]
    decoder_mode = st.selectbox("解码器", decoder_modes)
    selected_package_path: str | None = None
    if decoder_mode == "Model Package":
        labels_by_path = {str(option["path"]): str(option["label"]) for option in motor_package_options}
        package_choices = [*labels_by_path, "__custom__"]
        selected_package_choice = st.selectbox(
            "Model Package",
            package_choices,
            format_func=lambda value: "自定义路径..." if value == "__custom__" else labels_by_path[value],
        )
        if selected_package_choice == "__custom__":
            selected_package_path = st.text_input("Model Package 路径")
        else:
            selected_package_path = selected_package_choice
    else:
        st.caption("当前使用平滑 Demo Decoder。")

    requested_device = st.selectbox("计算设备", ["cpu", "cuda", "mps"], index=["cpu", "cuda", "mps"].index(defaults.device.lower()) if defaults.device.lower() in {"cpu", "cuda", "mps"} else 0)
    device, device_notice = resolve_device(requested_device)
    motor_decoder, motor_error = ensure_motor_decoder(mode=decoder_mode, package_path=selected_package_path, device=device)
    if device_notice:
        st.caption(device_notice)
    if motor_error:
        st.error("模型加载失败，请检查 Model Package。")
        with st.expander("开发详情"):
            st.code(motor_error)
    elif motor_decoder is not None:
        if isinstance(motor_decoder, ModelPackageMotorIntentDecoder):
            st.caption(
                f"模型已就绪 · {motor_decoder.display_name} · 输入窗口 {motor_decoder.window_sec:g}s / "
                f"{motor_decoder.target_sample_rate:g}Hz（由 Package 决定）"
            )
        else:
            st.caption("Demo Decoder 已就绪。")

    st.divider()
    start_col, pause_col, reset_col = st.columns(3)
    start = start_col.button("开始 / 继续", type="primary", use_container_width=True, disabled=motor_decoder is None)
    pause = pause_col.button("暂停", use_container_width=True)
    reset = reset_col.button("重置", use_container_width=True)
    st.caption("神经状态指标来自当前 EEG 窗口的传统特征。")

selected_trial_indices = list(range(count)) if trial_selection == "全部 Trial" else [int(trial_selection)]
window_sec = float(motor_decoder.window_sec) if isinstance(motor_decoder, ModelPackageMotorIntentDecoder) else float(selected_window_sec)
configuration = (data_path, session, tuple(selected_trial_indices), window_sec, step_sec, device)
if motor_decoder is None:
    st.stop()
ensure_decoder(configuration, selected_trial_indices[0], motor_decoder)
preview_trial = cached_trial(data_path, session, selected_trial_indices[0])
if window_count(preview_trial, window_sec, step_sec) == 0:
    st.error("当前 trial 短于所选窗口长度。")
    st.stop()
with st.sidebar:
    display_channels = st.multiselect(
        "波形显示通道",
        preview_trial.channel_names,
        default=preview_trial.channel_names[: min(6, len(preview_trial.channel_names))],
        max_selections=6,
    )
if start:
    st.session_state.demo_playing = True
    st.session_state.demo_finished = False
    st.session_state.demo_visual_state.pause()
    st.session_state.demo_static_revision += 1
if pause:
    st.session_state.demo_playing = False
    st.session_state.demo_visual_state.pause()
    st.session_state.demo_static_revision += 1
if reset:
    reset_replay(selected_trial_indices[0])
    st.session_state.demo_configuration = configuration
    st.session_state.demo_decoder = DemoStateDecoder(
        device=device, motor_decoder=motor_decoder, compute_motor_intent=SHOW_MOTOR_INTENT
    )


def decode_current_frame() -> None:
    current_trial_index = st.session_state.demo_current_trial_index
    current_window_index = st.session_state.demo_current_window_index
    trial = cached_trial(data_path, session, current_trial_index)
    total_frames = window_count(trial, window_sec, step_sec)
    if total_frames == 0:
        raise ValueError(f"Trial {current_trial_index + 1} is shorter than the selected window")
    window = trial_window(trial, current_window_index, window_sec=window_sec, step_sec=step_sec)
    result = st.session_state.demo_decoder.decode_window(
        DemoEEGWindow(
            samples=window,
            sample_rate=trial.sample_rate,
            channel_names=trial.channel_names,
            unit=trial.unit,
            timestamp=(current_trial_index * trial.samples.shape[-1] + current_window_index * step_sec * trial.sample_rate) / trial.sample_rate,
            montage_name="bnci_22",
        )
    )
    st.session_state.demo_result = result
    cortical = result.cortical_activity
    st.session_state.demo_visual_state.set_decode_targets(
        result.psd_values,
        None if cortical is None else cortical.left_rgba,
        None if cortical is None else cortical.right_rgba,
        cortical_available=bool(cortical is not None and cortical.available),
    )
    # The cortical mapper already smooths its target with decode-time EMA.
    # Re-render this media element only for a new decoder result, not on every
    # 15 Hz visual tick; this keeps both hemispheres in one stable frame.
    st.session_state.demo_cortical_revision += 1
    for name, value in result.states.items():
        history = st.session_state.demo_history.setdefault(name, [])
        history.append(float(value))
        del history[:-24]
    st.session_state.demo_last_trial_index = current_trial_index
    st.session_state.demo_last_window_index = current_window_index
    st.session_state.demo_static_revision += 1
    st.session_state.demo_current_window_index += 1
    if st.session_state.demo_current_window_index >= total_frames:
        st.session_state.demo_trial_decode_complete = True


def current_trial_frame_count() -> int:
    trial = cached_trial(data_path, session, st.session_state.demo_current_trial_index)
    return window_count(trial, window_sec, step_sec)


def advance_trial_if_ready() -> bool:
    """Move only at a stream-time boundary, preserving visual target arrays."""
    if not st.session_state.demo_trial_decode_complete:
        return False
    trial_frames = current_trial_frame_count()
    trial_duration = trial_frames * step_sec
    visual_state = st.session_state.demo_visual_state
    if visual_state.stream_time_sec < trial_duration:
        return False
    sequence_position = selected_trial_indices.index(st.session_state.demo_current_trial_index)
    if sequence_position + 1 >= len(selected_trial_indices):
        st.session_state.demo_playing = False
        st.session_state.demo_finished = True
        visual_state.stream_time_sec = trial_duration
        visual_state.pause()
        st.session_state.demo_static_revision += 1
        return True
    visual_state.stream_time_sec -= trial_duration
    st.session_state.demo_current_trial_index = selected_trial_indices[sequence_position + 1]
    st.session_state.demo_current_window_index = 0
    st.session_state.demo_next_decode_stream_time = 0.0
    st.session_state.demo_trial_decode_complete = False
    return True


def decode_due_frames() -> None:
    """Run decode only on its own stream-time clock, with bounded catch-up."""
    visual_state = st.session_state.demo_visual_state
    catchup = 0
    while st.session_state.demo_playing and catchup < MAX_DECODE_CATCHUP:
        if advance_trial_if_ready():
            continue
        if st.session_state.demo_trial_decode_complete:
            break
        if visual_state.stream_time_sec + 1e-9 < st.session_state.demo_next_decode_stream_time:
            break
        decode_current_frame()
        st.session_state.demo_next_decode_stream_time += step_sec
        catchup += 1
    if catchup >= MAX_DECODE_CATCHUP and st.session_state.demo_playing and not st.session_state.demo_trial_decode_complete:
        overdue = visual_state.stream_time_sec - st.session_state.demo_next_decode_stream_time
        if overdue >= 0.0:
            skipped = int(overdue // step_sec) + 1
            st.session_state.demo_current_window_index += skipped
            st.session_state.demo_next_decode_stream_time += skipped * step_sec
            if st.session_state.demo_current_window_index >= current_trial_frame_count():
                st.session_state.demo_current_window_index = current_trial_frame_count()
                st.session_state.demo_trial_decode_complete = True


def sparkline_svg(values: list[float]) -> str:
    if len(values) < 2:
        return '<svg class="state-spark" viewBox="0 0 48 16"></svg>'
    points = []
    for index, value in enumerate(values[-24:]):
        x = 48 * index / (min(len(values), 24) - 1)
        y = 15 - np.clip(value, 0.0, 100.0) * 0.13
        points.append(f"{x:.1f},{y:.1f}")
    return f'<svg class="state-spark" viewBox="0 0 48 16"><polyline fill="none" stroke="#00A6A6" stroke-width="1.5" points="{" ".join(points)}"/></svg>'


def render_dashboard(result) -> None:
    motor = result.motor_intent
    displayed_trial = st.session_state.demo_last_trial_index
    displayed_window = st.session_state.demo_last_window_index
    assert displayed_trial is not None and displayed_window is not None
    displayed_trial_data = cached_trial(data_path, session, displayed_trial)
    displayed_total_windows = window_count(displayed_trial_data, window_sec, step_sec)
    sequence_position = selected_trial_indices.index(displayed_trial)
    sequence_progress = (sequence_position + (displayed_window + 1) / displayed_total_windows) / len(selected_trial_indices)
    if st.session_state.demo_finished:
        status, status_class = "● 数据流结束", "done"
    elif st.session_state.demo_playing:
        status, status_class = "● EEG 数据流", "live"
    else:
        status, status_class = "● 数据流已暂停", ""
    st.markdown(
        f'<div class="demo-status"><span class="{status_class}">{status}</span>'
        f'<span class="item">Session {session} · Trial {displayed_trial + 1} / {count}</span>'
        f'<span class="item">窗口 {displayed_window + 1} / {displayed_total_windows}</span>'
        f'<span class="item">进度 {sequence_progress:.0%}</span><span class="item">{result.device}</span>'
        f'<span class="item">信号 {result.signal_quality:.0f}%</span><span class="item">延迟 {result.latency_ms:.1f} ms · P95 {result.p95_latency_ms:.1f} ms</span></div>',
        unsafe_allow_html=True,
    )

    if SHOW_MOTOR_INTENT:
        probability_rows = "".join(
            f'<div class="prob-row"><span>{MOTOR_LABELS_CN[label]}</span><div class="prob-track"><div class="prob-fill" style="width:{float(value) * 100:.1f}%"></div></div><span>{float(value):.0%}</span></div>'
            for label, value in ((label, motor["probabilities"][label]) for label in ("left_hand", "right_hand", "feet", "tongue"))
        )
        core_left, core_right = st.columns([1.12, 1], gap="small")
        with core_left:
            decoder_name = str(motor.get("decoder_display_name", "Demo"))
            st.markdown(
                f'<div class="core-card"><div class="core-title">运动意图 · Decoder {decoder_name} · 综合状态 {result.brain_state_score:.0f} · {score_level(result.brain_state_score)}</div>'
                f'<div class="motor-head"><span class="motor-label">{motor["label_cn"]}</span><span class="motor-confidence">{float(motor["confidence"]):.1%}</span></div>{probability_rows}</div>',
                unsafe_allow_html=True,
            )
        with core_right:
            st.markdown(
                f'<div class="core-card"><div class="core-title">AI 神经状态解读 · 综合状态 {result.brain_state_score:.0f} · {score_level(result.brain_state_score)}</div><div class="interpretation">{result.interpretation}</div>'
                '<div class="disclaimer">规则生成 · 研究演示 · 非医学结论</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f'<div class="core-card"><div class="core-title">AI 神经状态解读 · 综合状态 {result.brain_state_score:.0f} · {score_level(result.brain_state_score)}</div><div class="interpretation">{result.interpretation}</div>'
            '<div class="disclaimer">规则生成 · 研究演示 · 非医学结论</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">Neural State Index</div>', unsafe_allow_html=True)
    state_items = list(STATE_LABELS_CN.items())
    for row_start in range(0, len(state_items), 8):
        cards = st.columns(8, gap="small")
        for column, (name, chinese_name) in zip(cards, state_items[row_start : row_start + 8], strict=True):
            value = result.states[name]
            history = st.session_state.demo_history.get(name, [])
            with column:
                if name == "emotion_state" and result.emotion is not None:
                    st.markdown(
                        f'<div class="state-card"><div class="state-top"><span class="state-name">{chinese_name}</span><span class="emotion-emoji">{result.emotion.emoji}</span></div>'
                        f'<div class="state-bottom"><span class="emotion-label">{result.emotion.label_cn}</span>{sparkline_svg(history)}</div>'
                        f'<div class="state-track"><div class="state-fill" style="width:{value:.1f}%"></div></div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="state-card"><div class="state-top"><span class="state-name">{chinese_name}</span><span class="state-value">{value:.0f}</span></div>'
                        f'<div class="state-bottom"><span class="state-level">{score_level(value)}</span>{sparkline_svg(history)}</div>'
                        f'<div class="state-track"><div class="state-fill" style="width:{value:.1f}%"></div></div></div>',
                        unsafe_allow_html=True,
                    )

def visual_waveform() -> tuple[np.ndarray, list[str], float, str]:
    """Slice the current trial by stream time instead of decoder-window index."""
    trial = cached_trial(data_path, session, st.session_state.demo_current_trial_index)
    visible_samples = min(trial.samples.shape[1], max(2, int(round(window_sec * trial.sample_rate))))
    cursor = int(st.session_state.demo_visual_state.stream_time_sec * trial.sample_rate)
    start = min(max(0, cursor), max(0, trial.samples.shape[1] - visible_samples))
    return trial.samples[:, start : start + visible_samples], trial.channel_names, trial.sample_rate, trial.unit


def _polyline_points(x_values: np.ndarray, y_values: np.ndarray, *, width: float, height: float) -> str:
    return " ".join(f"{x * width:.1f},{y * height:.1f}" for x, y in zip(x_values, y_values, strict=True))


def waveform_svg(display: np.ndarray, names: list[str]) -> str:
    """Small SVG renderer avoids creating a Matplotlib figure for every visual tick."""
    max_points = 320
    if display.shape[0] > max_points:
        indices = np.linspace(0, display.shape[0] - 1, max_points, dtype=int)
        display = display[indices]
    width, height, left = 520.0, 150.0, 28.0
    x = np.linspace(left / width, 0.99, display.shape[0])
    channel_height = 0.86 / max(1, display.shape[1])
    paths: list[str] = []
    labels: list[str] = []
    for index, name in enumerate(names):
        values = display[:, index]
        amplitude = max(float(np.ptp(values)), 1e-6)
        center = 0.07 + channel_height * (index + 0.5)
        y = center - 0.70 * channel_height * (values - np.mean(values)) / amplitude
        paths.append(f'<polyline fill="none" stroke="#00A6A6" stroke-width="0.8" points="{_polyline_points(x, y, width=width, height=height)}"/>')
        labels.append(f'<text x="2" y="{height * (center + 0.02):.1f}" fill="#8293A1" font-size="7">{name}</text>')
    grids = "".join(f'<line x1="{left}" y1="{height * (0.07 + channel_height * index):.1f}" x2="{width * .99}" y2="{height * (0.07 + channel_height * index):.1f}" stroke="#263849" stroke-width="0.5"/>' for index in range(display.shape[1] + 1))
    return f'<svg viewBox="0 0 {width:.0f} {height:.0f}" style="width:100%;height:150px;display:block;background:#0E1117">{grids}{"".join(labels)}{"".join(paths)}</svg>'


def waveform_html() -> str:
    waveform, all_names, sample_rate, unit = visual_waveform()
    selected_indices = [all_names.index(name) for name in display_channels if name in all_names]
    selected_indices = selected_indices or list(range(min(6, waveform.shape[0])))
    display = waveform[selected_indices].T
    if unit.lower() == "v":
        display = display * 1e6
    names = [all_names[index] for index in selected_indices]
    del sample_rate
    return waveform_svg(display, names)


def render_waveform_visual() -> None:
    st.markdown(waveform_html(), unsafe_allow_html=True)


def psd_svg(frequencies: np.ndarray, values: np.ndarray) -> str:
    mask = (frequencies >= 1.0) & (frequencies <= 45.0)
    x_values = frequencies[mask]
    db = 10.0 * np.log10(values[mask] + 1e-12)
    if x_values.size > 180:
        indices = np.linspace(0, x_values.size - 1, 180, dtype=int)
        x_values, db = x_values[indices], db[indices]
    low, high = np.percentile(db, (2.0, 98.0))
    y_values = 0.92 - 0.80 * np.clip((db - low) / max(high - low, 1e-6), 0.0, 1.0)
    x_normalized = 0.08 + 0.90 * (x_values - 1.0) / 44.0
    points = _polyline_points(x_normalized, y_values, width=420.0, height=150.0)
    grids = "".join(f'<line x1="34" y1="{row}" x2="412" y2="{row}" stroke="#263849" stroke-width="0.5"/>' for row in (22, 54, 86, 118, 142))
    return f'<svg viewBox="0 0 420 150" style="width:100%;height:150px;display:block;background:#0E1117">{grids}<polyline fill="none" stroke="#00A6A6" stroke-width="1.2" points="{points}"/></svg>'


def psd_html(result, displayed_psd: np.ndarray | None) -> str:
    chart = ""
    if result.psd_frequencies is not None and displayed_psd is not None:
        chart = psd_svg(result.psd_frequencies, displayed_psd)
    labels = {"delta": "δ", "theta": "θ", "alpha": "α", "beta": "β", "gamma": "γ"}
    bands = " · ".join(f"{labels[name]} {value:.0%}" for name, value in result.band_power.items())
    return f'{chart}<div class="signal-caption">{bands}</div>'


def render_psd_visual(result, displayed_psd: np.ndarray | None) -> None:
    st.markdown(psd_html(result, displayed_psd), unsafe_allow_html=True)


def render_cortical_visual(visual_state: VisualState) -> None:
    left, right = visual_state.displayed_cortical_rgba("left"), visual_state.displayed_cortical_rgba("right")
    if not visual_state.cortical_available:
        st.caption("暂无可用皮层映射")
        return
    if left is None or right is None:
        return
    # One Streamlit image avoids the independent left/right media lifecycle that
    # could briefly drop the right hemisphere during a high-frequency rerun.
    combined = np.concatenate((left[::2, ::2], right[::2, ::2]), axis=1)
    st.markdown(
        '<div class="signal-caption" style="display:flex;justify-content:space-around"><span>左半球</span><span>右半球</span></div>',
        unsafe_allow_html=True,
    )
    st.image(combined, width="stretch")


decode_run_every = step_sec if st.session_state.demo_playing else None
visual_run_every = VISUAL_INTERVAL_SEC if st.session_state.demo_playing else None


@st.fragment(run_every=decode_run_every)
def render_decode_fragment() -> None:
    """Decoder-rate controller and state-index dashboard."""
    visual_state = st.session_state.demo_visual_state
    if st.session_state.demo_playing:
        st.session_state.demo_decode_fragment_run_count += 1
        # `stream_time_sec` is advanced solely by the visual clock. This
        # fragment reads that shared cursor and performs the bounded decoder
        # catchup without requesting a full-app rerun.
        decode_due_frames()

    st.markdown(
        f'<span class="demo-debug-counter demo-debug-decode" data-runs="{st.session_state.demo_decode_fragment_run_count}"></span>',
        unsafe_allow_html=True,
    )

    result = st.session_state.demo_result
    if result is None:
        st.info("选择 Trial 后点击“开始 / 继续”启动数据流。")
        return


    # Direct fragment output is scoped to this dashboard position. In
    # particular, no fragment writes into a placeholder owned by another
    # fragment, avoiding Streamlit's invalid-delta-path/front-end flicker.
    render_dashboard(result)


@st.fragment(run_every=visual_run_every)
def render_visual_fragment() -> None:
    """Visual-rate fragment: continuous waveform plus interpolated PSD only."""
    visual_state = st.session_state.demo_visual_state
    if st.session_state.demo_playing:
        st.session_state.demo_visual_fragment_run_count += 1
        visual_state.advance(time.monotonic(), float(speed))
        # PSD targets are installed by `render_decode_fragment`; interpolation
        # is intentionally visual-only. Cortical maps remain decoder-rate.
        visual_state.interpolate(decode_interval_sec=step_sec)
    st.markdown(
        f'<span class="demo-debug-counter demo-debug-visual" data-runs="{st.session_state.demo_visual_fragment_run_count}"></span>',
        unsafe_allow_html=True,
    )

    result = st.session_state.demo_result
    if result is None:
        st.markdown(
            '<div class="visual-signal-grid"><div class="visual-signal-pane"><div class="signal-title">实时 EEG 波形</div><div class="signal-caption">连续数据流 · μV · 偏移显示</div></div>'
            '<div class="visual-signal-pane"><div class="signal-title">神经频谱</div><div class="signal-caption">1–45 Hz · 平滑显示</div></div></div>',
            unsafe_allow_html=True,
        )
        return

    started = time.perf_counter()
    waveform = waveform_html()
    visual_state.waveform_render_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    spectrum = psd_html(result, visual_state.displayed_psd)
    visual_state.psd_render_ms = (time.perf_counter() - started) * 1000.0
    st.markdown(
        '<div class="visual-signal-grid">'
        '<div class="visual-signal-pane"><div class="signal-title">实时 EEG 波形</div><div class="signal-caption">连续数据流 · μV · 偏移显示</div>'
        f'{waveform}</div>'
        '<div class="visual-signal-pane"><div class="signal-title">神经频谱</div><div class="signal-caption">1–45 Hz · 平滑显示</div>'
        f'{spectrum}</div></div>',
        unsafe_allow_html=True,
    )


@st.fragment(run_every=decode_run_every)
def render_cortical_fragment() -> None:
    """Decode-rate cortical output, using the mapper's existing EMA target."""
    st.markdown('<div class="signal-title">皮层活动图</div>', unsafe_allow_html=True)
    visual_state = st.session_state.demo_visual_state
    if visual_state.displayed_cortical_left is None or visual_state.displayed_cortical_right is None:
        return
    started = time.perf_counter()
    render_cortical_visual(visual_state)
    visual_state.cortical_render_ms = (time.perf_counter() - started) * 1000.0


render_decode_fragment()
st.markdown('<div class="signal-section">神经信号分析</div>', unsafe_allow_html=True)
signal_visual_column, map_column = st.columns([2.40, 1.02], gap="small")
with signal_visual_column:
    render_visual_fragment()
with map_column:
    render_cortical_fragment()
