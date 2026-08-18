"""Streamlit EEG multi-state visualization demo, isolated from production UI."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from bci_dayloop.data.hdf5_dataset import EEGHDF5  # noqa: E402
from bci_dayloop.demo.input import list_demo_sessions, load_demo_trial, trial_window, window_count  # noqa: E402
from bci_dayloop.demo.model_package_discovery import discover_motor_intent_packages  # noqa: E402
from bci_dayloop.demo.motor_decoder import DemoMotorIntentDecoder, ModelPackageMotorIntentDecoder, MotorIntentDecoder  # noqa: E402
from bci_dayloop.demo.schemas import MOTOR_LABELS_CN, STATE_LABELS_CN  # noqa: E402
from bci_dayloop.demo.state_decoder import DemoStateDecoder  # noqa: E402
from bci_dayloop.demo.utils import score_level  # noqa: E402


st.set_page_config(page_title="Omni Neural Decoder", page_icon=":material/neurology:", layout="wide")

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
.signal-section {font-size:0.80rem; font-weight:600; color:#DCE6ED; margin:1.35rem 0 0.23rem;}
.signal-title {font-size:0.80rem; font-weight:600; color:#DCE6ED; margin:0 0 0.05rem;}.signal-caption {font-size:0.62rem; color:#8293A1; margin:0 0 0.08rem;}
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
    st.session_state.demo_last_tick = 0.0
    st.session_state.demo_current_trial_index = start_trial_index
    st.session_state.demo_current_window_index = 0
    st.session_state.demo_last_trial_index = None
    st.session_state.demo_last_window_index = None
    st.session_state.demo_finished = False


def ensure_decoder(
    configuration: tuple[object, ...],
    start_trial_index: int,
    motor_decoder: MotorIntentDecoder,
) -> None:
    if st.session_state.get("demo_configuration") != configuration:
        reset_replay(start_trial_index)
        st.session_state.demo_configuration = configuration
        st.session_state.demo_decoder = DemoStateDecoder(device=str(configuration[-1]), motor_decoder=motor_decoder)
    elif st.session_state.demo_decoder is None:
        st.session_state.demo_decoder = DemoStateDecoder(device=str(configuration[-1]), motor_decoder=motor_decoder)
    else:
        st.session_state.demo_decoder.set_motor_decoder(motor_decoder)


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
    "demo_last_tick": 0.0,
    "demo_configuration": None,
    "demo_current_trial_index": 0,
    "demo_current_window_index": 0,
    "demo_last_trial_index": None,
    "demo_last_window_index": None,
    "demo_finished": False,
    "demo_motor_decoder": None,
    "demo_motor_configuration": None,
    "demo_motor_error": None,
}.items():
    st.session_state.setdefault(key, value)

defaults = command_line_defaults()
available_files = discover_hdf5()
default_data = defaults.data_path or (available_files[0] if available_files else str(ROOT / "data/processed/bnci2014_001/subject_01.h5"))

st.title("Omni Neural Decoder · 多维脑状态实时解码")
st.markdown('<div class="demo-subtitle">一个 EEG 输入，多维神经状态同步解析</div>', unsafe_allow_html=True)
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
    step_sec = st.select_slider("推进步长", options=[0.25, 0.5, 0.75, 1.0], value=0.5)
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
    st.session_state.demo_last_tick = 0.0
if pause:
    st.session_state.demo_playing = False
if reset:
    reset_replay(selected_trial_indices[0])
    st.session_state.demo_configuration = configuration
    st.session_state.demo_decoder = DemoStateDecoder(device=device, motor_decoder=motor_decoder)


def decode_current_frame() -> None:
    current_trial_index = st.session_state.demo_current_trial_index
    current_window_index = st.session_state.demo_current_window_index
    trial = cached_trial(data_path, session, current_trial_index)
    total_frames = window_count(trial, window_sec, step_sec)
    if total_frames == 0:
        raise ValueError(f"Trial {current_trial_index + 1} is shorter than the selected window")
    window = trial_window(trial, current_window_index, window_sec=window_sec, step_sec=step_sec)
    result = st.session_state.demo_decoder.decode(
        window,
        sample_rate=trial.sample_rate,
        channel_names=trial.channel_names,
        unit=trial.unit,
        timestamp=(current_trial_index * trial.samples.shape[-1] + current_window_index * step_sec * trial.sample_rate) / trial.sample_rate,
    )
    st.session_state.demo_result = result
    for name, value in result.states.items():
        history = st.session_state.demo_history.setdefault(name, [])
        history.append(float(value))
        del history[:-24]
    st.session_state.demo_last_trial_index = current_trial_index
    st.session_state.demo_last_window_index = current_window_index
    st.session_state.demo_current_window_index += 1
    if st.session_state.demo_current_window_index >= total_frames:
        sequence_position = selected_trial_indices.index(current_trial_index)
        if sequence_position + 1 < len(selected_trial_indices):
            st.session_state.demo_current_trial_index = selected_trial_indices[sequence_position + 1]
            st.session_state.demo_current_window_index = 0
        else:
            st.session_state.demo_playing = False
            st.session_state.demo_finished = True


def sparkline_svg(values: list[float]) -> str:
    if len(values) < 2:
        return '<svg class="state-spark" viewBox="0 0 48 16"></svg>'
    points = []
    for index, value in enumerate(values[-24:]):
        x = 48 * index / (min(len(values), 24) - 1)
        y = 15 - np.clip(value, 0.0, 100.0) * 0.13
        points.append(f"{x:.1f},{y:.1f}")
    return f'<svg class="state-spark" viewBox="0 0 48 16"><polyline fill="none" stroke="#00A6A6" stroke-width="1.5" points="{" ".join(points)}"/></svg>'


def render_result(result) -> None:
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
            f'<div class="core-card"><div class="core-title">AI 神经状态解读</div><div class="interpretation">{result.interpretation}</div>'
            '<div class="disclaimer">规则生成 · 研究演示 · 非医学结论</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">Neural State Index</div>', unsafe_allow_html=True)
    cards = st.columns(8, gap="small")
    for column, (name, chinese_name) in zip(cards, STATE_LABELS_CN.items(), strict=True):
        value = result.states[name]
        history = st.session_state.demo_history.get(name, [])
        with column:
            st.markdown(
                f'<div class="state-card"><div class="state-top"><span class="state-name">{chinese_name}</span><span class="state-value">{value:.0f}</span></div>'
                f'<div class="state-bottom"><span class="state-level">{score_level(value)}</span>{sparkline_svg(history)}</div>'
                f'<div class="state-track"><div class="state-fill" style="width:{value:.1f}%"></div></div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="signal-section">神经信号分析</div>', unsafe_allow_html=True)
    waveform_column, spectrum_column, map_column = st.columns([1.45, 1.05, 0.82], gap="small")
    with waveform_column:
        st.markdown('<div class="signal-title">实时 EEG 波形</div><div class="signal-caption">当前窗口 · μV · 偏移显示</div>', unsafe_allow_html=True)
        waveform = result.waveform
        if waveform is not None:
            all_names = result.channel_names or [f"Ch {index + 1}" for index in range(waveform.shape[0])]
            selected_indices = [all_names.index(name) for name in display_channels if name in all_names]
            selected_indices = selected_indices or list(range(min(6, waveform.shape[0])))
            display = waveform[selected_indices].T
            if (result.waveform_unit or "").lower() == "v":
                display = display * 1e6
            names = [all_names[index] for index in selected_indices]
            time_axis = np.arange(display.shape[0]) / float(result.sample_rate or 1.0)
            fig, axis = plt.subplots(figsize=(5.2, 1.55), facecolor="#0E1117")
            axis.set_facecolor("#0E1117")
            offsets = np.ptp(display, axis=0).mean() * np.arange(len(selected_indices))
            offsets = offsets if np.any(offsets) else np.arange(len(selected_indices)) * 20.0
            for index, name in enumerate(names):
                axis.plot(time_axis, display[:, index] + offsets[index], linewidth=0.58, label=name)
            axis.tick_params(labelsize=6, length=2)
            axis.grid(alpha=0.12)
            axis.legend(loc="upper right", ncol=2, fontsize=5.5, frameon=False)
            fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.20)
            st.pyplot(fig, clear_figure=True, use_container_width=True)
    with spectrum_column:
        st.markdown('<div class="signal-title">神经频谱</div><div class="signal-caption">1–45 Hz · 当前窗口 PSD</div>', unsafe_allow_html=True)
        if result.psd_frequencies is not None and result.psd_values is not None:
            psd = 10.0 * np.log10(result.psd_values + 1e-12)
            fig, axis = plt.subplots(figsize=(4.2, 1.55), facecolor="#0E1117")
            axis.set_facecolor("#0E1117")
            axis.plot(result.psd_frequencies, psd, color="#00A6A6", linewidth=1.05)
            axis.set(xlim=(1, 45))
            axis.tick_params(labelsize=6, length=2)
            axis.grid(alpha=0.12)
            fig.subplots_adjust(left=0.12, right=0.99, top=0.96, bottom=0.20)
            st.pyplot(fig, clear_figure=True, use_container_width=True)
        labels = {"delta": "δ", "theta": "θ", "alpha": "α", "beta": "β", "gamma": "γ"}
        bands = " · ".join(f"{labels[name]} {value:.0%}" for name, value in result.band_power.items())
        st.markdown(f'<div class="signal-caption">{bands}</div>', unsafe_allow_html=True)
    with map_column:
        st.markdown('<div class="signal-title">脑区活动地图</div><div class="signal-caption">Alpha Power Topomap</div>', unsafe_allow_html=True)
        if result.topomap_values is not None and result.topomap_positions is not None:
            try:
                import mne

                fig, axis = plt.subplots(figsize=(2.45, 1.8), facecolor="#0E1117")
                axis.set_facecolor("#0E1117")
                mne.viz.plot_topomap(result.topomap_values, result.topomap_positions, axes=axis, show=False, contours=3, cmap="viridis")
                fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
                st.pyplot(fig, clear_figure=True, use_container_width=True)
            except Exception:  # pragma: no cover - defensive display fallback.
                fig, axis = plt.subplots(figsize=(2.45, 1.8), facecolor="#0E1117")
                axis.set_facecolor("#0E1117")
                axis.scatter(result.topomap_positions[:, 0], result.topomap_positions[:, 1], c=result.topomap_values, s=54, cmap="viridis")
                axis.add_patch(plt.Circle((0, 0), 1.04, fill=False, color="#F3F6F9", linewidth=0.8))
                axis.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), aspect="equal")
                axis.axis("off")
                fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
                st.pyplot(fig, clear_figure=True, use_container_width=True)
        else:
            st.caption("当前通道没有可用 montage。")


@st.fragment(run_every=0.25)
def live_demo() -> None:
    now = time.monotonic()
    interval = 0.55 / float(speed)
    should_decode = st.session_state.demo_playing and now - st.session_state.demo_last_tick >= interval
    if should_decode:
        st.session_state.demo_last_tick = now
        decode_current_frame()
    result = st.session_state.demo_result
    if result is not None:
        render_result(result)
    else:
        st.info("选择 Trial 后点击“开始 / 继续”启动数据流。")


live_demo()
