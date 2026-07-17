from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bci_dayloop.acquisition.factory import AcquirerFactory  # noqa: E402
from bci_dayloop.data.hdf5_dataset import EEGHDF5  # noqa: E402
from bci_dayloop.data.preprocessing import EEGPreprocessor  # noqa: E402
from bci_dayloop.inference.realtime import SlidingWindowDecoder  # noqa: E402
from bci_dayloop.models.factory import ModelFactory  # noqa: E402
from bci_dayloop.utils.config import load_yaml  # noqa: E402

st.set_page_config(page_title="BCI DayLoop", page_icon=":material/neurology:", layout="wide")


@st.cache_data(max_entries=16)
def describe_data(path: str) -> tuple[list[str], list[str], float, str]:
    dataset = EEGHDF5(path)
    metadata = dataset.metadata
    return dataset.sessions(), metadata.class_names, metadata.sample_rate, metadata.unit


@st.cache_resource(max_entries=4)
def load_model_package(path: str, device: str):
    return ModelFactory.load_package(path, device=device)


def discover_hdf5() -> list[str]:
    return [str(path) for path in sorted((ROOT / "data" / "processed").glob("*.h5"))]


def discover_packages() -> list[str]:
    return [str(path.parent) for path in sorted((ROOT / "runs").glob("*/model_package/model.yaml"))]


for key, value in {
    "running": False,
    "acquirer": None,
    "decoder": None,
    "history": [],
    "waveform": None,
    "last_result": None,
    "windows_seen": 0,
    "runtime_error": None,
}.items():
    st.session_state.setdefault(key, value)

st.title("BCI DayLoop")
st.caption("BNCI2014_001 · LaBraM Base frozen encoder · linear probe · pseudo-realtime replay")

data_files = discover_hdf5()
packages = discover_packages()
with st.sidebar:
    st.header("Replay setup")
    if not data_files:
        st.warning("No HDF5 file found under data/processed.")
        data_path = st.text_input("Data file", str(ROOT / "data" / "processed" / "bnci2014_001_s01.h5"))
    else:
        data_path = st.selectbox("Data file", data_files)
    if not packages:
        st.warning("No model package found under runs.")
        package_path = st.text_input("Model package", str(ROOT / "runs" / "day1_bnci_s01" / "model_package"))
    else:
        package_path = st.selectbox("Model package", packages)
    acquirer_name = st.selectbox("Acquirer", AcquirerFactory.list_acquirers())
    model_name = st.selectbox("Model", ModelFactory.list_models())
    device = st.segmented_control("Compute device", ["cuda", "cpu"], default="cuda")
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.55, 0.01)
    replay_speed = st.number_input("Replay speed", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
    max_windows = st.number_input("Maximum windows", min_value=1, max_value=10000, value=100, step=10)

sessions: list[str] = []
try:
    sessions, class_names, sample_rate, input_unit = describe_data(data_path)
except Exception as exc:  # noqa: BLE001
    st.error(f"Cannot read data file: {exc}")
    class_names, sample_rate, input_unit = [], 0.0, "V"
session = st.sidebar.selectbox("Session", sessions, index=max(0, len(sessions) - 1)) if sessions else None

with st.sidebar.container(horizontal=True):
    start_clicked = st.button("Start replay", type="primary", icon=":material/play_arrow:", disabled=st.session_state.running)
    stop_clicked = st.button("Stop", icon=":material/stop:", disabled=not st.session_state.running)

if start_clicked:
    try:
        package = Path(package_path)
        model_yaml = load_yaml(package / "model.yaml")
        if model_yaml.get("name") != model_name:
            raise ValueError(f"Package contains {model_yaml.get('name')}, not {model_name}")
        model = load_model_package(str(package.resolve()), str(device))
        preprocessing = EEGPreprocessor(load_yaml(package / "preprocessing.yaml"))
        with (package / "command_map.json").open("r", encoding="utf-8") as handle:
            command_map = json.load(handle)
        acquirer = AcquirerFactory.create(
            acquirer_name,
            data_path=data_path,
            session=str(session),
            speed=float(replay_speed),
            loop=False,
            window_sec=4.0,
            step_sec=0.5,
        )
        decoder = SlidingWindowDecoder(
            model,
            preprocessing,
            class_names,
            sample_rate=sample_rate,
            input_unit=input_unit,
            window_sec=4.0,
            step_sec=0.5,
            confidence_threshold=float(threshold),
            command_map=command_map,
        )
        acquirer.start_stream()
        st.session_state.update(
            running=True,
            acquirer=acquirer,
            decoder=decoder,
            history=[],
            waveform=None,
            last_result=None,
            windows_seen=0,
            runtime_error=None,
        )
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.session_state.runtime_error = str(exc)

if stop_clicked and st.session_state.acquirer is not None:
    st.session_state.acquirer.stop_stream()
    st.session_state.running = False


@st.fragment(run_every=0.1)
def live_dashboard() -> None:
    if st.session_state.running:
        try:
            samples, _ = st.session_state.acquirer.get_new_samples()
            if samples.shape[1] == 0:
                st.session_state.running = False
            else:
                st.session_state.waveform = samples
                result = st.session_state.decoder.push(
                    samples,
                    trial_id=st.session_state.acquirer.current_trial_id,
                    expected_class_id=st.session_state.acquirer.current_label,
                )
                if result is not None:
                    st.session_state.last_result = result
                    st.session_state.history.append(result.to_dict())
                    st.session_state.windows_seen += 1
                    if st.session_state.windows_seen >= int(max_windows):
                        st.session_state.acquirer.stop_stream()
                        st.session_state.running = False
        except Exception as exc:  # noqa: BLE001
            st.session_state.runtime_error = str(exc)
            st.session_state.running = False

    result = st.session_state.last_result
    history = st.session_state.history
    latencies = np.asarray([item["latency_ms"] for item in history], dtype=float)
    current_latency = float(latencies[-1]) if len(latencies) else 0.0
    average_latency = float(latencies.mean()) if len(latencies) else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if len(latencies) else 0.0
    with st.container(horizontal=True):
        st.metric("Current prediction", result.prediction if result else "—", border=True)
        st.metric("Confidence", f"{result.confidence:.1%}" if result else "—", border=True)
        st.metric("Vehicle command", result.command if result else "STOP", border=True)
        st.metric("Current latency", f"{current_latency:.1f} ms", border=True)
        st.metric("Average latency", f"{average_latency:.1f} ms", border=True)
        st.metric("P95 latency", f"{p95_latency:.1f} ms", border=True)

    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            st.subheader("EEG waveform")
            waveform = st.session_state.waveform
            if waveform is None:
                st.info("Start replay to display EEG samples.")
            else:
                names = st.session_state.acquirer.metadata.channel_names[: min(8, waveform.shape[0])]
                values = waveform[: len(names)].T
                if input_unit.lower() == "v":
                    values = values * 1e6
                frame = pd.DataFrame(values, columns=names)
                st.line_chart(frame)
                st.caption(f"Showing {len(names)} channels · display unit µV")
    with right:
        with st.container(border=True):
            st.subheader("Prediction history")
            if history:
                frame = pd.DataFrame(history)
                frame.index.name = "window"
                st.dataframe(
                    frame[["prediction", "confidence", "command", "latency_ms", "trial_id"]].tail(30),
                    hide_index=False,
                )
            else:
                st.info("No decoded window yet.")

    status = "Running" if st.session_state.running else "Stopped"
    st.caption(f"Status: {status} · decoded windows: {st.session_state.windows_seen}/{int(max_windows)}")


if st.session_state.runtime_error:
    st.error(st.session_state.runtime_error)
live_dashboard()

