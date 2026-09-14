from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from bci_dayloop.runtime.types import InputContract
from bci_dayloop.utils.config import load_yaml


def _benchmark_module():
    path = Path("scripts/run_device_window_latency_benchmark.py")
    spec = importlib.util.spec_from_file_location("device_window_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _loaded(model_type: str, window_sec: float) -> object:
    sample_rate = 100.0 if model_type == "model_50m" else 200.0
    channels = tuple(f"C{index}" for index in range(64 if model_type == "model_50m" else (19 if model_type == "labram" else 22)))
    return SimpleNamespace(
        model_type=model_type,
        runtime_model=SimpleNamespace(
            input_contract=InputContract(
                channel_names=channels,
                sample_rate=sample_rate,
                window_sec=window_sec,
                num_samples=round(window_sec * sample_rate),
                input_unit="uV",
                tensor_layout="BCT" if model_type == "model_50m" else "BCTP",
                model_input_keys=("signal",),
            )
        ),
    )


def test_grid_config_declares_exactly_twelve_package_driven_candidates() -> None:
    runner = _benchmark_module()
    payload = load_yaml(Path("configs/benchmarks/window_latency_live_1_2_3_4s.yaml"))
    benchmark = payload["benchmark"]
    fixed, allowed = runner._schedule_contract(
        benchmark["schedule"], benchmark["candidates"]
    )

    assert fixed is None
    assert allowed == (1.0, 2.0, 3.0, 4.0)
    candidates = benchmark["candidates"]
    assert len(candidates) == 12
    assert [candidate["id"] for candidate in candidates] == [
        "model_50m_1s", "model_50m_2s", "model_50m_3s", "model_50m_4s",
        "labram_live19_1s", "labram_live19_2s", "labram_live19_3s", "labram_live19_4s",
        "cbramod_live19_spline22_1s", "cbramod_live19_spline22_2s",
        "cbramod_live19_spline22_3s", "cbramod_live19_spline22_4s",
    ]


def test_frozen_4s_schedule_remains_supported() -> None:
    runner = _benchmark_module()
    payload = load_yaml(Path("configs/benchmarks/window_latency_live_4s.yaml"))
    benchmark = payload["benchmark"]

    fixed, allowed = runner._schedule_contract(
        benchmark["schedule"], benchmark["candidates"]
    )

    assert fixed == 4.0
    assert allowed == (4.0,)


def test_1b_latency_only_config_is_explicitly_non_classifying() -> None:
    payload = load_yaml(Path("configs/benchmarks/model_1b_backbone_latency_live_4s.yaml"))
    benchmark = payload["benchmark"]
    assert benchmark["mode"] == "device_backbone_latency_only"
    assert benchmark["latency_only"] is True
    assert benchmark["schedule"]["window_sec"] == 4.0
    assert benchmark["schedule"]["warmup_windows"] == 20
    assert benchmark["schedule"]["measured_windows"] == 200
    assert "candidates" not in benchmark


@pytest.mark.parametrize(
    ("model_type", "window_sec", "expected"),
    [
        ("model_50m", 1.0, (1, 64, 100)),
        ("model_50m", 2.0, (1, 64, 200)),
        ("model_50m", 3.0, (1, 64, 300)),
        ("model_50m", 4.0, (1, 64, 400)),
        ("labram", 1.0, (1, 19, 1, 200)),
        ("labram", 2.0, (1, 19, 2, 200)),
        ("labram", 3.0, (1, 19, 3, 200)),
        ("labram", 4.0, (1, 19, 4, 200)),
        ("cbramod", 1.0, (1, 22, 1, 200)),
        ("cbramod", 2.0, (1, 22, 2, 200)),
        ("cbramod", 3.0, (1, 22, 3, 200)),
        ("cbramod", 4.0, (1, 22, 4, 200)),
    ],
)
def test_manifest_prepared_contracts_cover_the_full_grid(
    model_type: str, window_sec: float, expected: tuple[int, ...]
) -> None:
    runner = _benchmark_module()
    assert runner._prepared_contract(_loaded(model_type, window_sec)) == expected


def test_package_driven_schedule_rejects_unapproved_windows() -> None:
    runner = _benchmark_module()
    with pytest.raises(ValueError, match="allowed_window_sec"):
        runner._schedule_contract(
            {
                "package_driven_windows": True,
                "allowed_window_sec": [1.0, 2.0, 3.0, 5.0],
                "step_sec": 0.5,
            },
            [{"id": str(index)} for index in range(12)],
        )
