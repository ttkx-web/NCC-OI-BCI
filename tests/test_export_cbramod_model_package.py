from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from bci_dayloop.data.sequential_dataset import (
    SequentialDataset,
    SequentialDatasetMetadata,
)
from bci_dayloop.models.cbramod.config import BCICIV2A_22_CHANNELS


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _exporter_module(name: str) -> object:
    sys.path.insert(0, str(SCRIPTS))
    try:
        path = SCRIPTS / "export_cbramod_model_package.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


def _training_report(
    *,
    policy: str = "spherical_spline",
    minimum: int | None = 21,
) -> dict[str, object]:
    return {
        "head_type": "linear",
        "preprocessing": {
            "source_unit": "V",
            "input_unit": "uV",
            "normalization": "fixed_100uv",
            "target_sample_rate": 200.0,
            "window_seconds": 2.0,
            "standard_channels": list(BCICIV2A_22_CHANNELS),
            "n_channels": 22,
            "time_segments": 2,
            "points_per_patch": 200,
            "missing_channel_policy": policy,
            "min_observed_channels": minimum,
            "spline_alpha": 1e-5,
            "filter_enabled": False,
            "reference_mode": "none",
            "strict_window_duration": True,
        },
    }


def _workload_dataset() -> SequentialDataset:
    data = np.stack(
        (
            np.full((2, 500), 11.0, dtype=np.float32),
            np.full((2, 500), 22.0, dtype=np.float32),
        )
    )
    return SequentialDataset(
        metadata=SequentialDatasetMetadata(
            sample_rate=250.0,
            channel_names=("C3", "C4"),
            class_names=("low_workload", "high_workload"),
            unit="V",
            dataset_name="workload_pbci_hackathon",
            window_sec=2.0,
        ),
        data=data,
        labels=np.asarray([1, 0], dtype=np.int64),
        subject_ids=np.asarray(["P01", "P01"]),
        session_ids=np.asarray(["S2", "S2"]),
        trial_ids=np.asarray(["trial-first", "trial-second"]),
        trial_ordinals=np.asarray([1, 2], dtype=np.int64),
        window_ids=np.asarray(["window-first", "window-second"]),
    )


def test_workload_training_report_package_and_reload_contract_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _exporter_module("test_cbramod_export_contract")
    runtime_kwargs = module.runtime_kwargs_from_training_report(
        _training_report()
    )

    assert runtime_kwargs["source_unit"] == "V"
    assert runtime_kwargs["input_unit"] == "uV"
    assert runtime_kwargs["normalization"] == "fixed_100uv"
    assert runtime_kwargs["window_seconds"] == 2.0
    assert runtime_kwargs["time_segments"] == 2
    assert runtime_kwargs["points_per_patch"] == 200
    assert runtime_kwargs["missing_channel_policy"] == "spherical_spline"
    assert runtime_kwargs["min_observed_channels"] == 21
    assert runtime_kwargs["spline_alpha"] == pytest.approx(1e-5)
    assert runtime_kwargs["standard_channels"] == BCICIV2A_22_CHANNELS

    backbone = tmp_path / "source_backbone.pt"
    classifier = tmp_path / "source_classifier.pt"
    report_path = tmp_path / "training_report.json"
    backbone.write_bytes(b"backbone")
    classifier.write_bytes(b"classifier")
    report_path.write_text("{}", encoding="utf-8")

    package_payload, preprocessing_payload, _ = module.build_package_payload(
        class_names=("low_workload", "high_workload"),
        command_map={},
        runtime_kwargs=runtime_kwargs,
        package_id="workload-cbramod-test",
        package_version="v1",
        dataset_name="workload_pbci_hackathon",
        step_sec=0.5,
        confidence_threshold=0.0,
        backbone_path=backbone,
        classifier_path=classifier,
        metrics={},
        training_report_path=report_path,
    )
    transform = preprocessing_payload["transform"]
    assert tuple(transform["standard_channels"]) == BCICIV2A_22_CHANNELS
    assert transform["normalization"] == "fixed_100uv"
    assert transform["missing_channel_policy"] == "spherical_spline"
    assert transform["min_observed_channels"] == 21
    assert transform["spline_alpha"] == pytest.approx(1e-5)
    assert package_payload["provenance"]["source_unit"] == "V"
    assert package_payload["input_contract"]["input_unit"] == "uV"

    package_dir = tmp_path / "package"
    package_dir.mkdir()
    (package_dir / "backbone.pt").write_bytes(backbone.read_bytes())
    (package_dir / "classifier.pt").write_bytes(classifier.read_bytes())
    module.dump_yaml(package_payload, package_dir / "package.yaml")
    module.dump_yaml(
        preprocessing_payload,
        package_dir / "preprocessing.yaml",
    )

    captured: dict[str, object] = {}
    sentinel = object()

    def fake_build_runtime(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(module, "build_cbramod_runtime", fake_build_runtime)
    loaded = module.load_package_runtime_for_smoke_test(
        package_path=package_dir,
        device="cpu",
        verify_hashes=True,
    )
    assert loaded is sentinel
    assert captured["input_unit"] == "uV"
    assert captured["normalization"] == "fixed_100uv"
    assert captured["window_seconds"] == 2.0
    assert captured["time_segments"] == 2
    assert captured["points_per_patch"] == 200
    assert captured["missing_channel_policy"] == "spherical_spline"
    assert captured["min_observed_channels"] == 21
    assert captured["spline_alpha"] == pytest.approx(1e-5)
    assert captured["standard_channels"] == BCICIV2A_22_CHANNELS


def test_strict_report_allows_none_minimum() -> None:
    module = _exporter_module("test_cbramod_export_strict")
    report = _training_report(policy="error", minimum=None)
    kwargs = module.runtime_kwargs_from_training_report(report)
    assert kwargs["missing_channel_policy"] == "error"
    assert kwargs["min_observed_channels"] is None
    assert kwargs["standard_channels"] == BCICIV2A_22_CHANNELS


def test_extract_first_trial_window_preserves_dataset_identity_and_samples() -> None:
    module = _exporter_module("test_cbramod_export_window")
    dataset = _workload_dataset()
    window = module.extract_first_trial_window(
        dataset=dataset,
        window_seconds=2.0,
        anchor="end",
    )

    np.testing.assert_array_equal(window.data, dataset.data[0])
    assert window.label == int(dataset.labels[0])
    assert window.trial_id == "trial-first"
    assert window.window_id == "window-first"
    assert window.metadata["session"] == "S2"
    assert window.data.shape == (2, 500)
    selection = window.metadata["training_source_trial_selection"]
    assert selection["selected_start_sample"] == 0
    assert selection["selected_end_sample_exclusive"] == 500

    with pytest.raises(ValueError, match="persisted trial window exactly"):
        module.extract_first_trial_window(
            dataset=dataset,
            window_seconds=1.0,
            anchor="end",
        )

