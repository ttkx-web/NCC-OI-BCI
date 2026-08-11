from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from bci_dayloop.data.hdf5_dataset import HDF5Metadata
from bci_dayloop.models.cbramod.config import (
    BCICIV2A_22_CHANNELS,
    CBRAMOD_NEURACLE_LIVE19_SPLINE22_PROFILE,
    CBRAMOD_STRICT22_PROFILE,
    CBraModConfig,
    resolve_cbramod_deployment_profile,
)
from bci_dayloop.models.cbramod.preprocessing import (
    CBraModPipelinePreprocessor,
)
from bci_dayloop.packages import loader as package_loader
from bci_dayloop.preprocessing.canonical import SignalCanonicalizer
from bci_dayloop.runtime.types import RawEEGWindow
from bci_dayloop.utils.config import dump_yaml
from scripts.export_cbramod_model_package import (
    build_package_payload,
    validate_deployment_prepared,
)
from scripts.train_cbramod_population_head import (
    build_argument_parser,
    prepare_cbramod_trials,
)


CLASS_NAMES = ("left_hand", "right_hand", "feet", "tongue")


def _profile_config(profile_name: str) -> tuple[object, CBraModConfig]:
    profile = resolve_cbramod_deployment_profile(profile_name)
    config = CBraModConfig(
        checkpoint_path=Path("unused.pth"),
        missing_channel_policy=profile.missing_channel_policy,
        min_observed_channels=profile.min_observed_channels,
        spline_alpha=profile.spline_alpha,
        filter_enabled=False,
        normalization="none",
    )
    return profile, config


def _package_payload(
    package_path: Path,
    *,
    spline: bool,
    include_minimum: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    package_path.mkdir()
    (package_path / "package.yaml").write_text("schema_version: 2\n")
    (package_path / "backbone.pt").write_bytes(b"backbone")
    (package_path / "classifier.pt").write_bytes(b"classifier")
    (package_path / "metrics.json").write_text("{}", encoding="utf-8")

    transform: dict[str, object] = {
        "type": "cbramod",
        "target_sample_rate": 200.0,
        "window_seconds": 4.0,
        "n_channels": 22,
        "standard_channels": list(BCICIV2A_22_CHANNELS),
        "time_segments": 4,
        "points_per_patch": 200,
        "strict_window_duration": True,
        "allow_missing_channels": True,
    }
    runtime: dict[str, object] = {
        "step_sec": 0.5,
        "confidence_threshold": 0.55,
        "command_map": {},
    }
    classifier_metadata: dict[str, object] = {}
    if spline:
        transform.update(
            missing_channel_policy="spherical_spline",
            spline_alpha=1e-5,
            completion_matrix_sha256="a" * 64,
        )
        if include_minimum:
            transform["min_observed_channels"] = 19
        profile = resolve_cbramod_deployment_profile(
            CBRAMOD_NEURACLE_LIVE19_SPLINE22_PROFILE
        )
        runtime["channel_completion"] = {
            "deployment_profile": profile.name,
            "observed_required": 19,
            "observed_channel_names": list(profile.observed_channel_names),
            "missing_expected": list(profile.simulated_missing_channels),
            "missing_channel_policy": "spherical_spline",
            "min_observed_channels": 19,
            "spline_alpha": 1e-5,
            "channel_completion_source": "shared_runtime_preprocessor",
            "completion_matrix_sha256": "a" * 64,
        }
        classifier_metadata = {
            "deployment_profile": profile.name,
            "training_channel_source_count": 22,
            "observed_channel_count": 19,
            "observed_channel_names": list(profile.observed_channel_names),
            "simulated_missing_channels": list(
                profile.simulated_missing_channels
            ),
            "missing_channel_policy": "spherical_spline",
            "min_observed_channels": 19,
            "spline_alpha": 1e-5,
            "channel_completion_source": "shared_runtime_preprocessor",
            "completion_matrix_sha256": "a" * 64,
        }

    dump_yaml(
        {
            "schema_version": 1,
            "canonicalizer": {"target_unit": "uV"},
            "transform": transform,
        },
        package_path / "preprocessing.yaml",
    )
    payload: dict[str, object] = {
        "schema_version": 2,
        "package": {"is_test_head": False},
        "model": {
            "type": "cbramod",
            "name": "cbramod-frozen-head",
            "num_classes": 4,
            "class_names": list(CLASS_NAMES),
            "head_type": "official_mlp",
        },
        "files": {
            "backbone": "backbone.pt",
            "classifier": "classifier.pt",
            "preprocessing": "preprocessing.yaml",
            "metrics": "metrics.json",
            "sha256": {},
        },
        "input_contract": {
            "channel_names": list(BCICIV2A_22_CHANNELS),
            "sample_rate": 200.0,
            "window_sec": 4.0,
            "num_samples": 800,
            "input_unit": "uV",
            "tensor_layout": "BCTP",
            "strict_window_duration": True,
            "model_input_keys": ["signal"],
        },
        "runtime": runtime,
        "provenance": {"training_channel_source_count": 22},
    }
    dump_yaml(payload, package_path / "package.yaml")
    return payload, classifier_metadata


def _patch_cbramod_loader(
    monkeypatch: pytest.MonkeyPatch,
    classifier_metadata: dict[str, object],
) -> None:
    import bci_dayloop.models.cbramod.backend as backend_module
    import bci_dayloop.models.cbramod.backbone as backbone_module
    import bci_dayloop.models.cbramod.classifier as classifier_module
    import bci_dayloop.models.cbramod.runtime as runtime_module

    class FakeBackbone:
        def __init__(self, config: CBraModConfig) -> None:
            self.config = config
            self.device = torch.device("cpu")

    class FakeClassifier:
        def to(self, device: object) -> "FakeClassifier":
            return self

        def eval(self) -> "FakeClassifier":
            return self

    class FakeBackend:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(backbone_module, "CBraModBackbone", FakeBackbone)
    monkeypatch.setattr(
        classifier_module,
        "build_cbramod_classifier",
        lambda config: FakeClassifier(),
    )
    monkeypatch.setattr(backend_module, "CBraModBackend", FakeBackend)
    monkeypatch.setattr(
        runtime_module,
        "load_cbramod_classifier_checkpoint",
        lambda *args, **kwargs: SimpleNamespace(
            metadata=dict(classifier_metadata)
        ),
    )


def test_strict_default_config_requires_all_22_channels() -> None:
    config = CBraModConfig(checkpoint_path=Path("unused.pth"))
    assert config.missing_channel_policy == "error"
    assert config.min_observed_channels == 22


def test_training_profile_is_explicit_and_strict22_remains_default() -> None:
    parser = build_argument_parser()
    strict_args = parser.parse_args(["--target-subject", "1"])
    live_args = parser.parse_args(
        [
            "--target-subject",
            "1",
            "--deployment-profile",
            CBRAMOD_NEURACLE_LIVE19_SPLINE22_PROFILE,
        ]
    )
    assert strict_args.deployment_profile == CBRAMOD_STRICT22_PROFILE
    assert (
        live_args.deployment_profile
        == CBRAMOD_NEURACLE_LIVE19_SPLINE22_PROFILE
    )


def test_loader_old_strict22_package_does_not_use_allow_missing_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload, classifier_metadata = _package_payload(
        tmp_path / "strict",
        spline=False,
    )
    _patch_cbramod_loader(monkeypatch, classifier_metadata)
    loaded = package_loader.load_runtime_package(
        tmp_path / "strict",
        device="cpu",
        verify_hashes=False,
    )
    config = loaded.runtime_model.input_transform.config
    assert config.missing_channel_policy == "error"
    assert config.min_observed_channels == 22
    assert "allow_missing_channels" not in inspect.getsource(
        package_loader._load_cbramod_package
    )


def test_loader_spline_package_reads_explicit_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload, classifier_metadata = _package_payload(
        tmp_path / "spline",
        spline=True,
    )
    _patch_cbramod_loader(monkeypatch, classifier_metadata)
    loaded = package_loader.load_runtime_package(
        tmp_path / "spline",
        device="cpu",
        verify_hashes=False,
    )
    config = loaded.runtime_model.input_transform.config
    assert config.missing_channel_policy == "spherical_spline"
    assert config.min_observed_channels == 19
    assert config.spline_alpha == pytest.approx(1e-5)


def test_loader_rejects_spline_package_without_explicit_minimum(
    tmp_path: Path,
) -> None:
    payload, _ = _package_payload(
        tmp_path / "missing-min",
        spline=True,
        include_minimum=False,
    )
    with pytest.raises(ValueError, match="min_observed_channels"):
        package_loader.load_runtime_package(
            tmp_path / "missing-min",
            device="cpu",
            verify_hashes=False,
        )


def test_live_profile_training_and_runtime_share_spline_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, config = _profile_config(
        CBRAMOD_NEURACLE_LIVE19_SPLINE22_PROFILE
    )
    preprocessor = CBraModPipelinePreprocessor(config)
    completion_sha256 = preprocessor.completion_matrix_sha256_for(
        profile.observed_channel_names
    )
    assert completion_sha256

    rng = np.random.default_rng(42)
    source_trials = rng.standard_normal((1, 22, 800), dtype=np.float32)
    removed_indices = [
        BCICIV2A_22_CHANNELS.index(name)
        for name in profile.simulated_missing_channels
    ]
    source_trials[:, removed_indices, :] = 1_000_000.0
    seen_channel_names: list[tuple[str, ...]] = []
    original_transform = preprocessor.transform

    def spy_transform(window: object) -> object:
        names = tuple(getattr(window, "channel_names"))
        seen_channel_names.append(names)
        assert not set(profile.simulated_missing_channels).intersection(names)
        return original_transform(window)  # type: ignore[arg-type]

    monkeypatch.setattr(preprocessor, "transform", spy_transform)
    prepared_trials = prepare_cbramod_trials(
        data=source_trials,
        metadata=HDF5Metadata(
            200.0,
            list(BCICIV2A_22_CHANNELS),
            list(CLASS_NAMES),
            "uV",
            "synthetic",
        ),
        trial_ids=np.asarray([1], dtype=np.int64),
        subject_id=2,
        session_name="0train",
        canonicalizer=SignalCanonicalizer(target_unit="uV"),
        preprocessor=preprocessor,
        observed_channel_names=profile.observed_channel_names,
        simulated_missing_channels=profile.simulated_missing_channels,
        expected_completion_matrix_sha256=completion_sha256,
    )
    assert seen_channel_names == [profile.observed_channel_names]
    assert prepared_trials.shape == (1, 22, 4, 200)
    assert prepared_trials.dtype == np.float32
    assert np.isfinite(prepared_trials).all()

    runtime_preprocessor = CBraModPipelinePreprocessor(config)
    observed_indices = [
        BCICIV2A_22_CHANNELS.index(name)
        for name in profile.observed_channel_names
    ]
    raw_window = RawEEGWindow(
        data=source_trials[0, observed_indices],
        channel_names=list(profile.observed_channel_names),
        sample_rate=200.0,
        unit="uV",
        layout="CT",
    )
    runtime_prepared = runtime_preprocessor.transform(
        SignalCanonicalizer(target_unit="uV").transform(raw_window)
    )
    runtime_kwargs = {
        "n_channels": 22,
        "time_segments": 4,
        "points_per_patch": 200,
        "observed_channel_names": profile.observed_channel_names,
        "simulated_missing_channels": profile.simulated_missing_channels,
        "missing_channel_policy": profile.missing_channel_policy,
        "completion_matrix_sha256": completion_sha256,
    }
    diagnostics = validate_deployment_prepared(
        runtime_prepared,
        runtime_kwargs=runtime_kwargs,
    )
    assert diagnostics["observed_channel_count"] == 19
    assert diagnostics["missing_channel_names"] == ["CPz", "P1", "P2"]
    assert diagnostics["completion_matrix_sha256"] == completion_sha256


def test_package_payload_records_live_profile_provenance(
    tmp_path: Path,
) -> None:
    profile = resolve_cbramod_deployment_profile(
        CBRAMOD_NEURACLE_LIVE19_SPLINE22_PROFILE
    )
    backbone = tmp_path / "backbone.pt"
    classifier = tmp_path / "head.pt"
    report = tmp_path / "training_report.json"
    backbone.write_bytes(b"backbone")
    classifier.write_bytes(b"classifier")
    report.write_text("{}", encoding="utf-8")
    runtime_kwargs = {
        "standard_channels": BCICIV2A_22_CHANNELS,
        "target_sample_rate": 200.0,
        "window_seconds": 4.0,
        "n_channels": 22,
        "time_segments": 4,
        "points_per_patch": 200,
        "input_unit": "uV",
        "strict_window_duration": True,
        "window_tolerance_seconds": 0.02,
        "filter_enabled": False,
        "filter_low_hz": 0.1,
        "filter_high_hz": 75.0,
        "filter_order": 4,
        "reference_mode": "none",
        "normalization": "none",
        "head_type": "official_mlp",
        "deployment_profile": profile.name,
        "training_channel_source_count": 22,
        "observed_channel_names": profile.observed_channel_names,
        "simulated_missing_channels": profile.simulated_missing_channels,
        "missing_channel_policy": "spherical_spline",
        "min_observed_channels": 19,
        "spline_alpha": 1e-5,
        "channel_completion_source": "shared_runtime_preprocessor",
        "completion_matrix_sha256": "b" * 64,
    }
    package, preprocessing, _ = build_package_payload(
        class_names=CLASS_NAMES,
        command_map={},
        runtime_kwargs=runtime_kwargs,
        package_id="cbramod-live19",
        package_version="v1",
        dataset_name="bnci2014_001",
        step_sec=0.5,
        confidence_threshold=0.55,
        backbone_path=backbone,
        classifier_path=classifier,
        metrics={},
        training_report_path=report,
    )
    transform = preprocessing["transform"]
    completion = package["runtime"]["channel_completion"]
    assert transform["missing_channel_policy"] == "spherical_spline"
    assert transform["min_observed_channels"] == 19
    assert completion["observed_required"] == 19
    assert completion["missing_expected"] == ["CPz", "P1", "P2"]
    assert completion["completion_matrix_sha256"] == "b" * 64


def test_strict_profile_remains_native_22_channels() -> None:
    profile = resolve_cbramod_deployment_profile(
        CBRAMOD_STRICT22_PROFILE
    )
    assert profile.observed_channel_names == BCICIV2A_22_CHANNELS
    assert profile.simulated_missing_channels == ()
    assert profile.missing_channel_policy == "error"
    assert profile.min_observed_channels == 22
