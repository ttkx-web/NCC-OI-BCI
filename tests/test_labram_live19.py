from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from torch import nn

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from bci_dayloop.data.channel_selection import (
    select_named_channels,
    strict_channel_indices,
)
from bci_dayloop.data.hdf5_dataset import (
    EEGHDF5,
    HDF5Metadata,
    write_hdf5,
)
from bci_dayloop.packages.loader import load_runtime_package
from bci_dayloop.runtime.types import InputContract
from scripts.export_labram_model_package import verify_package
from scripts.train_labram_population_head import (
    load_preprocessed_subject_session,
)


SOURCE_22 = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1",
    "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
)
LIVE_19 = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1",
    "Cz", "C2", "C4", "C6", "CP3", "CP1", "CP2", "CP4", "Pz",
    "POz",
)
CLASSES = ("left_hand", "right_hand", "feet", "tongue")


def _write_subject(path: Path) -> EEGHDF5:
    data = np.arange(22 * 1000, dtype=np.float32).reshape(1, 22, 1000)
    write_hdf5(
        path,
        data=data,
        labels=np.asarray([0], dtype=np.int64),
        subject_ids=np.asarray([2], dtype=np.int64),
        session_ids=["0train"],
        trial_ids=np.asarray([1], dtype=np.int64),
        metadata=HDF5Metadata(
            sample_rate=250.0,
            channel_names=list(SOURCE_22),
            class_names=list(CLASSES),
            unit="V",
            dataset_name="BNCI2014_001",
        ),
    )
    return EEGHDF5(path)


class _FakePreprocessor:
    config = SimpleNamespace(
        patch_samples=200,
        target_sample_rate=200.0,
    )

    def transform(
        self,
        data: np.ndarray,
        _sample_rate: float,
        _unit: str,
        *,
        reshape: bool,
    ) -> np.ndarray:
        assert reshape is True
        return np.zeros(
            (data.shape[0], data.shape[1], 4, 200),
            dtype=np.float32,
        )


def test_training_selects_and_reorders_22_to_19(tmp_path: Path) -> None:
    path = tmp_path / "subject_02.h5"
    _write_subject(path)
    requested = LIVE_19[::-1]
    X, _, metadata, summary = load_preprocessed_subject_session(
        subject_id=2,
        path=path,
        session_name="0train",
        preprocessor=_FakePreprocessor(),  # type: ignore[arg-type]
        reference_metadata=None,
        expected_window_sec=4.0,
        maximum_per_class=None,
        seed=42,
        requested_channel_names=requested,
    )
    assert X.shape == (1, 19, 4, 200)
    assert tuple(metadata.channel_names) == requested
    assert summary["source_channel_count"] == 22
    assert summary["selected_channel_count"] == 19


def test_default_training_selection_preserves_22_channels(tmp_path: Path) -> None:
    path = tmp_path / "subject_02.h5"
    _write_subject(path)
    X, _, metadata, summary = load_preprocessed_subject_session(
        subject_id=2,
        path=path,
        session_name="0train",
        preprocessor=_FakePreprocessor(),  # type: ignore[arg-type]
        reference_metadata=None,
        expected_window_sec=4.0,
        maximum_per_class=None,
        seed=42,
    )
    assert X.shape == (1, 22, 4, 200)
    assert tuple(metadata.channel_names) == SOURCE_22
    assert summary["source_channel_count"] == 22
    assert summary["selected_channel_count"] == 22


def test_channel_selection_rejects_missing_required_channel() -> None:
    with pytest.raises(ValueError, match="missing"):
        strict_channel_indices(SOURCE_22[:-1], LIVE_19)


def test_channel_selection_rejects_duplicate_requested_channel() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        strict_channel_indices(SOURCE_22, ("Fz", "Fz"))


def test_channel_selection_rejects_duplicate_source_channel() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        strict_channel_indices(("Fz", "Fz"), ("Fz",))


def test_channel_selection_rejects_invalid_requested_channel() -> None:
    with pytest.raises(ValueError, match="NOT_A_CHANNEL"):
        strict_channel_indices(SOURCE_22, ("NOT_A_CHANNEL",))


def test_select_named_channels_preserves_requested_values_and_order() -> None:
    data = np.arange(22 * 3, dtype=np.float32).reshape(22, 3)
    selected, names = select_named_channels(
        data,
        source_channel_names=SOURCE_22,
        requested_channel_names=("POz", "Fz", "Cz"),
        channel_axis=0,
    )
    assert names == ("POz", "Fz", "Cz")
    assert np.array_equal(selected[0], data[SOURCE_22.index("POz")])
    assert np.array_equal(selected[1], data[SOURCE_22.index("Fz")])
    assert np.array_equal(selected[2], data[SOURCE_22.index("Cz")])


def test_exporter_smoke_explicitly_selects_live19(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _write_subject(tmp_path / "subject_02.h5")
    captured: dict[str, object] = {}

    class FakeRuntime:
        input_contract = InputContract(
            channel_names=LIVE_19,
            sample_rate=200.0,
            window_sec=4.0,
            num_samples=800,
            input_unit="uV",
            tensor_layout="BCTP",
        )

        def predict(self, raw: object) -> object:
            captured["shape"] = tuple(raw.data.shape)  # type: ignore[attr-defined]
            captured["names"] = tuple(raw.channel_names)  # type: ignore[attr-defined]
            return SimpleNamespace(
                probabilities=torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
                predicted_class=3,
                confidence=0.4,
            )

    monkeypatch.setattr(
        "scripts.export_labram_model_package.load_runtime_package",
        lambda *_args, **_kwargs: SimpleNamespace(
            model_type="labram",
            runtime_model=FakeRuntime(),
            window_sec=4.0,
            class_names=CLASSES,
        ),
    )
    report = verify_package(
        package_path=tmp_path / "package",
        dataset=dataset,
        session_name="0train",
        device="cpu",
    )
    assert report["status"] == "passed"
    assert captured == {"shape": (19, 1000), "names": LIVE_19}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_schema_v2_package(path: Path) -> None:
    path.mkdir()
    backbone = path / "backbone.pt"
    classifier = path / "classifier.pt"
    backbone.write_bytes(b"fake-backbone")
    linear = nn.Linear(3, 4)
    torch.save(
        {
            "embedding_dim": 3,
            "n_classes": 4,
            "state_dict": linear.state_dict(),
        },
        classifier,
    )
    (path / "preprocessing.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "canonicalizer": {"target_unit": "uV"},
                "transform": {
                    "type": "labram",
                    "bandpass_hz": [0.1, 75.0],
                    "notch_hz": 50.0,
                    "target_sample_rate": 200.0,
                    "zscore_epsilon": 1e-6,
                    "patch_samples": 200,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (path / "metrics.json").write_text("{}", encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "package": {
            "id": "labram-live19-test",
            "version": "v1",
            "is_test_head": False,
            "warning_message": None,
        },
        "model": {
            "type": "labram",
            "name": "labram-linear",
            "num_classes": 4,
            "class_names": list(CLASSES),
            "n_patches": 4,
            "amp": False,
            "freeze_encoder": True,
            "embedding_batch_size": 4,
        },
        "files": {
            "backbone": backbone.name,
            "classifier": classifier.name,
            "preprocessing": "preprocessing.yaml",
            "metrics": "metrics.json",
            "sha256": {
                "backbone": _sha256(backbone),
                "classifier": _sha256(classifier),
            },
        },
        "input_contract": {
            "channel_names": list(LIVE_19),
            "sample_rate": 200.0,
            "window_sec": 4.0,
            "num_samples": 800,
            "input_unit": "uV",
            "tensor_layout": "BCTP",
            "strict_window_duration": True,
            "model_input_keys": ["signal"],
        },
        "runtime": {
            "step_sec": 0.5,
            "confidence_threshold": 0.55,
            "command_map": {},
        },
    }
    (path / "package.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )


def test_schema_v2_loader_preserves_live19_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_schema_v2_package(package)

    class FakeAdapter:
        def __init__(self, **kwargs: object) -> None:
            self.channel_names = list(kwargs["channel_names"])  # type: ignore[arg-type]
            self.n_classes = int(kwargs["n_classes"])  # type: ignore[arg-type]
            self.embedding_dim = 3
            self.device = torch.device("cpu")
            self.head = nn.Linear(3, self.n_classes)

    monkeypatch.setattr(
        "bci_dayloop.packages.loader.LaBraMLinearAdapter",
        FakeAdapter,
    )
    loaded = load_runtime_package(package, verify_hashes=True)
    contract = loaded.runtime_model.input_contract
    assert loaded.model_type == "labram"
    assert loaded.is_test_head is False
    assert contract.channel_names == LIVE_19
    assert contract.sample_rate == 200.0
    assert contract.window_sec == 4.0
    assert contract.num_samples == 800
    assert contract.input_unit == "uV"
    assert contract.tensor_layout == "BCTP"
