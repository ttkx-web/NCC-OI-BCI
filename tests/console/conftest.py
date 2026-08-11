from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONSOLE_API = ROOT / "apps" / "console-api"
if str(CONSOLE_API) not in sys.path:
    sys.path.insert(0, str(CONSOLE_API))


@pytest.fixture
def runtime_package(tmp_path: Path) -> Path:
    package = tmp_path / "model_packages" / "subject_01" / "population" / "v1"
    package.mkdir(parents=True)
    for name in ("backbone.pt", "classifier.pt"):
        (package / name).write_bytes(b"fixture")
    (package / "preprocessing.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    (package / "metrics.json").write_text(
        json.dumps({"final_test": {"balanced_accuracy": 0.72, "macro_f1": 0.71}}), encoding="utf-8"
    )
    manifest = {
        "schema_version": 2,
        "package": {"id": "fixture_50m", "version": "v1", "is_test_head": False},
        "model": {
            "type": "model_50m",
            "name": "50m-linear",
            "task": "motor_imagery",
            "dataset": "BNCI2014_001",
            "class_names": ["left_hand", "right_hand", "feet", "tongue"],
        },
        "files": {
            "backbone": "backbone.pt",
            "classifier": "classifier.pt",
            "preprocessing": "preprocessing.yaml",
            "metrics": "metrics.json",
        },
        "input_contract": {
            "channel_names": ["C3", "C4"],
            "sample_rate": 100,
            "window_sec": 4,
            "num_samples": 400,
            "input_unit": "uV",
            "tensor_layout": "BCT",
        },
        "runtime": {"step_sec": 0.5, "confidence_threshold": 0.55},
        "adaptation": {"offline": {"type": "none", "head_type": "population"}},
    }
    (package / "package.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return package


@pytest.fixture
def dataset_file(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "processed" / "bnci2014_001" / "subject_01.h5"
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=np.zeros((2, 2, 500), dtype=np.float32))
        handle.create_dataset("labels", data=np.asarray([0, 1], dtype=np.int64))
        handle.create_dataset("subject_ids", data=np.asarray([1, 1], dtype=np.int64))
        string_type = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset("session_ids", data=np.asarray(["1test", "1test"], dtype=string_type))
        handle.create_dataset("trial_ids", data=np.asarray([1, 2], dtype=np.int64))
        handle.attrs["sample_rate"] = 100.0
        handle.attrs["channel_names"] = json.dumps(["C3", "C4"])
        handle.attrs["class_names"] = json.dumps(["left_hand", "right_hand", "feet", "tongue"])
        handle.attrs["unit"] = "uV"
        handle.attrs["dataset_name"] = "BNCI2014_001"
    return path

