from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bci_dayloop.data.workload import (
    DIFF_CONDITION,
    EASY_CONDITION,
    WorkloadCondition,
    WorkloadHDF5,
    build_workload_session,
    write_workload_hdf5,
)
from bci_dayloop.runtime.types import ModelOutput
from scripts import smoke_test_workload_neuroonline as smoke


def _write_workload(path: Path) -> Path:
    easy = WorkloadCondition(
        condition=EASY_CONDITION,
        data=np.arange(2 * 2 * 4, dtype=np.float32).reshape(2, 2, 4),
        channel_names=("C3", "C4"),
        sample_rate=2.0,
        unit="V",
        source_set=path.parent / "easy.set",
    )
    diff = WorkloadCondition(
        condition=DIFF_CONDITION,
        data=(100 + np.arange(2 * 2 * 4, dtype=np.float32)).reshape(2, 2, 4),
        channel_names=("C3", "C4"),
        sample_rate=2.0,
        unit="V",
        source_set=path.parent / "diff.set",
    )
    session = build_workload_session(
        easy,
        diff,
        subject_id="P01",
        session_id="S12",
    )
    return write_workload_hdf5(
        path,
        [session],
        subject_id="P01",
        data_root=path.parent,
    )


class _Runtime:
    def __init__(self) -> None:
        self.windows = []

    def predict(self, raw_window):
        self.windows.append(raw_window)
        return ModelOutput(
            logits=torch.tensor([[0.0, 1.0]]),
            probabilities=torch.tensor([[0.25, 0.75]]),
            predicted_class=1,
            confidence=0.75,
            diagnostics={},
        )


def test_workload_smoke_preserves_multi_character_identity_and_runs_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = _write_workload(tmp_path / "subject_01.h5")
    persisted = WorkloadHDF5(data_path).load(session="S12")
    runtime = _Runtime()
    loaded = SimpleNamespace(
        runtime_model=runtime,
        model_type="test",
        model_name="test-runtime",
        class_names=("low_workload", "high_workload"),
        window_sec=2.0,
    )
    monkeypatch.setattr(smoke, "load_runtime_package", lambda *_args, **_kwargs: loaded)

    result = smoke.run_smoke(
        data_path=data_path,
        session="S12",
        model_package=tmp_path / "unused-package",
        device="cpu",
        trial_index=1,
    )

    assert result["adapter"] == "workload_hdf5"
    assert result["trial"] == {
        "index": 1,
        "subject_id": "P01",
        "session_id": "S12",
        "trial_id": "P01:S12:MATBdiff:000000",
        "window_id": "P01:S12:MATBdiff:000000",
        "label": 1,
        "trial_ordinal": 2,
    }
    assert result["runtime"]["class_names"] == ["low_workload", "high_workload"]
    assert result["prediction"] == {
        "class_index": 1,
        "class_name": "high_workload",
        "confidence": pytest.approx(0.75),
        "probabilities": [pytest.approx(0.25), pytest.approx(0.75)],
    }
    assert result["warning"] is None
    assert persisted["window_ids"].dtype == object
    assert persisted["window_ids"].tolist()[1] == "P01:S12:MATBdiff:000000"
    assert runtime.windows[0].trial_id == "P01:S12:MATBdiff:000000"
    assert runtime.windows[0].window_id == "P01:S12:MATBdiff:000000"
    assert runtime.windows[0].metadata["subject_id"] == "P01"
    assert runtime.windows[0].metadata["session_id"] == "S12"
    dataset = smoke.load_sequential_dataset(data_path, session="S12")
    assert dataset.subject_ids.dtype == object
    assert dataset.session_ids.dtype == object
    assert dataset.trial_ids.dtype == object
    assert dataset.window_ids.dtype == object
