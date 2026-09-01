from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bci_dayloop.data.hdf5_dataset import HDF5Metadata, write_hdf5
from bci_dayloop.data.trial_reader import open_trial_reader, reader_identity
from bci_dayloop.data.workload import WorkloadHDF5


ROOT = Path(__file__).resolve().parents[1]
REAL_WORKLOAD_S01 = ROOT / "data/processed/workload/subject_01.h5"


def test_default_eeg_reader_preserves_flat_hdf5_view(tmp_path: Path) -> None:
    path = tmp_path / "subject_01.h5"
    write_hdf5(
        path,
        np.zeros((2, 2, 4), dtype=np.float32),
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([1, 1], dtype=np.int64),
        ["train", "test"],
        np.asarray([11, 12], dtype=np.int64),
        HDF5Metadata(2.0, ["C3", "C4"], ["a", "b"], "uV", "synthetic"),
    )
    reader = open_trial_reader(
        data_reader="eeg", path=path, canonical_subject_id=1
    )
    assert reader.available_sessions() == ["test", "train"]
    np.testing.assert_array_equal(reader.load(session="train")["trial_ids"], [11])


def test_real_workload_reader_exposes_canonical_s1_and_s2_views() -> None:
    if not REAL_WORKLOAD_S01.is_file():
        pytest.skip("Real generated Workload H5 is not available in this checkout.")

    reader = open_trial_reader(
        data_reader="workload", path=REAL_WORKLOAD_S01, canonical_subject_id=1
    )
    assert isinstance(reader, WorkloadHDF5)
    assert reader.source_subject_id == "P01"
    assert reader.canonical_subject_id == 1
    assert reader.available_sessions() == ["S1", "S2"]
    assert reader.metadata.class_names == ["low_workload", "high_workload"]

    s1 = reader.load(session="S1")
    s2 = reader.load(session="S2")
    for loaded, session in ((s1, "S1"), (s2, "S2")):
        assert loaded["data"].shape == (298, 61, 500)
        assert int((loaded["labels"] == 0).sum()) == 149
        assert int((loaded["labels"] == 1).sum()) == 149
        assert set(loaded["subject_ids"].tolist()) == {1}
        assert set(loaded["session_ids"].tolist()) == {session}
        assert len(set(loaded["trial_ids"].tolist())) == 298

    assert int(s1["trial_ids"][0]) != int(s2["trial_ids"][0])
    assert reader_identity(
        reader, data_reader="workload", canonical_subject_id=1
    ) == {"canonical_subject_id": 1, "source_subject_id": "P01"}

    all_trials = reader.trial_metadata()
    assert all_trials["labels"].shape == (596,)
    assert len(set(all_trials["trial_ids"].tolist())) == 596
