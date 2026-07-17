from __future__ import annotations

import numpy as np

from bci_dayloop.acquisition.factory import AcquirerFactory
from bci_dayloop.acquisition.replay import ReplayAcquirer
from bci_dayloop.data.hdf5_dataset import HDF5Metadata, write_hdf5


def make_data(path):
    data = np.arange(4 * 2 * 40, dtype=np.float32).reshape(4, 2, 40)
    write_hdf5(
        path,
        data,
        np.array([0, 1, 0, 1]),
        np.ones(4, dtype=np.int64),
        ["0train", "0train", "1test", "1test"],
        np.arange(4),
        HDF5Metadata(20.0, ["C3", "C4"], ["left_hand", "right_hand"], "V", "synthetic"),
    )


def test_replay_acquirer_chunk_incremental_and_factory(tmp_path):
    path = tmp_path / "replay.h5"
    make_data(path)
    acquirer = ReplayAcquirer(path, "1test", speed=10000.0, window_sec=1.0, step_sec=0.5)
    acquirer.start_stream()
    new, timestamps = acquirer.get_new_samples()
    assert new.shape == (2, 10)
    assert timestamps.shape == (10,)
    window, _ = acquirer.get_chunk(1.0)
    assert window.shape == (2, 20)
    assert acquirer.current_label in {0, 1}
    acquirer.stop_stream()
    assert "replay" in AcquirerFactory.list_acquirers()


def test_replay_loop_wraps(tmp_path):
    path = tmp_path / "loop.h5"
    make_data(path)
    acquirer = ReplayAcquirer(path, "1test", speed=10000.0, loop=True, step_sec=3.0)
    acquirer.start_stream()
    samples, _ = acquirer.get_new_samples()
    assert samples.shape == (2, 60)
    samples, _ = acquirer.get_new_samples()
    assert samples.shape == (2, 60)
    acquirer.stop_stream()

