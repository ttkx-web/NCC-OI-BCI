from __future__ import annotations

import numpy as np

from bci_dayloop.data.hdf5_dataset import EEGHDF5, HDF5Metadata, write_hdf5
from bci_dayloop.data.preprocessing import EEGPreprocessor, PreprocessingConfig


def test_hdf5_roundtrip_and_session_split(tmp_path):
    path = tmp_path / "eeg.h5"
    rng = np.random.default_rng(1)
    data = rng.normal(size=(4, 2, 250)).astype(np.float32)
    write_hdf5(
        path,
        data,
        np.array([0, 1, 0, 1]),
        np.ones(4, dtype=np.int64),
        ["0train", "0train", "1test", "1test"],
        np.arange(4),
        HDF5Metadata(250.0, ["C3", "C4"], ["left_hand", "right_hand"], "V", "synthetic"),
    )
    dataset = EEGHDF5(path)
    assert dataset.sessions() == ["0train", "1test"]
    assert dataset.load("1test")["data"].shape == (2, 2, 250)
    trial_metadata = dataset.trial_metadata()
    np.testing.assert_array_equal(trial_metadata["labels"], [0, 1, 0, 1])
    np.testing.assert_array_equal(trial_metadata["subject_ids"], [1, 1, 1, 1])
    np.testing.assert_array_equal(
        trial_metadata["session_ids"],
        ["0train", "0train", "1test", "1test"],
    )
    np.testing.assert_array_equal(trial_metadata["trial_ids"], [0, 1, 2, 3])
    assert dataset.metadata.channel_names == ["C3", "C4"]


def test_shared_preprocessing_shape_dtype_and_zscore():
    rng = np.random.default_rng(2)
    data = (rng.normal(size=(3, 2, 1000)) * 1e-6).astype(np.float32)
    preprocessor = EEGPreprocessor(PreprocessingConfig())
    output = preprocessor.transform(data, 250.0, "V")
    assert output.shape == (3, 2, 4, 200)
    assert output.dtype == np.float32
    flattened = output.reshape(3, 2, -1)
    np.testing.assert_allclose(flattened.mean(axis=-1), 0.0, atol=2e-5)
    np.testing.assert_allclose(flattened.std(axis=-1), 1.0, atol=2e-5)


def test_non_eeg_channel_removal():
    data = np.zeros((3, 100))
    selected, names = EEGPreprocessor.select_eeg_channels(data, ["C3", "EOG1", "C4"])
    assert selected.shape == (2, 100)
    assert names == ["C3", "C4"]
