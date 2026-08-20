from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat

from bci_dayloop.data.hdf5_dataset import EEGHDF5
from bci_dayloop.data.trial_reader import open_trial_reader
from bci_dayloop.models.model_50m.config import Model50MConfig, STANDARD_64_CHANNELS
from bci_dayloop.models.model_50m.preprocessing import Model50MPreprocessor
from scripts.prepare_seed_preprocessed_eeg_hdf5 import (
    CLASS_NAMES,
    SEED_CHANNEL_NAMES,
    SOURCE_TO_CANONICAL,
    build_trial_plan,
    convert_subject,
    discover_subject_sessions,
)


def _write_fixture_session(path: Path, marker: float) -> None:
    payload: dict[str, np.ndarray] = {}
    for ordinal in range(1, 16):
        samples = 801 if ordinal == 1 else 400
        payload[f"fixture_eeg{ordinal}"] = np.full(
            (62, samples), marker + ordinal, dtype=np.float64
        )
    savemat(path, payload)


def _fixture_input(tmp_path: Path) -> tuple[Path, Path, np.ndarray]:
    data_root = tmp_path / "SEED"
    input_root = data_root / "Preprocessed_EEG"
    input_root.mkdir(parents=True)
    for date, marker in (("20140103", 30.0), ("20140101", 10.0), ("20140102", 20.0)):
        _write_fixture_session(input_root / f"fixture_{date}.mat", marker)
    labels = np.asarray([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1], dtype=np.int64)
    savemat(input_root / "label.mat", {"label": labels.reshape(1, -1)})
    return data_root, input_root, labels


def _convert_fixture(tmp_path: Path) -> Path:
    data_root, input_root, labels = _fixture_input(tmp_path)
    output_root = tmp_path / "output"
    result = convert_subject(
        input_root=input_root,
        data_root=data_root,
        output_root=output_root,
        subject_id=1,
        subject_order=["fixture"],
        source_labels=labels,
        overwrite=False,
        dry_run=False,
    )
    assert result["total_epochs"] == 48
    return output_root / "subject_01.h5"


def test_seed_trial_plan_sorts_numeric_suffixes_sessions_and_labels(tmp_path: Path) -> None:
    _data_root, input_root, labels = _fixture_input(tmp_path)
    sessions = discover_subject_sessions(input_root, "fixture")
    assert [session.canonical_session_id for session in sessions] == ["S1", "S2", "S3"]
    assert [session.date for session in sessions] == ["20140101", "20140102", "20140103"]
    plans = build_trial_plan(sessions=sessions, source_labels=labels)
    assert [plan.source_trial_ordinal for plan in plans[:15]] == list(range(1, 16))
    assert [plan.source_trial_id for plan in plans] == list(range(1, 46))
    assert plans[0].epoch_count == 2
    assert plans[0].remainder_samples == 1
    assert plans[1].epoch_count == 1
    assert plans[0].source_label == 1
    assert plans[0].canonical_label == SOURCE_TO_CANONICAL[1] == 2


def test_seed_converter_preserves_2s_epochs_labels_sessions_and_provenance(tmp_path: Path) -> None:
    path = _convert_fixture(tmp_path)
    with h5py.File(path, "r") as handle:
        assert handle["data"].shape == (48, 62, 400)
        assert handle["data"].dtype == np.dtype("float32")
        assert np.isfinite(handle["data"][:]).all()
        assert set(handle["labels"][:].tolist()) == {0, 1, 2}
        assert np.array_equal(handle["trial_ids"][:], np.arange(48))
        assert len(np.unique(handle["trial_ids"][:])) == 48
        assert set(handle["session_ids"].asstr()[:]) == {"S1", "S2", "S3"}
        assert np.array_equal(handle["source_trial_ids"][:2], [1, 1])
        assert np.array_equal(handle["source_epoch_indices"][:2], [0, 1])
        assert np.array_equal(handle["source_start_samples"][:2], [0, 400])
        assert np.array_equal(handle["source_end_samples"][:2], [400, 800])
        assert np.array_equal(handle["source_labels"][:2], [1, 1])
        assert np.array_equal(handle["labels"][:2], [2, 2])
        assert handle["source_trial_ordinals"][0] == 1
        assert handle["source_session_ids"].asstr()[0] == "fixture_20140101"
        assert handle.attrs["unit"] == "uV"
        assert handle.attrs["source_unit"] == "unknown"
        assert not bool(handle.attrs["unit_scaling_applied"])
        assert json.loads(handle.attrs["label_mapping"]) == {
            "0": "negative", "1": "neutral", "2": "positive"
        }


def test_seed_output_reads_through_existing_reader_and_50m_preprocessor(tmp_path: Path) -> None:
    path = _convert_fixture(tmp_path)
    reader = EEGHDF5(path)
    assert reader.metadata.class_names == CLASS_NAMES
    assert reader.available_sessions() == ["S1", "S2", "S3"]
    assert reader.load("S1")["data"].shape == (16, 62, 400)
    assert reader.load("S2")["data"].shape == (16, 62, 400)
    assert reader.load("S3")["data"].shape == (16, 62, 400)
    canonical = open_trial_reader(data_reader="eeg", path=path, canonical_subject_id=1)
    assert canonical.trial_metadata()["trial_ids"].shape == (48,)

    processed = Model50MPreprocessor(
        Model50MConfig(checkpoint_path="unused", window_seconds=2.0)
    )(
        signal=reader.load("S1")["data"][0],
        channel_names=reader.metadata.channel_names,
        original_sample_rate=reader.metadata.sample_rate,
        input_unit=reader.metadata.unit,
    )
    assert processed.signal.shape == (64, 200)
    assert processed.mapped_channel_count == 58
    assert processed.missing_channel_count == 6
    assert processed.unknown_channel_names == ("PO5", "PO6", "CB1", "CB2")
    assert len(STANDARD_64_CHANNELS) == 64

