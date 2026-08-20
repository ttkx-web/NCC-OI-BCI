from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat

from bci_dayloop.data.hdf5_dataset import EEGHDF5
from bci_dayloop.data.trial_reader import open_trial_reader
from bci_dayloop.models.model_50m.config import Model50MConfig, STANDARD_64_CHANNELS
from bci_dayloop.models.model_50m.preprocessing import Model50MPreprocessor


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_mema_for_dl_hdf5.py"


def _make_mema_input(root: Path) -> None:
    root.mkdir()
    labels = np.tile(np.asarray([2, 1, 0] * 4, dtype=np.int32), (20, 1))
    savemat(root / "label_attention.mat", {"label": labels})
    payload: dict[str, np.ndarray] = {}
    for trial in range(1, 13):
        samples = 2_001 if trial == 1 else 1_000
        payload[f"fixture_eeg{trial}"] = np.full(
            (32, samples), float(trial), dtype=np.float64
        )
    savemat(root / "Subject1.mat", payload)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_mema_converter_segments_source_trials_and_preserves_provenance(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _make_mema_input(input_root)
    _run("--input-root", str(input_root), "--output-root", str(output_root), "--subjects", "1")

    path = output_root / "subject_01.h5"
    with h5py.File(path, "r") as handle:
        assert handle["data"].shape == (13, 32, 1000)
        assert handle["data"].dtype == np.dtype("float32")
        assert np.array_equal(handle["labels"][:], np.asarray([2, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0]))
        assert np.array_equal(handle["trial_ids"][:], np.arange(13))
        assert np.array_equal(handle["source_trial_ids"][:2], np.asarray([1, 1]))
        assert np.array_equal(handle["source_epoch_indices"][:2], np.asarray([0, 1]))
        assert np.array_equal(handle["source_start_samples"][:2], np.asarray([0, 1000]))
        assert np.array_equal(handle["source_end_samples"][:2], np.asarray([1000, 2000]))
        assert set(handle["session_ids"].asstr()[:10]) == {"S1"}
        assert set(handle["session_ids"].asstr()[10:]) == {"S2"}
        assert handle.attrs["label_mapping_status"] == "working_assumption"
        assert json.loads(handle.attrs["label_mapping"]) == {
            "0": "relaxing", "1": "neutral", "2": "concentrating"
        }
        assert handle.attrs["source_unit"] == "unknown"
        assert handle.attrs["unit"] == "uV"
        assert not bool(handle.attrs["original_session_metadata_available"])

    summary = json.loads((output_root / "conversion_summary.json").read_text())
    assert summary["total_epochs"] == 13
    assert summary["discarded_remainder_total"] == 1
    assert summary["total_class_counts"] == {
        "relaxing": 4, "neutral": 4, "concentrating": 5
    }
    _run(
        "--input-root", str(input_root), "--output-root", str(output_root),
        "--subjects", "1", "--verify-existing",
    )


def test_mema_output_is_readable_by_existing_reader_and_preprocessor(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _make_mema_input(input_root)
    _run("--input-root", str(input_root), "--output-root", str(output_root), "--subjects", "1")
    path = output_root / "subject_01.h5"

    reader = EEGHDF5(path)
    assert reader.available_sessions() == ["S1", "S2"]
    assert reader.load("S1")["data"].shape == (10, 32, 1000)
    canonical = open_trial_reader(data_reader="eeg", path=path, canonical_subject_id=1)
    assert canonical.trial_metadata()["trial_ids"].shape == (13,)

    config = Model50MConfig(checkpoint_path="unused", window_seconds=2.0)
    processed = Model50MPreprocessor(config)(
        signal=reader.load("S1")["data"][0],
        channel_names=reader.metadata.channel_names,
        original_sample_rate=reader.metadata.sample_rate,
        input_unit=reader.metadata.unit,
    )
    assert processed.signal.shape == (64, 200)
    assert processed.mapped_channel_count == 30
    assert processed.missing_channel_count == 34
    assert processed.unknown_channel_names == ("HEOL", "HEOR")
    assert len(STANDARD_64_CHANNELS) == 64


def test_mema_converter_dry_run_does_not_write_output(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _make_mema_input(input_root)
    result = _run(
        "--input-root", str(input_root), "--output-root", str(output_root),
        "--subjects", "1", "--dry-run",
    )
    assert '"total_epochs": 13' in result.stdout
    assert '"discarded_remainder_total": 1' in result.stdout
    assert not output_root.exists()


def test_mema_converter_discards_whole_epoch_outside_float32_range(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _make_mema_input(input_root)
    source = input_root / "Subject1.mat"
    from scipy.io import loadmat

    payload = loadmat(source)
    payload["fixture_eeg12"][0, 0] = 1e300
    savemat(source, {key: value for key, value in payload.items() if not key.startswith("__")})
    _run("--input-root", str(input_root), "--output-root", str(output_root), "--subjects", "1")
    with h5py.File(output_root / "subject_01.h5", "r") as handle:
        assert handle["data"].shape == (12, 32, 1000)
        assert json.loads(handle.attrs["dropped_float32_overflow_epoch_indices"]) == {"12": [0]}
        assert handle.attrs["float32_overflow_policy"] == (
            "discard_whole_2s_epoch_if_any_value_exceeds_float32_range"
        )
