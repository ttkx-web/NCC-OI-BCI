from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from bci_dayloop.data.awakening import (
    CHANNEL_NAMES,
    N_CHANNELS,
    PARENT_ELEMENTS,
    PARENT_SAMPLES,
    WINDOW_SAMPLES,
    AwakeningHDF5Writer,
    ParentWindowPlan,
    assign_logical_sessions,
    extract_subwindows,
    inspect_lance_row,
    validate_awakening_hdf5,
    validate_fixed_window_contract,
)
from bci_dayloop.data.hdf5_dataset import EEGHDF5
from bci_dayloop.data.sequential_dataset import load_sequential_dataset
from bci_dayloop.models.model_50m.config import Model50MConfig
from bci_dayloop.models.model_50m.preprocessing import Model50MPreprocessor
from scripts.prepare_awakening_hdf5 import _preflight_existing_output_dir


def _parent(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(N_CHANNELS, PARENT_SAMPLES)).astype(np.float32)


def _inspect(data: np.ndarray, **overrides: object):
    values = {
        "data": data.reshape(-1),
        "shape": [N_CHANNELS, PARENT_SAMPLES],
        "original_shape": [N_CHANNELS, PARENT_SAMPLES],
        "valid_length": PARENT_ELEMENTS,
        "channel_names": list(CHANNEL_NAMES),
        "label": 1,
    }
    values.update(overrides)
    return inspect_lance_row(**values)


def test_valid_parent_produces_five_exact_non_overlapping_windows() -> None:
    parent = np.arange(PARENT_ELEMENTS, dtype=np.float32).reshape(
        N_CHANNELS, PARENT_SAMPLES
    )
    inspection = _inspect(parent)
    assert inspection.row_rejection_reason is None
    assert inspection.valid_subwindow_indices == (0, 1, 2, 3, 4)

    windows = extract_subwindows(parent, inspection.valid_subwindow_indices)
    assert windows.shape == (5, N_CHANNELS, WINDOW_SAMPLES)
    for index in range(5):
        np.testing.assert_array_equal(
            windows[index],
            parent[:, index * WINDOW_SAMPLES:(index + 1) * WINDOW_SAMPLES],
        )
    reconstructed = np.concatenate(list(windows), axis=-1)
    np.testing.assert_array_equal(reconstructed, parent)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        (
            {
                "data": np.empty(0),
                "shape": [0, 2000],
                "original_shape": [0, 2000],
                "valid_length": 0,
                "channel_names": [],
            },
            "empty_row",
        ),
        ({"shape": [61, 2000]}, "shape_mismatch"),
        ({"original_shape": [62, 1999]}, "original_shape_mismatch"),
        ({"valid_length": 123999}, "valid_length_mismatch"),
        (
            {"channel_names": list(CHANNEL_NAMES[:-1])},
            "channel_count_mismatch",
        ),
        ({"channel_names": list(reversed(CHANNEL_NAMES))}, "channel_order_mismatch"),
    ],
)
def test_invalid_parent_metadata_is_rejected(
    overrides: dict[str, object], reason: str
) -> None:
    values = dict(overrides)
    data = np.asarray(values.pop("data", _parent()))
    assert _inspect(data, **values).row_rejection_reason == reason


def test_nonfinite_and_all_zero_parents_are_rejected() -> None:
    nan_parent = _parent()
    nan_parent[0, 0] = np.nan
    assert _inspect(nan_parent).row_rejection_reason == "nan_or_inf"
    assert _inspect(np.zeros((62, 2000), dtype=np.float32)).row_rejection_reason == (
        "all_zero_row"
    )


def test_only_subwindow_with_all_channel_zero_run_is_discarded() -> None:
    parent = _parent()
    parent[:, 450:453] = 0.0
    inspection = _inspect(parent)
    assert inspection.row_rejection_reason is None
    assert inspection.zero_runs == ((450, 453),)
    assert inspection.zero_run_subwindow_indices == (1,)
    assert inspection.valid_subwindow_indices == (0, 2, 3, 4)


def _plans() -> list[ParentWindowPlan]:
    plans: list[ParentWindowPlan] = []
    row_id = 0
    for subject in ("sub-a", "sub-b"):
        for label in (0, 1):
            for _ in range(10):
                plans.append(
                    ParentWindowPlan(
                        row_position=row_id,
                        global_idx=row_id,
                        sample_id=f"sample-{row_id}",
                        source_subject_id=subject,
                        label=label,
                        valid_subwindow_indices=(0, 1, 2, 3, 4),
                        zero_run_subwindow_indices=(),
                    )
                )
                row_id += 1
    return plans


def test_parent_group_split_is_stratified_and_reproducible() -> None:
    plans = _plans()
    first, manifest = assign_logical_sessions(
        plans, validation_fraction=0.2, seed=42
    )
    second, _ = assign_logical_sessions(
        list(reversed(plans)), validation_fraction=0.2, seed=42
    )
    changed, _ = assign_logical_sessions(
        plans, validation_fraction=0.2, seed=43
    )
    assert first == second
    assert first != changed
    assert len(manifest) == 4
    for group in manifest:
        assert group["S1_parent_rows"] == 8
        assert group["S2_parent_rows"] == 2
    # One assignment per parent means all five children inherit one session.
    assert set(first.values()) == {"S1", "S2"}


def test_only_fixed_two_second_non_overlapping_contract_is_supported() -> None:
    validate_fixed_window_contract(window_seconds=2.0, step_seconds=2.0)
    with pytest.raises(ValueError, match="window-seconds"):
        validate_fixed_window_contract(window_seconds=1.0, step_seconds=2.0)
    with pytest.raises(ValueError, match="step-seconds"):
        validate_fixed_window_contract(window_seconds=2.0, step_seconds=1.0)


def test_existing_outputs_fail_before_formal_conversion(tmp_path: Path) -> None:
    output = tmp_path / "awakening_2s"
    output.mkdir()
    (output / "subject_01.h5").touch()
    with pytest.raises(FileExistsError, match="--overwrite"):
        _preflight_existing_output_dir(output, overwrite=False, dry_run=False)
    _preflight_existing_output_dir(output, overwrite=True, dry_run=False)
    _preflight_existing_output_dir(output, overwrite=False, dry_run=True)


def test_hdf5_schema_existing_loaders_dataloader_and_50m_preprocessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "subject_01.h5"
    first_parent = _parent(10)
    second_parent = _parent(11)
    writer = AwakeningHDF5Writer(
        path,
        total_windows=10,
        canonical_subject_id=1,
        source_subject_id="sub-a",
        seed=42,
        validation_fraction=0.2,
    )
    writer.append(
        windows=extract_subwindows(first_parent, range(5)),
        label=0,
        session_id="S1",
        source_lance_row_id=100,
        source_subwindow_indices=range(5),
        source_sample_id="Awakening:100",
    )
    writer.append(
        windows=extract_subwindows(second_parent, range(5)),
        label=1,
        session_id="S2",
        source_lance_row_id=101,
        source_subwindow_indices=range(5),
        source_sample_id="Awakening:101",
    )
    writer.close()

    validation = validate_awakening_hdf5(
        path,
        expected_windows=10,
        canonical_subject_id=1,
        source_subject_id="sub-a",
    )
    assert validation["windows"] == 10
    with h5py.File(path, "r") as handle:
        assert handle["data"].shape == (10, 62, 400)
        assert handle["data"].dtype == np.dtype("float32")
        assert handle["data"].compression == "gzip"
        assert handle["data"].compression_opts == 4
        assert handle["data"].shuffle
        assert handle["data"].chunks == (8, 62, 400)
        assert np.array_equal(handle["trial_ids"][:], np.arange(10))
        assert np.array_equal(handle["source_lance_row_ids"][:5], [100] * 5)
        assert np.array_equal(handle["source_subwindow_indices"][:5], range(5))
        assert handle["source_sample_ids"].asstr()[0] == "Awakening:100"
        assert json.loads(handle.attrs["class_names"]) == [
            "non_awakening", "awakening"
        ]
        assert bool(handle.attrs["no_temporal_padding"])

    reader = EEGHDF5(path)
    assert reader.available_sessions() == ["S1", "S2"]
    raw = reader.load("S1")
    assert raw["data"].shape == (5, 62, 400)
    sequential = load_sequential_dataset(path, session="S1")
    assert sequential.data.shape == (5, 62, 400)
    batch = next(
        iter(
            DataLoader(
                TensorDataset(
                    torch.from_numpy(raw["data"]),
                    torch.from_numpy(raw["labels"]),
                ),
                batch_size=2,
                shuffle=False,
            )
        )
    )
    assert batch[0].shape == (2, 62, 400)

    import bci_dayloop.models.model_50m.preprocessing as preprocessing

    original_resample = preprocessing._resample_signal
    calls = 0

    def counted_resample(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original_resample(*args, **kwargs)

    monkeypatch.setattr(preprocessing, "_resample_signal", counted_resample)
    config = Model50MConfig(checkpoint_path="unused", window_seconds=2.0)
    processed = Model50MPreprocessor(config)(
        signal=raw["data"][0],
        channel_names=reader.metadata.channel_names,
        original_sample_rate=reader.metadata.sample_rate,
        input_unit=reader.metadata.unit,
    )
    assert processed.signal.shape == (64, 200)
    assert processed.padded_points == 0
    assert processed.cropped_points == 0
    assert processed.original_sample_rate == 200.0
    assert processed.target_sample_rate == 100.0
    assert processed.mapped_channel_count == 60
    assert processed.missing_channel_count == 4
    assert set(processed.unknown_channel_names) == {"TP9", "TP10"}
    assert calls == 1
