from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from bci_dayloop.data.dataset_adapter_registry import (
    DEFAULT_DATASET_ADAPTER_REGISTRY,
    inspect_hdf5_dataset,
)
from bci_dayloop.data.hdf5_dataset import HDF5Metadata, write_hdf5
from bci_dayloop.data.sequential_dataset import load_sequential_dataset
from bci_dayloop.data.workload import (
    DIFF_CONDITION,
    EASY_CONDITION,
    WorkloadCondition,
    build_workload_session,
    write_workload_hdf5,
)


def _write_legacy(path: Path) -> Path:
    return write_hdf5(
        path,
        np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4),
        np.asarray([2, 0, 1], dtype=np.int64),
        np.asarray([7, 7, 7], dtype=np.int64),
        ["legacy_session"] * 3,
        np.asarray([30, 10, 20], dtype=np.int64),
        HDF5Metadata(
            sample_rate=2.0,
            channel_names=["C3", "C4"],
            class_names=["left", "right", "feet"],
            unit="uV",
            dataset_name="legacy_test",
        ),
    )


def _write_seed(path: Path) -> Path:
    with h5py.File(path, "w") as handle:
        handle.attrs["dataset_name"] = "seed"
        handle.attrs["subject_id"] = "SEED-02"
        handle.attrs["class_names"] = json.dumps(["negative", "neutral", "positive"])
        handle.attrs["unit"] = "uV"
        handle.attrs["window_sec"] = 2.0
        group = handle.create_group("sessions").create_group("S3")
        group.attrs["sample_rate"] = 2.0
        group.attrs["channel_names"] = json.dumps(["FP1", "FP2"])
        group.create_dataset(
            "data",
            data=np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4),
        )
        group.create_dataset("labels", data=np.asarray([2, 0, 1], dtype=np.int64))
        group.create_dataset(
            "trial_ids", data=np.asarray([b"trial-30", b"trial-10", b"trial-20"])
        )
        group.create_dataset("trial_ordinals", data=np.asarray([1, 2, 3]))
    return path


def _write_workload(path: Path) -> Path:
    easy = WorkloadCondition(
        condition=EASY_CONDITION,
        data=np.arange(2 * 2 * 4, dtype=np.float32).reshape(2, 2, 4),
        channel_names=("C3", "C4"),
        sample_rate=2.0,
        unit="uV",
        source_set=path.parent / "easy.set",
    )
    diff = WorkloadCondition(
        condition=DIFF_CONDITION,
        data=(100 + np.arange(2 * 2 * 4, dtype=np.float32)).reshape(2, 2, 4),
        channel_names=("C3", "C4"),
        sample_rate=2.0,
        unit="uV",
        source_set=path.parent / "diff.set",
    )
    session = build_workload_session(
        easy,
        diff,
        subject_id="P03",
        session_id="S1",
    )
    return write_workload_hdf5(
        path,
        [session],
        subject_id="P03",
        data_root=path.parent,
    )


@pytest.mark.parametrize(
    ("writer", "session", "adapter_name"),
    (
        (_write_legacy, "legacy_session", "legacy_hdf5"),
        (_write_seed, "S3", "seed_hdf5"),
        (_write_workload, "S1", "workload_hdf5"),
    ),
)
def test_registry_selects_the_adapter_for_each_supported_hdf5_layout(
    tmp_path: Path,
    writer,
    session: str,
    adapter_name: str,
) -> None:
    path = writer(tmp_path / f"{adapter_name}.h5")

    descriptor = inspect_hdf5_dataset(path)
    adapter = DEFAULT_DATASET_ADAPTER_REGISTRY.resolve(descriptor)
    dataset = load_sequential_dataset(path, session=session)

    assert adapter.name == adapter_name
    assert dataset.num_trials > 0


@pytest.mark.parametrize("include_root_data", (False, True))
def test_registry_fails_closed_for_unknown_hdf5_layout(
    tmp_path: Path,
    include_root_data: bool,
) -> None:
    path = tmp_path / "unknown.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["dataset_name"] = "unregistered_dataset"
        if include_root_data:
            handle.create_dataset("data", data=np.zeros((1, 1, 1)))
        handle.create_group("sessions")

    with pytest.raises(ValueError, match="no registered dataset adapter matches"):
        load_sequential_dataset(path, session="S1")


@pytest.mark.parametrize(
    (
        "writer",
        "session",
        "expected_subject_ids",
        "expected_channel_names",
        "expected_trial_ids",
        "expected_labels",
    ),
    (
        (
            _write_legacy,
            "legacy_session",
            [7, 7, 7],
            ("C3", "C4"),
            [30, 10, 20],
            [2, 0, 1],
        ),
        (
            _write_seed,
            "S3",
            ["SEED-02"] * 3,
            ("FP1", "FP2"),
            ["trial-30", "trial-10", "trial-20"],
            [2, 0, 1],
        ),
        (
            _write_workload,
            "S1",
            ["P03"] * 4,
            ("C3", "C4"),
            [
                "P03:S1:MATBeasy:000000",
                "P03:S1:MATBdiff:000000",
                "P03:S1:MATBeasy:000001",
                "P03:S1:MATBdiff:000001",
            ],
            [0, 1, 0, 1],
        ),
    ),
)
def test_registry_preserves_persisted_trial_order_and_identity(
    tmp_path: Path,
    writer,
    session: str,
    expected_subject_ids: list[int] | list[str],
    expected_channel_names: tuple[str, ...],
    expected_trial_ids: list[int] | list[str],
    expected_labels: list[int],
) -> None:
    path = writer(tmp_path / "subject.h5")

    dataset = load_sequential_dataset(path, session=session)

    assert dataset.trial_ids.tolist() == expected_trial_ids
    assert dataset.labels.tolist() == expected_labels
    assert dataset.subject_ids.tolist() == expected_subject_ids
    assert dataset.session_ids.tolist() == [session] * len(expected_labels)
    assert dataset.trial_ordinals.tolist() == list(range(1, len(expected_labels) + 1))
    assert dataset.metadata.sample_rate == pytest.approx(2.0)
    assert dataset.metadata.channel_names == expected_channel_names
