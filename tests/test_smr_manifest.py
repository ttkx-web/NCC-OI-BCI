from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from scripts.update_smr_control_canonical import apply

from bci_dayloop.data.hdf5_dataset import HDF5Metadata, write_hdf5
from bci_dayloop.data.smr_manifest import (
    MANIFEST_SCHEMA_VERSION,
    build_canonical,
    inspect_h5,
    load_manifest,
    qc_reference,
    rebuild_hash_index,
    write_json_atomic,
)


CLASSES = ["left_hand", "right_hand", "both_hand", "rest"]
CHANNELS = ["C3", "C4"]


def _source(path: Path, *, session: str = "neuracle_smr_S01_0827_160001", channels: list[str] | None = None, data: np.ndarray | None = None, labels: np.ndarray | None = None) -> np.ndarray:
    channels = channels or CHANNELS
    data = np.asarray(data if data is not None else np.arange(4 * len(channels) * 4, dtype=np.float32).reshape(4, len(channels), 4), dtype=np.float32)
    labels = np.asarray(labels if labels is not None else np.arange(4), dtype=np.int64)
    write_hdf5(path, data, labels, np.full(len(data), 2), [session] * len(data), np.arange(len(data)), HDF5Metadata(2.0, channels, CLASSES, "uV", "smr_control"))
    return data


def _manifest(tmp_path: Path, *, source: str, sessions: list[dict], qc: dict | None = None) -> Path:
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION, "canonical_output": "canonical.h5", "trial_hash_index": "hashes.json",
        "canonical_contract": {"dataset_name": "smr_control", "canonical_subject_name": "test", "canonical_subject_id": 1, "class_names": CLASSES, "label_semantics": {str(i): name for i, name in enumerate(CLASSES)}, "sample_rate": 2.0, "unit": "uV", "window_seconds": 2.0, "samples_per_trial": 4, "channel_names": CHANNELS, "allowed_auxiliary_channels": ["ECG"]},
        "source_files": [{"path": source, "status": "accepted", "source_subject_ids": [2]}], "sessions": sessions, "mapping_reference": {},
    }
    if qc:
        payload["qc_reference"] = qc
    path = tmp_path / "dataset_manifest.json"
    write_json_atomic(path, payload)
    return path


def _accepted(source: str, session: str = "neuracle_smr_S01_0827_160001", canonical: str = "S1") -> dict:
    return {"source_file": source, "source_session_id": session, "source_date": "08-27", "source_time": "16:00:01", "source_sort_key": "0827160001", "status": "accepted", "canonical_session_id": canonical, "trial_count": 4, "class_counts": [1, 1, 1, 1], "qc_status": "pass"}


def _index(manifest_path: Path) -> None:
    manifest = load_manifest(manifest_path)
    write_json_atomic(manifest_path.parent / "hashes.json", rebuild_hash_index(manifest_path=manifest_path, manifest=manifest))


def test_manifest_roundtrip_clean_inspection_auxiliary_and_semantics(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"
    raw = np.concatenate((_source(source), np.ones((4, 1, 4), dtype=np.float32)), axis=1)
    # Re-write with an approved auxiliary channel; canonical EEG remains exact.
    _source(source, channels=[*CHANNELS, "ECG"], data=raw)
    manifest_path = _manifest(tmp_path, source="source.h5", sessions=[_accepted("source.h5")])
    report = inspect_h5(input_path=source, manifest_path=manifest_path)
    assert report["overall_status"] == "PASS"
    assert report["contract"]["channels"] == "PASS"
    loaded = load_manifest(manifest_path)
    write_json_atomic(manifest_path, loaded)
    assert load_manifest(manifest_path)["schema_version"] == 1
    with h5py.File(source, "a") as handle:
        handle.attrs["class_names"] = json.dumps(["bad", *CLASSES[1:]])
    assert inspect_h5(input_path=source, manifest_path=manifest_path)["overall_status"] == "REJECT"


def test_unexpected_auxiliary_channel_is_warning_and_channel_mismatch_rejects(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"
    _source(source, channels=[*CHANNELS, "NEW_AUX"])
    manifest_path = _manifest(tmp_path, source="source.h5", sessions=[_accepted("source.h5")])
    report = inspect_h5(input_path=source, manifest_path=manifest_path)
    assert report["overall_status"] == "WARNING"
    _source(source, channels=["C3", "Cz"])
    assert inspect_h5(input_path=source, manifest_path=manifest_path)["overall_status"] == "REJECT"


def test_duplicate_index_reports_exact_and_mixed_duplicates(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"
    data = _source(source)
    manifest_path = _manifest(tmp_path, source="source.h5", sessions=[_accepted("source.h5")])
    _index(manifest_path)
    exact = inspect_h5(input_path=source, manifest_path=manifest_path)
    assert exact["overall_status"] == "REJECT"
    assert exact["duplicates"]["exact_duplicates"] == 4
    mixed = tmp_path / "mixed.h5"
    mixed_data = data.copy()
    mixed_data[-1] += 100
    _source(mixed, data=mixed_data)
    report = inspect_h5(input_path=mixed, manifest_path=manifest_path)
    assert report["overall_status"] == "WARNING"
    assert report["duplicates"]["exact_duplicates"] == 3
    assert report["duplicates"]["new_trials"] == 1


def test_session_partial_and_qc_warning(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"
    baseline = _source(source)
    manifest_path = _manifest(tmp_path, source="source.h5", sessions=[_accepted("source.h5")], qc=qc_reference(baseline, CHANNELS))
    outlier = tmp_path / "outlier.h5"
    _source(outlier, data=baseline * 100)
    report = inspect_h5(input_path=outlier, manifest_path=manifest_path)
    assert report["overall_status"] == "WARNING"
    partial = tmp_path / "partial.h5"
    _source(partial, labels=np.array([0, 0, 1, 1]))
    report = inspect_h5(input_path=partial, manifest_path=manifest_path)
    assert report["sessions"][0]["status"] == "WARNING"


def test_deterministic_build_reader_and_duplicate_invariant(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"
    _source(source)
    manifest_path = _manifest(tmp_path, source="source.h5", sessions=[_accepted("source.h5")])
    manifest = load_manifest(manifest_path)
    first, second = tmp_path / "one.h5", tmp_path / "two.h5"
    assert build_canonical(manifest_path=manifest_path, manifest=manifest, output=first)["reader_smoke"] == "PASS"
    build_canonical(manifest_path=manifest_path, manifest=manifest, output=second)
    with h5py.File(first, "r") as a, h5py.File(second, "r") as b:
        for name in ("data", "labels", "session_ids", "trial_ids", "source_session_ids"):
            assert np.array_equal(a[name][:], b[name][:])
    duplicate_source = tmp_path / "duplicate_source.h5"
    _source(duplicate_source, session="neuracle_smr_S01_0827_170001")
    broken = json.loads(json.dumps(manifest))
    broken["source_files"].append({"path": "duplicate_source.h5", "status": "accepted", "source_subject_ids": [2]})
    broken["sessions"].append(_accepted("duplicate_source.h5", "neuracle_smr_S01_0827_170001", "S2"))
    with pytest.raises(ValueError, match="duplicate invariant"):
        build_canonical(manifest_path=manifest_path, manifest=broken, output=tmp_path / "broken.h5")
    # The transactional apply path does not replace either public artifact if
    # the proposed deterministic build fails validation.
    public = tmp_path / "canonical.h5"
    build_canonical(manifest_path=manifest_path, manifest=manifest, output=public)
    old_manifest = manifest_path.read_bytes()
    old_canonical = public.read_bytes()
    with pytest.raises(ValueError, match="duplicate invariant"):
        apply(manifest_path, manifest, broken)
    assert manifest_path.read_bytes() == old_manifest
    assert public.read_bytes() == old_canonical
