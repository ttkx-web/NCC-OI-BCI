from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.combine_yaxin_0819_0820_h5 import (
    CANONICAL_SESSIONS,
    DROP_CHANNELS,
    SOURCE_0820_COUNTS,
    SOURCE_0820_SESSIONS,
    _session_rows,
    build_payload,
    read_0819,
    read_0820,
    retained_0820_channel_indices,
    verify_output,
)


def test_0820_auxiliary_filter_is_explicitly_name_based() -> None:
    canonical = ["Fpz", "Fp1", "PO5"]
    source = ["Fpz", "ECG", "Fp1", "HEOR", "PO5", "HEOL", "VEOU", "VEOL", "Trigger"]
    indices = retained_0820_channel_indices(
        canonical_channels=canonical,
        source_channels=source,
    )
    assert indices.tolist() == [0, 2, 4]
    assert DROP_CHANNELS == ("ECG", "HEOR", "HEOL", "VEOU", "VEOL", "Trigger")


def test_0820_session_selection_requires_exact_balanced_named_sessions() -> None:
    sessions = np.concatenate([
        np.full(count, session, dtype=object)
        for session, count in zip(SOURCE_0820_SESSIONS, SOURCE_0820_COUNTS, strict=True)
    ])
    labels = np.concatenate([
        np.repeat(np.arange(4, dtype=np.int64), count // 4)
        for count in SOURCE_0820_COUNTS
    ])
    for session, count in zip(SOURCE_0820_SESSIONS, SOURCE_0820_COUNTS, strict=True):
        rows = _session_rows(
            session_ids=sessions,
            labels=labels,
            source_session=session,
            expected_count=count,
        )
        assert len(rows) == count

    labels[0] = 1
    with pytest.raises(ValueError, match="balanced labels"):
        _session_rows(
            session_ids=sessions,
            labels=labels,
            source_session=SOURCE_0820_SESSIONS[0],
            expected_count=40,
        )


def test_generated_clean_yaxin_h5_contract_when_local_sources_are_available() -> None:
    root = Path(__file__).resolve().parents[1]
    source_0819 = root / "data/processed/yaxin/smr_control_yaxin_0819_combined.h5"
    source_0820 = root / "data/processed/yaxin/smr_control_s02_0820.h5"
    output = root / "data/processed/yaxin/smr_control_yaxin_0819_0820_combined.h5"
    if not all(path.is_file() for path in (source_0819, source_0820, output)):
        pytest.skip("Local yaxin source/output H5 files are not available.")

    s0819 = read_0819(source_0819)
    s0820 = read_0820(source_0820)
    payload, mapping = build_payload(s0819, s0820)
    object.__setattr__(s0819, "data", np.empty((0,), dtype=np.float32))
    object.__setattr__(s0820, "data", np.empty((0,), dtype=np.float32))
    verification = verify_output(
        output=output,
        s0819=s0819,
        s0820=s0820,
        payload=payload,
        session_mapping=mapping,
    )
    assert verification["data_shape"] == [680, 59, 2000]
    assert verification["class_counts"] == [170, 170, 170, 170]
    assert verification["exact_duplicate_trial_pairs"] == 0
    assert verification["reader_smoke"]["EEGHDF5"] == "PASS"
    assert verification["reader_smoke"]["TrialReader"] == "PASS"

    with h5py.File(output, "r") as handle:
        assert list(dict.fromkeys(handle["session_ids"].asstr()[:].tolist())) == list(CANONICAL_SESSIONS)
        assert "recommended_split" not in handle.attrs
        assert all("0821" not in value for value in handle["source_session_ids"].asstr()[:])
