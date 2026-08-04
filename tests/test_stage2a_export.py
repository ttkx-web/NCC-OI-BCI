from dataclasses import replace

import numpy as np
import pytest

from bci_dayloop.data.records import EEGEvent
from bci_dayloop.data.stage2a_export import (
    ELIGIBLE_FOR_ACCURACY,
    WINDOW_SEMANTICS,
    export_stage2a_trials_hdf5,
    read_stage2a_trials_hdf5,
)
from bci_dayloop.data.trial_extraction import EEGTrial


def _trial(*, end_sample: int = 1000) -> EEGTrial:
    start = EEGEvent(0, "imagery", code=1, label="left_hand", block_id=1, trial_id=1)
    end = EEGEvent(end_sample, "rest", code=20, block_id=1, trial_id=1)
    return EEGTrial(
        eeg=np.arange(59 * 1000, dtype=np.float32).reshape(59, 1000),
        label="left_hand",
        block_id=1,
        trial_id=1,
        start_sample=0,
        end_sample=end_sample,
        duration_seconds=4.0,
        start_event=start,
        end_event=end,
        canonical_start_sample=0,
        canonical_end_sample=1000,
        canonical_n_samples=1000,
        observed_rest_sample=end_sample,
        observed_event_n_samples=end_sample,
        rest_offset_samples=end_sample - 1000,
        rest_offset_seconds=(end_sample - 1000) / 250.0,
        endpoint_qc_passed=True,
        source_metadata={
            "bdf_sha256": "bdf-hash",
            "csv_sha256": "csv-hash",
            "source_format": "BDF",
            "conversion_tool": "unverified",
            "conversion_tool_version": None,
            "reader_name": "neuracle-bdf",
            "reader_version": "1",
            "unit_evidence_level": "vendor_confirmed",
        },
    )


def test_stage2a_export_round_trips_every_required_field(tmp_path) -> None:
    path = tmp_path / "stage2a.h5"
    manifest = export_stage2a_trials_hdf5(
        path,
        (_trial(),),
        channel_names=tuple(f"Ch{index}" for index in range(59)),
        sampling_rate=250.0,
        subject_id="sub-anon",
        session_id="ses-01",
    )

    restored = read_stage2a_trials_hdf5(path)

    assert manifest["window_semantics"] == "cue_plus_imagery_4s"
    assert manifest["eligible_for_accuracy"] is False
    assert WINDOW_SEMANTICS == "cue_plus_imagery_4s"
    assert ELIGIBLE_FOR_ACCURACY is False
    assert restored["eeg"].shape == (1, 59, 1000)
    assert restored["eeg"].dtype == np.float32
    assert restored["sampling_rate"] == 250.0
    assert restored["unit"] == "uV"
    assert restored["subject_id"] == "sub-anon"
    assert restored["session_id"] == "ses-01"
    assert restored["window_semantics"] == "cue_plus_imagery_4s"
    assert restored["eligible_for_accuracy"] is False
    assert restored["canonical_start_samples"].tolist() == [0]
    assert restored["canonical_end_samples"].tolist() == [1000]
    assert restored["observed_rest_samples"].tolist() == [1000]
    assert restored["observed_event_n_samples"].tolist() == [1000]
    assert restored["rest_offset_samples"].tolist() == [0]
    assert restored["endpoint_qc_passed"].tolist() == [True]
    assert restored["window_semantics_per_trial"].tolist() == ["cue_plus_imagery_4s"]
    assert restored["eligible_for_accuracy_per_trial"].tolist() == [False]
    assert restored["extraction_policy"].tolist() == ["fixed_duration_from_class_marker"]
    assert restored["labels"].tolist() == ["left_hand"]
    assert restored["block_ids"].tolist() == ["1"]
    assert restored["trial_ids"].tolist() == ["1"]
    assert restored["duration_seconds"].tolist() == [4.0]
    assert restored["bdf_sha256"].tolist() == ["bdf-hash"]
    assert restored["csv_sha256"].tolist() == ["csv-hash"]
    assert restored["source_format"].tolist() == ["BDF"]
    assert restored["conversion_tool"].tolist() == ["unverified"]
    assert restored["reader_name"].tolist() == ["neuracle-bdf"]
    assert restored["unit_evidence_level"].tolist() == ["vendor_confirmed"]


def test_stage2a_export_rejects_channel_mismatch_and_missing_provenance(tmp_path) -> None:
    with pytest.raises(ValueError, match="1000"):
        export_stage2a_trials_hdf5(
            tmp_path / "bad.h5",
            (_trial(),),
            channel_names=("C3",),
            sampling_rate=250.0,
            subject_id=None,
            session_id=None,
        )
    with pytest.raises(ValueError, match="provenance"):
        export_stage2a_trials_hdf5(
            tmp_path / "no-provenance.h5",
            (replace(_trial(), source_metadata={}),),
            channel_names=tuple(f"Ch{index}" for index in range(59)),
            sampling_rate=250.0,
            subject_id=None,
            session_id=None,
        )
