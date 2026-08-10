import csv
from pathlib import Path

import numpy as np
import pytest

from bci_dayloop.data.records import EEGEvent, RawEEGRecord, UnitEvidence
from bci_dayloop.data.stage2a_export import read_stage2a_trials_hdf5
from scripts.export_neuracle_stage2a_4s import main


class FakeBDFReader:
    def __init__(self, *_args: object) -> None:
        pass

    def load(self, _path: Path) -> RawEEGRecord:
        return RawEEGRecord(
            eeg=np.arange(59 * 1200, dtype=np.float32).reshape(59, 1200),
            channel_names=tuple(f"Ch{index}" for index in range(59)),
            sampling_rate=250.0,
            unit_evidence=UnitEvidence("uV", "uV", "vendor_confirmed"),
            source_sha256="bdf-hash",
            metadata={"source_format": "BDF", "reader_name": "neuracle-bdf", "reader_version": "1", "conversion_tool": "unverified", "conversion_tool_version": None},
            events=(
                EEGEvent(0, "imagery", code=1, label="left_hand"),
                EEGEvent(990, "rest", code=20),
            ),
        )


def test_export_cli_writes_fixed_1000_sample_hdf5_without_padding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("scripts.export_neuracle_stage2a_4s.NeuracleBDFReader", FakeBDFReader)
    csv_path = tmp_path / "sub-anon.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", "session", "block", "trial", "class", "event_code", "event_name", "flip_time", "lsl_timestamp", "trigger_transport"])
        writer.writeheader()
        writer.writerows([
            {"subject": "sub-anon", "session": "ses-01", "block": "1", "trial": "1", "class": "left_hand", "event_code": "1", "event_name": "imagery", "flip_time": "", "lsl_timestamp": "", "trigger_transport": ""},
            {"subject": "sub-anon", "session": "ses-01", "block": "1", "trial": "1", "class": "", "event_code": "20", "event_name": "rest", "flip_time": "", "lsl_timestamp": "", "trigger_transport": ""},
        ])
    output = tmp_path / "export.h5"

    assert main(["--bdf", str(tmp_path / "1.bdf"), "--csv", str(csv_path), "--output", str(output)]) == 0
    restored = read_stage2a_trials_hdf5(output)
    assert restored["eeg"].shape == (1, 59, 1000)
    assert np.array_equal(restored["eeg"][0], FakeBDFReader().load(tmp_path / "1.bdf").eeg[:, :1000])
    assert restored["observed_event_n_samples"].tolist() == [990]
    assert restored["rest_offset_samples"].tolist() == [-10]
