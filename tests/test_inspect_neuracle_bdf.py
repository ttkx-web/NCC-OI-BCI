import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.inspect_neuracle_bdf import inspect_neuracle_bdf, main


class FakeAnnotations:
    def __init__(self) -> None:
        self.description = np.array(["S 1", "Stimulus/S 3", "note"])


class FakeRaw:
    def __init__(self) -> None:
        self.ch_names = ["C3", " ECG ", "C4"]
        self._channel_types = ["eeg", "eeg", "eeg"]
        self.info = {"sfreq": 4.0}
        self.n_times = 10
        self.annotations = FakeAnnotations()
        self._data = np.array(
            [
                [0.0, np.nan, np.inf, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [10.0] * 10,
                [5.0] * 10,
            ]
        )
        self.get_data_calls: list[dict[str, object]] = []

    def get_channel_types(self) -> list[str]:
        return self._channel_types

    def get_data(
        self, *, picks: list[int], start: int, stop: int, units: str
    ) -> np.ndarray:
        self.get_data_calls.append(
            {"picks": picks, "start": start, "stop": stop, "units": units}
        )
        return self._data[picks, start:stop]


def _input_file(tmp_path: Path) -> Path:
    path = tmp_path / "placeholder.bdf"
    path.write_bytes(b"not a real BDF")
    return path


def test_default_inspection_reads_only_metadata_and_annotations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = FakeRaw()
    read_calls: list[dict[str, object]] = []

    def read_raw_bdf(path: str, **kwargs: object) -> FakeRaw:
        read_calls.append({"path": path, **kwargs})
        return raw

    monkeypatch.setattr("scripts.inspect_neuracle_bdf.mne.io.read_raw_bdf", read_raw_bdf)
    input_path = _input_file(tmp_path)

    report = inspect_neuracle_bdf(input_path)

    assert read_calls == [{"path": str(input_path), "preload": False, "verbose": "ERROR"}]
    assert raw.get_data_calls == []
    assert report["sha256"] == hashlib.sha256(b"not a real BDF").hexdigest()
    assert report["channel_names"] == ["C3", " ECG ", "C4"]
    assert report["channel_types"] == ["eeg", "eeg", "eeg"]
    assert report["channel_count"] == 3
    assert report["sampling_rate"] == 4.0
    assert report["n_times"] == 10
    assert report["duration_seconds"] == 2.5
    assert report["unit"] == "uV"
    assert report["event_count"] == 3
    assert report["marker_counts"] == {"1": 1, "3": 1, "unparsed": 1}
    assert report["excluded_channel_names"] == [" ECG "]
    assert report["short_segment_qc"]["checked"] is False
    assert report["checked_seconds"] == 0.0


def test_bounded_segment_check_uses_microvolts_without_scaling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = FakeRaw()
    monkeypatch.setattr("scripts.inspect_neuracle_bdf.mne.io.read_raw_bdf", lambda *_args, **_kwargs: raw)

    report = inspect_neuracle_bdf(_input_file(tmp_path), max_seconds=1)

    assert raw.get_data_calls == [{"picks": [0, 2], "start": 0, "stop": 4, "units": "uV"}]
    assert report["checked_seconds"] == 1.0
    assert report["short_segment_qc"] == {
        "checked": True,
        "has_nan": True,
        "has_inf": True,
        "constant_channel_names": ["C4"],
    }


def test_cli_writes_json_report_with_mocked_raw(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "scripts.inspect_neuracle_bdf.mne.io.read_raw_bdf", lambda *_args, **_kwargs: FakeRaw()
    )
    output_path = tmp_path / "report.json"

    result = main(["--input", str(_input_file(tmp_path)), "--output", str(output_path)])

    assert result == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["unit"] == "uV"


def test_missing_input_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        inspect_neuracle_bdf(tmp_path / "missing.bdf")
