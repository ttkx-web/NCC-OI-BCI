from pathlib import Path

import numpy as np
import pytest

from bci_dayloop.data.neuracle_bdf import (
    NeuracleBDFReader,
    annotations_to_events,
    parse_neuracle_marker,
)
from bci_dayloop.data.records import RawEEGRecord, UnitEvidence


class FakeAnnotations:
    def __init__(self, onset: list[float], duration: list[float], description: list[str]) -> None:
        self.onset = np.array(onset)
        self.duration = np.array(duration)
        self.description = np.array(description)


class FakeRaw:
    def __init__(
        self,
        *,
        channel_types: list[str] | None = None,
        annotations: FakeAnnotations | None = None,
    ) -> None:
        self.ch_names = [" C3 ", "ECG", "HEOR", "C4", "EMG"]
        self._channel_types = channel_types or ["eeg", "eeg", "eeg", "eeg", "emg"]
        self.info = {"sfreq": 250.0, "nchan": len(self.ch_names), "meas_date": None}
        self.n_times = 5
        self.annotations = annotations or FakeAnnotations([0.004], [0.0], ["100"])
        self._data = np.array(
            [
                [0.25, 0.5, 0.75, 1.0, 1.25],
                [2.0, 2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0, 3.0],
                [1.5, 1.75, 2.0, 2.25, 2.5],
                [4.0, 4.0, 4.0, 4.0, 4.0],
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


def _safe_uv_evidence() -> UnitEvidence:
    return UnitEvidence("uV", None, "vendor_confirmed")


def _placeholder_bdf(tmp_path: Path) -> Path:
    path = tmp_path / "placeholder.bdf"
    path.touch()
    return path


def test_load_uses_lazy_read_and_returns_only_usable_eeg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = FakeRaw()
    calls: list[dict[str, object]] = []

    def read_raw_bdf(path: str, **kwargs: object) -> FakeRaw:
        calls.append({"path": path, **kwargs})
        return raw

    monkeypatch.setattr("bci_dayloop.data.neuracle_bdf.mne.io.read_raw_bdf", read_raw_bdf)
    path = _placeholder_bdf(tmp_path)

    record = NeuracleBDFReader(_safe_uv_evidence()).load(
        path, subject_id="sub-001", session_id="ses-01", device_id="device-01"
    )

    assert calls == [{"path": str(path), "preload": False, "verbose": "ERROR"}]
    assert raw.get_data_calls == [{"picks": [0, 3], "start": 0, "stop": 5, "units": "uV"}]
    assert isinstance(record, RawEEGRecord)
    assert record.eeg.shape == (2, 5)
    assert record.eeg[0, 0] == pytest.approx(0.25)
    assert record.channel_names == (" C3 ", "C4")
    assert record.channel_types == ("eeg", "eeg")
    assert record.sampling_rate == 250.0
    assert np.array_equal(record.timestamps, np.arange(5, dtype=np.float64) / 250.0)
    assert record.subject_id == "sub-001"
    assert record.session_id == "ses-01"
    assert record.device_id == "device-01"
    assert record.metadata["all_channel_names"] == (" C3 ", "ECG", "HEOR", "C4", "EMG")
    assert record.metadata["all_channel_types"] == ("eeg", "eeg", "eeg", "eeg", "emg")
    assert record.metadata["excluded_channel_names"] == ("ECG", "HEOR", "EMG")
    assert record.metadata["original_channel_count"] == 5
    assert record.metadata["eeg_channel_count"] == 2
    assert record.metadata["source_format"] == "BDF"
    assert record.metadata["reader_name"] == "neuracle-bdf"


def test_non_numeric_annotations_are_preserved_as_custom_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = FakeRaw(annotations=FakeAnnotations([0.008], [0.2], ["start-marker"]))
    monkeypatch.setattr("bci_dayloop.data.neuracle_bdf.mne.io.read_raw_bdf", lambda *_args, **_kwargs: raw)

    record = NeuracleBDFReader(_safe_uv_evidence()).load(_placeholder_bdf(tmp_path))

    assert len(record.events) == 1
    event = record.events[0]
    assert event.sample_index == 2
    assert event.event_type == "custom"
    assert event.code == "start-marker"
    assert event.onset_seconds == 0.008
    assert event.duration_seconds == 0.2
    assert event.metadata == {
        "original_description": "start-marker",
        "marker_code": None,
    }


def test_non_model_safe_evidence_is_rejected() -> None:
    with pytest.raises(ValueError, match="model-safe"):
        NeuracleBDFReader(UnitEvidence("uV", None, "header_candidate"))


def test_non_uv_safe_evidence_is_rejected() -> None:
    with pytest.raises(ValueError, match="normalized_unit"):
        NeuracleBDFReader(UnitEvidence("V", None, "vendor_confirmed"))


def test_missing_bdf_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        NeuracleBDFReader(_safe_uv_evidence()).load(tmp_path / "missing.bdf")


def test_recording_without_usable_eeg_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = FakeRaw(channel_types=["ecg", "eog", "eog", "emg", "emg"])
    monkeypatch.setattr("bci_dayloop.data.neuracle_bdf.mne.io.read_raw_bdf", lambda *_args, **_kwargs: raw)

    with pytest.raises(ValueError, match="no usable EEG"):
        NeuracleBDFReader(_safe_uv_evidence()).load(_placeholder_bdf(tmp_path))


def test_out_of_bounds_annotation_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = FakeRaw(annotations=FakeAnnotations([0.02], [0.0], ["late"]))
    monkeypatch.setattr("bci_dayloop.data.neuracle_bdf.mne.io.read_raw_bdf", lambda *_args, **_kwargs: raw)

    with pytest.raises(ValueError, match="outside the recording"):
        NeuracleBDFReader(_safe_uv_evidence()).load(_placeholder_bdf(tmp_path))


@pytest.mark.parametrize(
    ("description", "code", "event_type", "label"),
    [
        (1, 1, "imagery", "left_hand"),
        ("2", 2, "imagery", "right_hand"),
        ("S 3", 3, "imagery", "feet"),
        ("Stimulus/S4", 4, "imagery", "tongue"),
        ("10", 10, "fixation", None),
        ("20", 20, "rest", None),
        ("90", 90, "block_start", None),
        ("91", 91, "block_end", None),
        ("100", 100, "recording_start", None),
        ("101", 101, "recording_end", None),
        ("127", 127, "abort", None),
    ],
)
def test_known_neuracle_markers_are_mapped(
    description: object, code: int, event_type: str, label: str | None
) -> None:
    parsed = parse_neuracle_marker(description)

    assert parsed["code"] == code
    assert parsed["event_type"] == event_type
    assert parsed["label"] == label
    assert parsed["metadata"]["marker_code"] == code


def test_marker_three_preserves_original_both_feet_label() -> None:
    parsed = parse_neuracle_marker(" S 3 ")

    assert parsed["code"] == 3
    assert parsed["label"] == "feet"
    assert parsed["metadata"]["original_label"] == "both_feet"


@pytest.mark.parametrize("description", ["S 1", "S1", "Stimulus/S 1"])
def test_neuracle_stimulus_string_variants_are_parsed(description: str) -> None:
    parsed = parse_neuracle_marker(description)

    assert parsed["code"] == 1
    assert parsed["event_type"] == "imagery"
    assert parsed["label"] == "left_hand"


def test_unknown_integer_marker_is_retained_as_custom() -> None:
    parsed = parse_neuracle_marker(" S 999 ")

    assert parsed["event_type"] == "custom"
    assert parsed["code"] == 999
    assert parsed["metadata"] == {
        "original_description": " S 999 ",
        "marker_code": 999,
    }


def test_non_numeric_marker_is_retained_as_custom() -> None:
    parsed = parse_neuracle_marker("vendor note")

    assert parsed["event_type"] == "custom"
    assert parsed["code"] == "vendor note"
    assert parsed["label"] is None
    assert parsed["metadata"] == {
        "original_description": "vendor note",
        "marker_code": None,
    }


def test_annotations_compute_sample_indices_with_rounding() -> None:
    events = annotations_to_events(
        FakeAnnotations([0.006], [0.0], ["1"]), sampling_rate=250.0, n_times=10
    )

    assert events[0].sample_index == 2
    assert events[0].metadata["original_description"] == "1"


@pytest.mark.parametrize(
    ("onset", "duration"),
    [
        (-0.004, 0.0),
        (0.0, -0.1),
        (np.nan, 0.0),
        (np.inf, 0.0),
        (0.0, np.nan),
        (0.0, np.inf),
    ],
)
def test_invalid_annotation_times_are_rejected(onset: float, duration: float) -> None:
    with pytest.raises(ValueError):
        annotations_to_events(
            FakeAnnotations([onset], [duration], ["1"]), sampling_rate=250.0, n_times=10
        )


def test_out_of_bounds_annotation_sample_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside the recording"):
        annotations_to_events(
            FakeAnnotations([0.02], [0.0], ["1"]), sampling_rate=250.0, n_times=5
        )


def test_reader_load_uses_formal_marker_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = FakeRaw(annotations=FakeAnnotations([0.004], [0.0], ["Stimulus/S 3"]))
    monkeypatch.setattr("bci_dayloop.data.neuracle_bdf.mne.io.read_raw_bdf", lambda *_args, **_kwargs: raw)

    record = NeuracleBDFReader(_safe_uv_evidence()).load(_placeholder_bdf(tmp_path))

    assert record.events[0].event_type == "imagery"
    assert record.events[0].code == 3
    assert record.events[0].label == "feet"
    assert record.events[0].metadata["original_label"] == "both_feet"
