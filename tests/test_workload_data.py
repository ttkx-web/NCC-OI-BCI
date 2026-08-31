from __future__ import annotations

import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

from bci_dayloop.data import workload


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def make_data(*, offset: float, epochs: int = 3, samples: int = 4) -> np.ndarray:
    return (
        np.arange(epochs * 2 * samples, dtype=np.float32).reshape(epochs, 2, samples)
        + offset
    )


def make_condition(
    condition: str,
    *,
    data: np.ndarray | None = None,
    channel_names: tuple[str, ...] = ("C3", "C4"),
    sample_rate: float = 2.0,
    unit: str = "V",
    path: Path | None = None,
) -> workload.WorkloadCondition:
    return workload.WorkloadCondition(
        condition=condition,
        data=make_data(offset=0.0) if data is None else data,
        channel_names=channel_names,
        sample_rate=sample_rate,
        unit=unit,
        source_set=path or Path(f"/source/{condition}.set"),
    )


def make_session(
    session_id: str,
    *,
    easy_data: np.ndarray | None = None,
    diff_data: np.ndarray | None = None,
    channel_names: tuple[str, ...] = ("C3", "C4"),
    sample_rate: float = 2.0,
    unit: str = "V",
    source_root: Path = Path("/source"),
) -> workload.WorkloadSession:
    easy = make_condition(
        workload.EASY_CONDITION,
        data=make_data(offset=0.0) if easy_data is None else easy_data,
        channel_names=channel_names,
        sample_rate=sample_rate,
        unit=unit,
        path=source_root / f"{session_id}_easy.set",
    )
    diff = make_condition(
        workload.DIFF_CONDITION,
        data=make_data(offset=1000.0) if diff_data is None else diff_data,
        channel_names=channel_names,
        sample_rate=sample_rate,
        unit=unit,
        path=source_root / f"{session_id}_diff.set",
    )
    return workload.build_workload_session(
        easy, diff, subject_id="P03", session_id=session_id
    )


def test_interleave_uses_fixed_labels_conditions_ordinals_and_source_indices() -> None:
    easy = make_data(offset=0.0)
    diff = make_data(offset=1000.0)

    stream = workload.interleave_easy_diff(
        easy, diff, subject_id="P03", session_id="S1"
    )

    assert stream.data.dtype == np.float32
    np.testing.assert_array_equal(stream.labels, [0, 1, 0, 1, 0, 1])
    np.testing.assert_array_equal(stream.condition_ids, [0, 1, 0, 1, 0, 1])
    np.testing.assert_array_equal(stream.source_epoch_indices, [0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(stream.trial_ordinals, [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(stream.data[0::2], easy)
    np.testing.assert_array_equal(stream.data[1::2], diff)
    assert stream.window_ids.tolist() == [
        "P03:S1:MATBeasy:000000",
        "P03:S1:MATBdiff:000000",
        "P03:S1:MATBeasy:000001",
        "P03:S1:MATBdiff:000001",
        "P03:S1:MATBeasy:000002",
        "P03:S1:MATBdiff:000002",
    ]


def test_hdf5_round_trip_keeps_sessions_separate_and_metadata_complete(tmp_path: Path) -> None:
    s1 = make_session("S1", source_root=tmp_path / "raw")
    s2 = make_session(
        "S2",
        easy_data=make_data(offset=2000.0),
        diff_data=make_data(offset=3000.0),
        source_root=tmp_path / "raw",
    )
    path = workload.write_workload_hdf5(
        tmp_path / "processed" / "subject_03.h5",
        [s1, s2],
        subject_id="P03",
        data_root=tmp_path / "raw",
    )

    dataset = workload.WorkloadHDF5(path)
    assert dataset.sessions() == ["S1", "S2"]
    loaded_s1 = dataset.load(session="S1")
    loaded_s2 = dataset.load(session="S2")
    np.testing.assert_array_equal(loaded_s1["data"], s1.stream.data)
    np.testing.assert_array_equal(loaded_s2["data"], s2.stream.data)
    assert not np.array_equal(loaded_s1["data"], loaded_s2["data"])
    np.testing.assert_array_equal(loaded_s1["labels"], [0, 1, 0, 1, 0, 1])
    np.testing.assert_array_equal(loaded_s1["source_epoch_indices"], [0, 0, 1, 1, 2, 2])
    assert loaded_s1["window_ids"].tolist() == s1.stream.window_ids.tolist()

    with h5py.File(path, "r") as handle:
        assert sorted(handle["sessions"].keys()) == ["S1", "S2"]
        assert handle["sessions"]["S1"]["data"].shape == (6, 2, 4)
        assert handle.attrs["dataset_name"] == "workload_pbci_hackathon"
        assert handle.attrs["input_is_preprocessed"]
        assert json.loads(handle.attrs["class_names"]) == ["low_workload", "high_workload"]
        assert json.loads(handle.attrs["label_map"]) == {"MATBeasy": 0, "MATBdiff": 1}
        assert json.loads(handle.attrs["ignored_conditions"]) == ["MATBmed", "RS", "RSraw"]
        assert handle.attrs["stream_construction"] == "synthetic_alternating_easy_diff"
        assert handle.attrs["preserves_within_condition_order"]
        assert not handle.attrs["preserves_original_cross_condition_timeline"]
        assert handle["sessions"]["S1"].attrs["shuffle"] == False
        assert handle["sessions"]["S1"].attrs["alternating"] == True
        assert handle["sessions"]["S1"].attrs["easy_source_set"] == "S1_easy.set"


def test_write_refuses_existing_file_without_overwrite(tmp_path: Path) -> None:
    path = workload.write_workload_hdf5(
        tmp_path / "subject_03.h5",
        [make_session("S1")],
        subject_id="P03",
        data_root=tmp_path,
    )
    before = path.read_bytes()

    with pytest.raises(FileExistsError, match="--overwrite"):
        workload.write_workload_hdf5(
            path,
            [make_session("S1", easy_data=make_data(offset=9.0))],
            subject_id="P03",
            data_root=tmp_path,
        )

    assert path.read_bytes() == before


def test_count_channel_order_rate_and_window_validation_errors() -> None:
    easy = make_condition(workload.EASY_CONDITION)
    with pytest.raises(ValueError, match="easy epochs=3.*diff epochs=2"):
        workload.build_workload_session(
            easy,
            make_condition(workload.DIFF_CONDITION, data=make_data(offset=1.0, epochs=2)),
            subject_id="P03",
            session_id="S1",
        )
    with pytest.raises(ValueError, match="Channel count mismatch"):
        workload.build_workload_session(
            make_condition(
                workload.EASY_CONDITION,
                channel_names=("C3",),
            ),
            make_condition(workload.DIFF_CONDITION),
            subject_id="P03",
            session_id="S1",
        )
    with pytest.raises(ValueError, match="Channel names/order mismatch"):
        workload.build_workload_session(
            easy,
            make_condition(workload.DIFF_CONDITION, channel_names=("C4", "C3")),
            subject_id="P03",
            session_id="S1",
        )
    with pytest.raises(ValueError, match="Sample rate mismatch"):
        workload.build_workload_session(
            easy,
            make_condition(
                workload.DIFF_CONDITION,
                data=make_data(offset=1.0, samples=8),
                sample_rate=4.0,
            ),
            subject_id="P03",
            session_id="S1",
        )
    with pytest.raises(ValueError, match="Expected a 2 s epoch"):
        workload.build_workload_session(
            make_condition(workload.EASY_CONDITION, data=make_data(offset=0.0, samples=3)),
            make_condition(workload.DIFF_CONDITION),
            subject_id="P03",
            session_id="S1",
        )


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_empty_and_nonfinite_conditions_are_rejected(invalid_value: float) -> None:
    empty = make_condition(workload.EASY_CONDITION, data=make_data(offset=0.0, epochs=0))
    with pytest.raises(ValueError, match="epoch count=0"):
        workload.build_workload_session(
            empty,
            make_condition(workload.DIFF_CONDITION, data=make_data(offset=1.0, epochs=0)),
            subject_id="P03",
            session_id="S1",
        )
    invalid = make_data(offset=0.0)
    invalid[0, 0, 0] = invalid_value
    with pytest.raises(ValueError, match="Non-finite"):
        workload.build_workload_session(
            make_condition(workload.EASY_CONDITION, data=invalid),
            make_condition(workload.DIFF_CONDITION),
            subject_id="P03",
            session_id="S1",
        )


def test_locate_sources_reads_only_target_conditions_and_reports_missing(tmp_path: Path) -> None:
    eeg_dir = tmp_path / "P03" / "S1" / "eeg"
    eeg_dir.mkdir(parents=True)
    easy = eeg_dir / "alldata_sbj03_sess1_MATBeasy.set"
    diff = eeg_dir / "alldata_sbj03_sess1_MATBdiff.set"
    for path in (
        easy,
        diff,
        eeg_dir / "alldata_sbj03_sess1_MATBmed.set",
        eeg_dir / "alldata_sbj03_sess1_RS.set",
        eeg_dir / "alldata_sbj03_sess1_RSraw.set",
    ):
        path.touch()

    sources = workload.locate_workload_sources(
        tmp_path, subject_id="P03", session_id="S1"
    )
    assert sources.easy_set == easy
    assert sources.diff_set == diff

    diff.unlink()
    with pytest.raises(FileNotFoundError, match="MATBdiff"):
        workload.locate_workload_sources(tmp_path, subject_id="P03", session_id="S1")


def test_prepare_subject_uses_requested_sessions_with_fake_eeglab_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for session_id in ("S1", "S2"):
        eeg_dir = tmp_path / "raw" / "P03" / session_id / "eeg"
        eeg_dir.mkdir(parents=True)
        (eeg_dir / f"published_{session_id}_MATBeasy.set").touch()
        (eeg_dir / f"published_{session_id}_MATBdiff.set").touch()

    requested_conditions: list[str] = []

    def fake_reader(
        set_path: str | Path,
        *,
        subject_id: str,
        session_id: str,
        condition: str,
    ) -> workload.WorkloadCondition:
        del subject_id
        requested_conditions.append(condition)
        return make_condition(
            condition,
            data=make_data(offset=0.0 if condition == workload.EASY_CONDITION else 1000.0),
            path=Path(set_path),
        )

    monkeypatch.setattr(workload, "read_eeglab_condition", fake_reader)
    output = workload.prepare_workload_subject(
        tmp_path / "raw",
        tmp_path / "processed",
        subject=3,
        sessions=["S1", "S2"],
    )

    assert output == tmp_path / "processed" / "subject_03.h5"
    assert requested_conditions == ["MATBeasy", "MATBdiff", "MATBeasy", "MATBdiff"]
    assert workload.WorkloadHDF5(output).sessions() == ["S1", "S2"]


def test_eeglab_import_failure_reports_source_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "MATBeasy.set"
    source.touch()

    class FakeMNEIO:
        @staticmethod
        def read_epochs_eeglab(path: Path, *, verbose: bool) -> object:
            del path, verbose
            raise ImportError("missing scipy sparse array")

    class FakeMNE:
        io = FakeMNEIO()

    monkeypatch.setitem(__import__("sys").modules, "mne", FakeMNE())
    with pytest.raises(RuntimeError, match="Could not load Workload EEGLAB epochs") as error:
        workload.read_eeglab_condition(
            source,
            subject_id="P03",
            session_id="S1",
            condition=workload.EASY_CONDITION,
        )
    assert str(source) in str(error.value)


def test_cli_reports_existing_output_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import prepare_workload_hdf5

    def existing_output(*args: object, **kwargs: object) -> Path:
        del args, kwargs
        raise FileExistsError("Workload output already exists: result.h5. Pass --overwrite to replace it.")

    monkeypatch.setattr(prepare_workload_hdf5, "prepare_workload_subject", existing_output)
    with pytest.raises(SystemExit, match="2") as error:
        prepare_workload_hdf5.main(
            [
                "--data-root",
                str(tmp_path / "raw"),
                "--output-root",
                str(tmp_path / "processed"),
                "--subjects",
                "3",
                "--sessions",
                "S1",
            ]
        )
    assert error.value.code == 2


def test_cli_summary_reports_each_session_stream(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import prepare_workload_hdf5

    path = workload.write_workload_hdf5(
        tmp_path / "subject_03.h5",
        [make_session("S1"), make_session("S2")],
        subject_id="P03",
        data_root=tmp_path,
    )

    prepare_workload_hdf5._print_subject_summary(path)

    output = capsys.readouterr().out
    assert "Subject: P03" in output
    assert "Session: S1" in output
    assert "Session: S2" in output
    assert "Data shape: (6, 2, 4)" in output
    assert "First 10 labels: 0 1 0 1 0 1" in output
    assert "P03:S1:MATBeasy:000000" in output
    assert f"Output path: {path}" in output


def test_cross_session_mismatch_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Channel names/order differs across sessions"):
        workload.write_workload_hdf5(
            tmp_path / "subject_03.h5",
            [make_session("S1"), make_session("S2", channel_names=("C4", "C3"))],
            subject_id="P03",
            data_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("session", "message"),
    [
        (make_session("S2", sample_rate=4.0, easy_data=make_data(offset=2.0, samples=8), diff_data=make_data(offset=3.0, samples=8)), "Sample rate differs across sessions"),
        (make_session("S2", unit="uV"), "Data unit differs across sessions"),
    ],
)
def test_cross_session_rate_and_unit_mismatches_are_rejected(
    tmp_path: Path,
    session: workload.WorkloadSession,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        workload.write_workload_hdf5(
            tmp_path / "subject_03.h5",
            [make_session("S1"), session],
            subject_id="P03",
            data_root=tmp_path,
        )
