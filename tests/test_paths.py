from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pytest

from bci_dayloop.utils import paths as path_utils


def _set_storage_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(path_utils, "DATA_ROOT", (tmp_path / "data").resolve())
    monkeypatch.setattr(
        path_utils,
        "CHECKPOINT_ROOT",
        (tmp_path / "checkpoints").resolve(),
    )
    monkeypatch.setattr(path_utils, "RUN_ROOT", (tmp_path / "runs").resolve())
    monkeypatch.setattr(
        path_utils,
        "MODEL_PACKAGE_ROOT",
        (tmp_path / "model_packages").resolve(),
    )
    monkeypatch.setattr(
        path_utils,
        "REGISTRY_ROOT",
        (tmp_path / "registries").resolve(),
    )


def test_project_root_matches_repository_root() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    assert path_utils.PROJECT_ROOT == repository_root


def test_subject_window_and_contract_tags() -> None:
    assert path_utils.subject_tag(1) == "subject_01"
    assert path_utils.subject_tag(12) == "subject_12"
    assert path_utils.window_tag(4.0) == "4s"
    assert path_utils.window_tag(4.25) == "4p25s"
    assert path_utils.contract_tag(4.0, " Flatten ") == "4s_flatten"
    assert path_utils.contract_tag(4.0, "Mean Pooling") == "4s_mean_pooling"


@pytest.mark.parametrize("subject_id", [0, -1])
def test_subject_tag_rejects_non_positive_ids(subject_id: int) -> None:
    with pytest.raises(ValueError, match="subject_id must be positive"):
        path_utils.subject_tag(subject_id)


@pytest.mark.parametrize("window_seconds", [0.0, -4.0])
def test_window_tag_rejects_non_positive_duration(
    window_seconds: float,
) -> None:
    with pytest.raises(ValueError, match="window_seconds must be positive"):
        path_utils.window_tag(window_seconds)


def test_timestamp_id_has_second_resolution_format() -> None:
    value = path_utils.timestamp_id()

    assert re.fullmatch(r"\d{8}_\d{6}", value)
    datetime.strptime(value, "%Y%m%d_%H%M%S")


def test_data_and_checkpoint_paths_follow_standard_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_storage_roots(monkeypatch, tmp_path)

    assert path_utils.processed_subject_path(
        "bnci2014_001",
        1,
    ) == (
        tmp_path
        / "data"
        / "processed"
        / "bnci2014_001"
        / "subject_01.h5"
    ).resolve()

    assert path_utils.backbone_checkpoint_path(
        "50m",
        "model_deploy.pt",
    ) == (
        tmp_path
        / "checkpoints"
        / "backbones"
        / "50m"
        / "model_deploy.pt"
    ).resolve()

    assert path_utils.population_head_path(
        stage="stage1",
        dataset="bnci2014_001",
        subject_id=1,
        window_seconds=4.0,
        aggregation="flatten",
    ) == (
        tmp_path
        / "checkpoints"
        / "heads"
        / "stage1"
        / "bnci2014_001"
        / "subject_01"
        / "population"
        / "4s_flatten"
        / "head.pt"
    ).resolve()

    assert path_utils.personal_head_path(
        stage="stage1",
        dataset="bnci2014_001",
        subject_id=1,
        window_seconds=4.0,
        aggregation="flatten",
        trials_per_class=20,
        seed=42,
    ) == (
        tmp_path
        / "checkpoints"
        / "heads"
        / "stage1"
        / "bnci2014_001"
        / "subject_01"
        / "personal"
        / "4s_flatten"
        / "trials_20"
        / "seed_42"
        / "head.pt"
    ).resolve()


def test_run_paths_include_contract_budget_seed_and_run_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_storage_roots(monkeypatch, tmp_path)
    run_id = "20260804_150000"

    assert path_utils.population_run_dir(
        stage="stage1",
        dataset="bnci2014_001",
        subject_id=1,
        window_seconds=4.0,
        aggregation="flatten",
        run_id=run_id,
    ) == (
        tmp_path
        / "runs"
        / "stage1"
        / "bnci2014_001"
        / "subject_01"
        / "population"
        / "4s_flatten"
        / run_id
    ).resolve()

    assert path_utils.personal_run_dir(
        stage="stage1",
        dataset="bnci2014_001",
        subject_id=1,
        window_seconds=4.0,
        aggregation="flatten",
        trials_per_class=20,
        seed=42,
        run_id=run_id,
    ) == (
        tmp_path
        / "runs"
        / "stage1"
        / "bnci2014_001"
        / "subject_01"
        / "personal"
        / "4s_flatten"
        / "trials_20"
        / "seed_42"
        / run_id
    ).resolve()

    assert path_utils.comparison_run_dir(
        stage="stage1",
        dataset="bnci2014_001",
        subject_id=1,
        session="1test",
        run_id=run_id,
    ) == (
        tmp_path
        / "runs"
        / "stage1"
        / "bnci2014_001"
        / "subject_01"
        / "comparisons"
        / "1test"
        / run_id
    ).resolve()


def test_package_and_registry_paths_are_versioned_and_separate_from_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_storage_roots(monkeypatch, tmp_path)
    version = "20260804_150000"

    population = path_utils.population_package_dir(
        stage="stage1",
        dataset="bnci2014_001",
        subject_id=1,
        window_seconds=4.0,
        aggregation="flatten",
        version=version,
    )
    personal = path_utils.personal_package_dir(
        stage="stage1",
        dataset="bnci2014_001",
        subject_id=1,
        window_seconds=4.0,
        aggregation="flatten",
        trials_per_class=20,
        seed=42,
        version=version,
    )
    registry = path_utils.personal_registry_path("stage1")

    assert population == (
        tmp_path
        / "model_packages"
        / "stage1"
        / "bnci2014_001"
        / "subject_01"
        / "population"
        / "4s_flatten"
        / version
    ).resolve()
    assert personal == (
        tmp_path
        / "model_packages"
        / "stage1"
        / "bnci2014_001"
        / "subject_01"
        / "personal"
        / "4s_flatten"
        / "trials_20"
        / "seed_42"
        / version
    ).resolve()
    assert registry == (
        tmp_path / "registries" / "stage1_personal_models.json"
    ).resolve()

    assert path_utils.RUN_ROOT not in population.parents
    assert path_utils.RUN_ROOT not in personal.parents
    assert version == population.name == personal.name
