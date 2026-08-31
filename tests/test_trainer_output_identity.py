from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from bci_dayloop.data.sequential_dataset import SequentialDatasetMetadata


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _script_module(filename: str, name: str) -> object:
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


def _metadata(dataset_name: str, window_sec: float) -> SequentialDatasetMetadata:
    return SequentialDatasetMetadata(
        sample_rate=200.0,
        channel_names=("C3", "C4"),
        class_names=("class_0", "class_1"),
        unit="uV",
        dataset_name=dataset_name,
        window_sec=window_sec,
    )


@pytest.mark.parametrize(
    ("dataset_name", "window_sec", "window_tag"),
    [
        ("workload_pbci_hackathon", 2.0, "2s"),
        ("seed", 2.0, "2s"),
        ("bnci2014_001", 4.0, "4s"),
    ],
)
def test_labram_default_output_identity_uses_loaded_dataset_metadata(
    dataset_name: str,
    window_sec: float,
    window_tag: str,
) -> None:
    trainer = _script_module(
        "train_labram_population_head.py",
        f"test_labram_output_{dataset_name}",
    )
    head_path, run_dir = trainer.resolve_default_artifact_paths(
        metadata=_metadata(dataset_name, window_sec),
        target_subject=1,
        window_seconds=window_sec,
    )

    assert head_path.as_posix().endswith(
        f"checkpoints/heads/stage0/{dataset_name}/subject_01/"
        f"population/{window_tag}_labram/head.pt"
    )
    assert run_dir.parent.as_posix().endswith(
        f"runs/stage0/{dataset_name}/subject_01/"
        f"population/{window_tag}_labram"
    )


@pytest.mark.parametrize(
    ("dataset_name", "window_sec", "window_tag"),
    [
        ("workload_pbci_hackathon", 2.0, "2s_flatten"),
        ("seed", 2.0, "2s_flatten"),
        ("bnci2014_001", 4.0, "4s_flatten"),
    ],
)
def test_cbramod_default_output_identity_uses_loaded_dataset_metadata(
    dataset_name: str,
    window_sec: float,
    window_tag: str,
) -> None:
    trainer = _script_module(
        "train_cbramod_population_head.py",
        f"test_cbramod_output_{dataset_name}",
    )
    head_path, run_dir = trainer.resolve_default_artifact_paths(
        metadata=_metadata(dataset_name, window_sec),
        target_subject=1,
        window_tag=window_tag,
    )

    assert head_path.as_posix().endswith(
        f"checkpoints/heads/stage1/{dataset_name}/subject_01/"
        f"cbramod/{window_tag}/head.pt"
    )
    assert run_dir.as_posix().endswith(
        f"runs/stage1/{dataset_name}/subject_01/"
        f"cbramod/{window_tag}"
    )
