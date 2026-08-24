from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_seed_loso_experiment.py"


def _module() -> object:
    scripts = str(SCRIPT.parent)
    sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location(
            "test_run_seed_loso_experiment_script", SCRIPT
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts)


def _plan(module: object, target: int = 1) -> object:
    return module.build_subject_plan(
        subjects=range(1, 16),
        target_subject=target,
        data_root=Path("data/processed/seed"),
        data_pattern="subject_{subject:02d}.h5",
        output_root=Path("experiments/seed_loso"),
        labram_checkpoint=Path("models/labram-base.pth"),
        cbramod_checkpoint=Path("models/cbramod.pth"),
        train_session="S1",
        validation_session="S2",
        final_test_session="S3",
        window_sec=2.0,
        device="cuda",
    )


def test_target_subject_is_excluded_from_population_subjects() -> None:
    module = _module()
    plan = _plan(module, target=7)

    assert plan.population_subjects == tuple(
        subject for subject in range(1, 16) if subject != 7
    )
    assert 7 not in plan.population_subjects


def test_subject_plan_order_is_stable() -> None:
    module = _module()
    args = module.build_parser().parse_args(
        [
            "--subjects",
            "3",
            "1",
            "2",
            "--labram-checkpoint",
            "labram.pth",
            "--cbramod-checkpoint",
            "cbramod.pth",
            "--dry-run",
        ]
    )
    plans = module.build_plans(args)

    assert [plan.target_subject for plan in plans] == [1, 2, 3]
    assert plans[0].population_subjects == (2, 3)
    assert plans[1].population_subjects == (1, 3)
    assert plans[2].population_subjects == (1, 2)


def test_subject_output_paths_preserve_seed_identity() -> None:
    module = _module()
    plan = _plan(module, target=1)
    commands = {command.name: command.argv for command in plan.commands}

    assert plan.output_dir.as_posix() == "experiments/seed_loso/subject_01"
    assert len(commands) == 6
    assert Path(_arg_after(commands["train_labram"], "--output")).as_posix() == (
        "experiments/seed_loso/subject_01/labram/head.pt"
    )
    assert Path(_arg_after(commands["export_labram"], "--output")).as_posix() == (
        "experiments/seed_loso/subject_01/labram/package"
    )
    assert Path(
        _arg_after(commands["train_cbramod"], "--output-head")
    ).as_posix() == "experiments/seed_loso/subject_01/cbramod/head.pt"
    assert Path(_arg_after(commands["export_cbramod"], "--output")).as_posix() == (
        "experiments/seed_loso/subject_01/cbramod/package"
    )
    assert Path(
        _arg_after(commands["evaluate_labram"], "--output-dir")
    ).as_posix() == "experiments/seed_loso/subject_01/labram/evaluation"


def test_dry_run_prints_commands_without_executing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()

    def unexpected_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run must not execute subprocesses")

    monkeypatch.setattr(module.subprocess, "run", unexpected_run)
    assert module.main(
        [
            "--subjects",
            "1",
            "2",
            "--labram-checkpoint",
            "labram.pth",
            "--cbramod-checkpoint",
            "cbramod.pth",
            "--dry-run",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "subject_01 population=2" in output
    assert "subject_02 population=1" in output
    assert "Dry-run complete: 2 subjects, no commands executed." in output


def _arg_after(argv: tuple[str, ...], flag: str) -> str:
    return argv[argv.index(flag) + 1]
